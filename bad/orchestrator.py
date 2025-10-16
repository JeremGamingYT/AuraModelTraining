"""
Orchestrator for coordinating neural and symbolic components.

The orchestrator is responsible for gluing together the ingestion,
schema generation, extraction, reasoning and language modules into a
coherent workflow.  It exposes two high‑level methods:

* ``ingest_document`` – incorporate a new document into the knowledge
  base, building or extending the knowledge graph and causal graph.
* ``answer_question`` – answer a user's question by delegating to the
  neural envelope for query parsing, the reasoning engine for
  computing results and the neural envelope again for answer
  synthesis.

The orchestrator maintains internal state (knowledge graph and causal
graph) across calls.  This allows it to accumulate knowledge from
multiple documents and reuse it for subsequent questions.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
import os
import re

from .document_ingestor import DocumentIngestor, DocumentChunk
from .schema_generator import SchemaGenerator, OntologySchema
from .entity_extractor import EntityRelationExtractor, Triple
from .entity_disambiguator import EntityDisambiguator
from .knowledge_graph import KnowledgeGraph
from .causal_graph import CausalGraph
from .reasoning_engine import ReasoningEngine
from .neural_envelope import NeuralEnvelope, ParsedQuery

logger = logging.getLogger(__name__)


class Orchestrator:
    """High‑level coordinator for document ingestion and question answering."""

    def __init__(
        self,
        *,
        ingestor: Optional[DocumentIngestor] = None,
        schema_generator: Optional[SchemaGenerator] = None,
        neural_envelope: Optional[NeuralEnvelope] = None,
        disambiguator: Optional[EntityDisambiguator] = None,
    ) -> None:
        self.ingestor = ingestor or DocumentIngestor()
        self.schema_generator = schema_generator or SchemaGenerator()
        self.neural_envelope = neural_envelope or NeuralEnvelope()
        self.disambiguator = disambiguator or EntityDisambiguator()
        # Shared state across documents and questions
        self.ontology: OntologySchema = OntologySchema()
        self.kg = KnowledgeGraph()
        self.cg = CausalGraph()
        self.engine = ReasoningEngine(self.kg, self.cg)
        # Retain raw chunks for summarisation and abbreviation detection
        self.chunks: List[DocumentChunk] = []
        self.ingested_sources: set[str] = set()

    def ingest_document(self, source: str | bytes, *, mime_type: Optional[str] = None) -> None:
        """Ingest a new document and update the knowledge and causal graphs.

        Parameters
        ----------
        source:
            Path to the document or bytes buffer.
        mime_type: str, optional
            Explicit MIME type to override detection.
        """
        logger.info("Ingesting document: %s", source)
        chunks = self.ingestor.ingest(source, mime_type=mime_type)
        # Retain chunks for later summarisation
        self.chunks.extend(chunks)
        # Extract abbreviations/synonyms and preload into disambiguator
        synonyms = self._extract_abbreviations(chunks)
        if synonyms:
            self.disambiguator.add_synonyms(synonyms)
        # Update or generate the schema
        schema = self.schema_generator.generate_schema(chunks)
        # Merge with existing ontology
        self.ontology.entity_types.update(schema.entity_types)
        self.ontology.relation_types.update(schema.relation_types)
        # Extract triples
        extractor = EntityRelationExtractor(self.ontology)
        triples = extractor.extract(chunks)
        # Disambiguate entities
        disambiguated = self.disambiguator.disambiguate(triples)
        # Store in knowledge graph
        self.kg.add_triples(disambiguated)
        # Build or update causal graph (only newly added triples)
        self.cg.build(disambiguated)
        logger.info("Document ingestion complete: %d triples added", len(disambiguated))

    def _extract_abbreviations(self, chunks: List[DocumentChunk]) -> Dict[str, str]:
        """Extract (acronym -> full phrase) mappings from text chunks.

        Looks for patterns like "Grands Modèles de Langage (GML)" or
        "Large Language Models (LLMs)" and builds a mapping from the
        acronym to the expanded phrase. Also attempts to map common
        domain abbreviations found in the document (e.g., LLM -> Grands
        Modèles de Langage).
        """
        mapping: Dict[str, str] = {}
        pattern = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ][^()]{3,}?)\s*\(\s*([A-Z]{2,}s?)\s*\)")
        for chunk in chunks:
            for m in pattern.finditer(chunk.content):
                phrase = m.group(1).strip().strip('"\'')
                abbr = m.group(2).strip()
                # Singularise trivial plural abbreviations like LLMs -> LLM
                if abbr.endswith('s') and len(abbr) > 3:
                    base = abbr[:-1]
                else:
                    base = abbr
                mapping[base] = phrase
        # Heuristic: if the canonical phrase "Grands Modèles de Langage" appears,
        # ensure LLM/LLMs point to it.
        doc_text = '\n'.join(c.content for c in chunks).lower()
        if 'grands modèles de langage' in doc_text or 'large language models' in doc_text:
            mapping.setdefault('LLM', 'Grands Modèles de Langage')
            mapping.setdefault('LLMs', 'Grands Modèles de Langage')
        return mapping

    def answer_question(self, question: str) -> str:
        """Answer a user question using the current knowledge base.

        The method parses the question via the neural envelope, decides
        which reasoning strategy to employ, executes the corresponding
        queries and synthesises a final answer.
        """
        parsed = self.neural_envelope.parse_query(question)
        # If the user referenced local documents like "@doc.txt", ingest them on the fly
        doc_refs = parsed.details.get('doc_refs') if isinstance(parsed.details, dict) else None
        if doc_refs:
            for ref in doc_refs:
                try:
                    if os.path.isfile(ref) and ref not in self.ingested_sources:
                        self.ingest_document(ref)
                        self.ingested_sources.add(ref)
                except Exception:
                    continue
        logger.info("Parsed query: %s", parsed)
        results: Dict[str, Any] = {}
        try:
            if parsed.intent == 'why':
                # Causal explanation: find causal paths between the first two entities
                if len(parsed.entities) >= 2:
                    cause, effect = parsed.entities[0], parsed.entities[1]
                    cpaths = self.engine.causal_paths(cause, effect)
                    results['causal_paths'] = cpaths
                else:
                    results['causal_paths'] = []
            elif parsed.intent in {'fact', 'who', 'list'}:
                # Fact retrieval or multi‑hop: if two entities, find paths; else list facts
                if len(parsed.entities) >= 2:
                    start_raw, end_raw = parsed.entities[0], parsed.entities[1]
                    # Resolve to canonical identifiers used in the KG
                    start = self.disambiguator.resolve_name(start_raw)
                    end = self.disambiguator.resolve_name(end_raw)
                    paths = []
                    try:
                        paths = self.engine.find_paths(start, end, max_hops=3)
                    except NotImplementedError:
                        # Backend does not support path search; we'll fall back to listing facts
                        paths = []
                    results['paths'] = paths
                    # Fallback: if no paths, surface nearby facts for each entity
                    if not paths:
                        facts: list[tuple[str, str, str]] = []
                        for subj in {start, end}:
                            if self.kg.backend == 'rdflib':
                                q = f"SELECT ?p ?o WHERE {{ <http://example.org/entity/{subj}> ?p ?o . }}"
                                try:
                                    for row in self.kg.query(q):
                                        p_uri = row['p']
                                        pred_name = p_uri.split('/')[-1]
                                        obj_name = row['o'].split('/')[-1]
                                        facts.append((subj, pred_name, obj_name))
                                except Exception:
                                    continue
                            else:
                                try:
                                    for u, v, key in self.kg.graph.edges(subj, keys=True):  # type: ignore[attr-defined]
                                        facts.append((u, key, v))
                                except Exception:
                                    continue
                        # Limit to a manageable number of facts
                        if facts:
                            results['facts'] = facts[:20]
                        # If still nothing useful, build a textual summary from chunks
                        if not facts:
                            summary = self._summarise_from_chunks(parsed.original)
                            if summary:
                                results['summary'] = summary
                elif len(parsed.entities) == 1:
                    # List all relations involving this entity
                    subj_raw = parsed.entities[0]
                    subj = self.disambiguator.resolve_name(subj_raw)
                    facts = []
                    if self.kg.backend == 'rdflib':
                        q = f"SELECT ?p ?o WHERE {{ <http://example.org/entity/{subj}> ?p ?o . }}"
                        for row in self.kg.query(q):
                            p_uri = row['p']
                            # Extract predicate name
                            pred_name = p_uri.split('/')[-1]
                            facts.append((subj, pred_name, row['o'].split('/')[-1]))
                    else:
                        for u, v, key in self.kg.graph.edges(subj, keys=True):  # type: ignore[attr-defined]
                            facts.append((u, key, v))
                    results['facts'] = facts
                    if not facts:
                        summary = self._summarise_from_chunks(parsed.original)
                        if summary:
                            results['summary'] = summary
                else:
                    # No entities detected; attempt a direct summary matching
                    summary = self._summarise_from_chunks(parsed.original)
                    if summary:
                        results['summary'] = summary
                    else:
                        results['facts'] = []
            else:
                # Subjective or other intents: return a placeholder
                summary = self._summarise_from_chunks(parsed.original)
                if summary:
                    results['summary'] = summary
                else:
                    results['facts'] = []
        except Exception as e:
            logger.error("Error during reasoning: %s", e)
        # Synthesize answer
        answer = self.neural_envelope.generate_answer(parsed, results)
        return answer

    def _summarise_from_chunks(self, question: str, top_k: int = 3) -> List[str]:
        """Return top-k relevant paragraphs from ingested chunks for the question.

        A simple keyword scoring is used as a Phase 1-2 fallback when the KG
        lacks explicit paths. This aligns with the doc's recommendation to
        ensure useful responses in early phases.
        """
        if not self.chunks:
            return []
        text = question.lower()
        # Extract keywords: drop very short/common words
        tokens = re.findall(r"[a-zà-öø-ÿA-ZÀ-ÖØ-ß]{3,}", text, flags=re.IGNORECASE)
        stop = {
            'les','des','une','un','le','la','de','du','et','que','qui','quoi','pourquoi','comment',
            'the','and','for','with','why','what','when','where','quel','quels','quelles','aux','dans'
        }
        keywords = {t.lower() for t in tokens if t.lower() not in stop}
        if not keywords:
            return []
        scored: List[tuple[int, str]] = []
        for c in self.chunks:
            para = c.content.strip()
            if not para:
                continue
            lower = para.lower()
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scored.append((score, para))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]