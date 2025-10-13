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

    def answer_question(self, question: str) -> str:
        """Answer a user question using the current knowledge base.

        The method parses the question via the neural envelope, decides
        which reasoning strategy to employ, executes the corresponding
        queries and synthesises a final answer.
        """
        parsed = self.neural_envelope.parse_query(question)
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
                    start, end = parsed.entities[0], parsed.entities[1]
                    paths = self.engine.find_paths(start, end, max_hops=3)
                    results['paths'] = paths
                elif len(parsed.entities) == 1:
                    # List all relations involving this entity
                    subj = parsed.entities[0]
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
                else:
                    results['facts'] = []
            else:
                # Subjective or other intents: return a placeholder
                results['facts'] = []
        except Exception as e:
            logger.error("Error during reasoning: %s", e)
        # Synthesize answer
        answer = self.neural_envelope.generate_answer(parsed, results)
        return answer