"""
Dynamic ontology generation.

This module encapsulates **Step 2** of the blueprint: generating a
schema or ontology specific to the context of the ingested document.
The goal of this step is to avoid producing an unstructured mass of
triples with heterogeneous predicates and entity types.  By
inferring a high–level model of the domain, subsequent extraction
and reasoning tasks become far more reliable and interpretable.

The default implementation leverages the OpenAI API to summarise the
contents of a document and propose a list of entity types and
relation types.  If no API key is available, the generator falls
back to a naive heuristic that scans the text for capitalised nouns
and simple verb phrases.

Example usage:

.. code-block:: python

    from blueprint_ai import DocumentIngestor, SchemaGenerator

    ingestor = DocumentIngestor()
    chunks = ingestor.ingest("paper.txt")
    schema = SchemaGenerator().generate_schema(chunks)
    print(schema.entity_types)
    print(schema.relation_types)

The returned ``OntologySchema`` can then be used to validate triples
extracted in later stages.
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Set, Tuple

from .document_ingestor import DocumentChunk

logger = logging.getLogger(__name__)

try:
    import openai  # type: ignore[import]
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False


@dataclass
class OntologySchema:
    """Represents the inferred ontology for a set of documents.

    Attributes
    ----------
    entity_types:
        A set of high–level entity type names (e.g. ``"Person"``,
        ``"Institution"``).
    relation_types:
        A set of predicate names describing relationships between
        entities (e.g. ``"has_inventor"``, ``"received_prize"``).
    """

    entity_types: Set[str] = field(default_factory=set)
    relation_types: Set[str] = field(default_factory=set)

    def add_entity(self, name: str) -> None:
        self.entity_types.add(name)

    def add_relation(self, name: str) -> None:
        self.relation_types.add(name)


class SchemaGenerator:
    """Generate an ontology from document chunks.

    The ``SchemaGenerator`` uses an LLM by default to infer a schema
    tailored to the contents of a document.  If OpenAI's API is
    available and the ``OPENAI_API_KEY`` environment variable is set,
    the generator will send a prompt containing concatenated document
    excerpts and ask the model to list entity and relation types.  If
    no API key is configured, it will fall back to a heuristic that
    extracts capitalised words as tentative entity types and simple
    verb phrases as relations.
    """

    def __init__(self, model: str = "gpt-3.5-turbo", max_chunk_chars: int = 8000):
        """
        Parameters
        ----------
        model: str, optional
            The OpenAI model to use.  Defaults to ``"gpt-3.5-turbo"``.
        max_chunk_chars: int, optional
            The maximum number of characters from the document to send
            to the LLM for schema inference.  Longer documents are
            truncated to fit within this limit.
        """
        self.model = model
        self.max_chunk_chars = max_chunk_chars

        if OPENAI_AVAILABLE:
            # Configure OpenAI API if a key is present.  We do not
            # raise an error if it isn't – the fallback will be used.
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                openai.api_key = api_key

    def generate_schema(self, chunks: Iterable[DocumentChunk]) -> OntologySchema:
        """Infer an ontology from the provided document chunks.

        Parameters
        ----------
        chunks:
            An iterable of ``DocumentChunk`` objects representing the
            document.  Only the textual content is used.

        Returns
        -------
        OntologySchema
            A schema containing inferred entity types and relation
            types.
        """
        # Concatenate chunk contents up to a maximum number of characters
        text = ''
        for chunk in chunks:
            if len(text) >= self.max_chunk_chars:
                break
            text += chunk.content + '\n'

        text = text.strip()
        if not text:
            return OntologySchema()

        if OPENAI_AVAILABLE and openai.api_key:
            return self._generate_with_openai(text)
        else:
            logger.info(
                "OpenAI API key not configured or openai library missing; "
                "falling back to heuristic schema generation"
            )
            return self._generate_with_heuristics(text)

    def _generate_with_openai(self, text: str) -> OntologySchema:
        """Invoke the OpenAI API to generate an ontology schema.

        The prompt instructs the model to return JSON with ``entity_types``
        and ``relation_types`` arrays.  The method includes basic error
        handling and falls back to heuristics in case of failure.
        """
        # Compose the prompt as a single string.  We avoid adjacent
        # string literals separated by newlines because that can lead to
        # syntax errors when the closing quote is not correctly
        # recognised.  The newline after "explanatory text." is
        # inserted manually.
        prompt = (
            "You are an ontology extraction engine.\n"
            "Given the following document text, identify high‑level entity types "
            "(e.g., Person, Organization, Prize) and relation types (e.g., "
            "developed_by, received_award) that describe relationships between "
            "these entities.\n"
            "Return your answer strictly as a JSON object with two arrays: "
            "'entity_types' and 'relation_types'. Do not include any explanatory text.\n\n"
            "Document:\n"
        ) + text
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = response.choices[0].message['content']  # type: ignore[index]
            import json
            data = json.loads(content)
            schema = OntologySchema(
                entity_types=set(data.get('entity_types', [])),
                relation_types=set(data.get('relation_types', [])),
            )
            return schema
        except Exception as e:
            logger.warning("OpenAI schema generation failed: %s", e)
            # Fall back to heuristic extraction
            return self._generate_with_heuristics(text)

    def _generate_with_heuristics(self, text: str) -> OntologySchema:
        """Generate a schema using simple regex heuristics.

        Capitalised words are treated as potential entity types and
        common verb phrases are treated as relation types.  The
        heuristic is intentionally broad – the downstream extraction
        module is responsible for validating triples against the schema.
        """
        # Extract candidate entity types: capitalised words that are
        # not sentence starters (avoid false positives by looking for
        # mid‑sentence capitalised nouns).
        candidate_entities: Set[str] = set()
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
            phrase = match.group(1).strip()
            # Filter out single words that are common stopwords
            if phrase.lower() in {'the', 'and', 'but', 'for', 'with', 'about', 'dans', 'les'}:
                continue
            candidate_entities.add(phrase)

        # Extract candidate relations: collect common verb forms.  We
        # adopt a simple heuristic by looking for French infinitive
        # endings (er, ir, re) and English past tense endings (ed)
        # within the text.  This is a crude approximation but
        # suffices to produce a non‑empty set of predicates for the
        # downstream extractor to refine.
        candidate_relations: Set[str] = set()
        for match in re.finditer(r'\b([A-Za-zéèêëïîàâçùû]+)\b', text, flags=re.IGNORECASE):
            word = match.group(1).lower()
            if word.endswith(('er', 'ir', 're')) or word.endswith('ed'):
                candidate_relations.add(word)

        # Normalise names: convert spaces to underscores and lowercase
        entity_types = {e.title().replace(' ', '_') for e in candidate_entities}
        relation_types = {r.lower().replace(' ', '_') for r in candidate_relations}
        return OntologySchema(entity_types=entity_types, relation_types=relation_types)