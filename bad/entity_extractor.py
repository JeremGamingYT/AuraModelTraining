"""
Entity and relation extraction.

This module implements **Step 3** of the blueprint: extracting
subject–predicate–object triples from document chunks according to a
predefined schema.  The extractor can operate in two modes:

* **LLM‑assisted extraction** – When the OpenAI API is available and
  configured, the extractor uses a language model to parse natural
  language text and return structured triples in JSON.  The model is
  guided by the provided ``OntologySchema`` to ensure that only
  expected entity and relation types are emitted.

* **Heuristic extraction** – In the absence of an API key, a very
  simple pattern matcher attempts to identify triples by scanning for
  capitalised noun phrases and verbs.  This fallback is intended
  merely as a placeholder; for production use the LLM‑assisted
  extraction is strongly recommended.

Example usage:

.. code-block:: python

    from blueprint_ai import DocumentIngestor, SchemaGenerator, EntityRelationExtractor

    ingestor = DocumentIngestor()
    chunks = ingestor.ingest("notes.txt")
    schema = SchemaGenerator().generate_schema(chunks)
    extractor = EntityRelationExtractor(schema)
    triples = extractor.extract(chunks)
    for t in triples:
        print(t.subject, t.predicate, t.object)

Note that the extractor returns a flat list of triples.  If you need
to keep track of the source chunk for each triple (for example,
assigning provenance metadata), you can inspect the ``metadata``
attribute on each ``Triple`` instance.
"""

from __future__ import annotations

import json
import os
import re
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .document_ingestor import DocumentChunk
from .schema_generator import OntologySchema

logger = logging.getLogger(__name__)

try:
    import openai  # type: ignore[import]
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False


@dataclass
class Triple:
    """Represents a subject–predicate–object statement.

    Attributes
    ----------
    subject:
        Canonical name of the subject entity.
    predicate:
        Name of the relation connecting the subject and object.
    object:
        Canonical name of the object entity.
    metadata:
        Optional dictionary containing provenance information (e.g.
        source chunk indices, confidence scores).
    """
    subject: str
    predicate: str
    object: str
    metadata: Optional[dict] = None


class EntityRelationExtractor:
    """Extract triples from document chunks.

    Parameters
    ----------
    schema: OntologySchema
        The schema defining allowed entity and relation types.  If
        provided, the extractor will instruct the language model to
        filter triples according to this schema.
    model: str, optional
        The OpenAI model to use for extraction.  Defaults to
        ``"gpt-3.5-turbo"``.  Ignored when no API key is configured.
    """

    def __init__(self, schema: OntologySchema, model: str = "gpt-3.5-turbo") -> None:
        self.schema = schema
        self.model = model
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                openai.api_key = api_key

    def extract(self, chunks: Iterable[DocumentChunk]) -> List[Triple]:
        """Extract triples from each document chunk.

        Parameters
        ----------
        chunks:
            An iterable of ``DocumentChunk`` objects.

        Returns
        -------
        List[Triple]
            A list of extracted triples.
        """
        triples: List[Triple] = []
        for idx, chunk in enumerate(chunks):
            text = chunk.content
            if not text:
                continue
            if OPENAI_AVAILABLE and openai.api_key:
                triples.extend(self._extract_with_openai(text, chunk, idx))
            else:
                triples.extend(self._extract_with_heuristics(text, chunk, idx))
        return triples

    def _extract_with_openai(self, text: str, chunk: DocumentChunk, idx: int) -> List[Triple]:
        """Use the OpenAI API to extract triples from the text.

        The prompt instructs the model to return a JSON array of
        triples, each with ``subject``, ``predicate`` and ``object``
        keys.  The schema is included in the prompt to constrain
        entity and relation names.
        """
        schema_prompt = ''
        if self.schema.entity_types or self.schema.relation_types:
            schema_prompt += "Entity types: " + ', '.join(sorted(self.schema.entity_types)) + "\n"
            schema_prompt += "Relation types: " + ', '.join(sorted(self.schema.relation_types)) + "\n"

        prompt = (
            "You are an information extraction system.  Given the following text, "
            "extract factual statements as a list of triples.  Each triple "
            "must have a 'subject', 'predicate' and 'object'.  Use only the "
            "entity and relation types listed below.  If a statement does not "
            "fit the schema, skip it.  Return only JSON.\n\n"
            f"{schema_prompt}\nText:\n{text}"
        )
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message['content']  # type: ignore[index]
            data = json.loads(content)
            triples: List[Triple] = []
            for item in data:
                subj = item.get('subject') or item.get('s')
                pred = item.get('predicate') or item.get('p')
                obj = item.get('object') or item.get('o')
                if not subj or not pred or not obj:
                    continue
                triples.append(
                    Triple(
                        subject=str(subj).strip(),
                        predicate=str(pred).strip(),
                        object=str(obj).strip(),
                        metadata={'chunk_index': idx},
                    )
                )
            return triples
        except Exception as e:
            logger.warning("OpenAI extraction failed, falling back to heuristics: %s", e)
            return self._extract_with_heuristics(text, chunk, idx)

    def _extract_with_heuristics(self, text: str, chunk: DocumentChunk, idx: int) -> List[Triple]:
        """Naively extract triples using regex heuristics.

        This fallback method scans for patterns of the form
        ``<ProperNoun> <verb> <ProperNoun>`` and uses these as
        ``subject predicate object``.  It makes no attempt to assign
        predicates to the schema; instead it simply lowercases the
        verb.  The extracted names are stripped of surrounding
        whitespace and capitalisation is preserved for entity names.
        """
        triples: List[Triple] = []
        # Split into sentences roughly on punctuation
        sentences = re.split(r'[\.!?]\s+', text)
        # Allow connectors inside multi-word names (e.g., "Modèles de Langage") and accented letters
        name_token = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9\-]*"
        connectors = r"(?:de|du|des|d['’]|of|the|and|et|la|le|les|l['’]|van|von|da|di|do)"
        name = rf"{name_token}(?:\s+(?:{connectors})\s+{name_token}|\s+{name_token})*"
        verb = r"[A-Za-zéèêëïîàâçùû]+"
        pattern = re.compile(rf"\b({name})\s+({verb})\s+({name})\b")
        for sentence in sentences:
            for match in pattern.finditer(sentence):
                subj, verb, obj = match.groups()
                triples.append(
                    Triple(
                        subject=subj.strip(),
                        predicate=verb.lower().replace(' ', '_'),
                        object=obj.strip(),
                        metadata={'chunk_index': idx},
                    )
                )
        return triples