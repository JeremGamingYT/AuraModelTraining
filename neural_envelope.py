"""
Neural language processing envelope.

The neural envelope acts as the "System 1" interface between the user
and the symbolic reasoning machinery.  It has two primary duties:

1. **Query parsing** – analyse the user's natural language question to
   identify the intent (fact retrieval, explanation, causal query,
   subjective judgement, etc.), extract the key entities mentioned and
   determine the required reasoning strategy.

2. **Answer synthesis** – once the reasoning engine has produced a
   structured answer (facts, chains of relations or causal
   explanations), convert it back into a fluid, natural language
   response suitable for the user.

The implementation provided here uses OpenAI's Chat API if available.
In the absence of an API key a set of heuristic rules performs
rudimentary parsing and answer synthesis.
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    import openai  # type: ignore[import]
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Represents a parsed natural language query.

    Attributes
    ----------
    original:
        The original question text.
    intent:
        High level category of the question (e.g. "fact", "why",
        "who", "subjective").
    entities:
        List of entity names extracted from the question.
    details:
        Additional freeform details useful for downstream reasoning
        (e.g. target property, time period).
    """

    original: str
    intent: str
    entities: List[str]
    details: Dict[str, str] = field(default_factory=dict)


class NeuralEnvelope:
    """A lightweight neural interface for query parsing and answer generation."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self.model = model
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                openai.api_key = api_key

    # --------------- Query parsing ---------------
    def parse_query(self, question: str) -> ParsedQuery:
        """Analyse a user's question and return a structured representation."""
        question = question.strip()
        if OPENAI_AVAILABLE and openai.api_key:
            return self._parse_with_openai(question)
        else:
            logger.info(
                "OpenAI API key not configured; using heuristic query parser"
            )
            return self._parse_with_heuristics(question)

    def _parse_with_openai(self, question: str) -> ParsedQuery:
        """Use the OpenAI API to parse a query into intent and entities."""
        prompt = (
            "You are a query understanding assistant.  Given a user's question, "
            "output a JSON object with keys 'intent' (one word describing the type "
            "of question: fact, why, who, subjective, list, compare), "
            "'entities' (list of named entities mentioned) and 'details' (object "
            "with any additional qualifiers such as time ranges or attributes).  "
            "Do not include any other fields.\n\nQuestion:\n" + question
        )
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message['content']  # type: ignore[index]
            import json
            data = json.loads(content)
            return ParsedQuery(
                original=question,
                intent=data.get('intent', 'fact'),
                entities=[str(e) for e in data.get('entities', [])],
                details={str(k): str(v) for k, v in data.get('details', {}).items()},
            )
        except Exception as e:
            logger.warning("OpenAI query parsing failed: %s", e)
            return self._parse_with_heuristics(question)

    def _parse_with_heuristics(self, question: str) -> ParsedQuery:
        """Basic fallback query parser using regex heuristics."""
        # Determine intent by looking for keywords
        q_lower = question.lower()
        intent = 'fact'
        if q_lower.startswith(('why', 'pourquoi')) or 'caus' in q_lower:
            intent = 'why'
        elif q_lower.startswith(('who', 'qui')):
            intent = 'who'
        elif any(word in q_lower for word in ['list', 'quels', 'quelles', 'montre', 'montrez']):
            intent = 'list'
        elif any(word in q_lower for word in ['compare', 'compar', 'difference']):
            intent = 'compare'
        elif any(word in q_lower for word in ['meilleur', 'good', 'best']) or 'bon' in q_lower:
            intent = 'subjective'
        # Extract capitalised entities, allowing common lowercase connectors inside names
        entity_token = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9\-]*"
        connectors = r"(?:de|du|des|d['’]|of|the|and|et|la|le|les|l['’]|van|von|da|di|do)"
        entity_pattern = rf"\b({entity_token}(?:\s+(?:{connectors})\s+{entity_token}|\s+{entity_token})*)\b"
        entities = re.findall(entity_pattern, question)
        # Remove leading determiners/articles
        cleaned: list[str] = []
        for e in entities:
            e = re.sub(r"^(?:Le|La|Les|L['’]|Un|Une|The|A|An)\s+", "", e).strip()
            if e and e not in cleaned:
                cleaned.append(e)
        entities = cleaned
        # Remove leading question word if misinterpreted as entity
        if entities and entities[0].lower() in {'why', 'who', 'what', 'where', 'when', 'comment', 'pourquoi', 'qui', 'que', 'quels', 'quelles'}:
            entities = entities[1:]
        # Detect inline document references like "@doc.txt"
        doc_refs = re.findall(r"@([^\s]+)", question)
        details: Dict[str, any] = {}
        if doc_refs:
            details['doc_refs'] = doc_refs
        return ParsedQuery(original=question, intent=intent, entities=entities, details=details)

    # --------------- Answer synthesis ---------------
    def generate_answer(self, parsed_query: ParsedQuery, results: Dict[str, any]) -> str:
        """Compose a natural language answer from structured reasoning results.

        Parameters
        ----------
        parsed_query:
            The structured representation returned by ``parse_query``.
        results:
            A dictionary containing the data needed to answer the
            question.  Its keys depend on the reasoning strategy.  For
            example:

            * ``facts`` – a list of triples (subject, predicate, object)
            * ``paths`` – a list of relation chains
            * ``causal_paths`` – a list of causal chains
            * ``counterfactuals`` – a list of (effect, path) tuples

        Returns
        -------
        str
            A natural language answer.
        """
        if OPENAI_AVAILABLE and openai.api_key:
            return self._generate_with_openai(parsed_query, results)
        else:
            return self._generate_with_templates(parsed_query, results)

    def _generate_with_openai(self, parsed_query: ParsedQuery, results: Dict[str, any]) -> str:
        """Use OpenAI to turn structured results into a natural answer."""
        # Compose a meta‑prompt describing the task
        prompt = "You are an assistant that turns structured data into a natural answer.\n"
        prompt += f"The user asked: {parsed_query.original}\n"
        prompt += "Here is the data extracted and inferred to answer the question in JSON: \n"
        import json
        prompt += json.dumps(results, indent=2)
        prompt += "\nCompose a concise and fluent answer in the same language as the question."
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message['content'].strip()  # type: ignore[index]
        except Exception as e:
            logger.warning("OpenAI answer synthesis failed: %s", e)
            return self._generate_with_templates(parsed_query, results)

    def _generate_with_templates(self, parsed_query: ParsedQuery, results: Dict[str, any]) -> str:
        """Fallback template‑based answer synthesis."""
        intent = parsed_query.intent
        # Fact retrieval: list statements
        if intent in {'fact', 'who', 'list'} and 'facts' in results:
            facts = results['facts']
            if not facts:
                return "Je n'ai trouvé aucun fait correspondant à votre question."
            lines = []
            for s, p, o in facts:
                lines.append(f"{s} {p.replace('_', ' ')} {o}.")
            return ' '.join(lines)
        # Multi‑hop reasoning: describe paths
        if intent in {'fact', 'who', 'list'} and 'paths' in results:
            paths = results['paths']
            if not paths:
                return "Je n'ai pas trouvé de relation entre ces entités."
            descriptions = []
            for path in paths:
                parts = [f"{subj} {pred.replace('_', ' ')} {obj}" for subj, pred, obj in path]
                descriptions.append(' puis '.join(parts))
            return ' ; '.join(descriptions)
        # Summary fallback from ingested document chunks
        if 'summary' in results:
            summary = results['summary']
            if not summary:
                return "Je n'ai pas pu extraire un résumé pertinent du document."
            return ' '.join(summary)
        # Causal explanation
        if intent == 'why' and 'causal_paths' in results:
            cpaths = results['causal_paths']
            if not cpaths:
                return "Je n'ai pas trouvé de relation causale pertinente."
            explanations = []
            for path in cpaths:
                explanations.append(' -> '.join(path))
            return "Voici les chaînes causales identifiées: " + ' ; '.join(explanations)
        # Counterfactuals
        if intent == 'why' and 'counterfactuals' in results:
            effects = results['counterfactuals']
            if not effects:
                return "Si la cause était supprimée, il n'y aurait pas de conséquences détectables."
            lines = []
            for eff, path in effects:
                lines.append(f"Sans cet événement, {eff} ne se produirait pas (chemin: {path})")
            return ' ; '.join(lines)
        # Fallback
        return "Je suis désolé, je n'ai pas pu formuler une réponse avec les informations fournies."