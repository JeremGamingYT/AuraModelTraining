"""
Symbolic reasoning engine.

This module contains the "System 2" of the architecture: a
deterministic, verifiable reasoning engine operating over the
knowledge graph.  Its responsibilities include answering multi‑hop
queries, performing simple logical deductions based on ontological
rules, and interfacing with the causal graph for "why" questions.

At present the ``ReasoningEngine`` supports the following operations:

* **Path finding** – discover one or more chains of relations linking
  two entities.
* **Predicate search** – find all (subject, object) pairs connected by a
  given relation.
* **Entity type search** – retrieve all entities belonging to a given
  type (requires ``rdf:type`` triples in the graph).
* **Causal queries** – delegate to a ``CausalGraph`` instance to
  compute cause–effect paths and simple counterfactuals.

The engine is deliberately conservative: it performs no probabilistic
reasoning and returns all possible answers.  It is up to the
``NeuralEnvelope`` or higher layers to prioritise or summarise results
for the end user.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Optional

from .knowledge_graph import KnowledgeGraph
from .causal_graph import CausalGraph

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """Perform logical and causal reasoning over knowledge graphs."""

    def __init__(self, kg: KnowledgeGraph, cg: Optional[CausalGraph] = None) -> None:
        self.kg = kg
        self.cg = cg

    # ----- Multi‑hop reasoning over the knowledge graph -----
    def find_paths(self, start: str, end: str, max_hops: int = 3) -> List[List[Tuple[str, str, str]]]:
        """Return all relation chains between two entities up to ``max_hops``.

        If the underlying KG uses rdflib, this method currently
        raises ``NotImplementedError``.  You can instead formulate a
        SPARQL query via ``execute_sparql``.
        """
        try:
            return self.kg.multi_hop_paths(start, end, max_hops=max_hops)
        except NotImplementedError as e:
            raise NotImplementedError(
                "Path finding is only available for the networkx backend. "
                "Use SPARQL queries with rdflib instead."
            ) from e

    def find_by_predicate(self, predicate: str) -> List[Tuple[str, str]]:
        """Return all (subject, object) pairs connected by the given relation."""
        results: List[Tuple[str, str]] = []
        if self.kg.backend == 'rdflib':
            q = f"SELECT ?s ?o WHERE {{ ?s <http://example.org/relation/{predicate}> ?o . }}"
            for row in self.kg.query(q):
                s = row['s']
                o = row['o']
                results.append((s, o))
        else:
            # networkx
            for u, v, key in self.kg.graph.edges(keys=True):  # type: ignore[attr-defined]
                if key == predicate:
                    results.append((u, v))
        return results

    def find_entities_of_type(self, entity_type: str) -> List[str]:
        """Return all entities declared as instances of ``entity_type``.

        This relies on triples of the form ``(entity, rdf:type, entity_type)``.
        The entity type should be provided using the same naming
        convention as in the ontology (e.g. ``Person``).
        """
        results: List[str] = []
        if self.kg.backend == 'rdflib':
            q = (
                "SELECT ?s WHERE { ?s a <http://example.org/ontology/" + entity_type + "> . }"
            )
            for row in self.kg.query(q):
                results.append(row['s'])
        else:
            # networkx: entity types are stored as edges with predicate 'rdf:type'
            for u, v, key in self.kg.graph.edges(keys=True):  # type: ignore[attr-defined]
                if key == 'rdf:type' and v == entity_type:
                    results.append(u)
        return results

    # ----- Causal queries -----
    def causal_paths(self, cause: str, effect: str) -> List[List[str]]:
        """Return causal paths from cause to effect.  Requires a CausalGraph."""
        if not self.cg:
            raise ValueError("No causal graph provided")
        return self.cg.causal_path(cause, effect)

    def counterfactual_effects(self, removed_cause: str) -> List[Tuple[str, str]]:
        """Return the effects lost when removing a cause node."""
        if not self.cg:
            raise ValueError("No causal graph provided")
        return self.cg.counterfactual(removed_cause)

    # ----- SPARQL execution -----
    def execute_sparql(self, query: str) -> List[Dict[str, str]]:
        """Execute a SPARQL query against the knowledge graph."""
        return self.kg.query(query)