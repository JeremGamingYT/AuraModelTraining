"""
Causal graph construction and reasoning.

This module provides a rudimentary implementation of the causal layer
described in **Part IV** of the blueprint.  It builds a directed
acyclic graph (DAG) representing cause–effect relations inferred from
the knowledge graph.  In a production system you would use
sophisticated causal discovery algorithms (e.g. from the ``causal-learn``
or ``dowhy`` libraries) and temporal information to infer causal
structure; here we implement a simple heuristic: any triple whose
predicate contains one of a set of causal keywords (``"cause"``,
``"because"``, ``"leads_to"``, ``"results_in"``) is treated as a causal
edge.

The resulting ``CausalGraph`` can be queried for causal paths and used
to perform naive counterfactual reasoning by virtually “removing” a
cause and observing reachable effects.  These operations are
illustrative and do not substitute for rigorous SCM analysis.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

try:
    import networkx as nx  # type: ignore[import]
except ImportError:  # pragma: no cover
    raise ImportError(
        "The causal graph requires the 'networkx' package.  Please install networkx to use this module."
    )

from .entity_extractor import Triple

logger = logging.getLogger(__name__)


class CausalGraph:
    """Simple directed acyclic graph of causal relations."""

    CAUSAL_KEYWORDS = {"cause", "causes", "because", "leads_to", "results_in", "induces"}

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def build(self, triples: List[Triple]) -> None:
        """Build the causal graph from a list of triples.

        This method populates the graph by adding directed edges for
        triples whose predicate contains one of the predefined causal
        keywords.  Nodes are created for all subjects and objects.
        """
        for triple in triples:
            # Normalise predicate
            predicate = triple.predicate.lower()
            if any(kw in predicate for kw in self.CAUSAL_KEYWORDS):
                self.graph.add_node(triple.subject)
                self.graph.add_node(triple.object)
                self.graph.add_edge(triple.subject, triple.object, predicate=triple.predicate)
        # Remove cycles if present (should not happen for pure cause–effect)
        if not nx.is_directed_acyclic_graph(self.graph):
            try:
                # Create a DAG by removing cycles (keep one direction)
                self.graph = nx.DiGraph(nx.DiGraph(self.graph).to_directed())  # type: ignore
            except Exception:
                logger.warning("Causal graph contains cycles; please verify predicates and data ordering.")

    def causal_path(self, source: str, target: str) -> List[List[str]]:
        """Return a list of causal paths from ``source`` to ``target``.

        Each path is represented as a list of node identifiers.  If
        no path exists, an empty list is returned.
        """
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target))
            return paths
        except nx.NetworkXNoPath:
            return []

    def counterfactual(self, removed_node: str) -> List[Tuple[str, str]]:
        """Perform a naive counterfactual query.

        Remove the specified cause node from the graph and return a
        list of nodes (effects) that are no longer reachable from the
        removed node.  This simulates asking “what if this cause
        did not occur?”.  The return value is a list of tuples
        (effect, path) where ``path`` is a string representation of
        the original path via which the effect was influenced.

        Note that this is a simplistic implementation and does not
        account for confounders or multiple causal paths.
        """
        effects: List[Tuple[str, str]] = []
        for target in self.graph.nodes:
            if target == removed_node:
                continue
            paths = list(nx.all_simple_paths(self.graph, removed_node, target))
            for path in paths:
                effects.append((target, ' -> '.join(path)))
        return effects