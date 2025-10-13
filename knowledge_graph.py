"""
Knowledge graph storage and query.

The knowledge graph (KG) is the central memory of the system.  It
stores factual triples and provides efficient querying capabilities
for multi‑hop reasoning.  This module attempts to use the
``rdflib`` library when available for standards‑compliant RDF storage
and SPARQL querying.  If ``rdflib`` is not installed, it falls back
to a lightweight implementation based on ``networkx`` directed graphs.

The KG is designed to be persistent – you can serialise it to disk
and reload it later – but persistence is optional and depends on
external libraries.  See the ``load`` and ``save`` methods for
details.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple, List, Dict, Any

from .entity_extractor import Triple

logger = logging.getLogger(__name__)

try:
    import rdflib  # type: ignore[import]
    from rdflib import Graph, URIRef, Literal, Namespace
    RDFLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    RDFLIB_AVAILABLE = False

try:
    import networkx as nx  # type: ignore[import]
    NETWORKX_AVAILABLE = True
except ImportError:  # pragma: no cover
    NETWORKX_AVAILABLE = False


class KnowledgeGraph:
    """A graph for storing and querying triples.

    When ``rdflib`` is available the graph uses its ``Graph`` class
    internally, enabling SPARQL queries and serialisation to standard
    formats (e.g. Turtle, N‑Triples).  Otherwise a NetworkX DiGraph is
    used with basic multi‑hop lookup capabilities.  In both cases the
    public API remains the same.
    """

    def __init__(self) -> None:
        if RDFLIB_AVAILABLE:
            self.backend = 'rdflib'
            self.graph: Graph = Graph()
            self.ns = Namespace("http://example.org/ontology/")
        elif NETWORKX_AVAILABLE:
            self.backend = 'networkx'
            self.graph = nx.MultiDiGraph()
        else:
            raise ImportError(
                "Neither rdflib nor networkx is installed; please install at least one."
            )

    def add_triples(self, triples: Iterable[Triple]) -> None:
        """Insert multiple triples into the graph.

        Parameters
        ----------
        triples:
            Iterable of ``Triple`` objects to add.
        """
        for triple in triples:
            self.add_triple(triple.subject, triple.predicate, triple.object)

    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        """Insert a single triple into the graph."""
        if self.backend == 'rdflib':
            s = URIRef(f"http://example.org/entity/{subject}")
            p = URIRef(f"http://example.org/relation/{predicate}")
            o = URIRef(f"http://example.org/entity/{obj}")
            self.graph.add((s, p, o))
        else:
            # networkx backend
            self.graph.add_node(subject)
            self.graph.add_node(obj)
            self.graph.add_edge(subject, obj, key=predicate)

    def query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """Execute a SPARQL query and return results.

        If the backend is NetworkX, a NotImplementedError is raised.
        """
        if self.backend != 'rdflib':
            raise NotImplementedError(
                "SPARQL querying requires rdflib; please install rdflib to use this feature."
            )
        results = []
        for row in self.graph.query(sparql_query):  # type: ignore[call-arg]
            record = {str(var): str(value) for var, value in row.asdict().items()}
            results.append(record)
        return results

    def multi_hop_paths(self, start: str, end: str, max_hops: int = 3) -> List[List[Tuple[str, str, str]]]:
        """Find paths between two entities up to a given number of hops.

        This method is available when the backend is NetworkX.  It
        returns a list of paths, each path being a list of triples
        (subject, predicate, object).  When using rdflib you should
        construct a SPARQL query instead.
        """
        if self.backend != 'networkx':
            raise NotImplementedError(
                "multi_hop_paths is only implemented for the networkx backend."
            )
        paths: List[List[Tuple[str, str, str]]] = []
        try:
            # Use breadth‑first search to find paths up to max_hops
            for target_path in nx.all_simple_edge_paths(self.graph, start, end, cutoff=max_hops):  # type: ignore[attr-defined]
                path: List[Tuple[str, str, str]] = []
                for u, v, key in target_path:
                    path.append((u, key, v))
                paths.append(path)
        except Exception as e:
            logger.debug("Error finding paths between %s and %s: %s", start, end, e)
        return paths

    def save(self, path: str, format: str = 'turtle') -> None:
        """Serialise the graph to a file.

        When using rdflib the ``format`` parameter controls the output
        serialisation (e.g. 'turtle', 'nt').  When using NetworkX,
        the graph is pickled using ``networkx.write_gpickle``.
        """
        if self.backend == 'rdflib':
            self.graph.serialize(destination=path, format=format)  # type: ignore[call-arg]
        else:
            nx.write_gpickle(self.graph, path)  # type: ignore[name-defined]

    def load(self, path: str, format: str = 'turtle') -> None:
        """Load a graph from a file, replacing the current contents.

        The behaviour mirrors ``save``.
        """
        if self.backend == 'rdflib':
            self.graph = Graph()
            self.graph.parse(path, format=format)  # type: ignore[call-arg]
        else:
            self.graph = nx.read_gpickle(path)  # type: ignore[name-defined]