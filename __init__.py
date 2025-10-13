"""
blueprint_ai
===================

This package implements a prototype of the architecture described in
the document *« IA Révolutionnaire : Compréhension Profonde, Peu de
Données »*.  The goal of this library is to provide a modular and
functional implementation of a **post‑LLM** system that combines
causal reasoning, dynamic knowledge graphs and efficient neural
language processing.

The package is organised around the major components described in the
documentation:

* **Document ingestion** – cleaning and segmenting arbitrary text or
  PDF sources into manageable chunks.
* **Dynamic schema generation** – building a context‑specific
  ontology from the contents of a document using a language model.
* **Entity and relation extraction** – extracting subject/predicate
  triples from each chunk according to the generated schema.
* **Entity disambiguation** – linking extracted names to canonical
  identifiers (e.g. Wikidata) or maintaining an internal mapping.
* **Knowledge graph** – storing facts in an RDF graph and providing
  query capabilities.
* **Causal graph** – building a directed acyclic graph from temporal
  relations and performing simple causal inference.
* **Reasoning engine** – answering multi‑hop and logical queries
  against the knowledge base.
* **Neural envelope** – a lightweight language processor that
  translates natural language questions into formal queries and
  synthesises results into human‑readable answers.
* **Orchestrator** – a controller that coordinates all of the above
  to iteratively answer complex user questions.

Each submodule is intended to be self contained and can be used
independently.  See the documentation within each module for usage
examples.
"""

from .document_ingestor import DocumentIngestor, DocumentChunk
from .schema_generator import SchemaGenerator, OntologySchema
from .entity_extractor import EntityRelationExtractor, Triple
from .entity_disambiguator import EntityDisambiguator
from .knowledge_graph import KnowledgeGraph
from .causal_graph import CausalGraph
from .reasoning_engine import ReasoningEngine
from .neural_envelope import NeuralEnvelope, ParsedQuery
from .orchestrator import Orchestrator

__all__ = [
    "DocumentIngestor",
    "DocumentChunk",
    "SchemaGenerator",
    "OntologySchema",
    "EntityRelationExtractor",
    "Triple",
    "EntityDisambiguator",
    "KnowledgeGraph",
    "CausalGraph",
    "ReasoningEngine",
    "NeuralEnvelope",
    "ParsedQuery",
    "Orchestrator",
]