# Blueprint AI — A Post‑LLM Architecture

This repository contains a reference implementation of the
architecture described in the French white paper *« IA
Révolutionnaire : Compréhension Profonde, Peu de Données »*.

The aim of this project is to build an intelligent system that
combines:

* **Document ingestion** — Cleanly extract and segment text from PDF,
  HTML or plain text sources into coherent chunks.
* **Dynamic schema generation** — Generate an ontology specific to the
  document using a language model or simple heuristics, preventing
  the proliferation of unstructured triples.
* **Entity and relation extraction** — Parse each chunk into
  subject–predicate–object triples that conform to the generated
  schema.
* **Entity disambiguation** — Resolve ambiguous names to canonical
  identifiers using the Wikidata API and an internal cache.
* **Knowledge graph** — Store facts in an RDF or directed graph
  structure, enabling SPARQL queries and multi‑hop traversals.
* **Causal graph** — Derive a directed acyclic graph of causal
  relations from the knowledge graph using simple heuristics.  This
  layer allows the system to answer “why” questions and perform
  counterfactual reasoning.
* **Reasoning engine** — Provide deterministic, verifiable reasoning
  over the knowledge and causal graphs.  Includes multi‑hop path
  finding, predicate search and causal queries.
* **Neural envelope** — A lightweight language processing wrapper
  around a language model (OpenAI GPT) that interprets user questions
  and synthesises natural language answers from structured results.
* **Orchestrator** — A controller that coordinates all of the above
  components to ingest new documents and answer questions iteratively.

## Layout

```
blueprint_ai/
├── document_ingestor.py       # Step 1: ingestion and segmentation
├── schema_generator.py        # Step 2: ontology/schema generation
├── entity_extractor.py        # Step 3: triple extraction
├── entity_disambiguator.py    # Step 4: entity linking
├── knowledge_graph.py         # Storage and SPARQL querying
├── causal_graph.py            # Causal DAG construction and queries
├── reasoning_engine.py        # Symbolic reasoning over the KG/causal graph
├── neural_envelope.py         # Neural interface for parsing and synthesis
├── orchestrator.py            # High‑level coordinator
├── main.py                    # CLI demonstration
└── README.md                  # This document
```

## Dependencies

The code has a modular set of dependencies.  At minimum you should
install:

```
pip install networkx requests
```

To unlock full functionality, including SPARQL queries and LLM
integration, install the optional packages as well:

```
pip install rdflib pdfminer.six beautifulsoup4 openai
```

If you intend to use the OpenAI API you must set the
`OPENAI_API_KEY` environment variable before running the code.

## Usage Example

```python
from blueprint_ai import Orchestrator

orch = Orchestrator()

# Ingest a document (PDF, HTML or plain text)
orch.ingest_document("path/to/report.pdf")

# Ask a question
answer = orch.answer_question(
    "Pourquoi le travail d'Einstein sur l'effet photoélectrique, et non la relativité, "
    "a-t-il été la raison principale de son prix Nobel ?"
)

print(answer)
```

You can also run the CLI directly:

```
python -m blueprint_ai.main --ingest path/to/report.pdf --question "Qui a publié la théorie de la relativité ?"
```

## Licence

This code is provided as a reference implementation for research
purposes.  You are free to modify and adapt it to your needs.