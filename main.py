"""
Example command‑line interface for the Blueprint AI system.

This script demonstrates how to use the various components exposed
through the ``blueprint_ai`` package to ingest documents and answer
questions.  It is not required to run the system, but provides a
convenient entry point for manual testing.

Usage::

    python -m blueprint_ai.main --ingest path/to/file.pdf
    python -m blueprint_ai.main --question "Pourquoi Einstein a‑t‑il reçu le prix Nobel ?"

When ingesting a document the script updates the internal knowledge
base.  Subsequent questions can then be answered without reloading
the document.  The knowledge base persists only for the duration of
the process; for long‑lived applications consider serialising the
graphs to disk using the methods on ``KnowledgeGraph``.
"""

import argparse
import sys

from .orchestrator import Orchestrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blueprint AI example CLI")
    parser.add_argument('--ingest', nargs='*', help='One or more documents to ingest')
    parser.add_argument('--question', help='Question to ask the system')
    args = parser.parse_args(argv)

    orchestrator = Orchestrator()
    # Ingest documents if any
    if args.ingest:
        for doc in args.ingest:
            orchestrator.ingest_document(doc)
    if args.question:
        answer = orchestrator.answer_question(args.question)
        print(answer)
    return 0


if __name__ == '__main__':
    sys.exit(main())