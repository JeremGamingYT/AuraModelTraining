"""
Document ingestion and segmentation.

This module implements **Step 1** of the pipeline outlined in the
documentation: *Analyse, nettoyage et segmentation du document*.  It
supports loading textual data from plain text, PDF files or string
buffers, cleaning it and splitting it into coherent semantic units
("chunks").  These chunks are later processed by the schema
generation and extraction modules.

The ingestion process is deliberately conservative – it does not
attempt to interpret the contents.  Its sole responsibility is to
produce a list of ``DocumentChunk`` instances containing raw text and
optional metadata about their position in the source document.  Any
further semantic analysis (entity recognition, relation extraction,
etc.) should happen in later stages of the pipeline.

Example usage:

.. code-block:: python

    from blueprint_ai import DocumentIngestor

    ingestor = DocumentIngestor()
    chunks = ingestor.ingest("/path/to/report.pdf")
    for chunk in chunks:
        print(chunk.content)

The ingestor relies on a handful of external libraries for PDF
parsing (``pdfminer.six``) and HTML stripping (``beautifulsoup4``).
These are optional; if they are not present, the ingestor will fall
back to simpler extraction methods and warn the user.
"""

from __future__ import annotations

import io
import os
import re
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# Optional dependencies for PDF and HTML extraction.  The user is
# expected to install these on their own system; the code will
# gracefully degrade if they are absent.
try:
    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore[import]
except ImportError:  # pragma: no cover
    pdf_extract_text = None

try:
    from bs4 import BeautifulSoup  # type: ignore[import]
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A semantic chunk of a document.

    Attributes
    ----------
    content:
        The raw textual content of the chunk.
    start_pos:
        Byte offset of the first character of the chunk in the
        original source.  This can be ``None`` if unknown.
    end_pos:
        Byte offset of the character immediately following the last
        character of the chunk in the original source.  This can be
        ``None`` if unknown.
    metadata:
        Arbitrary additional metadata associated with the chunk.  This
        can include page numbers, section headings, etc.
    """

    content: str
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        # Strip leading/trailing whitespace.  We keep internal line
        # breaks intact to preserve paragraph structure.
        self.content = self.content.strip()


class DocumentIngestor:
    """Loads and segments textual documents into manageable chunks.

    The default segmentation strategy splits documents at blank lines
    (two or more consecutive newline characters) and ensures that
    segments do not exceed a configurable maximum length.  For
    languages with clearly defined sentence boundaries (such as
    English and French), this heuristic usually produces coherent
    paragraphs.  Users can provide a custom segmentation function via
    the ``segmenter`` argument.
    """

    def __init__(self, max_chunk_length: int = 1000) -> None:
        """
        Parameters
        ----------
        max_chunk_length: int, optional
            The maximum number of characters per chunk.  If a
            paragraph exceeds this length it will be further split
            around sentence boundaries.
        """
        self.max_chunk_length = max_chunk_length

    def ingest(self, source: str | bytes, *, mime_type: Optional[str] = None) -> List[DocumentChunk]:
        """Ingest a document from a file path, bytes object or raw string.

        Parameters
        ----------
        source:
            Either a path to a file on disk, a bytes buffer
            containing file data, or a plain text string.  If a path
            is provided, its extension is used to infer the file type.
        mime_type: str, optional
            Explicit MIME type (e.g. ``"application/pdf"`` or
            ``"text/html"``).  If omitted, the ingestor will try to
            infer the type from the file extension or content.

        Returns
        -------
        List[DocumentChunk]
            A list of semantic chunks extracted from the document.
        """
        if isinstance(source, bytes):
            data = source
            path = None
        elif os.path.isfile(source):  # type: ignore[arg-type]
            path = source  # type: ignore[assignment]
            with open(source, 'rb') as f:  # type: ignore[path-like]
                data = f.read()
        else:
            # Assume raw text string
            text = str(source)
            return self._segment_text(text)

        # Determine MIME type from extension if not provided
        if mime_type is None and path is not None:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.pdf':
                mime_type = 'application/pdf'
            elif ext in {'.html', '.htm'}:
                mime_type = 'text/html'
            else:
                mime_type = 'text/plain'

        # Route to appropriate extractor
        if mime_type == 'application/pdf':
            text = self._extract_pdf(data)
        elif mime_type == 'text/html':
            text = self._extract_html(data)
        else:
            text = data.decode('utf-8', errors='ignore')

        return self._segment_text(text)

    def _extract_pdf(self, data: bytes) -> str:
        """Extract text from a PDF document.

        This method uses ``pdfminer.six`` when available.  If the
        dependency is missing it will fall back to a very naive
        extraction: reading raw bytes and decoding them as UTF‑8.  This
        fallback is unlikely to yield usable results for real PDFs and
        should be considered a last resort.
        """
        if pdf_extract_text is not None:
            try:
                with io.BytesIO(data) as buff:
                    return pdf_extract_text(buff)
            except Exception as e:
                logger.warning("Failed to extract PDF with pdfminer: %s", e)
        logger.warning(
            "pdfminer.six is not installed; using fallback PDF extractor. "
            "Results may be poor."
        )
        # naive fallback: attempt to decode bytes directly
        return data.decode('utf-8', errors='ignore')

    def _extract_html(self, data: bytes) -> str:
        """Extract text from an HTML document.

        Uses ``BeautifulSoup`` if available.  Without it the method
        strips simple tags using a regular expression.  Users are
        encouraged to install ``beautifulsoup4`` for robust HTML
        handling.
        """
        html = data.decode('utf-8', errors='ignore')
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text(separator='\n')
        logger.warning(
            "beautifulsoup4 is not installed; using regex based HTML cleaner."
        )
        # Strip tags using a naive regex; this may leave scripts/style.
        text = re.sub(r'<[^>]+>', '', html)
        return text

    def _segment_text(self, text: str) -> List[DocumentChunk]:
        """Split raw text into a list of ``DocumentChunk`` objects.

        Paragraphs are delimited by two or more newlines.  Very long
        paragraphs are further split at sentence boundaries using a
        simple period based heuristic.  This segmentation strategy is
        language agnostic but works best on languages with periods
        indicating sentence ends.
        """
        # Normalise line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        paragraphs = re.split(r'\n\s*\n', text)
        chunks: List[DocumentChunk] = []
        offset = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                offset += len(para) + 2  # skip blank lines
                continue
            # Further split long paragraphs
            while len(para) > self.max_chunk_length:
                # Find the last period within max_chunk_length
                split_at = para.rfind('.', 0, self.max_chunk_length)
                if split_at == -1:
                    # No period; hard split at max_chunk_length
                    split_at = self.max_chunk_length
                segment, para = para[:split_at + 1], para[split_at + 1:].lstrip()
                end_pos = offset + len(segment)
                chunks.append(DocumentChunk(segment, start_pos=offset, end_pos=end_pos))
                offset = end_pos
            end_pos = offset + len(para)
            chunks.append(DocumentChunk(para, start_pos=offset, end_pos=end_pos))
            offset = end_pos + 2  # account for removed blank lines
        return chunks