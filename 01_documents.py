"""
01_documents.py
================
Step 1 of the RAG pipeline: PDF ingestion.

Responsible for extracting raw text from uploaded PDF files, page by page,
and returning a normalized list of "document page" records that downstream
steps (cleaning, chunking) can consume regardless of the original file.

No cleaning or chunking happens here -- this module's only job is to get
text out of PDF bytes safely, with graceful error handling for corrupt,
encrypted, image-only, or empty PDFs.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import BinaryIO, List, Union

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from utils import logger, timed


class DocumentExtractionError(Exception):
    """Raised when a PDF cannot be parsed or contains no extractable text."""


@dataclass
class PageDocument:
    """A single extracted page of text plus provenance metadata."""

    source: str          # original file name
    page_number: int      # 1-indexed page number
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@timed("PDF Extraction")
def extract_pdf_text(
    file: Union[str, bytes, BinaryIO],
    source_name: str,
) -> List[PageDocument]:
    """Extract per-page text from a single PDF.

    Parameters
    ----------
    file:
        A file path, raw bytes, or a file-like object (e.g. Streamlit's
        ``UploadedFile``).
    source_name:
        Human-readable name of the document (used for citations later).

    Returns
    -------
    List of ``PageDocument`` objects, one per non-empty page.

    Raises
    ------
    DocumentExtractionError
        If the PDF is corrupt, encrypted without a usable password, or
        contains no extractable text at all (e.g. a scanned image PDF with
        no OCR layer).
    """
    try:
        if isinstance(file, (bytes, bytearray)):
            reader = PdfReader(io.BytesIO(file))
        else:
            reader = PdfReader(file)
    except PdfReadError as exc:
        logger.error("Failed to open PDF '%s': %s", source_name, exc)
        raise DocumentExtractionError(
            f"'{source_name}' could not be read. The file may be corrupted."
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error opening PDF '%s': %s", source_name, exc)
        raise DocumentExtractionError(
            f"'{source_name}' is not a valid PDF file."
        ) from exc

    if reader.is_encrypted:
        try:
            # Try an empty password first (common for "restricted" PDFs).
            reader.decrypt("")
        except Exception as exc:
            logger.error("PDF '%s' is password-protected: %s", source_name, exc)
            raise DocumentExtractionError(
                f"'{source_name}' is password-protected and cannot be read."
            ) from exc

    pages: List[PageDocument] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Could not extract text from page %d of '%s': %s", i, source_name, exc)
            raw_text = ""

        stripped = raw_text.strip()
        if stripped:
            pages.append(PageDocument(source=source_name, page_number=i, text=raw_text))

    if not pages:
        logger.error("No extractable text found in '%s'", source_name)
        raise DocumentExtractionError(
            f"'{source_name}' contains no extractable text. "
            "It may be a scanned/image-only PDF."
        )

    logger.info("Extracted %d text page(s) from '%s'", len(pages), source_name)
    return pages


@timed("Batch PDF Extraction")
def extract_multiple_pdfs(
    files: List[tuple[Union[str, bytes, BinaryIO], str]],
) -> List[PageDocument]:
    """Extract text from multiple PDFs, skipping any that fail (with a log).

    Parameters
    ----------
    files:
        List of ``(file, source_name)`` tuples.

    Returns
    -------
    Combined list of ``PageDocument`` across all successfully-parsed files.
    """
    all_pages: List[PageDocument] = []
    for file, source_name in files:
        try:
            all_pages.extend(extract_pdf_text(file, source_name))
        except DocumentExtractionError as exc:
            logger.warning("Skipping '%s': %s", source_name, exc)
            continue
    return all_pages
