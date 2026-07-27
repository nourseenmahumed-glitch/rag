"""
02_preprocessing.py
=====================
Step 2 of the RAG pipeline: text cleaning.

Raw text extracted from PDFs is noisy: broken hyphenation across line
breaks, repeated headers/footers, stray page numbers, multiple blank lines,
and inconsistent whitespace. This module normalizes that text before it
is handed to the chunker.
"""

from __future__ import annotations

import re
from typing import List

from utils import logger, timed

# --------------------------------------------------------------------------- #
# Regex patterns (compiled once)
# --------------------------------------------------------------------------- #
_HYPHEN_LINEBREAK = re.compile(r"(\w+)-\n(\w+)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_STANDALONE_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_DOT_LEADER = re.compile(r"\.{4,}")


def clean_text(raw_text: str) -> str:
    """Apply a pipeline of normalization steps to a single block of text.

    The transformations are conservative: they remove obvious extraction
    noise without altering the substantive content of the document, since
    downstream answers must remain faithful to the source.
    """
    if not raw_text:
        return ""

    text = raw_text

    # Remove control / non-printable characters left over from PDF extraction.
    text = _NON_PRINTABLE.sub(" ", text)

    # Re-join words that were hyphenated across a line break, e.g.
    # "market-\ning" -> "marketing".
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # Collapse long dot-leaders often found in tables of contents.
    text = _MULTI_DOT_LEADER.sub(" ", text)

    # Drop lines that are just a standalone page number.
    text = _STANDALONE_PAGE_NUMBER.sub("", text)

    # Normalize whitespace.
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    # Trim trailing whitespace on each line, then strip the whole block.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    return text


def deduplicate_repeated_lines(pages_text: List[str], min_occurrences: int = 3) -> List[str]:
    """Strip lines that repeat verbatim across many pages (headers/footers).

    A line that appears near-identically on ``min_occurrences`` or more
    pages is almost certainly a running header/footer rather than content,
    so it is removed from every page.
    """
    if len(pages_text) < min_occurrences:
        return pages_text

    line_counts: dict[str, int] = {}
    per_page_lines = [p.split("\n") for p in pages_text]

    for lines in per_page_lines:
        seen_this_page = set()
        for line in lines:
            key = line.strip().lower()
            if len(key) < 4:  # too short to be meaningful noise
                continue
            if key not in seen_this_page:
                line_counts[key] = line_counts.get(key, 0) + 1
                seen_this_page.add(key)

    noisy_lines = {k for k, v in line_counts.items() if v >= min_occurrences}

    cleaned_pages = []
    for lines in per_page_lines:
        kept = [line for line in lines if line.strip().lower() not in noisy_lines]
        cleaned_pages.append("\n".join(kept))

    return cleaned_pages


@timed("Text Preprocessing")
def preprocess_page_documents(page_documents: List) -> List:
    """Clean a list of ``PageDocument`` objects in place and return them.

    First strips repeated running headers/footers across the whole
    document, then applies per-page normalization.
    """
    if not page_documents:
        return []

    raw_texts = [doc.text for doc in page_documents]
    deduped = deduplicate_repeated_lines(raw_texts)

    cleaned_count = 0
    for doc, deduped_text in zip(page_documents, deduped):
        cleaned = clean_text(deduped_text)
        if cleaned != doc.text:
            cleaned_count += 1
        doc.text = cleaned

    # Drop any pages that became empty after cleaning.
    non_empty = [doc for doc in page_documents if doc.text.strip()]

    logger.info(
        "Preprocessed %d page(s), %d modified, %d empty pages dropped",
        len(page_documents),
        cleaned_count,
        len(page_documents) - len(non_empty),
    )
    return non_empty
