"""
03_chunking.py
================
Step 3 of the RAG pipeline: adaptive chunking.

Splits cleaned page text into overlapping, semantically-coherent chunks
using LlamaIndex's ``SentenceSplitter``, which chunks on sentence
boundaries and only falls back to hard token cuts when a single sentence
exceeds the target chunk size. This produces more coherent, better-grounded
retrieval units than naive fixed-width splitting.

Chunk size adapts to document length: very short documents use smaller
chunks (so a 2-page PDF still yields several retrievable units), while
longer documents use the configured default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from llama_index.core.node_parser import SentenceSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from utils import logger, timed


@dataclass
class Chunk:
    """A single retrievable unit of text plus provenance metadata."""

    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


def _adaptive_chunk_size(total_chars: int) -> tuple[int, int]:
    """Scale chunk size down for very small documents, up for huge ones."""
    if total_chars < 3_000:
        return 256, 32
    if total_chars < 20_000:
        return 384, 48
    return CHUNK_SIZE, CHUNK_OVERLAP


@timed("Adaptive Chunking")
def chunk_page_documents(page_documents: List) -> List[Chunk]:
    """Chunk a list of cleaned ``PageDocument`` objects into ``Chunk`` objects.

    Each page is split independently so that chunk metadata (source, page
    number) stays accurate for citations.
    """
    if not page_documents:
        return []

    total_chars = sum(doc.char_count for doc in page_documents)
    chunk_size, chunk_overlap = _adaptive_chunk_size(total_chars)

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
    )

    chunks: List[Chunk] = []
    running_index = 0
    for doc in page_documents:
        try:
            pieces = splitter.split_text(doc.text)
        except Exception as exc:
            logger.warning(
                "Splitter failed on page %d of '%s' (%s); falling back to whole page",
                doc.page_number, doc.source, exc,
            )
            pieces = [doc.text]

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            running_index += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}::p{doc.page_number}::c{running_index}",
                    text=piece,
                    source=doc.source,
                    page_number=doc.page_number,
                    chunk_index=running_index,
                )
            )

    logger.info(
        "Created %d chunk(s) (chunk_size=%d, overlap=%d) from %d page(s)",
        len(chunks), chunk_size, chunk_overlap, len(page_documents),
    )
    return chunks
