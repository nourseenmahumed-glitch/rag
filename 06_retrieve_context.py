"""
06_retrieve_context.py
=========================
Step 6 of the RAG pipeline: context retrieval.

Given a user query, embeds it and retrieves the Top-K most similar chunks
from the ChromaDB collection, returning them with similarity scores and
full provenance metadata for citation display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from chromadb.api.models.Collection import Collection

from config import TOP_K
from utils import logger, similarity_from_distance, timed


class RetrievalError(Exception):
    """Raised when retrieval fails (empty index, query failure, etc.)."""


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its similarity score and metadata."""

    rank: int
    text: str
    source: str
    page_number: int
    chunk_index: int
    similarity: float


@timed("Context Retrieval")
def retrieve_top_k(
    query: str,
    collection: Collection,
    embed_query_fn,
    top_k: int = TOP_K,
) -> List[RetrievedChunk]:
    """Retrieve the top-k most relevant chunks for a query.

    Parameters
    ----------
    query:
        The user's natural-language question.
    collection:
        An active Chroma ``Collection`` (see ``05_create_chroma_store.py``).
    embed_query_fn:
        Callable that embeds a single query string (see
        ``04_vector_representation.embed_query``). Passed in explicitly to
        keep this module decoupled from a specific embedding backend.
    top_k:
        Number of chunks to retrieve.

    Returns
    -------
    List of ``RetrievedChunk``, ranked best-first.
    """
    if not query or not query.strip():
        raise RetrievalError("Query cannot be empty.")

    try:
        count = collection.count()
    except Exception as exc:
        logger.error("Could not read collection count: %s", exc)
        raise RetrievalError(f"The knowledge base is unavailable: {exc}") from exc

    if count == 0:
        raise RetrievalError(
            "The knowledge base is empty. Please upload and process a PDF first."
        )

    effective_k = min(top_k, count)

    try:
        query_vector = embed_query_fn(query)
    except Exception as exc:
        logger.error("Failed to embed query: %s", exc)
        raise RetrievalError(f"Failed to process your question: {exc}") from exc

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("Chroma query failed: %s", exc)
        raise RetrievalError(f"Retrieval from the knowledge base failed: {exc}") from exc

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: List[RetrievedChunk] = []
    for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        retrieved.append(
            RetrievedChunk(
                rank=rank,
                text=doc,
                source=meta.get("source", "unknown"),
                page_number=meta.get("page_number", -1),
                chunk_index=meta.get("chunk_index", -1),
                similarity=similarity_from_distance(dist),
            )
        )

    logger.info("Retrieved %d chunk(s) for query: %r", len(retrieved), query[:80])
    return retrieved
