"""
04_vector_representation.py
=============================
Step 4 of the RAG pipeline: embedding generation.

Wraps ``sentence-transformers`` with the ``BAAI/bge-small-en-v1.5`` model to
turn chunk text (and later, user queries) into dense vector representations
suitable for similarity search in ChromaDB.

The model is loaded once per process (cached) since loading it is
comparatively expensive.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import BGE_QUERY_INSTRUCTION, EMBEDDING_MODEL_NAME
from utils import logger, timed


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load (and cache) the sentence-transformers embedding model."""
    logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        raise EmbeddingError(
            f"Could not load embedding model '{EMBEDDING_MODEL_NAME}'. "
            "Check your internet connection or model name."
        ) from exc
    logger.info("Embedding model loaded successfully.")
    return model


@timed("Embedding Generation (passages)")
def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed a batch of passage/chunk texts (no query instruction prefix)."""
    if not texts:
        return []
    try:
        model = _get_model()
        vectors: np.ndarray = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

    return vectors.tolist()


@timed("Embedding Generation (query)")
def embed_query(query: str) -> List[float]:
    """Embed a single user query, using BGE's recommended query instruction."""
    if not query or not query.strip():
        raise EmbeddingError("Cannot embed an empty query.")
    try:
        model = _get_model()
        instructed_query = f"{BGE_QUERY_INSTRUCTION}{query.strip()}"
        vector: np.ndarray = model.encode(
            instructed_query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        raise EmbeddingError(f"Failed to embed query: {exc}") from exc

    return vector.tolist()


def embedding_dimension() -> int:
    """Return the dimensionality of the loaded embedding model."""
    return _get_model().get_sentence_embedding_dimension()
