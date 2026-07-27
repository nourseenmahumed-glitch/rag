"""
05_create_chroma_store.py
============================
Step 5 of the RAG pipeline: persistent vector store management.

Wraps ChromaDB's ``PersistentClient`` to create, populate, reuse, and reset
the knowledge base collection. The store is persisted to disk under
``chroma_db/`` so it survives across app restarts and is rebuilt only when
the user uploads a new document or explicitly resets it.
"""

from __future__ import annotations

from typing import List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION_NAME
from utils import logger, timed


class VectorStoreError(Exception):
    """Raised when the vector store cannot be created, written to, or read."""


_client: Optional[chromadb.ClientAPI] = None


def get_client() -> chromadb.ClientAPI:
    """Return a cached persistent ChromaDB client."""
    global _client
    if _client is None:
        try:
            _client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB client: %s", exc)
            raise VectorStoreError(f"Could not initialize the vector store: {exc}") from exc
    return _client


def collection_exists(collection_name: str = COLLECTION_NAME) -> bool:
    """Check whether a collection already exists and has at least one item."""
    try:
        client = get_client()
        names = [c.name for c in client.list_collections()]
        if collection_name not in names:
            return False
        collection = client.get_collection(collection_name)
        return collection.count() > 0
    except Exception as exc:
        logger.warning("Could not check collection existence: %s", exc)
        return False


def get_or_create_collection(collection_name: str = COLLECTION_NAME) -> Collection:
    """Fetch the collection, creating an empty one if it doesn't exist."""
    try:
        client = get_client()
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        logger.error("Failed to get/create collection '%s': %s", collection_name, exc)
        raise VectorStoreError(f"Could not access the vector store collection: {exc}") from exc


@timed("Vector Store Build")
def build_vector_store(
    chunks: List,
    embeddings: List[List[float]],
    collection_name: str = COLLECTION_NAME,
    reset_existing: bool = True,
) -> Collection:
    """Create (or rebuild) the Chroma collection from chunks + their embeddings.

    Parameters
    ----------
    chunks:
        List of ``Chunk`` objects (see ``03_chunking.py``).
    embeddings:
        Parallel list of embedding vectors, one per chunk.
    collection_name:
        Name of the Chroma collection to populate.
    reset_existing:
        If True, any existing collection with this name is deleted first
        (used when a *new* PDF replaces the knowledge base). If False, new
        chunks are appended to the existing collection (multi-document mode).
    """
    if not chunks:
        raise VectorStoreError("No chunks to index -- the document may be empty.")
    if len(chunks) != len(embeddings):
        raise VectorStoreError(
            f"Chunk/embedding count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
        )

    client = get_client()

    if reset_existing:
        try:
            existing = [c.name for c in client.list_collections()]
            if collection_name in existing:
                client.delete_collection(collection_name)
                logger.info("Deleted existing collection '%s' for rebuild.", collection_name)
        except Exception as exc:
            logger.warning("Could not delete existing collection (continuing): %s", exc)

    collection = get_or_create_collection(collection_name)

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source": chunk.source,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    try:
        # Chroma has a practical batch size limit; write in batches to be safe.
        batch_size = 200
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
    except Exception as exc:
        logger.error("Failed to write to vector store: %s", exc)
        raise VectorStoreError(f"Failed to build the vector store: {exc}") from exc

    logger.info(
        "Vector store '%s' now contains %d item(s).", collection_name, collection.count()
    )
    return collection


def reset_vector_store(collection_name: str = COLLECTION_NAME) -> None:
    """Delete the collection entirely (used by the 'Reset Knowledge Base' button)."""
    try:
        client = get_client()
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name)
            logger.info("Collection '%s' reset.", collection_name)
        else:
            logger.info("Collection '%s' did not exist; nothing to reset.", collection_name)
    except Exception as exc:
        logger.error("Failed to reset vector store: %s", exc)
        raise VectorStoreError(f"Failed to reset the knowledge base: {exc}") from exc


def get_collection_stats(collection_name: str = COLLECTION_NAME) -> dict:
    """Return simple stats about the current collection for the sidebar."""
    try:
        if not collection_exists(collection_name):
            return {"exists": False, "count": 0, "sources": []}
        collection = get_or_create_collection(collection_name)
        count = collection.count()
        sample = collection.get(limit=min(count, 5000), include=["metadatas"])
        sources = sorted({m.get("source", "unknown") for m in sample.get("metadatas", [])})
        return {"exists": True, "count": count, "sources": sources}
    except Exception as exc:
        logger.warning("Could not fetch collection stats: %s", exc)
        return {"exists": False, "count": 0, "sources": []}
