"""
RAG Module — Retrieval Augmented Generation
Uses SentenceTransformers (all-MiniLM-L6-v2) for embeddings + FAISS for vector search.

Design:
- Embeddings model loaded once at module level (singleton pattern)
- FAISS index loaded lazily on first query
- Swap embedding model by changing EMBEDDING_MODEL constant
"""

import json
import logging
import os
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Constants — easy to swap
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 90MB, CPU-friendly, ~384-dim embeddings
INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.json"
DEFAULT_TOP_K = 3

# ── Singletons ────────────────────────────────────────────────────────────────

_embedding_model: SentenceTransformer | None = None
_faiss_index: faiss.Index | None = None
_chunks: list[str] | None = None


def _get_model() -> SentenceTransformer:
    """Load embedding model once and reuse across requests."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_store() -> tuple[faiss.Index, list[str]]:
    """Load FAISS index and chunk list once and reuse across requests."""
    global _faiss_index, _chunks

    if _faiss_index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at '{INDEX_PATH}'. "
                "Run `python ingest.py` first to build the index."
            )
        logger.info(f"Loading FAISS index from {INDEX_PATH}")
        _faiss_index = faiss.read_index(INDEX_PATH)

    if _chunks is None:
        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError(
                f"Chunks file not found at '{CHUNKS_PATH}'. "
                "Run `python ingest.py` first."
            )
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            _chunks = json.load(f)

    return _faiss_index, _chunks


# ── Public Interface ──────────────────────────────────────────────────────────

def retrieve_context(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Embed query and retrieve top-k similar medical text chunks from FAISS.

    Args:
        query:  Natural language query (combines user question + image findings).
        top_k:  Number of chunks to retrieve.

    Returns:
        Formatted string of retrieved medical context passages.
    """
    try:
        model = _get_model()
        index, chunks = _get_store()

        # Embed and normalize for cosine similarity (IndexFlatIP)
        embedding = model.encode([query], normalize_embeddings=True)
        embedding = np.array(embedding, dtype=np.float32)

        distances, indices = index.search(embedding, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx == -1 or idx >= len(chunks):
                continue
            results.append(f"[Source {rank} — relevance: {score:.2f}]\n{chunks[idx]}")

        if not results:
            return "No relevant medical context found in the knowledge base."

        return "\n\n---\n\n".join(results)

    except FileNotFoundError as e:
        logger.error(str(e))
        return f"Knowledge base unavailable: {str(e)}"

    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return f"Context retrieval failed: {str(e)}"
