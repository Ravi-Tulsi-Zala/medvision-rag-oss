import json
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DOCS_PATH = "data/docs.txt"
INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 200     
OVERLAP = 50          
MIN_CHUNK_WORDS = 30  

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if len(chunk_words) < MIN_CHUNK_WORDS:
            continue
        chunks.append(" ".join(chunk_words))

    return chunks


def build_faiss_index(chunks: list[str], model_name: str = EMBEDDING_MODEL) -> None:

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Embedding {len(chunks)} chunks (this may take a moment)...")
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,   
        show_progress_bar=True,
        batch_size=32,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    logger.info(f"FAISS index saved: {INDEX_PATH} ({index.ntotal} vectors, dim={dimension})")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Chunks saved: {CHUNKS_PATH} ({len(chunks)} chunks)")

def main():
    logger.info("Building FAISS index")
    logger.info(f"   Source: {DOCS_PATH}")

    if not os.path.exists(DOCS_PATH):
        logger.error(f"Knowledge base not found at '{DOCS_PATH}'")
        sys.exit(1)

    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"   Document length: {len(text.split())} words")

    chunks = chunk_text(text)
    logger.info(f"   Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={OVERLAP})")

    build_faiss_index(chunks)

if __name__ == "__main__":
    main()
