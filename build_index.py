"""
build_index.py — One-time script to build the FAISS vector index.

Usage:
    python build_index.py

This reads all JSON files from knowledge_base/, generates OpenAI embeddings
for each document, and saves the FAISS index to index/faiss_index.{faiss,pkl}.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

KB_DIR = Path("knowledge_base")
INDEX_DIR = Path("index")
INDEX_PATH = INDEX_DIR / "faiss_index"


def load_knowledge_base() -> list[dict]:
    docs = []
    for json_file in sorted(KB_DIR.glob("*.json")):
        with open(json_file, "r") as f:
            entries = json.load(f)
        for entry in entries:
            docs.append(entry)
        print(f"  Loaded {len(entries)} entries from {json_file.name}")
    return docs


def build_index(docs: list[dict]) -> None:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    texts = [d["content"] for d in docs]
    print(f"\nGenerating embeddings for {len(texts)} documents...")

    batch_size = 50
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = embeddings_model.embed_documents(batch)
        all_vectors.extend(vectors)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    matrix = np.array(all_vectors, dtype="float32")

    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH) + ".faiss")
    with open(str(INDEX_PATH) + ".pkl", "wb") as f:
        pickle.dump(docs, f)

    print(f"\nIndex saved to {INDEX_PATH}.faiss and {INDEX_PATH}.pkl")
    print(f"Total documents indexed: {index.ntotal}")


def verify_index(docs: list[dict]) -> None:
    """Quick smoke test — query a few known questions."""
    from langchain_openai import OpenAIEmbeddings

    index = faiss.read_index(str(INDEX_PATH) + ".faiss")
    with open(str(INDEX_PATH) + ".pkl", "rb") as f:
        stored_docs = pickle.load(f)

    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    test_queries = [
        "How do I reset my password?",
        "What is the maternity leave policy?",
        "How do I set up VPN?",
        "When is payday?",
        "How do I enroll in health insurance?",
    ]

    print("\n--- Index Verification ---")
    for q in test_queries:
        vec = np.array(emb.embed_query(q), dtype="float32").reshape(1, -1)
        vec /= np.linalg.norm(vec) or 1
        dists, idxs = index.search(vec, 1)
        top_doc = stored_docs[idxs[0][0]]
        score = float(dists[0][0])
        print(f"\nQuery: {q}")
        print(f"  → [{top_doc['category']}] {top_doc['title']} (score: {score:.4f})")


if __name__ == "__main__":
    print("=== SmartDesk AI — Building Knowledge Base Index ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Create a .env file.")
        sys.exit(1)

    print("Loading knowledge base documents...")
    docs = load_knowledge_base()
    print(f"\nTotal documents: {len(docs)}")

    build_index(docs)
    verify_index(docs)

    print("\n✓ Index built successfully. You can now run the app: streamlit run app.py")
