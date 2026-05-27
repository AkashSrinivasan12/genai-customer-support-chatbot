"""
Vector Store — FAISS-based local store (mirrors Azure AI Search used in production).
Handles embedding, indexing, and similarity search over healthcare KPI documents.
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_PATH = Path("data/faiss_index.bin")
DOCS_PATH = Path("data/docs_store.pkl")


def get_embedding(text: str) -> List[float]:
    """Get OpenAI embedding for a text string."""
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def build_document_text(doc: Dict) -> str:
    """Convert a KPI record into a searchable text chunk."""
    return (
        f"Doctor: {doc['doctor']} | Department: {doc['department']} | "
        f"Metric: {doc['metric']} | Value: {doc['value']} | "
        f"Trend: {doc['trend']} | Period: {doc['period']} | "
        f"Notes: {doc['notes']}"
    )


def build_index(data_path: str = "data/healthcare_kpis.json") -> Tuple[faiss.Index, List[Dict]]:
    """Build FAISS index from KPI JSON data."""
    with open(data_path, "r") as f:
        documents = json.load(f)

    texts = [build_document_text(doc) for doc in documents]
    embeddings = [get_embedding(text) for text in texts]

    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)

    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    index.add(vectors)

    # Persist
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"Index built: {len(documents)} documents, dim={dim}")
    return index, documents


def load_index() -> Tuple[faiss.Index, List[Dict]]:
    """Load persisted FAISS index and document store."""
    if not INDEX_PATH.exists() or not DOCS_PATH.exists():
        return build_index()
    index = faiss.read_index(str(INDEX_PATH))
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
    return index, documents


def search(query: str, top_k: int = 3) -> List[Dict]:
    """Search for most relevant KPI documents given a natural language query."""
    index, documents = load_index()
    query_vec = np.array([get_embedding(query)], dtype="float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            doc = documents[idx].copy()
            doc["relevance_score"] = float(score)
            results.append(doc)
    return results
