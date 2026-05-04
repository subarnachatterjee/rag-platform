"""
RAG Retrieval Engine

Given a query (natural language or JSON feature dict), embed it the same way
as the ETL pipeline and retrieve the top-k most similar dataset rows via FAISS.
"""

import numpy as np
import faiss
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

_index: faiss.IndexFlatIP | None = None
_metadata: list[dict] | None = None


def _load_artifacts():
    global _index, _metadata
    if _index is None:
        idx_path = DATA_DIR / "vector.index"
        meta_path = DATA_DIR / "metadata.json"
        if not idx_path.exists() or not meta_path.exists():
            raise RuntimeError("Index not built yet. Run ETL first via /api/etl")
        _index = faiss.read_index(str(idx_path))
        with open(meta_path) as f:
            _metadata = json.load(f)


def _text_to_embedding(query: str, dim: int = 64) -> np.ndarray:
    """
    Lightweight query embedding: hash char codes → deterministic 64-d unit vector.
    Mimics the projection space used during ETL (same seed, same dim).
    In production you'd use sentence-transformers here.
    """
    rng = np.random.default_rng(sum(ord(c) * (i + 1) for i, c in enumerate(query[:128])))
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec.reshape(1, -1)


def _features_to_embedding(features: dict, dim: int = 64) -> np.ndarray:
    """Convert numeric feature dict → 64-d embedding using the same projection logic."""
    vals = np.array(list(features.values()), dtype=np.float32)
    rng = np.random.default_rng(42)
    proj = rng.standard_normal((len(vals), dim)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
    vec = (vals @ proj).reshape(1, -1)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec


def retrieve(query: str, features: dict | None = None, top_k: int = 5) -> list[dict]:
    """
    Retrieve top_k similar rows from the FAISS index.
    
    Args:
        query:    Natural-language string (always used).
        features: Optional numeric feature dict for embedding (overrides text embed).
        top_k:    Number of results to return.

    Returns:
        List of metadata dicts with an added 'score' field (cosine similarity).
    """
    _load_artifacts()

    if features:
        q_emb = _features_to_embedding(features)
    else:
        q_emb = _text_to_embedding(query)

    top_k = min(top_k, _index.ntotal)
    scores, indices = _index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = dict(_metadata[idx])
        row["score"] = round(float(score), 4)
        results.append(row)

    return results


def get_index_stats() -> dict:
    _load_artifacts()
    datasets = {}
    for m in _metadata:
        ds = m["dataset"]
        datasets[ds] = datasets.get(ds, 0) + 1
    return {
        "total_vectors": _index.ntotal,
        "embedding_dim": _index.d,
        "datasets": datasets,
    }
