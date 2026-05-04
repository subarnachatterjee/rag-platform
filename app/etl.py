"""
ETL Pipeline — loads public datasets, extracts features, builds FAISS embedding index.

Datasets used (all from sklearn / built-in — no download needed):
  iris, wine, breast_cancer, digits, diabetes, linnerud,
  california_housing (regression proxy used as classification).
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import faiss
import json
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Dataset registry ──────────────────────────────────────────────────────────
def load_all_datasets() -> list[dict]:
    loaders = [
        ("iris",           _wrap(datasets.load_iris)),
        ("wine",           _wrap(datasets.load_wine)),
        ("breast_cancer",  _wrap(datasets.load_breast_cancer)),
        ("digits",         _wrap(datasets.load_digits)),
        ("diabetes",       _load_diabetes_cls),      # regression → binned
        ("linnerud_pulse", _load_linnerud),
    ]
    records = []
    for name, loader in loaders:
        try:
            rec = loader()
            rec["name"] = name
            records.append(rec)
            print(f"  ✓ {name:20s}  shape={rec['X'].shape}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return records

def _wrap(loader_fn):
    """Convert sklearn Bunch → plain dict with X/y keys."""
    def _inner():
        d = loader_fn()
        return {
            "X": d.data,
            "y": d.target,
            "feature_names": list(d.feature_names),
            "target_names":  list(d.target_names),
        }
    return _inner


def _load_diabetes_cls():
    d = datasets.load_diabetes()
    y = pd.qcut(d.target, q=3, labels=[0, 1, 2]).astype(int)
    return {"X": d.data, "y": np.array(y),
            "feature_names": d.feature_names, "target_names": ["low","mid","high"]}


def _load_linnerud():
    d = datasets.load_linnerud()
    X = d.data          # physiological features
    y = (X[:, 0] > X[:, 0].mean()).astype(int)   # pulse above mean → binary
    return {"X": X, "y": y,
            "feature_names": d.feature_names, "target_names": ["low_pulse","high_pulse"]}


# ── Feature selection ─────────────────────────────────────────────────────────
def select_features(X, y, k: int = 5) -> tuple[np.ndarray, list[int]]:
    k = min(k, X.shape[1])
    sel = SelectKBest(f_classif, k=k)
    X_sel = sel.fit_transform(X, y)
    indices = sel.get_support(indices=True).tolist()
    return X_sel, indices


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_dataset(X: np.ndarray) -> np.ndarray:
    """Normalize + L2-project to fixed 64-d embedding."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    dim_out = 64
    rng = np.random.default_rng(42)
    proj = rng.standard_normal((X_scaled.shape[1], dim_out)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
    emb = X_scaled @ proj
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb / norms


# ── FAISS index ───────────────────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product on unit vecs = cosine sim
    index.add(embeddings)
    return index


# ── Evaluate dataset with LR cross-val ───────────────────────────────────────
def evaluate_dataset(X, y) -> dict:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    scores = cross_val_score(
        LogisticRegression(max_iter=500, random_state=42),
        X_s, y, cv=5, scoring="f1_weighted"
    )
    precision_scores = cross_val_score(
        LogisticRegression(max_iter=500, random_state=42),
        X_s, y, cv=5, scoring="precision_weighted"
    )
    return {
        "f1_mean":        float(np.mean(scores)),
        "f1_std":         float(np.std(scores)),
        "precision_mean": float(np.mean(precision_scores)),
        "precision_std":  float(np.std(precision_scores)),
    }


# ── Master ETL run ────────────────────────────────────────────────────────────
def run_etl() -> dict:
    print("\n🔄  Running ETL pipeline …")
    t0 = time.time()

    records = load_all_datasets()
    all_embeddings = []
    all_metadata   = []
    results        = []

    for rec in records:
        X, y = rec["X"], rec["y"]
        name = rec["name"]
        feat_names = list(rec.get("feature_names", [f"f{i}" for i in range(X.shape[1])]))
        target_names = list(rec.get("target_names", [str(c) for c in np.unique(y)]))

        # Feature selection
        X_sel, sel_idx = select_features(X, y)
        sel_names = [feat_names[i] for i in sel_idx]

        # Embed each sample
        emb = embed_dataset(X_sel)
        all_embeddings.append(emb)

        # Evaluate
        metrics = evaluate_dataset(X_sel, y)

        # Store per-row metadata for retrieval
        for i in range(len(X)):
            lbl = int(y[i])
            all_metadata.append({
                "dataset": name,
                "row_idx": int(i),
                "label": lbl,
                "label_name": str(target_names[lbl]) if lbl < len(target_names) else str(lbl),
                "features": {str(sel_names[j]): round(float(X_sel[i, j]), 5) for j in range(X_sel.shape[1])},
            })

        results.append({
            "dataset":      name,
            "n_samples":    int(X.shape[0]),
            "n_features":   int(X.shape[1]),
            "n_selected":   int(X_sel.shape[1]),
            "n_classes":    int(len(np.unique(y))),
            "selected_features": [str(n) for n in sel_names],
            "target_names": [str(n) for n in target_names],
            "metrics":      metrics,
        })
        print(f"  📊 {name:20s}  precision={metrics['precision_mean']:.3f}  f1={metrics['f1_mean']:.3f}")

    # Build combined FAISS index
    combined_emb = np.vstack(all_embeddings).astype(np.float32)
    index = build_faiss_index(combined_emb)
    faiss.write_index(index, str(DATA_DIR / "vector.index"))

    # Persist metadata & results
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(all_metadata, f)

    with open(DATA_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    summary = {
        "datasets_processed": len(results),
        "total_rows_indexed": len(all_metadata),
        "etl_seconds": round(elapsed, 2),
        "results": results,
    }
    with open(DATA_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅  ETL complete in {elapsed:.1f}s — {len(all_metadata)} rows indexed across {len(results)} datasets\n")
    return summary


if __name__ == "__main__":
    run_etl()
