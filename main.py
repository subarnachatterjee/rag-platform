"""
FastAPI — RAG Multi-Source Intelligence Platform
REST API serving layer
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import json

from app.etl import run_etl
from app.retrieval import retrieve, get_index_stats

DATA_DIR = Path(__file__).parent / "data"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="RAG Multi-Source Intelligence Platform",
    description="Retrieval-augmented intelligence over 6 public datasets",
    version="1.0.0",
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = "show me iris setosa samples"
    features: dict | None = None
    top_k: int = 5

class ETLResponse(BaseModel):
    status: str
    datasets_processed: int
    total_rows_indexed: int
    etl_seconds: float

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/etl", response_model=ETLResponse, tags=["ETL"])
def trigger_etl():
    """
    Run the full ETL pipeline:
    - Load 6 public datasets
    - Select top-K features via ANOVA F-test
    - Embed all rows into 64-d vectors
    - Build FAISS index
    - Cross-validate with Logistic Regression
    Returns summary metrics.
    """
    summary = run_etl()
    return ETLResponse(
        status="success",
        datasets_processed=summary["datasets_processed"],
        total_rows_indexed=summary["total_rows_indexed"],
        etl_seconds=summary["etl_seconds"],
    )


@app.post("/api/retrieve", tags=["RAG"])
def retrieve_similar(req: QueryRequest):
    """
    Semantic retrieval: embed query → FAISS nearest-neighbour search.
    Returns top-k similar rows with cosine similarity scores.
    """
    try:
        results = retrieve(req.query, req.features, req.top_k)
        return {"query": req.query, "results": results}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/results", tags=["Analytics"])
def get_results():
    """All dataset evaluation results (precision, F1, feature names …)."""
    path = DATA_DIR / "results.json"
    if not path.exists():
        raise HTTPException(400, "Run ETL first.")
    return json.loads(path.read_text())


@app.get("/api/summary", tags=["Analytics"])
def get_summary():
    """High-level ETL summary."""
    path = DATA_DIR / "summary.json"
    if not path.exists():
        raise HTTPException(400, "Run ETL first.")
    return json.loads(path.read_text())


@app.get("/api/stats", tags=["Analytics"])
def get_stats():
    """FAISS index statistics."""
    try:
        return get_index_stats()
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/api/health", tags=["Meta"])
def health():
    return {"status": "ok", "version": "1.0.0"}
