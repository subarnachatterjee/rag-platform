"""
FastAPI — RAG Multi-Source Intelligence Platform
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
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

etl_status = {"state": "idle", "message": "ETL not yet run"}

class QueryRequest(BaseModel):
    query: str = "show me iris setosa samples"
    features: dict | None = None
    top_k: int = 5

def run_etl_background():
    global etl_status
    try:
        etl_status = {"state": "running", "message": "ETL in progress..."}
        summary = run_etl()
        etl_status = {
            "state": "done",
            "status": "success",
            "datasets_processed": summary["datasets_processed"],
            "total_rows_indexed": summary["total_rows_indexed"],
            "etl_seconds": summary["etl_seconds"],
        }
    except Exception as e:
        etl_status = {"state": "error", "message": str(e)}

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/etl", tags=["ETL"])
def trigger_etl(background_tasks: BackgroundTasks):
    if etl_status["state"] == "running":
        return {"status": "already running"}
    background_tasks.add_task(run_etl_background)
    return {"status": "started", "message": "ETL running in background. Poll /api/etl/status"}

@app.get("/api/etl/status", tags=["ETL"])
def etl_status_check():
    return etl_status

@app.post("/api/retrieve", tags=["RAG"])
def retrieve_similar(req: QueryRequest):
    try:
        results = retrieve(req.query, req.features, req.top_k)
        return {"query": req.query, "results": results}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/results", tags=["Analytics"])
def get_results():
    path = DATA_DIR / "results.json"
    if not path.exists():
        raise HTTPException(400, "Run ETL first.")
    return json.loads(path.read_text())

@app.get("/api/summary", tags=["Analytics"])
def get_summary():
    path = DATA_DIR / "summary.json"
    if not path.exists():
        raise HTTPException(400, "Run ETL first.")
    return json.loads(path.read_text())

@app.get("/api/stats", tags=["Analytics"])
def get_stats():
    try:
        return get_index_stats()
    except RuntimeError as e:
        raise HTTPException(400, str(e))

@app.get("/api/health", tags=["Meta"])
def health():
    return {"status": "ok", "version": "1.0.0"}
