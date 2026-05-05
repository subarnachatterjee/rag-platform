# RAG Multi-Source Intelligence Platform

Retrieval-augmented intelligence system fusing 6 public datasets through automated ETL pipelines, embedding-based retrieval and statistical feature selection.

## Live Demo
🚀 [https://rag-platform-1s91.onrender.com](https://rag-platform-1s91.onrender.com)

## Tech Stack
Python · FastAPI · scikit-learn · FAISS · NumPy · pandas

## Features
- 6 public datasets fused via automated ETL pipeline
- SelectKBest feature selection (ANOVA F-test)
- 64-dimensional embeddings indexed with FAISS
- 5-fold cross-validation with Logistic Regression
- REST API with 6 endpoints
- Live dashboard with semantic search

## API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/api/etl` | POST | Run ETL pipeline |
| `/api/etl/status` | GET | Poll ETL progress |
| `/api/retrieve` | POST | Semantic search |
| `/api/results` | GET | Dataset metrics |
| `/api/stats` | GET | FAISS index stats |
| `/api/health` | GET | Health check |

## How to Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

## Project Structure
rag-platform/
├── app/
│   ├── etl.py          # ETL pipeline, feature selection, embeddings
│   └── retrieval.py    # FAISS retrieval engine
├── static/
│   └── index.html      # dashboard UI
├── main.py             # FastAPI app
└── requirements.txt
