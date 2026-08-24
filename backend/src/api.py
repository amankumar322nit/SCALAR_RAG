"""
FastAPI REST Endpoints and Web Server for Scaler RAG Pipeline.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
ROOT_DIR = BACKEND_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.pipeline import RAGPipeline
from src.instrumentation import StructuredTracer

app = FastAPI(
    title="Scaler Learner Support RAG API",
    description="Grounded AI Support Assistant with Source Attribution and Observability",
    version="1.0.0"
)

# Enable CORS for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline instances
rag_pipeline = RAGPipeline()
tracer = StructuredTracer()

# Robust Frontend Directory Discovery
CANDIDATE_FRONTEND_PATHS = [
    ROOT_DIR / "frontend",
    BACKEND_DIR / "frontend",
    Path.cwd() / "frontend",
    BACKEND_DIR / "static",
]
FRONTEND_DIR = next((p for p in CANDIDATE_FRONTEND_PATHS if p.exists() and (p / "index.html").exists()), CANDIDATE_FRONTEND_PATHS[0])


class QuestionRequest(BaseModel):
    question: Optional[str] = Field(None, description="Learner natural language query")
    query: Optional[str] = Field(None, description="Alternative field for query")
    top_k: Optional[int] = Field(4, description="Top-k chunks to retrieve")

    def get_text(self) -> str:
        return (self.question or self.query or "").strip()


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "scaler-rag-support",
        "total_chunks_in_store": len(rag_pipeline.vector_store.chunks),
        "frontend_dir": str(FRONTEND_DIR)
    }


@app.post("/ask")
def ask_question(req: QuestionRequest):
    """
    Accepts learner query, retrieves grounded context, and returns answer + citations.
    """
    query_text = req.get_text()
    if not query_text:
        raise HTTPException(status_code=400, detail="Field 'question' or 'query' must not be empty.")

    result = rag_pipeline.query(query_text, top_k=req.top_k or 4, emit_stdout=True)
    return result


@app.post("/ingest", response_model=IngestResponse)
def reindex_corpus():
    """Trigger re-indexing of the document corpus."""
    count = rag_pipeline.ingest()
    return {"status": "success", "chunks_indexed": count}


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., description="Target trace ID")
    rating: int = Field(..., description="+1 for thumbs up, -1 for thumbs down")
    comment: Optional[str] = Field("", description="Optional user comment")


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """Record learner positive/negative feedback for online evaluation."""
    success = tracer.record_user_feedback(
        trace_id=req.trace_id,
        rating=req.rating,
        feedback=req.comment or ""
    )
    if not success:
        raise HTTPException(status_code=404, detail="Trace ID not found or feedback error.")
    return {"status": "success", "message": "Feedback recorded for online evaluation."}


@app.get("/evals/online")
def get_online_evaluations():
    """Retrieve real-time aggregated online evaluation metrics."""
    return tracer.get_online_eval_summary()


@app.get("/traces")
def get_traces(limit: int = 25):
    """Retrieve recent query execution traces for observability."""
    return {"traces": tracer.get_recent_traces(limit=limit)}


@app.get("/")
def serve_ui():
    """Serve the Web UI HTML."""
    index_html = FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return {"message": "Scaler RAG API is running. Visit /docs for OpenAPI specs."}


# Mount static files (serves style.css and app.js)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
