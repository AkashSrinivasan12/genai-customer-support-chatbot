"""
FastAPI Backend — RAG Chatbot API
Exposes /chat endpoint for the Streamlit frontend and external integrations (e.g. Power BI).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time

from src.rag_pipeline import query
from src.vector_store import build_index

app = FastAPI(
    title="Healthcare KPI RAG Chatbot",
    description="RAG-powered chatbot for doctor-specific KPI queries. Built with FAISS + OpenAI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    chat_history: Optional[List[ChatMessage]] = []

class Source(BaseModel):
    label: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    latency_ms: float
    tokens_used: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Healthcare KPI RAG Chatbot is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main RAG query endpoint.
    Accepts a natural language question and returns an AI-generated answer
    grounded in retrieved KPI data.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()
    try:
        history = [{"role": m.role, "content": m.content} for m in request.chat_history]
        result = query(
            user_question=request.question,
            top_k=request.top_k,
            chat_history=history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.time() - start) * 1000, 2)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=latency_ms,
        tokens_used=result["tokens_used"],
    )


@app.post("/rebuild-index")
def rebuild_index():
    """Rebuild the FAISS vector index from the latest KPI data."""
    try:
        index, docs = build_index()
        return {"status": "ok", "documents_indexed": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
