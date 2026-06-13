"""FastAPI backend exposing ingestion, query, observability, and eval endpoints."""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agent.orchestrator import Agent
from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger
from app.core.schemas import IngestResponse, QueryRequest, QueryResponse
from app.db import models
from app.ingestion.chunking import chunk_text
from app.ingestion.loaders import load_document
from app.retrieval.hybrid import get_retriever

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_agent: Agent | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        models.init_engine()
        _agent = Agent()
    return _agent


@app.get("/health")
def health():
    return {"status": "ok", "provider": settings.llm_provider, "vector_backend": settings.vector_backend}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename or "doc.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        pages = load_document(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    doc_id = str(uuid.uuid4())
    all_chunks = []
    for page_no, text in pages:
        all_chunks.extend(chunk_text(text, doc_id, source=file.filename or "uploaded", page=page_no))

    if all_chunks:
        get_retriever().index(all_chunks)
    tmp_path.unlink(missing_ok=True)
    return IngestResponse(doc_id=doc_id, chunks=len(all_chunks), source=file.filename or "uploaded")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    agent = get_agent()
    resp = agent.run(req)
    # Persist for observability
    try:
        with models.get_session() as session:
            session.add(models.QueryLog(
                user_id=req.user_id, query=req.query, answer=resp.answer,
                blocked=resp.blocked, block_reason=resp.block_reason,
                redactions=resp.redactions,
                n_tool_calls=sum(1 for s in resp.steps if s.tool_call),
                n_citations=len(resp.citations), latency_ms=resp.latency_ms,
            ))
            session.commit()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to log query: %s", exc)
    return resp


@app.get("/metrics")
def metrics():
    """Aggregate observability metrics for the dashboard."""
    from sqlalchemy import func, select

    with models.get_session() as session:
        ql = models.QueryLog
        total = session.scalar(select(func.count()).select_from(ql)) or 0
        avg_latency = session.scalar(select(func.avg(ql.latency_ms))) or 0.0
        blocked = session.scalar(select(func.count()).where(ql.blocked.is_(True))) or 0
        redactions = session.scalar(select(func.sum(ql.redactions))) or 0
        tool_calls = session.scalar(select(func.sum(ql.n_tool_calls))) or 0
        recent = session.execute(
            select(ql.created_at, ql.query, ql.latency_ms, ql.n_tool_calls, ql.blocked)
            .order_by(ql.created_at.desc()).limit(20)
        ).all()
    return {
        "total_queries": total,
        "avg_latency_ms": round(float(avg_latency), 1),
        "blocked_queries": blocked,
        "total_redactions": int(redactions),
        "total_tool_calls": int(tool_calls),
        "recent": [
            {"time": str(r[0]), "query": r[1][:80], "latency_ms": round(r[2], 1),
             "tool_calls": r[3], "blocked": r[4]} for r in recent
        ],
    }
