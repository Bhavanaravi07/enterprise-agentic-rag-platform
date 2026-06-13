"""Shared data schemas."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    source: str
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str = "hybrid"


class Citation(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    snippet: str


class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    roles: list[str] = Field(default_factory=lambda: ["employee"])
    top_k: int | None = None
    use_agent: bool = True


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: Any = None


class AgentStep(BaseModel):
    thought: str
    tool_call: ToolCall | None = None
    observation: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    redactions: int = 0
    blocked: bool = False
    block_reason: str | None = None
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    source: str
