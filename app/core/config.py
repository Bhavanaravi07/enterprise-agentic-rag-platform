"""Central application configuration loaded from environment variables."""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "Enterprise Agentic RAG Platform"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"

    # --- LLM provider ---
    # provider: one of openai | azure | anthropic | gemini
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    # --- Vector store ---
    # backend: faiss | pinecone
    vector_backend: str = "faiss"
    faiss_index_path: str = "data/faiss.index"
    pinecone_api_key: str | None = None
    pinecone_index: str = "enterprise-rag"
    embedding_dim: int = 1536

    # --- Datastores ---
    database_url: str = "postgresql+psycopg://rag:rag@postgres:5432/ragdb"
    redis_url: str = "redis://redis:6379/0"

    # --- Retrieval ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k_vector: int = 20
    top_k_keyword: int = 20
    top_k_final: int = 6
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_alpha: float = 0.5  # weight of vector vs keyword in fusion

    # --- Agent ---
    max_agent_steps: int = 6

    # --- Guardrails ---
    enable_pii_redaction: bool = True
    enable_injection_guard: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
