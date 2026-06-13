"""Embedding generation with pluggable providers.

Defaults to a deterministic local hashing embedder so the platform runs with
zero API keys for development and CI. Set EMBEDDING_PROVIDER appropriately and
provide keys to use OpenAI/Azure embeddings in production.
"""
from __future__ import annotations

import hashlib

import numpy as np

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class LocalHashEmbedder:
    """Deterministic, dependency-free embeddings for dev/CI.

    Not semantically meaningful at scale, but stable and fast — good enough to
    exercise the full pipeline end-to-end without external calls.
    """

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for token in t.lower().split():
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


def get_embedder():
    s = get_settings()
    if s.llm_provider == "openai" and s.openai_api_key:
        logger.info("Using OpenAI embedder: %s", s.embedding_model)
        return OpenAIEmbedder(s.embedding_model, s.openai_api_key)
    logger.warning("Falling back to LocalHashEmbedder (no API key configured).")
    return LocalHashEmbedder(dim=s.embedding_dim)
