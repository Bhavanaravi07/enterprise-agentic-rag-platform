"""Reranking of fused candidates.

Uses a cross-encoder (sentence-transformers) when available; otherwise falls
back to a lexical overlap reranker so the pipeline runs without heavy models.
"""
from __future__ import annotations

import re
from collections import Counter

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.schemas import ScoredChunk

logger = get_logger(__name__)
_TOKEN = re.compile(r"[a-z0-9]+")


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.score = float(s)
            c.retriever = "rerank"
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:k]


class LexicalReranker:
    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        q = Counter(_TOKEN.findall(query.lower()))
        for c in candidates:
            d = Counter(_TOKEN.findall(c.chunk.text.lower()))
            overlap = sum((q & d).values())
            c.score = overlap / (1 + len(d))
            c.retriever = "rerank-lexical"
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:k]


def get_reranker():
    s = get_settings()
    try:
        return CrossEncoderReranker(s.rerank_model)
    except Exception as exc:  # pragma: no cover
        logger.warning("Cross-encoder unavailable (%s); using lexical reranker.", exc)
        return LexicalReranker()
