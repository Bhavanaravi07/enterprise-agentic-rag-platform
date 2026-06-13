"""Hybrid retriever: vector + BM25, fused with Reciprocal Rank Fusion, reranked.

This is the retrieval engine the agent and the simple RAG path both call.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.schemas import Chunk, ScoredChunk
from app.retrieval.embeddings import get_embedder
from app.retrieval.keyword import BM25
from app.retrieval.rerank import get_reranker
from app.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """RRF: score = sum 1/(k + rank). Robust to differing score scales."""
    fused: dict[str, ScoredChunk] = {}
    agg: dict[str, float] = {}
    for results in result_lists:
        for rank, sc in enumerate(results):
            cid = sc.chunk.id
            agg[cid] = agg.get(cid, 0.0) + 1.0 / (k + rank + 1)
            fused.setdefault(cid, sc)
    out = []
    for cid, score in agg.items():
        sc = fused[cid]
        out.append(ScoredChunk(chunk=sc.chunk, score=score, retriever="hybrid"))
    out.sort(key=lambda x: x.score, reverse=True)
    return out


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = get_embedder()
        self.store = get_vector_store()
        self.bm25 = BM25()
        self.reranker = get_reranker()
        self._refit_bm25()

    def _refit_bm25(self) -> None:
        self.bm25.fit(self.store.all_chunks())

    def index(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self._refit_bm25()

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        s = self.settings
        top_k = top_k or s.top_k_final
        q_vec = self.embedder.embed([query])[0]
        vector_hits = self.store.search(q_vec, s.top_k_vector)
        keyword_hits = self.bm25.search(query, s.top_k_keyword)
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits])
        candidates = fused[: max(s.top_k_vector, s.top_k_keyword)]
        reranked = self.reranker.rerank(query, candidates, top_k)
        logger.info(
            "retrieve: %d vector, %d keyword, %d fused -> %d final",
            len(vector_hits), len(keyword_hits), len(fused), len(reranked),
        )
        return reranked


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
