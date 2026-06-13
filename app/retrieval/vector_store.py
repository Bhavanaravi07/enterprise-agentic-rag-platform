"""Vector store abstraction. FAISS (local) by default; Pinecone optional."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.schemas import Chunk, ScoredChunk

logger = get_logger(__name__)


class FaissStore:
    """In-process FAISS index with a parallel chunk metadata list."""

    def __init__(self, dim: int, index_path: str) -> None:
        import faiss

        self.faiss = faiss
        self.dim = dim
        self.index_path = Path(index_path)
        self.meta_path = self.index_path.with_suffix(".meta.pkl")
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                self.chunks: list[Chunk] = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
            self.chunks = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.persist()

    def search(self, query_vec: np.ndarray, k: int) -> list[ScoredChunk]:
        if self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(query_vec.reshape(1, -1), min(k, self.index.ntotal))
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            out.append(ScoredChunk(chunk=self.chunks[idx], score=float(score), retriever="vector"))
        return out

    def persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)


class PineconeStore:
    def __init__(self, api_key: str, index_name: str, dim: int) -> None:
        from pinecone import Pinecone

        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self._local_meta: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        items = [
            {"id": c.id, "values": v.tolist(), "metadata": {"text": c.text, "source": c.source}}
            for c, v in zip(chunks, vectors)
        ]
        self.index.upsert(vectors=items)
        for c in chunks:
            self._local_meta[c.id] = c

    def search(self, query_vec: np.ndarray, k: int) -> list[ScoredChunk]:
        res = self.index.query(vector=query_vec.tolist(), top_k=k, include_metadata=True)
        out = []
        for m in res["matches"]:
            md = m.get("metadata", {})
            chunk = Chunk(id=m["id"], doc_id="", text=md.get("text", ""), source=md.get("source", ""))
            out.append(ScoredChunk(chunk=chunk, score=float(m["score"]), retriever="vector"))
        return out

    def all_chunks(self) -> list[Chunk]:
        return list(self._local_meta.values())


def get_vector_store():
    s = get_settings()
    if s.vector_backend == "pinecone" and s.pinecone_api_key:
        logger.info("Using Pinecone vector store: %s", s.pinecone_index)
        return PineconeStore(s.pinecone_api_key, s.pinecone_index, s.embedding_dim)
    logger.info("Using FAISS vector store at %s", s.faiss_index_path)
    return FaissStore(s.embedding_dim, s.faiss_index_path)
