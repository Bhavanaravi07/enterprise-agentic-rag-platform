"""BM25 keyword retrieval over the chunk corpus."""
from __future__ import annotations

import math
import re
from collections import Counter

from app.core.schemas import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_tokens: list[list[str]] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def fit(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.doc_tokens = [_tokenize(c.text) for c in chunks]
        self.df = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] += 1
        lengths = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        q_terms = _tokenize(query)
        scored = []
        for chunk, toks in zip(self.chunks, self.doc_tokens):
            if not toks:
                continue
            freqs = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in q_terms:
                if term not in freqs:
                    continue
                idf = self._idf(term)
                tf = freqs[term]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score, retriever="keyword"))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]
