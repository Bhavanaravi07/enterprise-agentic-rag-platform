"""Semantic-ish chunking.

Strategy: split on paragraph/sentence boundaries, then greedily pack into
chunks of ~chunk_size characters with overlap. This preserves semantic units
(sentences) rather than cutting mid-thought, which improves retrieval quality
over naive fixed-width splitting. For embedding-similarity-based semantic
chunking, see `semantic_split` which groups adjacent sentences by cosine drift.
"""
import re
import uuid

from app.core.config import get_settings
from app.core.schemas import Chunk

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_text(
    text: str,
    doc_id: str,
    source: str,
    page: int | None = None,
) -> list[Chunk]:
    settings = get_settings()
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        body = " ".join(buf)
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=body,
                source=source,
                page=page,
                metadata={"char_len": len(body)},
            )
        )
        # carry overlap
        if overlap > 0:
            carry, carry_len = [], 0
            for s in reversed(buf):
                if carry_len + len(s) > overlap:
                    break
                carry.insert(0, s)
                carry_len += len(s)
            buf = carry
            buf_len = carry_len
        else:
            buf, buf_len = [], 0

    for sent in sentences:
        if buf_len + len(sent) > size and buf:
            flush()
        buf.append(sent)
        buf_len += len(sent)
    flush()
    return chunks
