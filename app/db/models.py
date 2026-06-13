"""SQLAlchemy models + session for persisting query logs (observability)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    user_id: Mapped[str] = mapped_column(String(128))
    query: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    blocked: Mapped[bool] = mapped_column(default=False)
    block_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    redactions: Mapped[int] = mapped_column(Integer, default=0)
    n_tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    n_citations: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)


_engine = None
_SessionLocal = None


def init_engine():
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine
    s = get_settings()
    try:
        _engine = create_engine(s.database_url, pool_pre_ping=True)
        Base.metadata.create_all(_engine)
    except Exception as exc:
        logger.warning("DB unavailable (%s); falling back to SQLite.", exc)
        _engine = create_engine("sqlite:///./observability.db")
        Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)
    return _engine


def get_session() -> Session:
    if _SessionLocal is None:
        init_engine()
    return _SessionLocal()
