"""Document loading. Extracts text + page metadata from supported file types."""
from pathlib import Path

from app.core.logging_config import get_logger

logger = get_logger(__name__)


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text). Lazy import so pypdf is optional."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def load_text(path: Path) -> list[tuple[int, str]]:
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
}


def load_document(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    logger.info("Loading %s via %s", path.name, loader.__name__)
    return loader(path)
