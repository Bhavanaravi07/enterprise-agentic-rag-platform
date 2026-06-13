"""PII detection and redaction.

Uses regex-based detection for common PII so the platform has zero external
dependencies by default. Swap in Microsoft Presidio for production by
implementing the same `redact` interface.
"""
import re
from dataclasses import dataclass

PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


@dataclass
class RedactionResult:
    text: str
    count: int
    entities: list[str]


def redact(text: str) -> RedactionResult:
    count = 0
    entities: list[str] = []
    out = text
    for label, pattern in PATTERNS.items():
        def _sub(match: re.Match) -> str:
            nonlocal count
            count += 1
            entities.append(label)
            return f"[REDACTED_{label}]"

        out = pattern.sub(_sub, out)
    return RedactionResult(text=out, count=count, entities=entities)
