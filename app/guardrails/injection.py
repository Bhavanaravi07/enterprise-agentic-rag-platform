"""Prompt-injection / jailbreak heuristic guardrail.

Layered defense: (1) heuristic pattern match on user input, (2) retrieved-context
sanitization to neutralize instructions embedded in documents. For production,
add a classifier model behind the same interface.
"""
import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|prompts)",
    r"disregard (all |the )?(previous|prior|system)",
    r"you are now",
    r"forget (everything|all previous)",
    r"reveal (your |the )?(system )?(prompt|instructions)",
    r"print (your |the )?(system )?(prompt|instructions)",
    r"act as (?:an? )?(?:DAN|jailbroken|unrestricted)",
    r"developer mode",
    r"override (the )?(safety|guardrails|rules)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class InjectionVerdict:
    blocked: bool
    reason: str | None = None
    matched: list[str] | None = None


def check_user_input(text: str) -> InjectionVerdict:
    hits = [p.pattern for p in _COMPILED if p.search(text)]
    if hits:
        return InjectionVerdict(
            blocked=True,
            reason="Potential prompt-injection detected in user query.",
            matched=hits,
        )
    return InjectionVerdict(blocked=False)


def sanitize_context(text: str) -> str:
    """Neutralize imperative instructions found inside retrieved documents.

    We don't drop content (it may be legitimately relevant) but we defang
    instruction-like lines so the LLM treats them as data, not commands.
    """
    sanitized = text
    for pattern in _COMPILED:
        sanitized = pattern.sub("[neutralized-instruction]", sanitized)
    return sanitized
