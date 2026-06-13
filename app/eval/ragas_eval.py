"""RAG evaluation. Uses RAGAS when installed + an API key is present; otherwise
computes lightweight proxy metrics so the eval harness always runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging_config import get_logger

logger = get_logger(__name__)
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


@dataclass
class EvalReport:
    metrics: dict[str, float] = field(default_factory=dict)
    per_sample: list[dict] = field(default_factory=list)
    backend: str = "fallback"


def _tok(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _fallback_eval(samples: list[EvalSample]) -> EvalReport:
    """Proxy metrics:
    - context_recall: ground-truth tokens covered by retrieved contexts
    - faithfulness: answer tokens supported by contexts
    - answer_relevancy: token overlap between answer and question
    """
    rows = []
    for s in samples:
        ctx_tokens = set().union(*[_tok(c) for c in s.contexts]) if s.contexts else set()
        ans_tokens = _tok(s.answer)
        q_tokens = _tok(s.question)
        faith = len(ans_tokens & ctx_tokens) / (len(ans_tokens) or 1)
        relev = len(ans_tokens & q_tokens) / (len(q_tokens) or 1)
        recall = (
            len(_tok(s.ground_truth) & ctx_tokens) / (len(_tok(s.ground_truth)) or 1)
            if s.ground_truth else float("nan")
        )
        rows.append({"question": s.question, "faithfulness": round(faith, 3),
                     "answer_relevancy": round(relev, 3), "context_recall": round(recall, 3)})

    def avg(key):
        vals = [r[key] for r in rows if r[key] == r[key]]  # filter NaN
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return EvalReport(
        metrics={"faithfulness": avg("faithfulness"),
                 "answer_relevancy": avg("answer_relevancy"),
                 "context_recall": avg("context_recall")},
        per_sample=rows, backend="fallback",
    )


def evaluate(samples: list[EvalSample]) -> EvalReport:
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth or "" for s in samples],
        }
        ds = Dataset.from_dict(data)
        result = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall])
        return EvalReport(metrics=dict(result), backend="ragas")
    except Exception as exc:
        logger.warning("RAGAS unavailable (%s); using fallback evaluator.", exc)
        return _fallback_eval(samples)
