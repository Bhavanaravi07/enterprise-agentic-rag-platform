"""Seed the index with the sample corpus and run a demo query + eval."""
from pathlib import Path

from app.agent.orchestrator import Agent
from app.core.schemas import QueryRequest
from app.eval.ragas_eval import EvalSample, evaluate
from app.ingestion.chunking import chunk_text
from app.ingestion.loaders import load_document
from app.retrieval.hybrid import get_retriever

SAMPLE_DIR = Path("data/sample")


def seed() -> None:
    retriever = get_retriever()
    for path in SAMPLE_DIR.glob("*.txt"):
        pages = load_document(path)
        chunks = []
        for page_no, text in pages:
            chunks.extend(chunk_text(text, doc_id=path.stem, source=path.name, page=page_no))
        retriever.index(chunks)
        print(f"Indexed {path.name}: {len(chunks)} chunks")


def demo() -> None:
    agent = Agent()
    questions = [
        "What is our PTO policy?",
        "What's the status of TKT-1003?",
        "Calculate 7% tax on a $1200 invoice",
        "How many days can I work remotely?",
    ]
    samples = []
    for q in questions:
        resp = agent.run(QueryRequest(query=q, use_agent=True, roles=["employee", "finance"]))
        print(f"\nQ: {q}\nA: {resp.answer}\n  citations={len(resp.citations)} "
              f"tools={sum(1 for s in resp.steps if s.tool_call)} latency={resp.latency_ms:.0f}ms")
        samples.append(EvalSample(
            question=q, answer=resp.answer,
            contexts=[c.snippet for c in resp.citations],
        ))
    report = evaluate(samples)
    print(f"\n=== Eval ({report.backend}) ===")
    for k, v in report.metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    seed()
    demo()
