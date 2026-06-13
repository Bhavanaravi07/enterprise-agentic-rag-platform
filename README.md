# Enterprise Agentic RAG Platform

An enterprise assistant that answers questions across company documents (policies, invoices, tickets, product docs), retrieves evidence with **hybrid search + reranking**, **reasons across sources with a multi-step agent**, calls **MCP-style tools**, and returns **cited answers** — with **PII redaction**, **prompt-injection guardrails**, **RBAC**, **RAGAS evaluation**, and an **observability dashboard**.

> Runs **fully offline with zero API keys** out of the box (local embeddings + stub LLM + FAISS), so you can demo the entire pipeline immediately. Plug in OpenAI / Azure OpenAI / Claude / Gemini and Pinecone for production.

## Architecture

```
                    ┌──────────────┐         ┌──────────────────────┐
  Streamlit UI ───► │ FastAPI API  │ ──────► │  Agent Orchestrator  │
  (chat / ingest /  │  /query      │         │  plan→tool→observe→  │
   observability)   │  /ingest     │         │  synthesize (cited)  │
                    │  /metrics    │         └─────────┬────────────┘
                    └──────┬───────┘                   │
                           │            ┌──────────────┼───────────────┐
                           │            ▼              ▼               ▼
                           │     Guardrails      Hybrid Retrieval   Tools (MCP)
                           │   (PII + injection)  vector + BM25      calculator
                           │                       → RRF fusion      sql_lookup
                           ▼                       → rerank          ticket_lookup
                  Postgres (query logs)                              policy_lookup
                  Redis (cache)         FAISS / Pinecone
```

## Features

| Capability | Implementation |
|---|---|
| Document ingestion | PDF / TXT / MD loaders (`app/ingestion/loaders.py`) |
| Semantic chunking | Sentence-aware packing with overlap (`app/ingestion/chunking.py`) |
| Hybrid search | Vector (FAISS/Pinecone) + BM25, fused via Reciprocal Rank Fusion |
| Reranking | Cross-encoder (sentence-transformers) with lexical fallback |
| Multi-step agent | Plan → tool call → observe → synthesize loop (`app/agent/orchestrator.py`) |
| MCP tools | calculator, sql_lookup, ticket_lookup, policy_lookup; servable over MCP |
| RAG evaluation | RAGAS (faithfulness, answer relevancy, context recall) + offline fallback |
| Guardrails | Regex PII redaction + prompt-injection detection & context sanitization |
| RBAC | Role-based filtering of retrieved chunks |
| Observability | Postgres query logs + `/metrics` endpoint + dashboard |
| Deployment | Docker Compose (api, frontend, postgres, redis) + GitHub Actions CI |

## Quick start (Docker)

```bash
cp .env.example .env        # optional: add an LLM API key for full generation
docker compose up --build
# API   → http://localhost:8000/docs
# UI    → http://localhost:8501
```

## Quick start (local, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/seed_and_demo.py          # seed sample corpus + run demo + eval
uvicorn app.api.main:app --reload        # terminal 1
streamlit run frontend/streamlit_app.py  # terminal 2
```

## Configuration

All settings are environment variables (see `.env.example`). Key switches:

- `LLM_PROVIDER` = `openai` | `azure` | `anthropic` | `gemini` (falls back to offline stub without a key)
- `VECTOR_BACKEND` = `faiss` | `pinecone`
- `ENABLE_PII_REDACTION`, `ENABLE_INJECTION_GUARD` = `true` | `false`

## Running the MCP tool server

```bash
pip install mcp
python -m app.tools.mcp_server
```

This exposes `calculator`, `sql_lookup`, `ticket_lookup`, and `policy_lookup` over the Model Context Protocol so external MCP clients (e.g. Claude Desktop, IDEs) can call the same tools the in-process agent uses.

## Tests

```bash
pytest -q
```

The suite covers chunking, PII redaction, injection blocking, all tools, BM25 ranking, and an offline end-to-end agent run — no network or API keys required.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload a document (multipart); chunks + indexes it |
| `POST` | `/query` | Ask a question; returns answer, citations, reasoning trace |
| `GET`  | `/metrics` | Aggregate observability metrics |
| `GET`  | `/health` | Liveness + active provider/backend |

Example:

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query":"What is our PTO policy?","roles":["employee"],"use_agent":true}'
```

## Project layout

```
app/
  api/         FastAPI app + endpoints
  agent/       LLM abstraction + multi-step orchestrator
  ingestion/   loaders + semantic chunking
  retrieval/   embeddings, vector store, BM25, rerank, hybrid fusion
  tools/       tool framework, built-in tools, MCP server
  guardrails/  PII redaction + prompt-injection defense
  eval/        RAGAS evaluation harness
  db/          SQLAlchemy models for observability
  core/        config, logging, schemas
frontend/      Streamlit UI (chat / ingest / dashboard)
docker/        Dockerfiles
scripts/       seed + demo + eval runner
tests/         offline test suite
```

## Notes on production hardening

- Swap regex PII for **Microsoft Presidio** behind the same `redact` interface.
- Add a **classifier-based** injection detector alongside the heuristics.
- Use **Pinecone** or **Azure AI Search** for the vector backend at scale.
- Wire `sql_lookup` to a read-replica with row-level security for true RBAC at the data layer.

## License

MIT
