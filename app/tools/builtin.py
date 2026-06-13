"""Concrete tool implementations."""
from __future__ import annotations

import ast
import operator
from typing import Any

from app.core.logging_config import get_logger
from app.tools.base import Tool, ToolRegistry

logger = get_logger(__name__)

# --- Calculator (safe AST eval, no builtins) ---
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants allowed")
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate an arithmetic expression. Supports + - * / ** % //."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. '1200 * 0.07'"}},
        "required": ["expression"],
    }

    def run(self, expression: str, **_: Any) -> Any:
        tree = ast.parse(expression, mode="eval")
        return _safe_eval(tree.body)


class SQLLookupTool(Tool):
    name = "sql_lookup"
    description = "Run a read-only SQL SELECT against the analytics database."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "A SELECT statement"}},
        "required": ["query"],
    }

    def __init__(self, engine=None) -> None:
        self.engine = engine

    def run(self, query: str, **_: Any) -> Any:
        q = query.strip().rstrip(";")
        if not q.lower().startswith("select"):
            raise ValueError("Only SELECT statements are permitted.")
        if self.engine is None:
            # Demo fallback when no DB is wired up.
            return [{"note": "No DB connected; returning stub", "query": q}]
        from sqlalchemy import text

        with self.engine.connect() as conn:
            rows = conn.execute(text(q)).mappings().all()
            return [dict(r) for r in rows]


class TicketLookupTool(Tool):
    name = "ticket_lookup"
    description = "Look up a support ticket by ID and return its status and summary."
    parameters = {
        "type": "object",
        "properties": {"ticket_id": {"type": "string"}},
        "required": ["ticket_id"],
    }

    # In-memory demo store; replace with real ticketing API client.
    _DB = {
        "TKT-1001": {"status": "open", "priority": "high", "summary": "VPN access failing for remote staff"},
        "TKT-1002": {"status": "resolved", "priority": "low", "summary": "Password reset request"},
        "TKT-1003": {"status": "in_progress", "priority": "medium", "summary": "Invoice #INV-552 disputed"},
    }

    def run(self, ticket_id: str, **_: Any) -> Any:
        return self._DB.get(ticket_id.upper(), {"error": "ticket not found", "ticket_id": ticket_id})


class PolicyLookupTool(Tool):
    name = "policy_lookup"
    description = "Retrieve relevant company policy passages for a question using RAG."
    parameters = {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    }

    def run(self, question: str, **_: Any) -> Any:
        # Lazy import to avoid a circular import at module load.
        from app.retrieval.hybrid import get_retriever

        hits = get_retriever().retrieve(question, top_k=4)
        return [
            {"source": h.chunk.source, "page": h.chunk.page, "text": h.chunk.text[:500]}
            for h in hits
        ]


def build_registry(sql_engine=None) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(SQLLookupTool(engine=sql_engine))
    reg.register(TicketLookupTool())
    reg.register(PolicyLookupTool())
    return reg
