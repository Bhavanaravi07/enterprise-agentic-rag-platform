"""Expose the tool registry as an MCP server.

Run standalone: `python -m app.tools.mcp_server`
Requires the `mcp` package. The same tools the in-process agent uses are served
here so external MCP clients (Claude Desktop, IDEs, other agents) can call them.
"""
from __future__ import annotations

from app.core.logging_config import configure_logging, get_logger
from app.tools.builtin import build_registry

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise SystemExit("Install the 'mcp' package to run the MCP server.")

    server = FastMCP("enterprise-rag-tools")
    registry = build_registry()

    for tool in registry.all():
        def make_handler(t):
            def handler(**kwargs):
                return t.run(**kwargs)
            handler.__name__ = t.name
            handler.__doc__ = t.description
            return handler

        server.tool(name=tool.name, description=tool.description)(make_handler(tool))

    logger.info("Serving %d tools over MCP", len(registry.all()))
    server.run()


if __name__ == "__main__":
    main()
