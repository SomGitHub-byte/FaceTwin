"""MCP stdio client for ADK agents — agents never import face code directly."""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _server_params() -> StdioServerParameters:
    command = os.getenv("MCP_SERVER_COMMAND", sys.executable)
    args_str = os.getenv("MCP_SERVER_ARGS", "-m,mcp_server.server")
    args = [a.strip() for a in args_str.split(",") if a.strip()]
    return StdioServerParameters(
        command=command,
        args=args,
        env={
            **os.environ,
            "PYTHONPATH": PROJECT_ROOT,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    )


@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    """Open a stdio MCP session to the FaceTwin tool server."""
    params = _server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_mcp_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a single MCP tool and parse the JSON result."""
    async with mcp_session() as session:
        result = await session.call_tool(name, arguments)
        if result.isError:
            return {"ok": False, "error": "MCP tool call failed"}
        if not result.content:
            return {"ok": False, "error": "Empty MCP response"}
        text = result.content[0].text
        return json.loads(text)


ALLOWED_TOOLS = frozenset(
    {"detect_face", "generate_embedding", "query_celebrity_db", "moderate_image"}
)
