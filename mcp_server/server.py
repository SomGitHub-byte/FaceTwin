"""FaceTwin MCP server — exposes face processing tools via FastMCP."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.celebrity_db import query_celebrity_db as _query_db
from mcp_server.tools.embeddings import generate_embedding as _generate_embedding
from mcp_server.tools.face_detection import detect_face as _detect_face
from mcp_server.tools.moderation import moderate_image as _moderate_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_server")

mcp = FastMCP("FaceTwin")


def _log_tool(name: str, success: bool) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    status = "success" if success else "fail"
    logger.info("tool=%s timestamp=%s status=%s", name, ts, status)


@mcp.tool()
def detect_face(image_b64: str) -> dict:
    """Detect a single face and return landmark features. Input: base64-encoded JPG/PNG."""
    result = _detect_face(image_b64)
    _log_tool("detect_face", result.get("ok", False))
    return result


@mcp.tool()
def generate_embedding(image_b64: str) -> dict:
    """Generate a face embedding vector. Input: base64-encoded JPG/PNG."""
    result = _generate_embedding(image_b64)
    _log_tool("generate_embedding", result.get("ok", False))
    return result


@mcp.tool()
def query_celebrity_db(vector: list[float], top_k: int = 5) -> dict:
    """Query precomputed celebrity embeddings. Returns top-k matches by cosine similarity."""
    result = _query_db(vector, top_k=top_k)
    _log_tool("query_celebrity_db", result.get("ok", False))
    return result


@mcp.tool()
def moderate_image(image_b64: str) -> dict:
    """Offline content moderation gate. Returns {safe: bool, reason: str}."""
    result = _moderate_image(image_b64)
    _log_tool("moderate_image", result.get("safe", False))
    return result


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
