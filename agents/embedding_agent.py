"""Embedding agent — generates face vector and queries celebrity DB via MCP."""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from agents.mcp_client import call_mcp_tool


class EmbeddingAgent(BaseAgent):
    """Generates embedding and retrieves raw celebrity matches."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("error"):
            yield Event(author=self.name)
            return

        image_b64 = ctx.session.state.get("image_b64")
        if not image_b64:
            ctx.session.state["error"] = "Missing image data"
            yield Event(
                author=self.name,
                output={"error": "Missing image data"},
                actions=EventActions(state_delta={"error": "Missing image data"}),
            )
            return

        emb = await call_mcp_tool("generate_embedding", {"image_b64": image_b64})
        # Dereference image from state after embedding — no persistent storage
        ctx.session.state.pop("image_b64", None)

        if not emb.get("ok"):
            reason = emb.get("error", "Embedding generation failed")
            ctx.session.state["error"] = reason
            yield Event(
                author=self.name,
                output={"error": reason},
                actions=EventActions(state_delta={"error": reason, "image_b64": None}),
            )
            return

        vector = emb["vector"]
        ctx.session.state["embedding"] = vector

        matches = await call_mcp_tool(
            "query_celebrity_db", {"vector": vector, "top_k": 5}
        )
        if not matches.get("ok"):
            reason = matches.get("error", "Database query failed")
            ctx.session.state["error"] = reason
            yield Event(
                author=self.name,
                output={"error": reason},
                actions=EventActions(state_delta={"error": reason}),
            )
            return

        ctx.session.state["raw_matches"] = matches["matches"]
        ctx.session.state["all_scores"] = matches.get("all_scores", [])
        yield Event(
            author=self.name,
            output={"status": "embedding_complete"},
            actions=EventActions(
                state_delta={
                    "embedding": vector,
                    "raw_matches": matches["matches"],
                    "all_scores": matches.get("all_scores", []),
                    "image_b64": None,
                }
            ),
        )


embedding_agent = EmbeddingAgent(name="embedding_agent")
