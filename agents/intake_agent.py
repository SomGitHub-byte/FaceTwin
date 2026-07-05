"""Intake agent — validates upload, moderation, and face detection via MCP."""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from agents.mcp_client import call_mcp_tool


class IntakeAgent(BaseAgent):
    """Runs security checks and face detection before embedding."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        image_b64 = ctx.session.state.get("image_b64")
        if not image_b64:
            ctx.session.state["error"] = "No image provided"
            yield Event(
                author=self.name,
                output={"error": "No image provided"},
                actions=EventActions(state_delta={"error": "No image provided"}),
            )
            return

        mod = await call_mcp_tool("moderate_image", {"image_b64": image_b64})
        if not mod.get("safe"):
            reason = mod.get("reason", "Content moderation failed")
            ctx.session.state["error"] = reason
            yield Event(
                author=self.name,
                output={"error": reason},
                actions=EventActions(state_delta={"error": reason}),
            )
            return

        face = await call_mcp_tool("detect_face", {"image_b64": image_b64})
        if not face.get("ok"):
            reason = face.get("error", "Face detection failed")
            ctx.session.state["error"] = reason
            yield Event(
                author=self.name,
                output={"error": reason},
                actions=EventActions(state_delta={"error": reason}),
            )
            return

        ctx.session.state["landmarks"] = face.get("features", {})
        ctx.session.state["intake_ok"] = True
        yield Event(
            author=self.name,
            output={"status": "intake_passed"},
            actions=EventActions(
                state_delta={
                    "landmarks": face.get("features", {}),
                    "intake_ok": True,
                }
            ),
        )


intake_agent = IntakeAgent(name="intake_agent")
