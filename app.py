"""FaceTwin — Gradio frontend (fully offline, no API keys)."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import socket
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from agents.orchestrator import orchestrator_agent
from config import MAX_UPLOAD_BYTES, RATE_LIMIT_PER_MINUTE

APP_NAME = "FaceTwin"
DISCLAIMER = (
    "*For entertainment — based on facial feature similarity, not personality traits.*"
)

_session_service = InMemorySessionService()
_runner = Runner(
    agent=orchestrator_agent,
    app_name=APP_NAME,
    session_service=_session_service,
)

# Per-session rate limiting
_request_log: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(session_id: str) -> str | None:
    now = time.time()
    window = _request_log[session_id]
    window[:] = [t for t in window if now - t < 60]
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        return f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} requests/minute). Please wait."
    window.append(now)
    return None


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    raw = buf.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")
    return base64.b64encode(raw).decode("ascii")


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No matches found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"**{i}. {r['match_percent']}% match with {r['name']}**")
        lines.append(f"   {r['explanation']}")
        lines.append("")
    return "\n".join(lines)


async def _run_pipeline(image_b64: str, session_id: str) -> dict:
    session = await _session_service.create_session(
        app_name=APP_NAME,
        user_id=session_id,
        state={"image_b64": image_b64},
    )

    final_state: dict = {}
    async for event in _runner.run_async(
        user_id=session_id,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="analyze face")]),
    ):
        if hasattr(event, "actions") and event.actions and event.actions.state_delta:
            final_state.update(event.actions.state_delta)

    # Read final session state
    updated = await _session_service.get_session(
        app_name=APP_NAME, user_id=session_id, session_id=session.id
    )
    return updated.state if updated else final_state


def analyze(image: Image.Image | None, request: gr.Request) -> tuple[str, str]:
    session_id = getattr(request, "session_hash", None) or "default"

    if image is None:
        return "Please upload a photo.", DISCLAIMER

    limit_msg = _check_rate_limit(session_id)
    if limit_msg:
        return limit_msg, DISCLAIMER

    try:
        image_b64 = _image_to_b64(image.convert("RGB"))
    except ValueError as exc:
        return str(exc), DISCLAIMER

    try:
        state = asyncio.run(_run_pipeline(image_b64, session_id))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return f"Pipeline error: {exc}", DISCLAIMER

    if state.get("error"):
        return f"❌ {state['error']}", DISCLAIMER

    results = state.get("final_results", [])
    return _format_results(results), DISCLAIMER


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FaceTwin") as demo:
        gr.Markdown("# 🎭 FaceTwin")
        gr.Markdown("Upload a photo to find your celebrity look-alikes.")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Your photo", sources=["upload"])
                btn = gr.Button("Find My Face Twin", variant="primary")
            with gr.Column():
                output = gr.Markdown(label="Results")
                disclaimer_output = gr.Markdown(DISCLAIMER)

        btn.click(fn=analyze, inputs=[img_input], outputs=[output, disclaimer_output])

        gr.Markdown("---")
        gr.Markdown(
            "**Privacy:** Photos are processed in memory only and never stored. "
            "**Offline:** No external API keys required."
        )
    return demo


def _find_free_port(start_port: int = 7860, end_port: int = 7880) -> int:
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Cannot find empty port in range: {start_port}-{end_port}")


if __name__ == "__main__":
    requested_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    try:
        port = requested_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
    except OSError:
        port = _find_free_port(requested_port + 1, 7880)
        print(f"Port {requested_port} is in use. Starting on port {port} instead.")

    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"Launching FaceTwin at http://{display_host}:{port}")
    build_ui().launch(server_name=host, server_port=port, theme=gr.themes.Soft())
