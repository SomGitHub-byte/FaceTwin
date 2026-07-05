"""Narrator agent — offline template (or optional Ollama) explanations."""

from __future__ import annotations

import random
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from config import NARRATOR_MODE, OLLAMA_BASE_URL, OLLAMA_MODEL

JAW_PHRASES = [
    "similar jaw structure",
    "comparable jawline width",
    "matching lower-face contour",
]
EYE_PHRASES = [
    "similar eye spacing",
    "aligned eye placement",
    "comparable upper-face proportions",
]
BROW_PHRASES = [
    "similar brow shape",
    "matching brow arch",
    "comparable forehead framing",
]
SMILE_PHRASES = [
    "similar smile curve",
    "comparable mouth shape",
    "matching lip line",
]


class NarratorAgent(BaseAgent):
    """Writes tasteful feature-only explanations for each match."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("error"):
            yield Event(author=self.name)
            return

        ranked = ctx.session.state.get("ranked_matches", [])
        landmarks = ctx.session.state.get("landmarks", {})

        if not ranked:
            ctx.session.state["error"] = "No ranked matches for narration"
            yield Event(
                author=self.name,
                output={"error": "No ranked matches"},
                actions=EventActions(state_delta={"error": "No ranked matches"}),
            )
            return

        results = []
        for i, match in enumerate(ranked):
            if NARRATOR_MODE == "ollama":
                explanation = await _ollama_explain(match, landmarks, i)
            elif NARRATOR_MODE == "gemini":
                explanation = await _gemini_explain(match, landmarks, i)
            else:
                explanation = _template_explain(match, landmarks, i)
            results.append(
                {
                    "name": match["name"],
                    "match_percent": match["match_percent"],
                    "explanation": explanation,
                }
            )

        ctx.session.state["final_results"] = results
        yield Event(
            author=self.name,
            output={"status": "complete", "results": results},
            actions=EventActions(state_delta={"final_results": results}),
        )


async def _gemini_explain(match: dict, landmarks: dict, seed: int) -> str:
    """Uses Google GenAI SDK to generate a tasteful explanation."""
    import os
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return _template_explain(match, landmarks, seed)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return _template_explain(match, landmarks, seed)

    prompt = (
        f"Write ONE short sentence for a face-similarity entertainment app. "
        f"Match: {match['match_percent']}% with {match['name']}. "
        f"ONLY mention observable features: jawline, eyes, brow, smile. "
        f"NEVER mention race, gender, age, or attractiveness. "
        f"Do not use placeholders. Landmark metrics context: {landmarks}"
    )

    try:
        client = genai.Client(api_key=api_key)
        import asyncio
        loop = asyncio.get_running_loop()
        def _call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=60,
                )
            )
        response = await loop.run_in_executor(None, _call)
        text = response.text.strip() if response.text else ""
        if text:
            return text
    except Exception as exc:
        import sys
        print(f"Warning: Gemini API call failed: {exc}", file=sys.stderr)
    return _template_explain(match, landmarks, seed)


def _template_explain(match: dict, landmarks: dict, seed: int) -> str:
    """Generate explanation from observable landmark ratios only."""
    rng = random.Random(seed + match["match_percent"])
    features = _pick_features(landmarks, rng)
    pct = match["match_percent"]
    name = match["name"]
    feature_text = " and ".join(features[:2])
    return f"{pct}% match with {name} — {feature_text}."


def _pick_features(landmarks: dict, rng: random.Random) -> list[str]:
    pools: list[tuple[str, list[str]]] = [
        ("jaw_ratio", JAW_PHRASES),
        ("eye_spacing", EYE_PHRASES),
        ("brow_angle", BROW_PHRASES),
        ("smile_curve", SMILE_PHRASES),
    ]
    scored = sorted(
        pools,
        key=lambda p: abs(float(landmarks.get(p[0], 0.5))),
        reverse=True,
    )
    chosen = []
    for _, phrases in scored[:3]:
        chosen.append(rng.choice(phrases))
    return chosen


async def _ollama_explain(match: dict, landmarks: dict, seed: int) -> str:
    """Optional local Ollama narration — no cloud API key."""
    import urllib.error
    import urllib.request

    prompt = (
        f"Write ONE short sentence for a face-similarity entertainment app. "
        f"Match: {match['match_percent']}% with {match['name']}. "
        f"ONLY mention observable features: jawline, eyes, brow, smile. "
        f"NEVER mention race, gender, age, or attractiveness. "
        f"Landmark hints: {landmarks}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        import json

        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = data.get("response", "").strip()
            if text:
                return text
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    return _template_explain(match, landmarks, seed)


narrator_agent = NarratorAgent(name="narrator_agent")
