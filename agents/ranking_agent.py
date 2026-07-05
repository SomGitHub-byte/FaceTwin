"""Ranking agent — percentile-normalizes similarity scores to display percentages."""

from __future__ import annotations

from typing import AsyncGenerator

import numpy as np
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions


class RankingAgent(BaseAgent):
    """Normalizes raw cosine scores against DB distribution (55–92% display range)."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("error"):
            yield Event(author=self.name)
            return

        raw_matches = ctx.session.state.get("raw_matches")
        all_scores = ctx.session.state.get("all_scores")

        if not raw_matches:
            ctx.session.state["error"] = "No matches to rank"
            yield Event(
                author=self.name,
                output={"error": "No matches to rank"},
                actions=EventActions(state_delta={"error": "No matches to rank"}),
            )
            return

        scores_arr = np.asarray(all_scores, dtype=np.float64) if all_scores else None
        ranked = []

        for match in raw_matches:
            raw = float(match["score"])
            pct = _to_display_percent(raw, scores_arr)
            ranked.append(
                {
                    "name": match["name"],
                    "raw_score": round(raw, 6),
                    "match_percent": pct,
                }
            )

        ctx.session.state["ranked_matches"] = ranked
        yield Event(
            author=self.name,
            output={"status": "ranking_complete", "count": len(ranked)},
            actions=EventActions(state_delta={"ranked_matches": ranked}),
        )


def _to_display_percent(raw_score: float, all_scores: np.ndarray | None) -> int:
    """
    Map cosine similarity to a display percentage using percentile rank
    against the full DB score distribution — avoids clustering near 95%+.
    """
    if all_scores is not None and len(all_scores) > 1:
        percentile = float(np.mean(all_scores <= raw_score) * 100)
    else:
        # Fallback: map typical Facenet512 cosine range [0.3, 0.85]
        percentile = (raw_score - 0.3) / (0.85 - 0.3) * 100
        percentile = max(0.0, min(100.0, percentile))

    # Spread into 55–92% entertainment range
    display = 55 + (percentile / 100.0) * 37
    return int(round(max(55, min(92, display))))


ranking_agent = RankingAgent(name="ranking_agent")
