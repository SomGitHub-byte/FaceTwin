"""Orchestrator — SequentialAgent pipeline for FaceTwin."""

from __future__ import annotations

from google.adk.agents import SequentialAgent

from agents.embedding_agent import embedding_agent
from agents.intake_agent import intake_agent
from agents.narrator_agent import narrator_agent
from agents.ranking_agent import ranking_agent


def create_orchestrator() -> SequentialAgent:
    return SequentialAgent(
        name="orchestrator_agent",
        sub_agents=[
            intake_agent,
            embedding_agent,
            ranking_agent,
            narrator_agent,
        ],
    )


orchestrator_agent = create_orchestrator()
root_agent = orchestrator_agent
