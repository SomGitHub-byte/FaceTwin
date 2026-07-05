"""Test script for FaceTwin multi-agent pipeline and MCP tools."""

from __future__ import annotations

import asyncio
import base64
import io
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import agents and app runner helpers
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agents.orchestrator import orchestrator_agent
from mcp_server.tools.validation import validate_image_bytes, ValidationError
from mcp_server.tools.moderation import moderate_image
from mcp_server.tools.celebrity_db import query_celebrity_db


def make_test_image() -> str:
    """Create a 100x100 test image in base64 format."""
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_input_validation():
    """Verify size limits and validation logic on image inputs."""
    print("Running validation tests...")
    
    # 1. Test empty base64
    try:
        validate_image_bytes("")
        assert False, "Empty base64 should fail validation"
    except ValidationError as exc:
        print("  - OK: Empty string caught:", exc)

    # 2. Test invalid base64
    try:
        validate_image_bytes("invalid_b64!")
        assert False, "Invalid base64 should fail validation"
    except ValidationError as exc:
        print("  - OK: Invalid base64 caught:", exc)

    # 3. Test valid base64 image
    b64_img = make_test_image()
    bgr, raw = validate_image_bytes(b64_img)
    assert bgr.shape == (100, 100, 3), f"Unexpected BGR shape: {bgr.shape}"
    print("  - OK: Valid image parsed successfully")


def test_moderation_gate():
    """Verify that empty/corrupted images are rejected by the moderation gate."""
    print("Running moderation tests...")
    
    # A plain white image should fail because np.std(gray) is 0 (blank/corrupted check)
    b64_img = make_test_image()
    res = moderate_image(b64_img)
    assert not res["safe"], "Blank image should be marked unsafe"
    assert "blank or corrupted" in res["reason"]
    print("  - OK: Blank image moderation test passed:", res["reason"])


def test_celebrity_query():
    """Verify querying the bootstrapped database works."""
    print("Running database query tests...")
    
    # Create a dummy query vector (512 dimensions)
    import numpy as np
    rng = np.random.default_rng(42)
    vec = (rng.standard_normal(512) / 10.0).tolist()
    
    res = query_celebrity_db(vec, top_k=5)
    assert res["ok"], f"DB query failed: {res.get('error')}"
    assert len(res["matches"]) == 5, f"Expected 5 matches, got {len(res['matches'])}"
    print("  - OK: DB query returned top 5 matches")
    for i, m in enumerate(res["matches"]):
        print(f"    {i+1}. {m['name']} (score: {m['score']})")


async def run_pipeline_test():
    """Verify the multi-agent session state transitions and ranking/narration."""
    print("Running multi-agent pipeline tests...")
    
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator_agent,
        app_name="FaceTwin",
        session_service=session_service,
    )
    
    user_id = "test-user-123"
    session = await session_service.create_session(
        app_name="FaceTwin",
        user_id=user_id,
        state={"image_b64": make_test_image()},
    )
    
    # We mock the MCP tool calls to run the agent logic offline without MediaPipe/DeepFace dependencies
    mock_mcp = AsyncMock()
    
    def mock_mcp_side_effect(name, arguments):
        if name == "moderate_image":
            return {"safe": True, "reason": "ok"}
        elif name == "detect_face":
            return {
                "ok": True,
                "face_count": 1,
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
                "features": {
                    "jaw_ratio": 0.65,
                    "eye_spacing": 0.42,
                    "brow_angle": 0.05,
                    "smile_curve": 3.2,
                }
            }
        elif name == "generate_embedding":
            # Generate a 512-d list
            import numpy as np
            rng = np.random.default_rng(42)
            v = (rng.standard_normal(512) / 10.0).tolist()
            return {"ok": True, "vector": v, "model": "Facenet512"}
        elif name == "query_celebrity_db":
            # Return some mock matches and scores
            return {
                "ok": True,
                "matches": [
                    {"name": "Elon Musk", "score": 0.82},
                    {"name": "Jeff Bezos", "score": 0.75},
                    {"name": "Tim Cook", "score": 0.68},
                    {"name": "Satya Nadella", "score": 0.61},
                    {"name": "Zendaya", "score": 0.55},
                ],
                "all_scores": [0.82, 0.75, 0.68, 0.61, 0.55, 0.40, 0.35, 0.20],
                "score_stats": {"min": 0.1, "max": 0.9, "mean": 0.5}
            }
        return {"ok": False, "error": "Unknown tool"}

    mock_mcp.side_effect = mock_mcp_side_effect

    with patch("agents.intake_agent.call_mcp_tool", mock_mcp), \
         patch("agents.embedding_agent.call_mcp_tool", mock_mcp):
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="analyze face")]),
        ):
            print("EVENT YIELDED:", event)

    # Read final session state to verify values
    updated_session = await session_service.get_session(
        app_name="FaceTwin", user_id=user_id, session_id=session.id
    )
    
    assert updated_session is not None
    state = updated_session.state
    print("DEBUG: Final session state is:", state)
    
    # 1. Intake should pass
    assert state.get("intake_ok"), f"Intake agent did not succeed. Error: {state.get('error')}"
    assert "landmarks" in state, "Landmarks missing from session state"
    
    # 2. Embedding should be deleted/popped and raw matches saved
    assert not state.get("image_b64"), "Image base64 buffer was not dereferenced from memory"
    assert "raw_matches" in state, "Raw matches missing from session state"
    
    # 3. Ranking should normalize scores
    assert "ranked_matches" in state, "Ranked matches missing from session state"
    for r in state["ranked_matches"]:
        assert 55 <= r["match_percent"] <= 92, f"Match percent {r['match_percent']} out of 55-92% range"
        print(f"  - Ranked: {r['name']} -> {r['match_percent']}% (raw: {r['raw_score']})")
        
    # 4. Narrator should create explanations
    assert "final_results" in state, "Final results missing from session state"
    print("  - Narrative results:")
    for res in state["final_results"]:
        print(f"    {res['match_percent']}% with {res['name']}: {res['explanation']}")
        
    print("  - OK: Pipeline completed successfully!")


def main():
    test_input_validation()
    print("-" * 40)
    test_moderation_gate()
    print("-" * 40)
    test_celebrity_query()
    print("-" * 40)
    asyncio.run(run_pipeline_test())


if __name__ == "__main__":
    main()
