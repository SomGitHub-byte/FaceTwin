"""Offline content moderation — heuristic + face-presence gate."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from config import MODERATION_THRESHOLD
from mcp_server.tools.validation import ValidationError, validate_image_bytes

logger = logging.getLogger(__name__)


def moderate_image(image_b64: str) -> dict[str, Any]:
    """
    Offline moderation gate.
    Checks: valid image, not mostly blank, skin-tone ratio heuristic, face presence.
    """
    try:
        bgr, raw = validate_image_bytes(image_b64)
    except ValidationError as exc:
        return {"safe": False, "reason": str(exc)}

    if not _magic_bytes_ok(raw):
        return {"safe": False, "reason": "Unrecognized image format"}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 5.0:
        return {"safe": False, "reason": "Image appears blank or corrupted"}

    nsfw_score = _skin_exposure_heuristic(bgr)
    if nsfw_score >= MODERATION_THRESHOLD:
        return {
            "safe": False,
            "reason": "Image flagged by content moderation (inappropriate content)",
        }

    if not _has_human_face(bgr):
        return {"safe": False, "reason": "No human face detected — upload a clear face photo"}

    return {"safe": True, "reason": "ok"}


def _magic_bytes_ok(raw: bytes) -> bool:
    return raw[:3] == b"\xff\xd8\xff" or raw[:8] == b"\x89PNG\r\n\x1a\n"


def _has_human_face(bgr: np.ndarray) -> bool:
    try:
        from retinaface.RetinaFace import detect_faces
    except ImportError:
        return False

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        faces = detect_faces(rgb, threshold=0.6)
    except Exception:
        return False

    if not faces:
        return False
    if isinstance(faces, dict):
        return bool(faces)
    return len(faces) > 0


def _skin_exposure_heuristic(bgr: np.ndarray) -> float:
    """
    Simple YCrCb skin-region ratio as a lightweight NSFW proxy.
    Returns 0.0 (safe) – 1.0 (likely inappropriate). Fully offline.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    skin_ratio = float(np.mean(skin_mask))

    h, w = bgr.shape[:2]
    # Large skin area in non-portrait aspect ratios raises flag
    aspect = max(w, h) / (min(w, h) + 1e-6)
    score = skin_ratio
    if aspect > 2.5 and skin_ratio > 0.55:
        score = min(1.0, skin_ratio * 1.3)
    if skin_ratio > 0.72:
        score = min(1.0, skin_ratio * 1.15)
    return round(score, 4)
