"""Local embedding generation using RetinaFace and lightweight image features."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from mcp_server.tools.face_detection import _detect_faces
from mcp_server.tools.validation import ValidationError, validate_image_bytes

logger = logging.getLogger(__name__)


def generate_embedding(image_b64: str) -> dict[str, Any]:
    """Generate a face embedding vector from image bytes."""
    try:
        bgr, _ = validate_image_bytes(image_b64)
    except ValidationError as exc:
        return {"ok": False, "error": str(exc)}

    try:
        faces = _detect_faces(bgr)
    except Exception as exc:
        logger.exception("Embedding face detection failed")
        return {"ok": False, "error": "Embedding generation failed"}

    if not faces:
        return {"ok": False, "error": "No face detected for embedding generation"}

    face = faces[0]
    bbox = face.get("facial_area", [0, 0, bgr.shape[1], bgr.shape[0]])
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
    crop = bgr[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else bgr

    vector = _compute_embedding(crop)
    return {"ok": True, "vector": vector, "model": "local-retinaface"}


def _compute_embedding(face_bgr: np.ndarray) -> list[float]:
    """Compute a fixed-size embedding from a face crop using DCT and color features."""
    if face_bgr.size == 0:
        face_bgr = np.zeros((64, 64, 3), dtype=np.uint8)

    face = cv2.resize(face_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dct = cv2.dct(gray)
    dct_flat = dct.flatten()[:256]

    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 1, 0, 1]).flatten()
    hist = hist / (np.linalg.norm(hist) + 1e-6)

    vector = np.concatenate([dct_flat, hist, np.zeros(512 - dct_flat.size - hist.size, dtype=np.float32)])
    vector = vector.astype(np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.ones(512, dtype=np.float64).tolist()
    return (vector / norm).tolist()


def cosine_similarity(a: list[float], b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
