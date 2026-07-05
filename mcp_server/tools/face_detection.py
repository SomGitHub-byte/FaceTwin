"""Face detection and landmark extraction using RetinaFace."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from mcp_server.tools.validation import ValidationError, validate_image_bytes

logger = logging.getLogger(__name__)


def detect_face(image_b64: str) -> dict[str, Any]:
    """Detect a single face and return landmark features or an error."""
    try:
        bgr, _ = validate_image_bytes(image_b64)
    except ValidationError as exc:
        return {"ok": False, "error": str(exc)}

    h, w = bgr.shape[:2]
    try:
        faces = _detect_faces(bgr)
    except Exception as exc:
        logger.exception("Face detection failed")
        return {"ok": False, "error": "Face detection failed"}

    if not faces:
        return {"ok": False, "error": "No face detected in the image"}

    if len(faces) > 1:
        return {"ok": False, "error": "Multiple faces detected — please upload a single-face photo"}

    face = faces[0]
    bbox = face.get("facial_area", [0, 0, w, h])
    landmarks_summary = _extract_landmark_features(face, w, h)

    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)

    return {
        "ok": True,
        "face_count": 1,
        "bbox": {
            "x": round(float(x1 / w), 4),
            "y": round(float(y1 / h), 4),
            "width": round(float(width / w), 4),
            "height": round(float(height / h), 4),
        },
        "features": landmarks_summary,
    }


def _detect_faces(bgr: np.ndarray) -> list[dict[str, Any]]:
    try:
        from retinaface.RetinaFace import detect_faces
    except ImportError as exc:
        raise RuntimeError("RetinaFace is required for face detection") from exc

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    faces = detect_faces(rgb, threshold=0.6)

    if not faces:
        return []
    if isinstance(faces, dict):
        return [faces[k] for k in sorted(faces.keys())]
    if isinstance(faces, list):
        return faces
    return []


def _extract_landmark_features(face: dict[str, Any], w: int, h: int) -> dict[str, float]:
    landmarks = face.get("landmarks", {})
    left_eye = np.asarray(landmarks.get("left_eye", [0.0, 0.0]), dtype=np.float32)
    right_eye = np.asarray(landmarks.get("right_eye", [0.0, 0.0]), dtype=np.float32)
    mouth_left = np.asarray(landmarks.get("mouth_left", [0.0, 0.0]), dtype=np.float32)
    mouth_right = np.asarray(landmarks.get("mouth_right", [0.0, 0.0]), dtype=np.float32)

    area = face.get("facial_area", [0.0, 0.0, float(w), float(h)])
    x1, y1, x2, y2 = area
    face_width = max(float(x2 - x1), 1.0)
    face_height = max(float(y2 - y1), 1.0)

    eye_dist = float(np.linalg.norm(right_eye - left_eye)) or 1.0
    mouth_width = float(np.linalg.norm(mouth_right - mouth_left)) or 1.0
    brow_angle = (right_eye[1] - left_eye[1]) / (abs(right_eye[0] - left_eye[0]) + 1e-6)

    jaw_ratio = face_width / face_height
    eye_spacing = eye_dist / face_width
    smile_curve = mouth_width / eye_dist

    return {
        "jaw_ratio": round(min(max(jaw_ratio, 0.1), 2.0), 4),
        "eye_spacing": round(min(max(eye_spacing, 0.1), 1.0), 4),
        "brow_angle": round(min(max(brow_angle, -1.0), 1.0), 4),
        "smile_curve": round(min(max(smile_curve, 0.1), 4.0), 4),
    }
