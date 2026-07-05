"""Input validation helpers for MCP tools."""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image

from config import (
    ALLOWED_EXTENSIONS,
    MAX_IMAGE_DIMENSION,
    MAX_UPLOAD_BYTES,
    MIN_IMAGE_DIMENSION,
)


class ValidationError(Exception):
    """Raised when image input fails validation."""


def validate_image_bytes(image_b64: str) -> Tuple[np.ndarray, bytes]:
    """Decode base64 image, validate size/type/dimensions, return BGR array + raw bytes."""
    import base64

    if not image_b64 or not isinstance(image_b64, str):
        raise ValidationError("Missing or invalid image data")

    try:
        raw = base64.b64decode(image_b64, validate=True)
    except Exception as exc:
        raise ValidationError("Invalid base64 encoding") from exc

    if len(raw) == 0:
        raise ValidationError("Empty image payload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValidationError(f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.verify()
        pil = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise ValidationError("File is not a valid JPEG or PNG") from exc

    if pil.format not in ("JPEG", "PNG"):
        raise ValidationError("Only JPG and PNG images are supported")

    w, h = pil.size
    if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
        raise ValidationError(f"Image too small (min {MIN_IMAGE_DIMENSION}px)")
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        raise ValidationError(f"Image too large (max {MAX_IMAGE_DIMENSION}px)")

    rgb = np.array(pil.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    return bgr, raw


def sniff_extension(raw: bytes) -> bool:
    """Quick magic-byte check for JPEG/PNG."""
    if raw[:3] == b"\xff\xd8\xff":
        return True
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    return False
