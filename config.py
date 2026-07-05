"""Shared configuration loaded from environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Facenet512")
NARRATOR_MODE = os.getenv("NARRATOR_MODE", "template")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "4096"))
MIN_IMAGE_DIMENSION = int(os.getenv("MIN_IMAGE_DIMENSION", "64"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
MODERATION_THRESHOLD = float(os.getenv("MODERATION_THRESHOLD", "0.85"))

CELEBRITY_DB_PATH = Path(
    os.getenv("CELEBRITY_DB_PATH", str(PROJECT_ROOT / "data" / "celebrity_embeddings.pkl"))
)

ALLOWED_MIME = {"image/jpeg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
