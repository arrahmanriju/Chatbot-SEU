"""Configuration shared by the build and serving processes."""
import os
from pathlib import Path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parents[1]


def load_environment():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def embedding_model():
    load_environment()
    return os.getenv("SEU_EMBEDDING_MODEL") or DEFAULT_MODEL
