"""
config.py
=========
Central configuration for MarketMind AI.

Reads runtime configuration from (in priority order):
    1. Streamlit secrets (``.streamlit/secrets.toml``) via ``st.secrets``
    2. Environment variables (``.env`` file loaded via python-dotenv)
    3. Sane hard-coded defaults (never used for API keys)

No secret value is ever hard-coded in this file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load a local .env file if present (no-op if it doesn't exist).
load_dotenv()

# --------------------------------------------------------------------------- #
# Streamlit secrets are only available inside a running Streamlit app. This
# module must also be importable from plain Python (e.g. unit tests, CLI
# scripts), so we import streamlit defensively.
# --------------------------------------------------------------------------- #
try:
    import streamlit as st

    _HAS_STREAMLIT = True
except Exception:  # pragma: no cover - streamlit always present in prod
    st = None  # type: ignore
    _HAS_STREAMLIT = False


def _get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch a config value from st.secrets first, then environment vars."""
    if _HAS_STREAMLIT:
        try:
            value = st.secrets.get(key)  # type: ignore[union-attr]
            if value not in (None, ""):
                return str(value)
        except Exception:
            # st.secrets raises if no secrets.toml exists at all; fall through.
            pass
    return os.environ.get(key, default)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
ASSETS_DIR: Path = BASE_DIR / "assets"
DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DIR: Path = BASE_DIR / "chroma_db"
LOG_DIR: Path = BASE_DIR / "logs"

for _dir in (DATA_DIR, CHROMA_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

LOGO_PATH: Path = ASSETS_DIR / "logo.png"
FAVICON_PATH: Path = ASSETS_DIR / "favicon.png"

# --------------------------------------------------------------------------- #
# App metadata
# --------------------------------------------------------------------------- #
APP_NAME: str = "MarketMind AI"
APP_SUBTITLE: str = "AI-Powered Marketing & Market Research Assistant"
APP_VERSION: str = "1.0.0"

# --------------------------------------------------------------------------- #
# LLM / OpenRouter configuration
# --------------------------------------------------------------------------- #
OPENROUTER_API_KEY: Optional[str] = _get_setting("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = _get_setting(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
) or "https://openrouter.ai/api/v1"
OPENROUTER_MODEL: str = _get_setting(
    "OPENROUTER_MODEL", "openai/gpt-4o-mini"
) or "openai/gpt-4o-mini"

# Optional headers OpenRouter recommends for attribution/rankings.
OPENROUTER_SITE_URL: str = _get_setting("OPENROUTER_SITE_URL", "https://marketmind.ai") or ""
OPENROUTER_SITE_NAME: str = _get_setting("OPENROUTER_SITE_NAME", APP_NAME) or APP_NAME

LLM_TEMPERATURE: float = float(_get_setting("LLM_TEMPERATURE", "0.2") or 0.2)
LLM_MAX_TOKENS: int = int(_get_setting("LLM_MAX_TOKENS", "1200") or 1200)

# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL_NAME: str = _get_setting(
    "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
) or "BAAI/bge-small-en-v1.5"

# BGE models recommend prefixing queries (not passages) with an instruction.
BGE_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE: int = int(_get_setting("CHUNK_SIZE", "512") or 512)
CHUNK_OVERLAP: int = int(_get_setting("CHUNK_OVERLAP", "64") or 64)

# --------------------------------------------------------------------------- #
# Retrieval / Vector store
# --------------------------------------------------------------------------- #
TOP_K: int = int(_get_setting("TOP_K", "5") or 5)
COLLECTION_NAME: str = _get_setting("COLLECTION_NAME", "marketmind_kb") or "marketmind_kb"

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
MAX_UPLOAD_SIZE_MB: int = int(_get_setting("MAX_UPLOAD_SIZE_MB", "200") or 200)


def is_llm_configured() -> bool:
    """True if an OpenRouter API key is available from any source."""
    return bool(OPENROUTER_API_KEY)


def as_dict() -> dict[str, Any]:
    """Return a redacted snapshot of the active configuration (for the UI)."""
    return {
        "app_name": APP_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": OPENROUTER_MODEL,
        "top_k": TOP_K,
        "collection_name": COLLECTION_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "api_key_configured": is_llm_configured(),
    }
