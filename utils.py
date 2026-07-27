"""
utils.py
========
Shared utility functions used across the MarketMind AI pipeline:

* Structured logging setup
* A dynamic module loader (needed because pipeline files are named with
  numeric prefixes like ``01_documents.py``, which are not valid Python
  identifiers and therefore cannot be imported with a plain ``import``
  statement)
* A performance timing decorator/context manager
* Small formatting helpers used by the Streamlit UI
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator, TypeVar

from config import LOG_DIR

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = "marketmind") -> logging.Logger:
    """Create (or fetch) a configured logger.

    Logs to both stdout and a rotating-by-run file under ``logs/``.
    Safe to call multiple times; handlers are only attached once.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        log_file = LOG_DIR / "marketmind.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:  # pragma: no cover - file logging is best-effort
        pass

    logger.propagate = False
    return logger


logger = setup_logger()

# --------------------------------------------------------------------------- #
# Dynamic module loader for numerically-prefixed pipeline files
# --------------------------------------------------------------------------- #
_MODULE_CACHE: dict[str, ModuleType] = {}


def load_module(module_alias: str, file_name: str) -> ModuleType:
    """Load a pipeline file (e.g. ``01_documents.py``) as an importable module.

    Parameters
    ----------
    module_alias:
        The name to register the module under in ``sys.modules`` (any valid
        Python identifier, e.g. ``"documents"``).
    file_name:
        The file name of the module, relative to the project root
        (e.g. ``"01_documents.py"``).

    Returns
    -------
    The loaded module object, with all of its top-level functions/classes
    accessible as attributes.
    """
    if module_alias in _MODULE_CACHE:
        return _MODULE_CACHE[module_alias]

    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Pipeline module not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_alias, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    spec.loader.exec_module(module)

    _MODULE_CACHE[module_alias] = module
    return module


# --------------------------------------------------------------------------- #
# Performance timing
# --------------------------------------------------------------------------- #
F = TypeVar("F", bound=Callable)


def timed(step_name: str) -> Callable[[F], F]:
    """Decorator that logs how long a pipeline step took to run."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("[%s] completed in %.2fs", step_name, elapsed)
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.error("[%s] failed after %.2fs: %s", step_name, elapsed, exc)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def timer(step_name: str) -> Iterator[None]:
    """Context-manager version of ``timed`` for inline blocks."""
    start = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - start
        logger.info("[%s] completed in %.2fs", step_name, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error("[%s] failed after %.2fs: %s", step_name, elapsed, exc)
        raise


# --------------------------------------------------------------------------- #
# Formatting helpers (used by the Streamlit UI)
# --------------------------------------------------------------------------- #
def format_bytes(num_bytes: float) -> str:
    """Human-readable byte size, e.g. 1536000 -> '1.46 MB'."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def truncate(text: str, max_chars: int = 320) -> str:
    """Truncate text to max_chars, breaking on a word boundary."""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0]
    return clipped.rstrip() + "…"


def similarity_from_distance(distance: float) -> float:
    """Convert a Chroma cosine *distance* into an intuitive 0-1 similarity."""
    similarity = 1.0 - distance
    return max(0.0, min(1.0, similarity))
