"""
utils.py
========
Reusable helper functions shared across notebooks, src/, and app/.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, List, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "ecommerce_llm", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout.

    Safe to call repeatedly (won't duplicate handlers).

    Args:
        name: Logger name.
        level: Logging level, e.g. logging.INFO.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# GPU / environment detection
# ---------------------------------------------------------------------------
def detect_device() -> str:
    """Detect the best available compute device.

    Returns:
        One of "cuda", "mps", or "cpu".
    """
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {name}")
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            logger.info("Apple MPS backend detected.")
            return "mps"
    except ImportError:
        logger.warning("PyTorch not installed yet; defaulting device check to CPU.")
    logger.warning("No GPU detected — falling back to CPU (training will be slow).")
    return "cpu"


def supports_bf16() -> bool:
    """Check whether the current GPU supports bfloat16 (Ampere+)."""
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NOISE_RE = re.compile(r"[^\w\s.,!?'\"%$€£@#:/()-]")


def remove_html(text: str) -> str:
    """Strip HTML tags from a string."""
    return _HTML_TAG_RE.sub(" ", text)


def remove_urls(text: str) -> str:
    """Strip URLs from a string."""
    return _URL_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFKC form and drop non-printable characters."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if ch.isprintable())


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces and strip ends."""
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def remove_noise_chars(text: str, keep_punct: bool = True) -> str:
    """Remove characters that aren't alphanumeric or common punctuation.

    Args:
        text: Input string.
        keep_punct: If True, keeps common sentence punctuation.

    Returns:
        Cleaned string.
    """
    if keep_punct:
        return _NOISE_RE.sub(" ", text)
    return re.sub(r"[^\w\s]", " ", text)


def clean_text(text: str, lowercase: bool = False) -> str:
    """Full cleaning pipeline: HTML -> URLs -> unicode -> noise -> whitespace.

    Args:
        text: Raw input text.
        lowercase: Whether to lowercase the result. Off by default because
            casing is useful signal for an instruct-tuned LLM.

    Returns:
        Cleaned text, or empty string if input was falsy/NaN.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = normalize_unicode(text)
    text = remove_noise_chars(text)
    text = normalize_whitespace(text)
    if lowercase:
        text = text.lower()
    return text


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def save_jsonl(records: Iterable[dict], path: Union[str, Path]) -> None:
    """Save an iterable of dicts to a JSON-Lines file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {sum(1 for _ in open(path))} records -> {path}")


def load_jsonl(path: Union[str, Path]) -> List[dict]:
    """Load a JSON-Lines file into a list of dicts."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def word_count(text: str) -> int:
    """Count whitespace-separated words in a string."""
    return len(text.split()) if isinstance(text, str) else 0


def safe_mkdir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it doesn't exist; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
