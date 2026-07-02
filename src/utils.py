"""
Shared, dependency-light helper functions used across the pipeline.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src import config

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured once per process.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A logger writing to stdout (and to ``config.LOG_FILE`` if the
        outputs directory exists) using ``config.LOG_FORMAT``.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(config.LOG_FORMAT)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        try:
            config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(config.LOG_FILE)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            # Read-only filesystem or missing permissions - stdout only.
            pass

    _LOGGERS[name] = logger
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Context manager that logs the wall-clock duration of a code block.

    Example
    -------
    >>> with timer("embedding candidates", logger):
    ...     embed_candidates(records)
    """
    log = logger or get_logger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("%s took %.2fs", label, elapsed)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records from a ``.jsonl`` or ``.jsonl.gz`` file.

    Streaming (rather than loading the whole file into a list) keeps peak
    memory bounded regardless of how large the candidate pool is.

    Parameters
    ----------
    path:
        Path to a newline-delimited JSON file, optionally gzip-compressed.

    Yields
    ------
    dict
        One parsed JSON object per non-blank line.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON at {path}:{line_number}: {exc}"
                ) from exc


def resolve_candidates_path() -> Path:
    """Return whichever candidates file actually exists on disk.

    Prefers the uncompressed ``candidates.jsonl``; falls back to the
    gzipped variant so the pipeline works with either bundle layout.
    """
    if config.CANDIDATES_JSONL.exists():
        return config.CANDIDATES_JSONL
    if config.CANDIDATES_JSONL_GZ.exists():
        return config.CANDIDATES_JSONL_GZ
    raise FileNotFoundError(
        f"Neither {config.CANDIDATES_JSONL} nor {config.CANDIDATES_JSONL_GZ} exists. "
        "Place the candidate dataset in the data/ directory."
    )


_WHITESPACE_RE = re.compile(r"\s+")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x1f\x7f]")


def clean_text(text: str | None) -> str:
    """Normalize free text for embedding / keyword matching.

    Strips control characters, collapses whitespace, and trims. Does NOT
    lowercase by default since some downstream uses (embeddings) benefit
    from natural casing; use :func:`normalize_for_matching` for
    case-insensitive keyword search.
    """
    if not text:
        return ""
    text = _NON_PRINTABLE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_for_matching(text: str | None) -> str:
    """Lowercase, whitespace-collapsed text suitable for substring/keyword search."""
    return clean_text(text).lower()


def contains_any(haystack: str, needles: list[str]) -> bool:
    """True if any of ``needles`` (case-insensitive) appears in ``haystack``."""
    hay = haystack.lower()
    return any(needle.lower() in hay for needle in needles)


def count_matches(haystack: str, needles: list[str]) -> int:
    """Count how many distinct ``needles`` appear (case-insensitive) in ``haystack``."""
    hay = haystack.lower()
    return sum(1 for needle in needles if needle.lower() in hay)


def safe_get(dictionary: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested-safe dict lookup: ``safe_get(d, "a", "b")`` == ``d["a"]["b"]``."""
    current: Any = dictionary
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def minmax_normalize(value: float, low: float, high: float) -> float:
    """Scale ``value`` linearly from ``[low, high]`` to ``[0, 1]``, clamped."""
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low))
