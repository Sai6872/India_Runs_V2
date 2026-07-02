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
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from src import config

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
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
            pass

    _LOGGERS[name] = logger
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    log = logger or get_logger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("%s took %.2fs", label, elapsed)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
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
    if not text:
        return ""
    text = _NON_PRINTABLE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_for_matching(text: str | None) -> str:
    return clean_text(text).lower()


@lru_cache(maxsize=128)
def _get_boundary_regex(needles_tuple: tuple[str, ...]) -> re.Pattern:
    escaped = [re.escape(n.lower()) for n in needles_tuple]
    pattern = r'(?<![a-z0-9])(' + '|'.join(escaped) + r')(?![a-z0-9])'
    return re.compile(pattern)


def contains_any(haystack: str, needles: list[str]) -> bool:
    if not haystack or not needles:
        return False
    hay = haystack.lower()
    pattern = _get_boundary_regex(tuple(needles))
    return pattern.search(hay) is not None


def count_matches(haystack: str, needles: list[str]) -> int:
    if not haystack or not needles:
        return 0
    hay = haystack.lower()
    
    count = 0
    for needle in needles:
        pattern = _get_boundary_regex(tuple([needle]))
        if pattern.search(hay):
            count += 1
    return count


def safe_get(dictionary: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = dictionary
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def minmax_normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low))