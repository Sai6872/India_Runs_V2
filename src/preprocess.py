"""
Data loading and normalization.

Turns the raw job description and raw candidate JSONL records into a
clean, uniform in-memory representation the rest of the pipeline can
consume without re-parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from src import config, utils

logger = utils.get_logger(__name__)


@dataclass
class CandidateRecord:
    """A single candidate, with the raw JSON kept alongside derived text.

    Keeping ``raw`` around means every later stage (feature engineering,
    behavior scoring, reasoning generation) can reach into the original
    profile without re-loading the file.
    """

    candidate_id: str
    raw: dict[str, Any]
    document: str = field(default="")


def load_job_description(path: str | Path | None = None) -> str:
    """Load and clean the job description text.

    Parameters
    ----------
    path:
        Path to the job description file (``.md`` or ``.docx``-extracted
        text). Defaults to ``config.JOB_DESCRIPTION_PATH``.

    Returns
    -------
    str
        Cleaned, whitespace-normalized job description text.
    """
    jd_path = Path(path) if path is not None else config.JOB_DESCRIPTION_PATH
    if not jd_path.exists():
        raise FileNotFoundError(
            f"Job description not found at {jd_path}. "
            "Pass an explicit path or place the file in data/."
        )

    raw_text = jd_path.read_text(encoding="utf-8")
    cleaned = utils.clean_text(raw_text)
    logger.info("Loaded job description (%d chars) from %s", len(cleaned), jd_path)
    return cleaned


def load_candidates(path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Stream raw candidate records from the candidates JSONL/JSONL.GZ file.

    Parameters
    ----------
    path:
        Explicit path to the candidates file. If omitted, resolves
        automatically via :func:`utils.resolve_candidates_path`.

    Yields
    ------
    dict
        Raw candidate JSON objects, one per candidate.
    """
    candidates_path = Path(path) if path is not None else utils.resolve_candidates_path()
    logger.info("Loading candidates from %s", candidates_path)
    count = 0
    for record in utils.read_jsonl(candidates_path):
        count += 1
        yield record
    logger.info("Loaded %d candidate records", count)


def normalize_text(text: str | None) -> str:
    """Public wrapper around :func:`utils.clean_text` used by callers of this module."""
    return utils.clean_text(text)


def _format_career_history(career_history: list[dict[str, Any]]) -> str:
    """Render career history entries as short natural-language sentences."""
    lines = []
    for entry in career_history or []:
        title = entry.get("title", "")
        company = entry.get("company", "")
        duration = entry.get("duration_months")
        duration_str = f"{duration} months" if duration is not None else "unknown duration"
        description = normalize_text(entry.get("description", ""))
        lines.append(f"{title} at {company} ({duration_str}). {description}")
    return " ".join(lines)


def _format_skills(skills: list[dict[str, Any]]) -> str:
    """Render skills as ``name (proficiency)`` tokens for embedding context."""
    tokens = []
    for skill in skills or []:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "")
        if name:
            tokens.append(f"{name} ({proficiency})" if proficiency else name)
    return ", ".join(tokens)


def build_candidate_document(candidate: dict[str, Any]) -> str:
    """Merge headline, summary, skills and career text into one embedding document.

    This is the text that gets embedded and compared against the job
    description embedding for semantic similarity. It intentionally
    concatenates the most JD-relevant free text fields.

    Parameters
    ----------
    candidate:
        A raw candidate JSON record (schema: ``candidate_schema.json``).

    Returns
    -------
    str
        A single normalized string representing the candidate for
        semantic matching.
    """
    profile = candidate.get("profile", {}) or {}

    headline = normalize_text(profile.get("headline", ""))
    summary = normalize_text(profile.get("summary", ""))
    current_title = normalize_text(profile.get("current_title", ""))
    current_company = normalize_text(profile.get("current_company", ""))
    current_industry = normalize_text(profile.get("current_industry", ""))
    years_of_experience = profile.get("years_of_experience", "")

    skills_text = _format_skills(candidate.get("skills", []))
    career_text = _format_career_history(candidate.get("career_history", []))

    parts = [
        headline,
        f"Current role: {current_title} at {current_company} ({current_industry}).",
        f"{years_of_experience} years of experience.",
        summary,
        f"Skills: {skills_text}." if skills_text else "",
        career_text,
    ]

    document = " ".join(part for part in parts if part)
    document = normalize_text(document)
    return document[: config.MAX_DOCUMENT_CHARS]


def prepare_candidate_record(candidate: dict[str, Any]) -> CandidateRecord:
    """Build a :class:`CandidateRecord` from a raw candidate JSON object."""
    candidate_id = candidate.get("candidate_id", "")
    document = build_candidate_document(candidate)
    return CandidateRecord(candidate_id=candidate_id, raw=candidate, document=document)


def prepare_all_candidates(
    candidates: Iterable[dict[str, Any]],
) -> Iterator[CandidateRecord]:
    """Lazily convert an iterable of raw candidate dicts into :class:`CandidateRecord`.

    Streaming keeps this compatible with :func:`load_candidates`, which
    yields records one at a time rather than materializing all 100K in
    memory at once.
    """
    skipped = 0
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not candidate_id:
            skipped += 1
            continue
        yield prepare_candidate_record(candidate)
    if skipped:
        logger.warning("Skipped %d candidate(s) missing candidate_id", skipped)
