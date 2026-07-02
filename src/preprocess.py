"""
Data loading and normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from src import config, utils

logger = utils.get_logger(__name__)


@dataclass
class CandidateRecord:
    candidate_id: str
    raw: dict[str, Any]
    document: str = field(default="")


def load_job_description(path: str | Path | None = None) -> str:
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
    candidates_path = Path(path) if path is not None else utils.resolve_candidates_path()
    logger.info("Loading candidates from %s", candidates_path)
    count = 0
    for record in utils.read_jsonl(candidates_path):
        count += 1
        yield record
    logger.info("Loaded %d candidate records", count)


def normalize_text(text: str | None) -> str:
    return utils.clean_text(text)


def _format_career_history(career_history: list[dict[str, Any]]) -> str:
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
    tokens = []
    for skill in skills or []:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "")
        if name:
            tokens.append(f"{name} ({proficiency})" if proficiency else name)
    return ", ".join(tokens)


def build_candidate_document(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) or {}

    headline = normalize_text(profile.get("headline", ""))
    summary = normalize_text(profile.get("summary", ""))
    current_title = normalize_text(profile.get("current_title", ""))
    current_company = normalize_text(profile.get("current_company", ""))
    current_industry = normalize_text(profile.get("current_industry", ""))
    years_of_experience = profile.get("years_of_experience", "")

    skills_text = _format_skills(candidate.get("skills", []))
    career_text = _format_career_history(candidate.get("career_history", []))

    # Strict ordering to prioritize dense info and avoid tokenizer truncation issues
    parts = [
        f"Current role: {current_title} at {current_company} ({current_industry}).",
        f"Skills: {skills_text}." if skills_text else "",
        career_text,
        f"{years_of_experience} years of experience.",
        headline,
        summary,
    ]

    document = " ".join(part for part in parts if part)
    document = normalize_text(document)
    return document[: config.MAX_DOCUMENT_CHARS]


def prepare_candidate_record(candidate: dict[str, Any]) -> CandidateRecord:
    candidate_id = candidate.get("candidate_id", "")
    document = build_candidate_document(candidate)
    return CandidateRecord(candidate_id=candidate_id, raw=candidate, document=document)


def prepare_all_candidates(
    candidates: Iterable[dict[str, Any]],
) -> Iterator[CandidateRecord]:
    skipped = 0
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not candidate_id:
            skipped += 1
            continue
        yield prepare_candidate_record(candidate)
    if skipped:
        logger.warning("Skipped %d candidate(s) missing candidate_id", skipped)