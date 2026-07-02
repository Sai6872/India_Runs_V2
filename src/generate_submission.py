"""
Submission generation: factual reasoning text + final CSV writer.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src import config, utils
from src.behavior_score import BehaviorFeatures
from src.feature_engineering import CandidateFeatures
from src.scorer import CandidateScore

logger = utils.get_logger(__name__)

CSV_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def _fmt_years(years: float) -> str:
    return f"{years:g}"


def _positive_fact(
    profile: dict[str, Any],
    features: CandidateFeatures,
) -> str:
    title = profile.get("current_title", "").strip()
    company = profile.get("current_company", "").strip()
    years = features.years_of_experience

    matched_groups = [g for g, hit in features.must_have_group_hits.items() if hit]
    group_labels = {
        "embeddings_retrieval": "embeddings/retrieval",
        "vector_db": "vector database",
        "python": "Python",
        "evaluation": "ranking evaluation",
    }
    skill_phrase = ", ".join(group_labels.get(g, g) for g in matched_groups[:3])

    role_clause = f"{_fmt_years(years)} years of experience"
    if title and company:
        role_clause += f", currently {title} at {company}"

    if skill_phrase:
        return f"{role_clause}, with demonstrated {skill_phrase} background matching the JD's core requirements."
    if features.implicit_ranking_signal:
        return f"{role_clause}; profile shows ranking/search/recommendation-system substance even without exact keyword matches."
    return f"{role_clause}, but limited direct evidence of the JD's core embeddings/retrieval/vector-DB stack."


def _concern_or_highlight(
    features: CandidateFeatures,
    behavior: BehaviorFeatures,
    score: CandidateScore,
) -> str | None:
    if score.penalty_reasons:
        return "Concern: " + score.penalty_reasons[0] + "."
    if behavior.is_stale:
        return "Concern: has been inactive on the platform recently, so availability is uncertain."
    if behavior.behavior_score >= 0.65:
        return "Strong platform engagement (recruiter response and interview follow-through) supports genuine availability."
        
    if features.location_fit == 1.0:
        return "Based in the JD's preferred Pune/Noida hub, which helps with in-person cadence."
    if features.location_fit >= 0.85:
        return "Based in an acceptable Tier-1 tech city."
        
    return None


def generate_reasoning(
    candidate: dict[str, Any],
    features: CandidateFeatures,
    behavior: BehaviorFeatures,
    score: CandidateScore,
) -> str:
    profile = candidate.get("profile", {}) or {}

    sentence_one = _positive_fact(profile, features)
    sentence_two = _concern_or_highlight(features, behavior, score)

    reasoning = sentence_one
    if sentence_two:
        reasoning = f"{sentence_one} {sentence_two}"

    return utils.clean_text(reasoning)


def build_submission_rows(
    ranked_scores: list[CandidateScore],
    candidates_by_id: dict[str, dict[str, Any]],
    features_by_id: dict[str, CandidateFeatures],
    behavior_by_id: dict[str, BehaviorFeatures],
) -> list[dict[str, Any]]:
    rows = []
    for rank, score in enumerate(ranked_scores, start=1):
        candidate = candidates_by_id.get(score.candidate_id, {})
        features = features_by_id.get(score.candidate_id)
        behavior = behavior_by_id.get(score.candidate_id)
        reasoning = (
            generate_reasoning(candidate, features, behavior, score)
            if features is not None and behavior is not None
            else ""
        )
        rows.append(
            {
                "candidate_id": score.candidate_id,
                "rank": rank,
                "score": round(score.final_score, 4),
                "reasoning": reasoning,
            }
        )
    return rows


def write_submission_csv(
    rows: list[dict[str, Any]],
    path: str | Path = config.SUBMISSION_CSV,
    expected_rows: int = config.TOP_N,
) -> Path:
    _validate_rows(rows, expected_rows=expected_rows)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote submission CSV with %d rows to %s", len(rows), output_path)
    return output_path


def _validate_rows(rows: list[dict[str, Any]], expected_rows: int = config.TOP_N) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} rows, got {len(rows)}")

    ranks = [row["rank"] for row in rows]
    if sorted(ranks) != list(range(1, expected_rows + 1)):
        raise ValueError("Ranks must be exactly 1..N, each used once")

    candidate_ids = [row["candidate_id"] for row in rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Duplicate candidate_id detected in submission rows")

    scores = [row["score"] for row in rows]
    for previous, current in zip(scores, scores[1:]):
        if current > previous:
            raise ValueError("Scores must be non-increasing as rank increases")