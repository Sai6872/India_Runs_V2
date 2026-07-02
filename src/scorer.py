"""
Hybrid scorer.

Combines semantic similarity, technical/skills features, career
features, behavioral score, and a penalties/bonuses term into a single
final score per candidate, using the weights defined in ``config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src import config, utils
from src.behavior_score import BehaviorFeatures
from src.feature_engineering import CandidateFeatures

logger = utils.get_logger(__name__)


@dataclass
class CandidateScore:
    """Final hybrid score for one candidate, with component breakdown retained
    for reasoning generation and debugging."""

    candidate_id: str
    semantic_score: float
    skills_score: float
    career_score: float
    behavior_score: float
    penalty_bonus_score: float
    final_score: float
    honeypot_flag: bool
    penalty_reasons: list[str] = field(default_factory=list)
    bonus_reasons: list[str] = field(default_factory=list)


def _penalty_bonus_term(
    features: CandidateFeatures,
    behavior: BehaviorFeatures,
) -> tuple[float, list[str], list[str]]:
    """Compute the penalties/bonuses component (0-1) plus human-readable reasons.

    Starts at a neutral 0.5 and is nudged up for bonuses (location fit,
    strong verification, low notice period) and down for JD-defined
    disqualifiers and honeypot suspicion.
    """
    score = 0.5
    penalty_reasons: list[str] = []
    bonus_reasons: list[str] = []

    if features.is_title_chaser:
        score -= 0.12
        penalty_reasons.append("frequent job-hopping pattern (title-chaser signal)")
    if features.is_consulting_only:
        score -= 0.15
        penalty_reasons.append("career limited to consulting/services firms")
    if features.is_cv_speech_robotics_only:
        score -= 0.15
        penalty_reasons.append("background in CV/speech/robotics without NLP/IR exposure")
    if features.is_langchain_only_recent:
        score -= 0.15
        penalty_reasons.append("recent AI experience looks LangChain-only with no deeper ML history")
    if features.is_stale_leadership:
        score -= 0.15
        penalty_reasons.append("senior/leadership title with no recent hands-on coding evidence")
    if features.is_research_only:
        score -= 0.15
        penalty_reasons.append("research-only background without production deployment")

    if behavior.is_stale:
        score -= 0.15
        penalty_reasons.append("inactive on platform for an extended period")
    if behavior.notice_penalty > 0.3:
        score -= 0.08
        penalty_reasons.append("long notice period beyond the JD's buyout window")

    honeypot_flag = features.honeypot_suspicion_score >= 0.5
    if honeypot_flag:
        score -= 0.6
        penalty_reasons.append("profile shows internally inconsistent facts")

    if features.location_fit >= 0.85:
        score += 0.10
        bonus_reasons.append("based in or near the JD's preferred Pune/Noida hub")
    if features.implicit_ranking_signal and features.skills_score >= 0.4:
        score += 0.05
        bonus_reasons.append("has ranking/retrieval/search substance beyond keyword matches")
    if features.production_ml_evidence:
        score += 0.05
        bonus_reasons.append("clear evidence of shipping ML to production")

    return utils.clamp(score), penalty_reasons, bonus_reasons


def compute_final_score(
    candidate_id: str,
    semantic_similarity: float,
    features: CandidateFeatures,
    behavior: BehaviorFeatures,
) -> CandidateScore:
    """Combine all signals into one final hybrid score for a candidate.

    Parameters
    ----------
    candidate_id:
        The candidate's ID.
    semantic_similarity:
        Cosine similarity between the candidate document and the JD,
        typically in ``[0, 1]``.
    features:
        Structured features from :func:`feature_engineering.extract_features`.
    behavior:
        Behavioral score from :func:`behavior_score.compute_behavior_score`.

    Returns
    -------
    CandidateScore
        Final weighted score plus a component breakdown.
    """
    semantic_score = utils.clamp(semantic_similarity)
    penalty_bonus_score, penalty_reasons, bonus_reasons = _penalty_bonus_term(features, behavior)
    honeypot_flag = features.honeypot_suspicion_score >= 0.5

    weights = config.SCORE_WEIGHTS
    final_score = (
        weights["semantic"] * semantic_score
        + weights["skills"] * features.skills_score
        + weights["career"] * features.career_score
        + weights["behavior"] * behavior.behavior_score
        + weights["penalty_bonus"] * penalty_bonus_score
    )

    # Honeypots are further hard-suppressed on top of the penalty term so a
    # single high semantic score can't accidentally carry one into the top 100.
    if honeypot_flag:
        final_score *= 0.3

    final_score = utils.clamp(final_score)

    return CandidateScore(
        candidate_id=candidate_id,
        semantic_score=semantic_score,
        skills_score=features.skills_score,
        career_score=features.career_score,
        behavior_score=behavior.behavior_score,
        penalty_bonus_score=penalty_bonus_score,
        final_score=final_score,
        honeypot_flag=honeypot_flag,
        penalty_reasons=penalty_reasons,
        bonus_reasons=bonus_reasons,
    )


def score_candidates(
    candidate_ids: list[str],
    semantic_similarities: Any,
    features_by_id: dict[str, CandidateFeatures],
    behavior_by_id: dict[str, BehaviorFeatures],
) -> list[CandidateScore]:
    """Vectorized-friendly batch wrapper around :func:`compute_final_score`.

    Parameters
    ----------
    candidate_ids:
        Ordered list of candidate IDs, aligned with ``semantic_similarities``.
    semantic_similarities:
        Array-like of cosine similarities, same order/length as ``candidate_ids``.
    features_by_id, behavior_by_id:
        Lookup dicts keyed by candidate_id.

    Returns
    -------
    list[CandidateScore]
        One score object per candidate, in the same order as ``candidate_ids``.
    """
    scores = []
    missing = 0
    for candidate_id, similarity in zip(candidate_ids, semantic_similarities):
        features = features_by_id.get(candidate_id)
        behavior = behavior_by_id.get(candidate_id)
        if features is None or behavior is None:
            missing += 1
            continue
        scores.append(compute_final_score(candidate_id, float(similarity), features, behavior))
    if missing:
        logger.warning("Skipped %d candidate(s) missing features/behavior data", missing)
    return scores
