"""
Behavioral scoring from Redrob platform signals.

The JD is explicit that behavioral availability matters: "a
perfect-on-paper candidate who hasn't logged in for 6 months and has a
5% recruiter response rate is, for hiring purposes, not actually
available." This module turns the 23 raw ``redrob_signals`` fields into
one normalized 0-1 behavior score, treated as a *modifier* rather than
the primary ranking criterion (per the architecture spec, section 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from src import config, utils

logger = utils.get_logger(__name__)


@dataclass
class BehaviorFeatures:
    """Normalized (0-1, unless noted) behavioral sub-scores for one candidate."""

    candidate_id: str
    availability_score: float = 0.0     # open_to_work, recency, responsiveness
    engagement_score: float = 0.0       # recruiter interest: views, saves, search appearance
    reliability_score: float = 0.0      # interview completion, verification, profile completeness
    notice_penalty: float = 0.0         # 0 = no penalty, 1 = max penalty
    inactivity_penalty: float = 0.0     # 0 = active, 1 = fully stale
    behavior_score: float = 0.0         # combined 0-1 score fed into the hybrid scorer
    is_stale: bool = False              # last_active_date older than the inactivity threshold


def _days_since(date_str: str | None, reference: date | None = None) -> float | None:
    if not date_str:
        return None
    reference = reference or date.today()
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    return (reference - parsed).days


def compute_behavior_score(candidate: dict[str, Any]) -> BehaviorFeatures:
    """Compute a normalized behavior score from a candidate's ``redrob_signals``.

    Parameters
    ----------
    candidate:
        Raw candidate JSON record containing a ``redrob_signals`` object.

    Returns
    -------
    BehaviorFeatures
        Normalized sub-scores plus the combined ``behavior_score``.
    """
    candidate_id = candidate.get("candidate_id", "")
    signals = candidate.get("redrob_signals", {}) or {}

    open_to_work = bool(signals.get("open_to_work_flag", False))
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    avg_response_hours = float(signals.get("avg_response_time_hours") or 0.0)
    days_inactive = _days_since(signals.get("last_active_date")) or 9999

    responsiveness = utils.clamp(1.0 - utils.minmax_normalize(avg_response_hours, 1, 168))  # 1hr..1wk
    recency = utils.clamp(1.0 - utils.minmax_normalize(days_inactive, 0, config.INACTIVITY_DAYS_THRESHOLD))
    is_stale = days_inactive > config.INACTIVITY_DAYS_THRESHOLD

    availability_score = utils.clamp(
        0.40 * float(open_to_work)
        + 0.30 * response_rate
        + 0.15 * responsiveness
        + 0.15 * recency
    )

    profile_views = float(signals.get("profile_views_received_30d") or 0)
    search_appearance = float(signals.get("search_appearance_30d") or 0)
    saved_by_recruiters = float(signals.get("saved_by_recruiters_30d") or 0)

    engagement_score = utils.clamp(
        0.40 * utils.minmax_normalize(profile_views, 0, 50)
        + 0.30 * utils.minmax_normalize(search_appearance, 0, 50)
        + 0.30 * utils.minmax_normalize(saved_by_recruiters, 0, 20)
    )

    profile_completeness = float(signals.get("profile_completeness_score") or 0) / 100.0
    interview_completion = float(signals.get("interview_completion_rate") or 0.0)
    github_activity = float(signals.get("github_activity_score") or -1)
    github_normalized = 0.0 if github_activity < 0 else utils.minmax_normalize(github_activity, 0, 100)
    verified_bonus = (
        float(bool(signals.get("verified_email")))
        + float(bool(signals.get("verified_phone")))
        + float(bool(signals.get("linkedin_connected")))
    ) / 3.0

    reliability_score = utils.clamp(
        0.35 * profile_completeness
        + 0.25 * interview_completion
        + 0.25 * github_normalized
        + 0.15 * verified_bonus
    )

    notice_days = float(signals.get("notice_period_days") or 0)
    notice_penalty = utils.minmax_normalize(
        max(0.0, notice_days - config.NOTICE_PERIOD_BUYOUT_DAYS),
        0, 150,
    )

    inactivity_penalty = utils.minmax_normalize(days_inactive, config.INACTIVITY_DAYS_THRESHOLD, 720)

    behavior_score = utils.clamp(
        0.45 * availability_score
        + 0.25 * engagement_score
        + 0.30 * reliability_score
        - 0.15 * notice_penalty
        - 0.20 * inactivity_penalty
    )

    return BehaviorFeatures(
        candidate_id=candidate_id,
        availability_score=availability_score,
        engagement_score=engagement_score,
        reliability_score=reliability_score,
        notice_penalty=notice_penalty,
        inactivity_penalty=inactivity_penalty,
        behavior_score=behavior_score,
        is_stale=is_stale,
    )
