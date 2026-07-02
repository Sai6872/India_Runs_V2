"""
Structured feature engineering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from src import config, utils

logger = utils.get_logger(__name__)


@dataclass
class CandidateFeatures:
    candidate_id: str

    must_have_group_hits: dict[str, bool] = field(default_factory=dict)
    must_have_coverage: float = 0.0
    nice_to_have_coverage: float = 0.0
    implicit_ranking_signal: bool = False
    production_ml_evidence: bool = False
    python_strength: float = 0.0
    skills_score: float = 0.0

    years_of_experience: float = 0.0
    in_experience_band: bool = True
    avg_tenure_months: float = 0.0
    is_title_chaser: bool = False
    is_consulting_only: bool = False
    is_cv_speech_robotics_only: bool = False
    is_langchain_only_recent: bool = False
    is_stale_leadership: bool = False
    is_research_only: bool = False
    career_score: float = 0.0

    location_fit: float = 0.0

    honeypot_suspicion_score: float = 0.0
    honeypot_reasons: list[str] = field(default_factory=list)

    disqualifier_count: int = 0


def _skills_text(candidate: dict[str, Any]) -> str:
    names = [s.get("name", "") for s in candidate.get("skills", []) or []]
    return utils.normalize_for_matching(" ".join(names))


def _full_text(candidate: dict[str, Any], document: str | None = None) -> str:
    if document:
        return document.lower()
    profile = candidate.get("profile", {}) or {}
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        " ".join(e.get("description", "") for e in candidate.get("career_history", []) or []),
        " ".join(e.get("title", "") for e in candidate.get("career_history", []) or []),
    ]
    return utils.normalize_for_matching(" ".join(parts))


def _extract_skill_features(candidate: dict[str, Any], text: str) -> tuple[dict[str, bool], float, float, bool, bool, float]:
    skills_text = _skills_text(candidate)
    combined_text = text + " " + skills_text

    must_have_hits = {
        group: utils.contains_any(combined_text, keywords)
        for group, keywords in config.MUST_HAVE_SKILL_GROUPS.items()
    }
    must_have_coverage = sum(must_have_hits.values()) / len(must_have_hits)

    nice_to_have_hits = sum(
        1 for keywords in config.NICE_TO_HAVE_SKILL_GROUPS.values()
        if utils.contains_any(combined_text, keywords)
    )
    nice_to_have_coverage = nice_to_have_hits / len(config.NICE_TO_HAVE_SKILL_GROUPS)

    implicit_ranking_signal = utils.contains_any(text, config.IMPLICIT_RANKING_SIGNALS)
    production_ml_evidence = utils.contains_any(text, config.PRODUCTION_ML_SIGNALS)

    python_strength = 0.0
    for skill in candidate.get("skills", []) or []:
        if skill.get("name", "").strip().lower() == "python":
            proficiency_scores = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.8, "expert": 1.0}
            python_strength = proficiency_scores.get(skill.get("proficiency", "").lower(), 0.4)
            duration = skill.get("duration_months") or 0
            python_strength = utils.clamp(python_strength + min(duration / 240.0, 0.15))
            break
            
    if python_strength == 0.0 and utils.contains_any(combined_text, ["python"]):
        python_strength = 0.4  

    return (
        must_have_hits, must_have_coverage, nice_to_have_coverage,
        implicit_ranking_signal, production_ml_evidence, python_strength,
    )


def _compute_avg_tenure_months(career_history: list[dict[str, Any]]) -> float:
    durations = [
        e.get("duration_months") for e in career_history or []
        if isinstance(e.get("duration_months"), (int, float))
    ]
    if not durations:
        return 0.0
    return sum(durations) / len(durations)


def _is_consulting_only(career_history: list[dict[str, Any]]) -> bool:
    if not career_history:
        return False
    companies = [e.get("company", "").lower() for e in career_history]
    return all(
        any(firm in company for firm in config.CONSULTING_FIRMS)
        for company in companies
    ) if companies else False


def _months_since(date_str: str | None, reference: date | None = None) -> float | None:
    if not date_str:
        return None
    reference = reference or date.today()
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    return (reference.year - parsed.year) * 12 + (reference.month - parsed.month)


def _is_stale_leadership(profile: dict[str, Any], career_history: list[dict[str, Any]]) -> bool:
    current_title = profile.get("current_title", "").lower()
    if not utils.contains_any(current_title, config.LEADERSHIP_TITLES_NO_CODE):
        return False
    current_entries = [e for e in career_history or [] if e.get("is_current")]
    if not current_entries:
        return False
    description = current_entries[0].get("description", "").lower()
    hands_on = utils.contains_any(description, config.PRODUCTION_ML_SIGNALS + ["code", "coding", "implemented", "built"])
    start_months_ago = _months_since(current_entries[0].get("start_date"))
    long_tenure_no_code = (start_months_ago or 0) >= config.LEADERSHIP_STALE_MONTHS and not hands_on
    return long_tenure_no_code


def _is_langchain_only_recent(career_history: list[dict[str, Any]], years_of_experience: float) -> bool:
    if not career_history:
        return False
        
    sorted_history = sorted(
        career_history, 
        key=lambda x: x.get("start_date", ""), 
        reverse=True
    )
    
    recent_entries = sorted_history[:2]
    recent_text = " ".join(e.get("description", "") for e in recent_entries).lower()
    has_langchain_recent = utils.contains_any(recent_text, config.LANGCHAIN_ONLY_SIGNALS)
    
    if not has_langchain_recent:
        return False
        
    older_entries = sorted_history[2:]
    older_text = " ".join(e.get("description", "") for e in older_entries).lower()
    has_deep_prior_ml = utils.contains_any(
        older_text, config.PRODUCTION_ML_SIGNALS + ["machine learning", "ml pipeline", "recommendation"]
    )
    return not has_deep_prior_ml and years_of_experience < 6


def _is_research_only(career_history: list[dict[str, Any]], text: str) -> bool:
    if not career_history:
        return False
    has_production = utils.contains_any(text, config.PRODUCTION_ML_SIGNALS)
    has_research_signal = utils.contains_any(text, config.RESEARCH_ONLY_SIGNALS)
    return has_research_signal and not has_production


def _location_fit(profile: dict[str, Any]) -> float:
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()

    if any(city in location for city in config.PREFERRED_LOCATIONS):
        return 1.0
    if any(city in location for city in config.ACCEPTABLE_LOCATIONS):
        return 0.85
    if config.PREFERRED_COUNTRY in country:
        return 0.6
    return 0.35


def _honeypot_suspicion(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    skills = candidate.get("skills", []) or []
    expert_zero_duration = [
        s for s in skills
        if s.get("proficiency") == "expert"
        and (s.get("duration_months") or 0) < config.HONEYPOT_EXPERT_SKILL_MIN_MONTHS
    ]
    if len(expert_zero_duration) >= config.HONEYPOT_EXPERT_SKILL_COUNT_THRESHOLD:
        score += 0.5
        reasons.append(
            f"{len(expert_zero_duration)} skills marked 'expert' with under "
            f"{config.HONEYPOT_EXPERT_SKILL_MIN_MONTHS} months of use"
        )

    profile = candidate.get("profile", {}) or {}
    years_of_experience = profile.get("years_of_experience") or 0
    career_history = candidate.get("career_history", []) or []
    total_tenure_years = sum((e.get("duration_months") or 0) for e in career_history) / 12.0
    slack_years = config.HONEYPOT_MAX_TENURE_VS_EXPERIENCE_SLACK_MONTHS / 12.0
    if total_tenure_years > years_of_experience + slack_years and years_of_experience > 0:
        score += 0.5
        reasons.append(
            f"career history totals {total_tenure_years:.1f} yrs but profile states "
            f"{years_of_experience} yrs of experience"
        )

    return utils.clamp(score), reasons


def extract_features(candidate: dict[str, Any], document: str | None = None) -> CandidateFeatures:
    candidate_id = candidate.get("candidate_id", "")
    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []
    text = _full_text(candidate, document)

    (must_have_hits, must_have_coverage, nice_to_have_coverage,
     implicit_ranking_signal, production_ml_evidence, python_strength) = _extract_skill_features(candidate, text)

    years_of_experience = float(profile.get("years_of_experience") or 0.0)
    lower_band = config.JD_EXPERIENCE_MIN_YEARS - config.JD_EXPERIENCE_SOFT_MARGIN_YEARS
    upper_band = config.JD_EXPERIENCE_MAX_YEARS + config.JD_EXPERIENCE_SOFT_MARGIN_YEARS
    in_experience_band = lower_band <= years_of_experience <= upper_band

    avg_tenure_months = _compute_avg_tenure_months(career_history)
    is_title_chaser = (
        len(career_history) >= 3
        and avg_tenure_months > 0
        and avg_tenure_months < config.TITLE_CHASER_MAX_AVG_TENURE_MONTHS
    )
    is_consulting_only = _is_consulting_only(career_history)
    is_cv_speech_robotics_only = (
        utils.contains_any(text, config.CV_SPEECH_ROBOTICS_SIGNALS)
        and not utils.contains_any(text, config.IMPLICIT_RANKING_SIGNALS + ["nlp", "text", "language"])
    )
    is_langchain_only_recent = _is_langchain_only_recent(career_history, years_of_experience)
    is_stale_leadership = _is_stale_leadership(profile, career_history)
    is_research_only = _is_research_only(career_history, text)

    disqualifier_count = sum([
        is_consulting_only, is_cv_speech_robotics_only, is_langchain_only_recent,
        is_stale_leadership, is_research_only,
    ])

    skills_score = utils.clamp(
        0.55 * must_have_coverage
        + 0.15 * nice_to_have_coverage
        + 0.15 * python_strength
        + 0.10 * float(implicit_ranking_signal)
        + 0.05 * float(production_ml_evidence)
    )

    career_score = 1.0
    if not in_experience_band:
        distance = min(abs(years_of_experience - lower_band), abs(years_of_experience - upper_band))
        career_score -= utils.minmax_normalize(distance, 0, config.JD_EXPERIENCE_SOFT_MARGIN_YEARS) * 0.4
    career_score -= 0.15 * float(is_title_chaser)
    career_score -= 0.20 * float(is_consulting_only)
    career_score -= 0.20 * float(is_cv_speech_robotics_only)
    career_score -= 0.20 * float(is_langchain_only_recent)
    career_score -= 0.20 * float(is_stale_leadership)
    career_score -= 0.25 * float(is_research_only)
    career_score += 0.10 * float(production_ml_evidence)
    career_score = utils.clamp(career_score)

    location_fit = _location_fit(profile)
    honeypot_score, honeypot_reasons = _honeypot_suspicion(candidate)

    return CandidateFeatures(
        candidate_id=candidate_id,
        must_have_group_hits=must_have_hits,
        must_have_coverage=must_have_coverage,
        nice_to_have_coverage=nice_to_have_coverage,
        implicit_ranking_signal=implicit_ranking_signal,
        production_ml_evidence=production_ml_evidence,
        python_strength=python_strength,
        skills_score=skills_score,
        years_of_experience=years_of_experience,
        in_experience_band=in_experience_band,
        avg_tenure_months=avg_tenure_months,
        is_title_chaser=is_title_chaser,
        is_consulting_only=is_consulting_only,
        is_cv_speech_robotics_only=is_cv_speech_robotics_only,
        is_langchain_only_recent=is_langchain_only_recent,
        is_stale_leadership=is_stale_leadership,
        is_research_only=is_research_only,
        career_score=career_score,
        location_fit=location_fit,
        honeypot_suspicion_score=honeypot_score,
        honeypot_reasons=honeypot_reasons,
        disqualifier_count=disqualifier_count,
    )