"""
Ranking: sort scored candidates and select the top 100 with deterministic
tie-breaking, matching the submission_spec.md rules (score non-increasing
as rank increases, ties broken deterministically).
"""

from __future__ import annotations

from src import config, utils
from src.scorer import CandidateScore

logger = utils.get_logger(__name__)


def score_candidates(scores: list[CandidateScore]) -> list[CandidateScore]:
    """Pass-through hook kept for pipeline symmetry with the roadmap's
    naming (``score_candidates`` / ``sort_candidates`` / ``select_top100``).

    Scoring itself happens in ``scorer.py``; this function exists so
    ``rank.py`` can be the single entry point the CLI calls, in case
    future re-scoring logic (e.g. re-ranking) needs to live here.
    """
    return scores


def sort_candidates(scores: list[CandidateScore]) -> list[CandidateScore]:
    """Sort candidates by final score descending, breaking ties deterministically.

    Tie-break rule (matches ``submission_spec.md`` Section 3): when final
    scores are equal, sort by ``candidate_id`` ascending so the ordering
    is 100% reproducible across runs and machines.

    Parameters
    ----------
    scores:
        Unsorted list of :class:`~src.scorer.CandidateScore`.

    Returns
    -------
    list[CandidateScore]
        Sorted descending by ``final_score``, ties broken by ``candidate_id``.
    """
    return sorted(scores, key=lambda s: (-s.final_score, s.candidate_id))


def select_top100(sorted_scores: list[CandidateScore], top_n: int = config.TOP_N) -> list[CandidateScore]:
    """Take the top N candidates from an already-sorted list.

    Parameters
    ----------
    sorted_scores:
        Output of :func:`sort_candidates`.
    top_n:
        Number of candidates to keep (default 100, per the competition spec).

    Returns
    -------
    list[CandidateScore]
        The top ``top_n`` candidates.

    Raises
    ------
    ValueError
        If fewer than ``top_n`` candidates are available to rank.
    """
    if len(sorted_scores) < top_n:
        raise ValueError(
            f"Only {len(sorted_scores)} scored candidates available, need at least {top_n}."
        )
    top = sorted_scores[:top_n]
    logger.info(
        "Selected top %d candidates (score range %.4f - %.4f)",
        top_n, top[0].final_score, top[-1].final_score,
    )
    return top


def honeypot_rate(top_scores: list[CandidateScore]) -> float:
    """Fraction of the given candidates flagged as suspected honeypots.

    Used as a sanity check before writing the submission: the spec
    disqualifies submissions with a honeypot rate above 10% in the top 100.
    """
    if not top_scores:
        return 0.0
    flagged = sum(1 for s in top_scores if s.honeypot_flag)
    return flagged / len(top_scores)


def rank_candidates(scores: list[CandidateScore], top_n: int = config.TOP_N) -> list[CandidateScore]:
    """Full ranking pipeline: score pass-through -> sort -> select top N.

    Parameters
    ----------
    scores:
        All computed :class:`~src.scorer.CandidateScore` objects.
    top_n:
        How many to keep (default: ``config.TOP_N``).

    Returns
    -------
    list[CandidateScore]
        Final ranked top-N list, index 0 == rank 1.
    """
    scored = score_candidates(scores)
    sorted_scores = sort_candidates(scored)
    top = select_top100(sorted_scores, top_n=top_n)

    rate = honeypot_rate(top)
    if rate > 0.10:
        logger.warning(
            "Honeypot rate in top %d is %.1f%%, above the 10%% disqualification threshold!",
            top_n, rate * 100,
        )
    else:
        logger.info("Honeypot rate in top %d: %.1f%%", top_n, rate * 100)

    return top
