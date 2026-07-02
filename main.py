"""
End-to-end CLI entry point for the Redrob candidate ranking engine.

Usage
-----
    python main.py --candidates data/candidates.jsonl --job-description data/job_description.md --out outputs/submission.csv

Runs the full pipeline: preprocessing -> embeddings -> feature
engineering -> behavior scoring -> hybrid scoring -> ranking ->
submission CSV, all CPU-only, all local, within the competition's
runtime/memory budget.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src import config, utils
from src import preprocess
from src import embeddings as emb
from src import feature_engineering as fe
from src import behavior_score as bscore
from src import scorer
from src import rank as rank_mod
from src import generate_submission as gensub

logger = utils.get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redrob candidate ranking pipeline")
    parser.add_argument(
        "--candidates", type=str, default=None,
        help="Path to candidates.jsonl or candidates.jsonl.gz (default: auto-detect under data/)",
    )
    parser.add_argument(
        "--job-description", type=str, default=None,
        help="Path to the job description text file (default: data/job_description.md)",
    )
    parser.add_argument(
        "--out", type=str, default=str(config.SUBMISSION_CSV),
        help="Output CSV path (default: outputs/submission.csv)",
    )
    parser.add_argument(
        "--top-n", type=int, default=config.TOP_N,
        help="Number of candidates to rank and output (default: 100)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.EMBEDDING_BATCH_SIZE,
        help="Embedding batch size (default: 128)",
    )
    return parser.parse_args(argv)


def run_pipeline(
    candidates_path: str | None,
    job_description_path: str | None,
    output_path: str,
    top_n: int = config.TOP_N,
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
) -> Path:
    """Run the complete ranking pipeline and write the submission CSV.

    Returns the path the CSV was written to.
    """
    pipeline_start = time.perf_counter()

    # --- Phase 3: preprocessing ---
    job_description = preprocess.load_job_description(job_description_path)
    raw_candidates = list(preprocess.load_candidates(candidates_path))
    records = list(preprocess.prepare_all_candidates(raw_candidates))
    candidates_by_id = {c.get("candidate_id"): c for c in raw_candidates}
    logger.info("Prepared %d candidate documents", len(records))

    # --- Phase 4: embeddings ---
    model = emb.load_embedding_model()
    jd_vector = emb.embed_job_description(model, job_description)
    documents = [r.document for r in records]
    candidate_vectors = emb.embed_candidates(model, documents, batch_size=batch_size)
    similarities = emb.cosine_similarity(jd_vector, candidate_vectors)
    similarity_by_id = {r.candidate_id: float(sim) for r, sim in zip(records, similarities)}

    # --- Phase 5: feature engineering ---
    features_by_id = {}
    with utils.timer("feature engineering", logger):
        for record in records:
            features_by_id[record.candidate_id] = fe.extract_features(record.raw, record.document)

    # --- Phase 6: behavioral scoring ---
    behavior_by_id = {}
    with utils.timer("behavior scoring", logger):
        for record in records:
            behavior_by_id[record.candidate_id] = bscore.compute_behavior_score(record.raw)

    # --- Phase 7: hybrid scoring ---
    candidate_ids = [r.candidate_id for r in records]
    sim_values = [similarity_by_id[cid] for cid in candidate_ids]
    scores = scorer.score_candidates(candidate_ids, sim_values, features_by_id, behavior_by_id)
    logger.info("Computed hybrid scores for %d candidates", len(scores))

    # --- Phase 8: ranking ---
    top_scores = rank_mod.rank_candidates(scores, top_n=top_n)

    # --- Phase 9: submission ---
    rows = gensub.build_submission_rows(top_scores, candidates_by_id, features_by_id, behavior_by_id)
    output = gensub.write_submission_csv(rows, path=output_path, expected_rows=top_n)

    elapsed = time.perf_counter() - pipeline_start
    logger.info("Full pipeline completed in %.1fs", elapsed)
    if elapsed > 300:
        logger.warning("Pipeline exceeded the 5-minute (300s) competition runtime budget!")

    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_pipeline(
            candidates_path=args.candidates,
            job_description_path=args.job_description,
            output_path=args.out,
            top_n=args.top_n,
            batch_size=args.batch_size,
        )
    except Exception:
        logger.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
