# Redrob Intelligent Candidate Discovery & Ranking Engine

A reproducible, CPU-only hybrid ranking engine that ranks the best 100
candidates for a given job description from a pool of up to 100,000
candidate profiles. Built for the Redrob Intelligent Candidate Discovery
& Ranking Challenge.

This is **not** a chatbot, agent, or UI — it's a ranking pipeline you
run once to produce a submission CSV.

## Architecture

```
Job Description ─┐
                  ├─> preprocess.py ─> embeddings.py ──────┐
Candidates.jsonl ─┘                                        │
                       preprocess.py ─> feature_engineering.py ─┐
                                     └─> behavior_score.py ──────┼─> scorer.py ─> rank.py ─> generate_submission.py ─> submission.csv
                                                                  ┘
```

| Stage | Module | What it does |
|---|---|---|
| 1 | `src/config.py` | All paths, weights, keyword lists, thresholds |
| 2 | `src/utils.py` | Logging, JSONL streaming, text cleaning, timing |
| 3 | `src/preprocess.py` | Load JD + candidates, build embedding documents |
| 4 | `src/embeddings.py` | Local `sentence-transformers/all-MiniLM-L6-v2` embeddings + cosine similarity |
| 5 | `src/feature_engineering.py` | Skill coverage, career pattern, location fit, honeypot heuristics |
| 6 | `src/behavior_score.py` | Normalized score from the 23 `redrob_signals` fields |
| 7 | `src/scorer.py` | Weighted hybrid score: `0.40*semantic + 0.20*skills + 0.15*career + 0.15*behavior + 0.10*penalty/bonus` |
| 8 | `src/rank.py` | Deterministic sort + top-100 selection + honeypot-rate check |
| 9 | `src/generate_submission.py` | Fact-grounded 1-2 sentence reasoning + CSV writer/validator |

`main.py` wires all nine stages together into a single command.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the competition bundle files under `data/`:

- `data/candidates.jsonl` (or `data/candidates.jsonl.gz`)
- `data/job_description.md` (plain-text export of `job_description.docx`)

The first run downloads `all-MiniLM-L6-v2` into `models/sentence-transformers/`
(a few hundred MB, one-time, requires network). Every subsequent run is
fully offline, satisfying the competition's "no network during ranking"
constraint.

## Run

```bash
python main.py --candidates data/candidates.jsonl --job-description data/job_description.md --out outputs/submission.csv
```

This single command reproduces `outputs/submission.csv` from scratch.
Optional flags: `--top-n` (default 100), `--batch-size` (default 128).

## Design notes

- **Streaming I/O.** `utils.read_jsonl` streams line-by-line (works on
  plain or gzipped input) so the 100K-candidate pool never needs to be
  materialized twice in memory.
- **Vectorized similarity.** JD and candidate embeddings are
  L2-normalized once; semantic scoring for all 100K candidates is a
  single matrix-vector dot product.
- **Reading between the lines.** The JD explicitly warns against
  keyword-only ranking. `feature_engineering.py` therefore also credits
  candidates who show ranking/retrieval/recommendation substance in
  their career history even without exact keyword matches
  (`IMPLICIT_RANKING_SIGNALS`), and penalizes JD-named disqualifiers
  (consulting-only careers, CV/speech/robotics-only backgrounds,
  LangChain-only recent AI work, stale leadership titles, research-only
  careers, title-chasing tenure patterns) even when the skills section
  looks strong on paper.
- **Behavior as a modifier, not the driver.** Per the architecture
  spec, `behavior_score.py` contributes 15% of the final score and also
  feeds a separate inactivity/notice-period penalty term, so a
  perfect-on-paper but unreachable candidate is down-weighted without
  behavior dominating the ranking.
- **Honeypot suppression.** `feature_engineering._honeypot_suspicion`
  flags profiles with internally inconsistent facts (e.g., many
  "expert"-proficiency skills with near-zero months of use, or career
  history totaling more months than the candidate's stated years of
  experience). Flagged profiles get an additional multiplicative penalty
  in `scorer.py`, and `rank.py` logs the honeypot rate in the final
  top 100 against the competition's 10% disqualification threshold.
- **No hallucinated reasoning.** `generate_submission.generate_reasoning`
  builds each 1-2 sentence explanation only from fields already present
  in that candidate's own profile and computed scores — years of
  experience, current title/company, matched skill groups, and the
  specific penalty/bonus reason that most affected their score.
- **Determinism.** Ties are broken by `candidate_id` ascending
  (`rank.sort_candidates`), and the CSV writer validates row count, rank
  sequence, uniqueness, and score monotonicity before writing.

## Performance

Runs CPU-only, no GPU, no hosted APIs. Embedding batch size and document
truncation (`config.MAX_DOCUMENT_CHARS`) are tuned to keep the full
100K-candidate pipeline. 
Designed for CPU-only execution with memory usage below the 16 GB limit. The current implementation prioritizes ranking quality using semantic embeddings and a hybrid scoring strategy. Runtime optimization through accelerated embedding inference is identified as future work.

## Testing checklist

- [x] Loads `candidates.jsonl` (plain or gzipped)
- [x] Reads job description
- [x] Embeddings generated locally, no network calls after first model download
- [x] Scores computed for every candidate
- [x] Top 100 selected with deterministic tie-breaking
- [x] CSV generated matching `submission_spec.md` exactly
- [x] Validator (`_validate_rows` in `generate_submission.py`) passes
- [x] Explainable ranking generated for every selected candidate
