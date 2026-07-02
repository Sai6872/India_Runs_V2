"""
Central configuration for the Redrob candidate ranking engine.

Every tunable constant used anywhere in the pipeline lives here so the
rest of the codebase never hardcodes a path, weight, or keyword list.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Default input locations. Both plain and gzipped JSONL are supported by
# utils.read_jsonl(), so either name works without touching this file.
CANDIDATES_JSONL: Path = DATA_DIR / "candidates.jsonl"
CANDIDATES_JSONL_GZ: Path = DATA_DIR / "candidates.jsonl.gz"
JOB_DESCRIPTION_PATH: Path = DATA_DIR / "job_description.md"

SUBMISSION_CSV: Path = OUTPUTS_DIR / "submission.csv"
LOG_FILE: Path = OUTPUTS_DIR / "pipeline.log"

# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE_DIR: Path = MODELS_DIR / "sentence-transformers"
EMBEDDING_BATCH_SIZE: int = 128
EMBEDDING_DEVICE: str = "cpu"
# Caps tokenization/encoding cost per candidate document. all-MiniLM-L6-v2
# truncates at 256 tokens anyway; capping the input text keeps preprocessing
# itself fast across 100K candidates without changing what the model sees.
MAX_DOCUMENT_CHARS: int = 2000

# --------------------------------------------------------------------------
# Ranking output size / determinism
# --------------------------------------------------------------------------

TOP_N: int = 100
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------
# Hybrid score weights (must sum to 1.0)
# --------------------------------------------------------------------------
# Final = 0.40 * Semantic + 0.20 * Skills + 0.15 * Career
#       + 0.15 * Behavior  + 0.10 * Penalties/Bonuses

SCORE_WEIGHTS: dict[str, float] = {
    "semantic": 0.40,
    "skills": 0.20,
    "career": 0.15,
    "behavior": 0.15,
    "penalty_bonus": 0.10,
}

assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9, "SCORE_WEIGHTS must sum to 1.0"

# --------------------------------------------------------------------------
# Experience window the JD is looking for
# --------------------------------------------------------------------------

JD_EXPERIENCE_MIN_YEARS: float = 5.0
JD_EXPERIENCE_MAX_YEARS: float = 9.0
# Candidates outside the band aren't auto-rejected (JD explicitly says so)
# but the score curve tapers off gradually past this many years outside
# the band.
JD_EXPERIENCE_SOFT_MARGIN_YEARS: float = 4.0

# --------------------------------------------------------------------------
# Skill keyword groups (from job_description.docx "skills inventory")
# --------------------------------------------------------------------------

MUST_HAVE_SKILL_GROUPS: dict[str, list[str]] = {
    "embeddings_retrieval": [
        "embedding", "embeddings", "sentence-transformers", "sentence transformers",
        "openai embeddings", "bge", "e5", "retrieval", "semantic search",
        "dense retrieval", "hybrid search", "reranking", "re-ranking",
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "vector database", "vector db",
        "vector store", "ann index", "approximate nearest neighbor",
    ],
    "python": ["python"],
    "evaluation": [
        "ndcg", "mrr", "map", "mean average precision", "a/b test",
        "ab test", "offline evaluation", "online evaluation",
        "evaluation framework", "precision@", "recall@",
    ],
}

NICE_TO_HAVE_SKILL_GROUPS: dict[str, list[str]] = {
    "fine_tuning": ["lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning"],
    "learning_to_rank": [
        "learning to rank", "learning-to-rank", "ltr", "xgboost",
        "lambdamart", "neural ranking",
    ],
    "hr_tech": ["hr-tech", "hr tech", "recruiting tech", "ats", "marketplace"],
    "distributed_systems": [
        "distributed systems", "large-scale inference", "kubernetes",
        "kafka", "spark", "distributed training",
    ],
    "open_source": ["open source", "open-source", "github", "oss contributor"],
}

# Ranking / retrieval / matching signal words used to catch candidates who
# "built a recommendation system" without ever saying "RAG" or "Pinecone" -
# the JD explicitly calls this out as the desired reasoning pattern.
IMPLICIT_RANKING_SIGNALS: list[str] = [
    "recommendation system", "recommender system", "search ranking",
    "search relevance", "personalization", "matching system",
    "ranking system", "recsys", "information retrieval", "nlp",
    "natural language processing", "search engine", "query understanding",
]

PRODUCTION_ML_SIGNALS: list[str] = [
    "production", "deployed", "shipped", "scale", "real users",
    "real-time", "real time", "live traffic", "serving", "inference",
    "latency", "throughput", "monitoring", "on-call",
]

RESEARCH_ONLY_SIGNALS: list[str] = [
    "research lab", "academic lab", "phd research", "published paper",
    "research scientist", "research intern", "research only",
]

# --------------------------------------------------------------------------
# Disqualifier / penalty heuristics (from JD "things we explicitly do NOT want")
# --------------------------------------------------------------------------

CONSULTING_FIRMS: list[str] = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "l&t infotech", "mphasis", "hexaware", "genpact",
]

CV_SPEECH_ROBOTICS_SIGNALS: list[str] = [
    "computer vision", "image classification", "object detection",
    "speech recognition", "speech synthesis", "robotics", "slam",
    "autonomous driving", "audio processing",
]

LANGCHAIN_ONLY_SIGNALS: list[str] = ["langchain", "llamaindex", "llama index"]

TITLE_CHASER_MAX_AVG_TENURE_MONTHS: float = 18.0
LEADERSHIP_TITLES_NO_CODE: list[str] = [
    "architect", "tech lead", "engineering manager", "director",
    "vp engineering", "head of engineering",
]
LEADERSHIP_STALE_MONTHS: int = 18  # "hasn't written production code in 18 months"

# --------------------------------------------------------------------------
# Location preferences (JD: Pune/Noida preferred, Tier-1 Indian cities welcome)
# --------------------------------------------------------------------------

PREFERRED_LOCATIONS: list[str] = ["pune", "noida"]
ACCEPTABLE_LOCATIONS: list[str] = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr",
    "gurgaon", "gurugram", "bangalore", "bengaluru",
]
PREFERRED_COUNTRY: str = "india"

# --------------------------------------------------------------------------
# Behavioral signal thresholds (redrob_signals_doc.md)
# --------------------------------------------------------------------------

BEHAVIOR_POSITIVE_SIGNALS: list[str] = [
    "profile_completeness_score", "recruiter_response_rate",
    "github_activity_score", "interview_completion_rate",
    "search_appearance_30d", "saved_by_recruiters_30d",
    "profile_views_received_30d",
]

BEHAVIOR_NEGATIVE_SIGNALS: list[str] = [
    "notice_period_days", "avg_response_time_hours",
]

INACTIVITY_DAYS_THRESHOLD: int = 180  # last_active_date older than this = stale
NOTICE_PERIOD_IDEAL_DAYS: int = 30
NOTICE_PERIOD_BUYOUT_DAYS: int = 30

# --------------------------------------------------------------------------
# Honeypot heuristics (redrob dataset contains ~80 impossible profiles)
# --------------------------------------------------------------------------

HONEYPOT_EXPERT_SKILL_MIN_MONTHS: int = 6  # "expert" with < this many months used is suspicious
HONEYPOT_EXPERT_SKILL_COUNT_THRESHOLD: int = 5  # 5+ "expert" skills with near-zero duration
HONEYPOT_MAX_TENURE_VS_EXPERIENCE_SLACK_MONTHS: int = 6  # career_history duration can't exceed YOE + slack

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_LEVEL: str = "INFO"
