"""
Semantic layer: local sentence embeddings and cosine similarity.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src import config, utils

logger = utils.get_logger(__name__)

_MODEL_CACHE: dict[str, Any] = {}


def load_embedding_model(model_name: str = config.EMBEDDING_MODEL_NAME):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    from sentence_transformers import SentenceTransformer  # imported lazily

    config.EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with utils.timer(f"loading embedding model '{model_name}'", logger):
        model = SentenceTransformer(
            model_name,
            cache_folder=str(config.EMBEDDING_CACHE_DIR),
            device=config.EMBEDDING_DEVICE,
        )
    _MODEL_CACHE[model_name] = model
    return model


def embed_job_description(model, job_description_text: str) -> np.ndarray:
    embedding = model.encode(
        [job_description_text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return embedding.astype(np.float32)


def embed_candidates(
    model,
    documents: Sequence[str],
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    if not documents:
        return np.empty((0, 0), dtype=np.float32)

    with utils.timer(f"embedding {len(documents)} candidate documents", logger):
        embeddings = model.encode(
            list(documents),
            batch_size=batch_size,
            show_progress_bar=len(documents) > 1000,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return embeddings.astype(np.float32)


def cosine_similarity(query_vector: np.ndarray, candidate_matrix: np.ndarray) -> np.ndarray:
    if candidate_matrix.size == 0:
        return np.empty((0,), dtype=np.float32)

    return candidate_matrix @ query_vector