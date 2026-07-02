"""
Semantic layer: local sentence embeddings and cosine similarity.

Uses ``sentence-transformers/all-MiniLM-L6-v2`` (CPU-only, no network
calls once the model is cached locally) to embed the job description and
every candidate document, then scores semantic fit via cosine similarity.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src import config, utils

logger = utils.get_logger(__name__)

_MODEL_CACHE: dict[str, Any] = {}


def load_embedding_model(model_name: str = config.EMBEDDING_MODEL_NAME):
    """Load (and cache) a local SentenceTransformer model.

    The model is downloaded once and cached under
    ``config.EMBEDDING_CACHE_DIR``; subsequent calls in the same process
    reuse the in-memory instance, and subsequent runs reuse the on-disk
    cache, so no network access is needed after the first run.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier. Defaults to the configured
        ``all-MiniLM-L6-v2`` model.

    Returns
    -------
    sentence_transformers.SentenceTransformer
        A ready-to-use, CPU-pinned embedding model.
    """
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
    """Embed the job description text into a single dense vector.

    Parameters
    ----------
    model:
        A loaded SentenceTransformer (see :func:`load_embedding_model`).
    job_description_text:
        Cleaned job description text (see ``preprocess.load_job_description``).

    Returns
    -------
    np.ndarray
        A 1-D float32 embedding vector, L2-normalized.
    """
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
    """Embed a batch of candidate documents.

    Parameters
    ----------
    model:
        A loaded SentenceTransformer.
    documents:
        Candidate document strings (see ``preprocess.build_candidate_document``).
    batch_size:
        Number of documents encoded per forward pass. Larger batches are
        faster but use more memory; 128 is a safe default on CPU for
        short (< 512 token) documents.

    Returns
    -------
    np.ndarray
        Shape ``(len(documents), embedding_dim)`` float32 array,
        L2-normalized row-wise.
    """
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
    """Compute cosine similarity between one query vector and many candidate vectors.

    Assumes both inputs are already L2-normalized (as produced by
    :func:`embed_job_description` / :func:`embed_candidates`), in which
    case cosine similarity reduces to a dot product - this is what makes
    batch scoring over 100K candidates a single fast matrix multiply.

    Parameters
    ----------
    query_vector:
        Shape ``(embedding_dim,)``.
    candidate_matrix:
        Shape ``(n_candidates, embedding_dim)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_candidates,)``, values in ``[-1, 1]`` (typically
        ``[0, 1]`` for semantically related text).
    """
    if candidate_matrix.size == 0:
        return np.empty((0,), dtype=np.float32)

    # Defensive re-normalization in case inputs weren't pre-normalized.
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    matrix_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True) + 1e-12
    normalized_matrix = candidate_matrix / matrix_norms

    return normalized_matrix @ query_norm
