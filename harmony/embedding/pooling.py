"""Pool chunk embeddings into track-level vectors."""

from __future__ import annotations

import numpy as np


def mean_pool(chunk_embeddings: np.ndarray) -> np.ndarray:
    """Mean-pool chunk embeddings into a single track vector."""
    if chunk_embeddings.ndim == 1:
        return chunk_embeddings
    return np.mean(chunk_embeddings, axis=0)


def max_norm_pool(chunk_embeddings: np.ndarray) -> np.ndarray:
    """Select the chunk with highest L2 norm."""
    if chunk_embeddings.ndim == 1:
        return chunk_embeddings
    norms = np.linalg.norm(chunk_embeddings, axis=1)
    return chunk_embeddings[int(np.argmax(norms))]


def first_pool(chunk_embeddings: np.ndarray) -> np.ndarray:
    """Use the first chunk only."""
    if chunk_embeddings.ndim == 1:
        return chunk_embeddings
    return chunk_embeddings[0]


_POOLERS = {
    "mean": mean_pool,
    "max_norm": max_norm_pool,
    "first": first_pool,
}


def pool_chunks(chunk_embeddings: np.ndarray, strategy: str) -> np.ndarray:
    """Pool chunk embeddings using the named strategy."""
    try:
        pooler = _POOLERS[strategy]
    except KeyError as e:
        supported = ", ".join(sorted(_POOLERS))
        raise ValueError(f"Unknown pooling strategy {strategy!r}. Supported: {supported}") from e
    return pooler(chunk_embeddings)
