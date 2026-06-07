"""Tests for chunk pooling strategies."""

from __future__ import annotations

import numpy as np
import pytest

from harmony.embedding.pooling import max_norm_pool, mean_pool, pool_chunks


def test_pool_chunks_mean() -> None:
    chunks = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = pool_chunks(chunks, "mean")
    np.testing.assert_allclose(result, mean_pool(chunks))


def test_pool_chunks_max_norm() -> None:
    chunks = np.array([[0.1, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = pool_chunks(chunks, "max_norm")
    np.testing.assert_allclose(result, max_norm_pool(chunks))


def test_pool_chunks_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown pooling strategy"):
        pool_chunks(np.zeros((2, 4), dtype=np.float32), "median")
