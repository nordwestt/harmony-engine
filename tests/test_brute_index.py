"""Tests for brute-force index."""

import numpy as np

from harmony.index.brute import BruteForceIndex


def test_brute_search_returns_nearest() -> None:
    index = BruteForceIndex()
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    index.add(["a", "b", "c"], vectors)

    ids, scores = index.search(np.array([1.0, 0.0, 0.0]), k=2)
    assert ids[0] == "a"
    assert ids[1] == "c"
    assert scores[0] > scores[1]


def test_brute_remove_tombstones() -> None:
    index = BruteForceIndex()
    vectors = np.eye(2, dtype=np.float32)
    index.add(["a", "b"], vectors)
    index.remove(["a"])

    ids, _ = index.search(np.array([1.0, 0.0]), k=2)
    assert ids == ["b"]
