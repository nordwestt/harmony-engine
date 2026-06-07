"""Index backend protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class IndexBackend(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Add vectors with string IDs."""
        ...

    def search(self, query: np.ndarray, k: int) -> tuple[list[str], list[float]]:
        """Return (ids, scores) for top-k nearest neighbors."""
        ...

    def remove(self, ids: list[str]) -> None:
        """Mark IDs as removed (tombstone or rebuild)."""
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...

    @property
    def size(self) -> int:
        ...
