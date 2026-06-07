"""FAISS-backed ANN index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaissIndex:
    def __init__(self, *, metric: str = "cosine", dimension: int = 512) -> None:
        self.metric = metric
        self.dimension = dimension
        self._index = None
        self._ids: list[str] = []
        self._tombstones: set[str] = set()

    def _ensure_faiss(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FAISS index requires optional dependencies. "
                "Install with: pip install harmony-engine[index]"
            ) from e

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        self._ensure_faiss()
        raise NotImplementedError("FAISS index add() not yet implemented")

    def search(self, query: np.ndarray, k: int) -> tuple[list[str], list[float]]:
        self._ensure_faiss()
        raise NotImplementedError("FAISS index search() not yet implemented")

    def remove(self, ids: list[str]) -> None:
        self._tombstones.update(ids)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {"ids": self._ids, "tombstones": list(self._tombstones), "metric": self.metric}
        p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, path: str) -> None:
        meta_path = Path(path).with_suffix(".json")
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._ids = meta["ids"]
        self._tombstones = set(meta.get("tombstones", []))
        self.metric = meta.get("metric", self.metric)

    @property
    def size(self) -> int:
        return len(self._ids) - len(self._tombstones)
