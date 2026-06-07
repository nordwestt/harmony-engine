"""Brute-force cosine similarity index for MVP / small libraries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class BruteForceIndex:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._tombstones: set[str] = set()

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        normed = np.stack([_normalize(v) for v in vectors])

        if self._vectors is None:
            self._ids = list(ids)
            self._vectors = normed
            return

        # Replace existing IDs, append new ones.
        id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}
        for i, id_ in enumerate(ids):
            if id_ in self._tombstones:
                self._tombstones.discard(id_)
            if id_ in id_to_idx:
                self._vectors[id_to_idx[id_]] = normed[i]
            else:
                self._ids.append(id_)
                self._vectors = np.vstack([self._vectors, normed[i]])

    def search(self, query: np.ndarray, k: int) -> tuple[list[str], list[float]]:
        if self._vectors is None or len(self._ids) == 0:
            return [], []

        q = _normalize(np.asarray(query, dtype=np.float32).reshape(-1))
        scores = self._vectors @ q

        ranked = sorted(
            [
                (self._ids[i], float(scores[i]))
                for i in range(len(self._ids))
                if self._ids[i] not in self._tombstones
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        return [r[0] for r in ranked], [r[1] for r in ranked]

    def remove(self, ids: list[str]) -> None:
        self._tombstones.update(ids)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self._vectors is None:
            np.save(p.with_suffix(".npy"), np.array([]))
        else:
            np.save(p.with_suffix(".npy"), self._vectors)
        meta = {"ids": self._ids, "tombstones": list(self._tombstones)}
        p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        vectors_path = p.with_suffix(".npy")
        meta_path = p.with_suffix(".json")
        if not meta_path.exists():
            self._ids = []
            self._vectors = None
            self._tombstones = set()
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._ids = meta["ids"]
        self._tombstones = set(meta.get("tombstones", []))
        if vectors_path.exists():
            loaded = np.load(vectors_path)
            self._vectors = loaded if loaded.size else None
        else:
            self._vectors = None

    @property
    def size(self) -> int:
        return len(self._ids) - len(self._tombstones)
