"""Shared test double implementing the Embedder protocol."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from harmony.embedding.keep_alive import parse_keep_alive


class FakeEmbedder:
    name = "fake"
    pooling_strategy = "mean"

    is_loaded = False
    is_loading = False
    load_error: str | None = None
    device = "cpu"
    keep_alive_policy = parse_keep_alive("forever")

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self._audio_vector = np.zeros(dimension, dtype=np.float32)
        self._audio_vector[0] = 1.0
        self._text_vector = np.zeros(dimension, dtype=np.float32)
        if dimension > 1:
            self._text_vector[1] = 1.0
        else:
            self._text_vector[0] = 1.0

    def preload(self) -> None:
        self.is_loaded = True

    def preload_background(self) -> None:
        self.is_loaded = True

    def unload(self) -> None:
        self.is_loaded = False

    @contextmanager
    def session(self) -> Iterator[None]:
        yield

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return self._audio_vector.copy()

    def embed_audio_batch(
        self,
        waveforms: list[np.ndarray],
        *,
        sample_rate: int,
    ) -> np.ndarray:
        return np.tile(self._audio_vector, (len(waveforms), 1))

    def embed_text(self, text: str) -> np.ndarray:
        return self._text_vector.copy()

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        return np.tile(self._text_vector, (len(texts), 1))
