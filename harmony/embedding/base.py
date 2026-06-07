"""Embedder protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embed a single audio waveform."""
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query."""
        ...

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple text queries."""
        ...
