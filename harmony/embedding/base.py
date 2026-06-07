"""Embedder protocol."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

import numpy as np

from harmony.embedding.keep_alive import KeepAlivePolicy


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def is_loaded(self) -> bool: ...

    @property
    def is_loading(self) -> bool: ...

    @property
    def load_error(self) -> str | None: ...

    @property
    def device(self) -> str: ...

    @property
    def keep_alive_policy(self) -> KeepAlivePolicy: ...

    def preload(self) -> None:
        """Load model weights into memory."""
        ...

    def preload_background(self) -> None:
        """Start loading model weights in a background thread."""
        ...

    def unload(self) -> None:
        """Release model weights from memory."""
        ...

    def session(self) -> AbstractContextManager[None]:
        """Group multiple embed calls before applying keep-alive policy."""
        ...

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embed a single audio waveform."""
        ...

    def embed_audio_batch(
        self,
        waveforms: list[np.ndarray],
        *,
        sample_rate: int,
    ) -> np.ndarray:
        """Embed multiple audio waveforms."""
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query."""
        ...

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple text queries."""
        ...
