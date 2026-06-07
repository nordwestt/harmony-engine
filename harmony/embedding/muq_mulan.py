"""MuQ-MuLan embedder wrapper (not yet implemented)."""

from __future__ import annotations

import numpy as np

from harmony.config import EmbeddingConfig


class MuQMuLanEmbedder:
    """Wrapper around MuQ-MuLan for audio and text embeddings."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = None

    @property
    def name(self) -> str:
        return self.config.model

    @property
    def dimension(self) -> int:
        # Placeholder until model is loaded.
        return 512

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        raise NotImplementedError(
            "MuQ-MuLan embedding is not yet implemented. "
            "Install with: pip install harmony-engine[embed]"
        )

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_model()
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        self._ensure_model()
        raise NotImplementedError

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_text(t) for t in texts])
