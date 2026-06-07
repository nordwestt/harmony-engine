"""MuQ-MuLan embedder wrapper."""

from __future__ import annotations

import numpy as np

from harmony.config import EmbeddingConfig

EMBEDDING_DIM = 512
DEFAULT_CHECKPOINT = "OpenMuQ/MuQ-MuLan-large"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class MuQMuLanEmbedder:
    """Wrapper around MuQ-MuLan for audio and text embeddings."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return self.config.model

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self.config.device)
        return self._device

    def _checkpoint(self) -> str:
        return self.config.checkpoint or DEFAULT_CHECKPOINT

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        import sys

        try:
            import torch
            from muq import MuQMuLan
        except ImportError as e:
            raise ImportError(
                "MuQ-MuLan requires optional dependencies. "
                "Install with: uv sync --extra embed"
            ) from e

        checkpoint = self._checkpoint()
        print(f"Loading MuQ-MuLan ({checkpoint}) on {self.device}…", file=sys.stderr)
        model = MuQMuLan.from_pretrained(checkpoint)
        print("Moving model to device…", file=sys.stderr)
        model = model.to(self.device).eval()
        print("Model ready.", file=sys.stderr)
        self._model = model
        return model

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        vectors = self.embed_audio_batch([waveform], sample_rate=sample_rate)
        return vectors[0]

    def embed_audio_batch(
        self,
        waveforms: list[np.ndarray],
        *,
        sample_rate: int,
    ) -> np.ndarray:
        if not waveforms:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        from harmony.audio.resample import resample

        model = self._ensure_model()
        import torch

        target_sr = 24000
        prepared: list[np.ndarray] = []
        for waveform in waveforms:
            wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if sample_rate != target_sr:
                wav = resample(wav, sample_rate, target_sr)
            prepared.append(wav)

        max_len = max(len(wav) for wav in prepared)
        padded = np.zeros((len(prepared), max_len), dtype=np.float32)
        for i, wav in enumerate(prepared):
            padded[i, : len(wav)] = wav

        tensor = torch.tensor(padded, device=self.device)
        with torch.no_grad():
            embeds = model(wavs=tensor)

        if isinstance(embeds, torch.Tensor):
            arr = embeds.detach().cpu().numpy()
        else:
            arr = np.asarray(embeds, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        vectors = self.embed_text_batch([text])
        return vectors[0]

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        model = self._ensure_model()
        import torch

        with torch.no_grad():
            embeds = model(texts=texts)

        if isinstance(embeds, torch.Tensor):
            arr = embeds.detach().cpu().numpy()
        else:
            arr = np.asarray(embeds, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32)
