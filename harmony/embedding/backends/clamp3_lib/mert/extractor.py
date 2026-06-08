"""In-memory MERT feature extraction (adapted from clamp3 preprocessing/audio)."""

from __future__ import annotations

import torch

from harmony.embedding.backends.clamp3_lib.constants import (
    DEFAULT_MERT_CHECKPOINT,
    MERT_SAMPLE_RATE,
    MERT_SLIDING_OVERLAP_PERCENT,
    MERT_SLIDING_WINDOW_SEC,
)
from harmony.embedding.backends.clamp3_lib.mert.hf_pretrains import HuBERTFeature


class MertExtractor:
    """Extract MERT features from in-memory waveforms."""

    def __init__(
        self,
        *,
        checkpoint: str = DEFAULT_MERT_CHECKPOINT,
        device: str = "cpu",
        sample_rate: int = MERT_SAMPLE_RATE,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self._feature_extractor = HuBERTFeature(
            checkpoint,
            sample_rate,
            force_half=False,
            disable_backprop=True,
            processor_normalize=True,
        )
        self._feature_extractor.to(device)
        self._feature_extractor.eval()

    def _chunk_features(self, wav: torch.Tensor) -> torch.Tensor:
        features = self._feature_extractor(wav, layer=None, reduction="mean")
        return features.mean(dim=0, keepdim=True)

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return MERT feature sequence of shape ``[T, hidden]``."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("Expected mono waveform tensor [1, T] or [T]")

        wav = self._feature_extractor.process_wav(waveform)
        wav = wav.to(self.device)

        window_samples = int(self.sample_rate * MERT_SLIDING_WINDOW_SEC)
        if MERT_SLIDING_WINDOW_SEC > 0 and wav.shape[-1] > window_samples:
            overlap_samples = int(window_samples * MERT_SLIDING_OVERLAP_PERCENT / 100)
            stride = max(1, window_samples - overlap_samples)
            chunk_features: list[torch.Tensor] = []
            for start in range(0, wav.shape[-1], stride):
                chunk = wav[:, start : start + window_samples]
                if chunk.shape[-1] < int(self.sample_rate * 1):
                    break
                chunk_features.append(self._chunk_features(chunk))
            if chunk_features and chunk_features[-1].shape[-1] < window_samples:
                if len(chunk_features) > 1:
                    chunk_features = chunk_features[:-1]
            if chunk_features:
                features = torch.cat(chunk_features, dim=1)
                return features.reshape(-1, features.size(-1))

        features = self._chunk_features(wav)
        return features.reshape(-1, features.size(-1))
