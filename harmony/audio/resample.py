"""Audio resampling."""

from __future__ import annotations

import numpy as np


def resample(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample waveform to target sample rate."""
    if source_sr == target_sr:
        return waveform

    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            "Resampling requires optional dependencies. "
            "Install with: pip install harmony-engine[audio]"
        ) from e

    return librosa.resample(waveform, orig_sr=source_sr, target_sr=target_sr)
