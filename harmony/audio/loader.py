"""Audio file loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path: Path | str, *, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio file and return (waveform, sample_rate).

    Requires optional `harmony-engine[audio]` dependencies (soundfile, librosa).
    """
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "Audio loading requires optional dependencies. "
            "Install with: pip install harmony-engine[audio]"
        ) from e

    path = Path(path)
    waveform, sample_rate = sf.read(path, always_2d=True)
    data = waveform.T  # (channels, samples)

    if mono and data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)

    return data.squeeze(), int(sample_rate)
