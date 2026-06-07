"""Audio file loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from harmony.config import AudioConfig


def load_audio(
    path: Path | str,
    *,
    mono: bool = True,
    config: AudioConfig | None = None,
) -> tuple[np.ndarray, int]:
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

    audio_config = config or AudioConfig()
    path = Path(path)

    size = path.stat().st_size
    if size > audio_config.max_file_size_bytes:
        raise ValueError(
            f"Audio file exceeds maximum size "
            f"({size} > {audio_config.max_file_size_bytes} bytes): {path}"
        )

    waveform, sample_rate = sf.read(path, always_2d=True)
    data = waveform.T  # (channels, samples)

    if mono and data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)

    result = data.squeeze(), int(sample_rate)
    duration_seconds = len(result[0]) / result[1]
    if duration_seconds > audio_config.max_duration_seconds:
        raise ValueError(
            f"Audio duration exceeds maximum "
            f"({duration_seconds:.0f}s > {audio_config.max_duration_seconds}s): {path}"
        )

    return result
