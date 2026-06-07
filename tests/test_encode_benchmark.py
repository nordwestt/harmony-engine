"""Tests for single-file encode benchmarking."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from harmony.config import Config
from harmony.embedding.benchmark import benchmark_encode
from tests.fake_embedder import FakeEmbedder


class SlowPreloadEmbedder(FakeEmbedder):
    def preload(self) -> None:
        time.sleep(0.05)
        super().preload()


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    pytest.importorskip("soundfile")
    import soundfile as sf

    sr = 24000
    t = np.linspace(0, 12, sr * 12, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, audio, sr)
    return path


def test_benchmark_encode_reports_timings(tmp_path: Path, wav_file: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    embedder = FakeEmbedder()
    embedder.preload()
    result = benchmark_encode(wav_file, cfg, embedder=embedder)

    assert result.path == str(wav_file)
    assert result.duration_s == pytest.approx(12.0, rel=0.01)
    assert result.chunks >= 1
    assert result.total_ms >= result.embed_ms
    assert result.vector_dim == 4
    assert result.resample_ms == 0.0
    assert result.model_load_ms == 0.0


def test_benchmark_model_load_separate_from_embed(tmp_path: Path, wav_file: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    result = benchmark_encode(wav_file, cfg, embedder=SlowPreloadEmbedder())

    assert result.model_load_ms >= 50.0
    assert result.embed_ms < 50.0
