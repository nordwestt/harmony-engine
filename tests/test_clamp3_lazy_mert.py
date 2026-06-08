"""Tests for CLaMP3 lazy MERT loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from harmony.config import Config
from harmony.embedding.backends.clamp3 import Clamp3Embedder


def test_text_embed_does_not_load_mert(tmp_path) -> None:
    cfg = Config(data_dir=tmp_path)
    embedder = Clamp3Embedder(cfg)
    runtime = MagicMock()
    runtime.model = MagicMock()
    runtime.tokenizer = MagicMock()
    runtime.mert = None

    with (
        patch.object(embedder, "_ensure_core", return_value=runtime),
        patch("harmony.embedding.backends.clamp3_lib.inference.embed_text") as embed_text,
        patch.object(embedder, "_ensure_mert") as ensure_mert,
    ):
        vector = np.zeros(768, dtype=np.float32)
        tensor = MagicMock()
        tensor.detach.return_value.cpu.return_value.numpy.return_value = vector
        embed_text.return_value = tensor
        embedder.embed_text("dreamy piano")
        ensure_mert.assert_not_called()
