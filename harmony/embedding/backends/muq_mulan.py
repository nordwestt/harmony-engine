"""MuQ-MuLan embedder backend."""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from harmony.config import Config
from harmony.embedding.base import resolve_device
from harmony.embedding.keep_alive import KeepAlivePolicy, parse_keep_alive

EMBEDDING_DIM = 512
DEFAULT_CHECKPOINT = "OpenMuQ/MuQ-MuLan-large"


class MuQMuLanEmbedder:
    """Wrapper around MuQ-MuLan for audio and text embeddings."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Any = None
        self._device: str | None = None
        self._unload_timer: threading.Timer | None = None
        self._session_depth = 0
        self._loading = False
        self._load_error: str | None = None
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return self._config.embedding.model

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self._config.embedding.device)
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def keep_alive_policy(self) -> KeepAlivePolicy:
        return parse_keep_alive(self._config.embedding.keep_alive)

    def _checkpoint(self) -> str:
        return self._config.embedding.checkpoint or DEFAULT_CHECKPOINT

    def preload(self) -> None:
        """Load model weights into memory."""
        self._ensure_model()

    def preload_background(self) -> None:
        """Start loading model weights in a background thread."""
        with self._lock:
            if self._model is not None or self._loading:
                return
            self._loading = True
            self._load_error = None

        def _run() -> None:
            try:
                self._ensure_model()
            except Exception as e:
                with self._lock:
                    self._load_error = str(e)
            finally:
                with self._lock:
                    self._loading = False

        thread = threading.Thread(target=_run, daemon=True, name="muq-preload")
        thread.start()

    def unload(self) -> None:
        """Release model weights from memory."""
        with self._lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    @contextmanager
    def session(self) -> Iterator[None]:
        """Group multiple embed calls before applying keep-alive policy."""
        self._session_depth += 1
        try:
            yield
        finally:
            self._session_depth -= 1
            if self._session_depth == 0:
                self._after_use()

    def _ensure_model(self) -> object:
        with self._lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None

            if self._model is not None:
                return self._model

            try:
                from muq import MuQMuLan
            except ImportError as e:
                raise ImportError(
                    "MuQ-MuLan requires optional dependencies. "
                    "Install with: uv sync --extra embed --extra embed-muq"
                ) from e

            checkpoint = self._checkpoint()
            print(f"Loading MuQ-MuLan ({checkpoint}) on {self.device}…", file=sys.stderr)
            model = MuQMuLan.from_pretrained(checkpoint)
            print("Moving model to device…", file=sys.stderr)
            model = model.to(self.device).eval()
            print("Model ready.", file=sys.stderr)
            self._model = model
            return model

    def _after_use(self) -> None:
        if self._session_depth > 0:
            return

        policy = self.keep_alive_policy
        if policy.mode == "immediate":
            self.unload()
        elif policy.mode == "timed":
            assert policy.minutes is not None
            self._schedule_unload(policy.minutes)

    def _schedule_unload(self, minutes: int) -> None:
        with self._lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()

            def _unload() -> None:
                print(f"Unloading model after {minutes} minutes idle.", file=sys.stderr)
                self.unload()

            self._unload_timer = threading.Timer(minutes * 60, _unload)
            self._unload_timer.daemon = True
            self._unload_timer.start()

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

        with self._lock:
            try:
                from harmony.audio.resample import resample

                model = self._ensure_model()
                import torch

                target_sr = self._config.audio.target_sample_rate
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
            finally:
                self._after_use()

    def embed_text(self, text: str) -> np.ndarray:
        vectors = self.embed_text_batch([text])
        return vectors[0]

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        with self._lock:
            try:
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
            finally:
                self._after_use()
