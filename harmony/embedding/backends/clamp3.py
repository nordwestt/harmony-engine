"""CLaMP3 SAAS embedder backend.

Loads MERT + CLaMP3 + XLM-RoBERTa (~several GB RAM). Install optional deps:
``uv sync --extra embed --extra embed-clamp3``
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from harmony.config import Config
from harmony.embedding.backends.clamp3_lib.constants import (
    CLAMP3_HIDDEN_SIZE,
    DEFAULT_MERT_CHECKPOINT,
    DEFAULT_WEIGHTS_FILENAME,
    MERT_SAMPLE_RATE,
    TEXT_MODEL_NAME,
)
from harmony.embedding.base import resolve_device
from harmony.embedding.keep_alive import KeepAlivePolicy, parse_keep_alive

EMBEDDING_DIM = CLAMP3_HIDDEN_SIZE
DEFAULT_CHECKPOINT = "sander-wood/clamp3"


@dataclass
class _Clamp3Runtime:
    model: Any
    tokenizer: Any
    mert: Any | None = None


class Clamp3Embedder:
    """Wrapper around CLaMP3 SAAS for audio and text embeddings."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._runtime: _Clamp3Runtime | None = None
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
    def pooling_strategy(self) -> str:
        return "mean"

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self._config.embedding.device)
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def keep_alive_policy(self) -> KeepAlivePolicy:
        return parse_keep_alive(self._config.embedding.keep_alive)

    def _clamp3_repo_id(self) -> str:
        checkpoint = (self._config.embedding.checkpoint or "").strip()
        if not checkpoint:
            return DEFAULT_CHECKPOINT
        path = Path(checkpoint).expanduser()
        if path.suffix == ".pth":
            return DEFAULT_CHECKPOINT
        if checkpoint.startswith(("OpenMuQ/", "muq-")):
            return DEFAULT_CHECKPOINT
        return checkpoint

    def _clamp3_checkpoint_path(self) -> Path:
        checkpoint = (self._config.embedding.checkpoint or "").strip()
        if checkpoint:
            path = Path(checkpoint).expanduser()
            if path.suffix == ".pth" and path.exists():
                return path

        cache_dir = self._config.data_dir / "models" / "clamp3"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / DEFAULT_WEIGHTS_FILENAME
        if cached.exists():
            return cached

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "CLaMP3 requires optional dependencies "
                f"({e.name or e}). "
                "Install with: uv sync --extra embed --extra embed-clamp3"
            ) from e

        repo_id = self._clamp3_repo_id()
        print(f"Downloading CLaMP3 weights from {repo_id}…", file=sys.stderr)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=DEFAULT_WEIGHTS_FILENAME,
            local_dir=str(cache_dir),
        )
        return Path(downloaded)

    def _mert_checkpoint(self) -> str:
        return DEFAULT_MERT_CHECKPOINT

    def preload(self) -> None:
        """Load model weights into memory."""
        self._ensure_core()

    def preload_background(self) -> None:
        """Start loading model weights in a background thread."""
        with self._lock:
            if self._runtime is not None or self._loading:
                return
            self._loading = True
            self._load_error = None

        def _run() -> None:
            try:
                self._ensure_core()
            except Exception as e:
                with self._lock:
                    self._load_error = str(e)
            finally:
                with self._lock:
                    self._loading = False

        thread = threading.Thread(target=_run, daemon=True, name="clamp3-preload")
        thread.start()

    def unload(self) -> None:
        """Release model weights from memory."""
        with self._lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None
            self._runtime = None
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

    def _cancel_unload_timer(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _ensure_core(self) -> _Clamp3Runtime:
        """Load CLaMP3 + tokenizer (text and audio encoder). MERT is lazy."""
        with self._lock:
            self._cancel_unload_timer()

            if self._runtime is not None:
                return self._runtime

            try:
                from transformers import AutoTokenizer

                from harmony.embedding.backends.clamp3_lib.model import (
                    build_clamp3_model,
                    load_clamp3_checkpoint,
                )
            except ImportError as e:
                raise ImportError(
                    "CLaMP3 requires optional dependencies "
                    f"({e.name or e}). "
                    "Install with: uv sync --extra embed --extra embed-clamp3"
                ) from e

            checkpoint_path = self._clamp3_checkpoint_path()
            print(f"Loading CLaMP3 ({checkpoint_path.name}) on {self.device}…", file=sys.stderr)

            model = build_clamp3_model()
            meta = load_clamp3_checkpoint(model, str(checkpoint_path))
            print(
                f"Loaded CLaMP3 epoch {meta.get('epoch')} "
                f"(eval loss {meta.get('min_eval_loss')}).",
                file=sys.stderr,
            )
            model = model.to(self.device).eval()

            tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
            print("CLaMP3 ready.", file=sys.stderr)

            self._runtime = _Clamp3Runtime(model=model, tokenizer=tokenizer, mert=None)
            return self._runtime

    def _ensure_mert(self) -> Any:
        """Load MERT feature extractor on first audio embed (not needed for text search)."""
        runtime = self._ensure_core()
        with self._lock:
            if runtime.mert is not None:
                return runtime.mert

            from harmony.embedding.backends.clamp3_lib.mert import MertExtractor

            print(f"Loading MERT ({self._mert_checkpoint()}) on {self.device}…", file=sys.stderr)
            runtime.mert = MertExtractor(checkpoint=self._mert_checkpoint(), device=self.device)
            return runtime.mert

    def _after_use(self) -> None:
        if self._session_depth > 0:
            return

        policy = self.keep_alive_policy
        if policy.mode == "immediate":
            self.unload()
        elif policy.mode == "timed":
            assert policy.minutes is not None
            self._schedule_unload(policy.minutes)
        else:
            self._cancel_unload_timer()

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
                from harmony.embedding.backends.clamp3_lib.inference import embed_audio_features

                runtime = self._ensure_core()
                mert = self._ensure_mert()
                import torch

                target_sr = MERT_SAMPLE_RATE
                vectors: list[np.ndarray] = []
                for waveform in waveforms:
                    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
                    if sample_rate != target_sr:
                        wav = resample(wav, sample_rate, target_sr)
                    tensor = torch.tensor(wav, device=self.device).unsqueeze(0)
                    mert_features = mert.extract(tensor)
                    with torch.no_grad():
                        embedding = embed_audio_features(
                            runtime.model,
                            mert_features,
                            device=self.device,
                        )
                    vectors.append(embedding.detach().cpu().numpy().astype(np.float32))

                return np.stack(vectors, axis=0)
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
                from harmony.embedding.backends.clamp3_lib.inference import embed_text

                runtime = self._ensure_core()
                import torch

                max_len = self._config.retrieval.max_query_length
                vectors: list[np.ndarray] = []
                for text in texts:
                    with torch.no_grad():
                        embedding = embed_text(
                            runtime.model,
                            runtime.tokenizer,
                            text,
                            device=self.device,
                            max_length=min(max_len, 512),
                        )
                    vectors.append(embedding.detach().cpu().numpy().astype(np.float32))

                return np.stack(vectors, axis=0)
            finally:
                self._after_use()
