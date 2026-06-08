"""Embedder factory and backend registry."""

from __future__ import annotations

from collections.abc import Callable

from harmony.config import Config
from harmony.embedding.backends.clamp3 import EMBEDDING_DIM as CLAMP3_EMBEDDING_DIM
from harmony.embedding.backends.muq_mulan import EMBEDDING_DIM as MUQ_EMBEDDING_DIM
from harmony.embedding.base import Embedder

_BACKEND_DIMENSIONS: dict[str, int] = {
    "muq-mulan": MUQ_EMBEDDING_DIM,
    "clamp3": CLAMP3_EMBEDDING_DIM,
}


def _create_muq_mulan(config: Config) -> Embedder:
    from harmony.embedding.backends.muq_mulan import MuQMuLanEmbedder

    return MuQMuLanEmbedder(config)


def _create_clamp3(config: Config) -> Embedder:
    from harmony.embedding.backends.clamp3 import Clamp3Embedder

    return Clamp3Embedder(config)


_REGISTRY: dict[str, Callable[[Config], Embedder]] = {
    "muq-mulan": _create_muq_mulan,
    "clamp3": _create_clamp3,
}


def list_backends() -> list[str]:
    """Return registered embedding backend names."""
    return sorted(_REGISTRY)


def backend_dimension(model: str) -> int:
    """Return the vector dimension for a registered backend without loading weights."""
    if model not in _BACKEND_DIMENSIONS:
        raise ValueError(
            f"Unknown embedding model {model!r}. Supported: {', '.join(list_backends())}"
        )
    return _BACKEND_DIMENSIONS[model]


def create_embedder(config: Config) -> Embedder:
    """Instantiate the embedder selected by ``config.embedding.model``."""
    backend = config.embedding.model
    if backend not in _REGISTRY:
        raise ValueError(
            f"Unknown embedding model {backend!r}. Supported: {', '.join(list_backends())}"
        )

    if config.embedding.dimension is None:
        config.embedding.dimension = backend_dimension(backend)

    return _REGISTRY[backend](config)
