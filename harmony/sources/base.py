"""Filesystem scanner protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

from harmony.models import ScannedFile


class FilesystemScannerProtocol(Protocol):
    def scan(self) -> Iterator[ScannedFile]:
        """Yield all discoverable audio files under configured paths."""
        ...

    def resolve_path(self, path: str) -> Path:
        """Return an absolute path readable by the audio pipeline."""
        ...
