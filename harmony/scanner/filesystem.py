"""Filesystem scanner — walk directories and hash audio files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

from harmony.config import FilesystemConfig
from harmony.models import ScannedFile


class FilesystemScanner:
    """Discover audio files by walking configured paths on disk."""

    def __init__(
        self,
        paths: list[str | Path],
        *,
        config: FilesystemConfig | None = None,
    ) -> None:
        self.paths = [Path(p).expanduser().resolve() for p in paths]
        self.config = config or FilesystemConfig()
        if self.paths:
            self.config.paths = [str(p) for p in self.paths]

    def scan(self) -> Iterator[ScannedFile]:
        extensions = {e.lower() for e in self.config.extensions}
        seen_hashes: set[str] = set()

        for root in self.paths:
            if not root.exists():
                continue
            for dirpath, _dirnames, filenames in os.walk(root, followlinks=self.config.follow_symlinks):
                for filename in filenames:
                    path = Path(dirpath) / filename
                    if path.suffix.lower() not in extensions:
                        continue
                    try:
                        content_hash = hash_file(path, chunk_mb=4)
                    except OSError:
                        continue

                    if content_hash in seen_hashes:
                        yield ScannedFile(path=str(path), content_hash=content_hash)
                        continue

                    seen_hashes.add(content_hash)
                    stat = path.stat()
                    yield ScannedFile(
                        path=str(path),
                        content_hash=content_hash,
                        size_bytes=stat.st_size,
                        mtime=stat.st_mtime,
                        title=path.stem,
                    )

    def resolve_path(self, path: str) -> Path:
        return Path(path).expanduser().resolve()


def hash_file(path: Path, *, chunk_mb: int = 4) -> str:
    """Streaming SHA-256 of file contents."""
    h = hashlib.sha256()
    chunk_size = chunk_mb * 1024 * 1024
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
