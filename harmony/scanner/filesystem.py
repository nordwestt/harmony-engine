"""Filesystem scanner — walk directories and hash audio files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

from harmony.config import FilesystemConfig
from harmony.errors import PathNotAllowedError
from harmony.models import ScannedFile
from harmony.scanner.tags import extract_tags


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

                    tags = extract_tags(path)

                    if content_hash in seen_hashes:
                        yield ScannedFile(
                            path=str(path),
                            content_hash=content_hash,
                            title=tags.title,
                            artist=tags.artist,
                            album=tags.album,
                            album_artist=tags.album_artist,
                            year=tags.year,
                            genre=tags.genre,
                            disc_number=tags.disc_number,
                            track_number=tags.track_number,
                            duration_ms=tags.duration_ms,
                        )
                        continue

                    seen_hashes.add(content_hash)
                    stat = path.stat()
                    yield ScannedFile(
                        path=str(path),
                        content_hash=content_hash,
                        size_bytes=stat.st_size,
                        mtime=stat.st_mtime,
                        title=tags.title,
                        artist=tags.artist,
                        album=tags.album,
                        album_artist=tags.album_artist,
                        year=tags.year,
                        genre=tags.genre,
                        disc_number=tags.disc_number,
                        track_number=tags.track_number,
                        duration_ms=tags.duration_ms,
                    )

    def resolve_path(self, path: str) -> Path:
        return Path(path).expanduser().resolve()


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except AttributeError:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def validate_scan_paths(
    paths: list[str | Path] | None,
    allowed_roots: list[str | Path],
) -> list[Path]:
    """Resolve and validate index scan paths against configured roots.

    When *paths* is None, returns resolved allowed roots.
    Raises PathNotAllowedError when a path is outside the allowlist or invalid.
    """
    roots = [Path(p).expanduser().resolve() for p in allowed_roots]
    if not roots:
        raise PathNotAllowedError(
            "No scan roots configured. Set filesystem.paths in config.yaml "
            "or HARMONY_INDEX_PATHS"
        )

    if paths is None:
        return roots

    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise PathNotAllowedError(f"Path does not exist: {raw}")
        if not path.is_dir():
            raise PathNotAllowedError(f"Path is not a directory: {raw}")
        if not any(_is_under_root(path, root) for root in roots):
            allowed = ", ".join(str(r) for r in roots)
            raise PathNotAllowedError(
                f"Path not allowed: {raw}. Must be under configured roots: {allowed}"
            )
        resolved.append(path)

    return resolved


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
