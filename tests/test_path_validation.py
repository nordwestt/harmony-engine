"""Tests for index path allowlist validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from harmony.errors import PathNotAllowedError
from harmony.scanner.filesystem import validate_scan_paths


def test_validate_scan_paths_uses_roots_when_omitted(tmp_path: Path) -> None:
    music = tmp_path / "music"
    music.mkdir()
    roots = [music]
    result = validate_scan_paths(None, roots)
    assert result == [music.resolve()]


def test_validate_scan_paths_accepts_subdirectory(tmp_path: Path) -> None:
    music = tmp_path / "music"
    sub = music / "albums"
    sub.mkdir(parents=True)
    result = validate_scan_paths([str(sub)], [music])
    assert result == [sub.resolve()]


def test_validate_scan_paths_rejects_outside_root(tmp_path: Path) -> None:
    music = tmp_path / "music"
    other = tmp_path / "other"
    music.mkdir()
    other.mkdir()
    with pytest.raises(PathNotAllowedError, match="not allowed"):
        validate_scan_paths([str(other)], [music])


def test_validate_scan_paths_rejects_missing_path(tmp_path: Path) -> None:
    music = tmp_path / "music"
    music.mkdir()
    with pytest.raises(PathNotAllowedError, match="does not exist"):
        validate_scan_paths(["/nonexistent/path"], [music])


def test_validate_scan_paths_rejects_file_not_directory(tmp_path: Path) -> None:
    music = tmp_path / "music"
    music.mkdir()
    file_path = tmp_path / "song.mp3"
    file_path.write_text("x")
    with pytest.raises(PathNotAllowedError, match="not a directory"):
        validate_scan_paths([str(file_path)], [music])


def test_validate_scan_paths_requires_configured_roots() -> None:
    with pytest.raises(PathNotAllowedError, match="No scan roots"):
        validate_scan_paths(None, [])
