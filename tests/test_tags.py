"""Tests for audio tag extraction."""

from pathlib import Path

import pytest

from harmony.scanner.tags import extract_tags


def test_extract_tags_from_path_structure(tmp_path: Path) -> None:
    track = tmp_path / "Radiohead" / "OK Computer" / "Paranoid Android.flac"
    track.parent.mkdir(parents=True)
    track.write_bytes(b"fake audio")

    tags = extract_tags(track)
    assert tags.title == "Paranoid Android"
    assert tags.artist == "Radiohead"
    assert tags.album == "OK Computer"


@pytest.mark.skipif(
    pytest.importorskip("mutagen") is None,
    reason="mutagen not installed",
)
def test_extract_tags_from_id3(tmp_path: Path) -> None:
    from mutagen.id3 import ID3, TALB, TIT2, TPE1

    path = tmp_path / "song.mp3"
    path.write_bytes(b"\x00")
    audio = ID3()
    audio.add(TPE1(encoding=3, text="Radiohead"))
    audio.add(TALB(encoding=3, text="OK Computer"))
    audio.add(TIT2(encoding=3, text="Paranoid Android"))
    audio.save(str(path), v2_version=3)

    tags = extract_tags(path)
    assert tags.artist == "Radiohead"
    assert tags.album == "OK Computer"
    assert tags.title == "Paranoid Android"
