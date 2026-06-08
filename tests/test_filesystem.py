"""Tests for filesystem scanner."""

from pathlib import Path

from harmony.scanner.filesystem import FilesystemScanner, hash_file


def test_hash_file(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello harmony")
    assert len(hash_file(f)) == 64


def test_scan_finds_audio(tmp_path: Path) -> None:
    music = tmp_path / "Artist" / "Album"
    music.mkdir(parents=True)
    track = music / "Song.flac"
    track.write_bytes(b"fake audio data")

    scanner = FilesystemScanner([tmp_path])
    files = list(scanner.scan())
    assert len(files) == 1
    assert files[0].path == str(track.resolve())
    assert files[0].title == "Song"
    assert files[0].artist == "Artist"
    assert files[0].album == "Album"
