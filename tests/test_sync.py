"""Tests for library sync."""

from pathlib import Path

from harmony.config import Config
from harmony.engine import Engine
from harmony.scanner.filesystem import FilesystemScanner, hash_file
from harmony.storage.metadata import MetadataStore
from harmony.storage.sync import LibrarySync


def _make_track(path: Path, content: bytes = b"track-a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_sync_adds_tracks(tmp_path: Path) -> None:
    music = tmp_path / "music"
    _make_track(music / "a.flac")

    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    sync = LibrarySync(cfg, store)

    report = sync.reconcile(FilesystemScanner([music]))
    assert report.added == 1
    assert store.count_tracks_by_status()["active"] == 1
    store.close()


def test_sync_detects_move_without_reembed(tmp_path: Path) -> None:
    music = tmp_path / "music"
    track = music / "old" / "song.flac"
    _make_track(track)

    cfg = Config(data_dir=tmp_path / "data")
    engine = Engine(cfg.data_dir)
    engine.init()
    engine.index(paths=[str(music)])

    new_path = music / "new" / "song.flac"
    new_path.parent.mkdir(parents=True)
    track.rename(new_path)

    report = engine.index(paths=[str(music)])
    assert report.moved == 1
    assert report.added == 0

    track_record = engine.store.get_track_by_content_hash(hash_file(new_path))
    assert track_record is not None
    assert track_record.primary_path == str(new_path.resolve())
    engine.close()
