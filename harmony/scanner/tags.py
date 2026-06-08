"""Audio file tag extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioTags:
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    album_artist: str | None = None
    year: int | None = None
    genre: str | None = None
    disc_number: int | None = None
    track_number: int | None = None
    duration_ms: int | None = None


def _first(value: list[str] | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    text = str(value).strip()
    return text or None


def _parse_int(value: str | list[str] | None) -> int | None:
    text = _first(value)
    if text is None:
        return None
    if "/" in text:
        text = text.split("/", 1)[0]
    try:
        return int(text)
    except ValueError:
        return None


def _parse_year(value: str | list[str] | None) -> int | None:
    text = _first(value)
    if text is None:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 4:
        try:
            return int(digits[:4])
        except ValueError:
            return None
    return None


def _tag_value(tag_map: object, easy_key: str, frame_id: str | None = None) -> str | None:
    value = tag_map.get(easy_key) if hasattr(tag_map, "get") else None
    if value is not None:
        return _first(value)

    if frame_id is None or not hasattr(tag_map, "get"):
        return None

    frame = tag_map.get(frame_id)
    if frame is None:
        return None
    text = getattr(frame, "text", None)
    return _first(text)


def _fallback_from_path(path: Path, tags: AudioTags) -> AudioTags:
    """Infer artist/album from directory structure: .../Artist/Album/Track.ext."""
    if tags.artist and tags.album:
        return tags

    parent = path.parent.name
    grandparent = path.parent.parent.name if path.parent.parent != path.parent else None

    # Require Artist/Album/Track layout — skip shallow paths like tmpdir/song.mp3.
    if parent == path.stem or not grandparent or parent == grandparent:
        return tags

    if not tags.album and parent:
        tags.album = parent
    if not tags.artist and grandparent:
        tags.artist = grandparent
    return tags


def _read_embedded_tags(tag_map: object, tags: AudioTags) -> bool:
    title = _tag_value(tag_map, "title", "TIT2")
    tags.title = title or tags.title
    tags.artist = _tag_value(tag_map, "artist", "TPE1")
    tags.album = _tag_value(tag_map, "album", "TALB")
    tags.album_artist = _tag_value(tag_map, "albumartist", "TPE2")
    tags.genre = _tag_value(tag_map, "genre", "TCON")
    tags.year = _parse_year(_tag_value(tag_map, "date", "TDRC")) or _parse_year(
        _tag_value(tag_map, "year")
    )
    tags.disc_number = _parse_int(_tag_value(tag_map, "discnumber", "TPOS"))
    tags.track_number = _parse_int(_tag_value(tag_map, "tracknumber", "TRCK"))

    if not tags.artist:
        tags.artist = tags.album_artist

    return any((tags.artist, tags.album, title, tags.genre, tags.year))


def _load_tag_map(path: Path) -> tuple[object | None, object | None]:
    """Load audio metadata and tag map, with format-specific fallbacks."""
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return None, None

    audio: object | None = None
    try:
        audio = MutagenFile(path)
    except Exception:
        audio = None

    if audio is not None and getattr(audio, "tags", None):
        return audio, audio.tags

    suffix = path.suffix.lower()
    if suffix in {".mp3", ".mp2", ".mpa"}:
        try:
            from mutagen.id3 import ID3

            tag_map = ID3(path)
            if tag_map:
                return None, tag_map
        except Exception:
            pass

    if audio is not None and getattr(audio, "tags", None):
        return audio, audio.tags
    return audio, None


def extract_tags(path: Path) -> AudioTags:
    """Read embedded tags from an audio file, with path-based fallbacks."""
    tags = AudioTags(title=path.stem)

    audio, tag_map = _load_tag_map(path)
    if audio is None and tag_map is None:
        return _fallback_from_path(path, tags)

    info = getattr(audio, "info", None) if audio is not None else None
    if info is not None and getattr(info, "length", None):
        tags.duration_ms = int(info.length * 1000)

    found_embedded = False
    if tag_map:
        found_embedded = _read_embedded_tags(tag_map, tags)

    if found_embedded:
        return tags
    return _fallback_from_path(path, tags)
