"""Tests for model keep-alive parsing."""

from harmony.embedding.keep_alive import parse_keep_alive


def test_parse_immediate() -> None:
    assert parse_keep_alive(False).mode == "immediate"
    assert parse_keep_alive(0).mode == "immediate"
    assert parse_keep_alive("off").mode == "immediate"


def test_parse_forever() -> None:
    assert parse_keep_alive(True).mode == "forever"
    assert parse_keep_alive("forever").mode == "forever"
    assert parse_keep_alive(-1).mode == "forever"


def test_parse_timed() -> None:
    policy = parse_keep_alive(30)
    assert policy.mode == "timed"
    assert policy.minutes == 30
    assert parse_keep_alive("15m").minutes == 15
