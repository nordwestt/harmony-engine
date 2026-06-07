"""Parse embedding model keep-alive configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class KeepAlivePolicy:
    mode: Literal["immediate", "timed", "forever"]
    minutes: int | None = None

    @property
    def label(self) -> str:
        if self.mode == "immediate":
            return "unload immediately after each use"
        if self.mode == "forever":
            return "keep loaded for the lifetime of the process"
        return f"keep loaded for {self.minutes} minutes since last use"


def parse_keep_alive(value: str | int | bool | None) -> KeepAlivePolicy:
    """Parse keep_alive from config.

    Accepted values:
    - false, 0, "immediate", "off" → unload after each embed/search
    - positive int or "30", "30m", "30min" → timed retention
    - true, -1, "forever", "always" → never unload while process is running
    """
    if value is None:
        return KeepAlivePolicy(mode="forever")

    if isinstance(value, bool):
        return KeepAlivePolicy(mode="forever" if value else "immediate")

    if isinstance(value, int):
        if value == -1:
            return KeepAlivePolicy(mode="forever")
        if value <= 0:
            return KeepAlivePolicy(mode="immediate")
        return KeepAlivePolicy(mode="timed", minutes=value)

    text = str(value).strip().lower()
    if text in {"false", "0", "immediate", "off", "no"}:
        return KeepAlivePolicy(mode="immediate")
    if text in {"true", "-1", "forever", "always", "on"}:
        return KeepAlivePolicy(mode="forever")

    if text.endswith("min"):
        text = text[:-3]
    if text.endswith("m"):
        text = text[:-1]

    minutes = int(text)
    if minutes <= 0:
        return KeepAlivePolicy(mode="immediate")
    return KeepAlivePolicy(mode="timed", minutes=minutes)
