"""HTTP client for a running Harmony API server."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def default_api_url() -> str | None:
    return os.environ.get("HARMONY_API_URL")


def search_text(api_url: str, query: str, *, k: int = 50) -> dict[str, Any]:
    """Search via a running harmony serve instance."""
    url = api_url.rstrip("/") + "/v1/search/text"
    body = json.dumps({"query": query, "k": k}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Harmony API at {api_url}. Is `harmony serve` running?"
        ) from e
