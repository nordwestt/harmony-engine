"""HTTP client for a running Harmony API server."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_API_HOST = "http://127.0.0.1:8000"


def default_api_url() -> str | None:
    return os.environ.get("HARMONY_API_URL")


def detect_api_url(*, timeout: float = 0.5) -> str | None:
    """Return API base URL if env is set or a local server responds to /health."""
    env_url = default_api_url()
    if env_url:
        return env_url.rstrip("/")

    health_url = DEFAULT_API_HOST.rstrip("/") + "/health"
    request = urllib.request.Request(health_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 200:
                return DEFAULT_API_HOST
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    return None


def _request(
    api_url: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 300,
) -> dict[str, Any]:
    url = api_url.rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(detail)
            message = payload.get("error") or payload.get("detail") or detail
        except json.JSONDecodeError:
            message = detail
        raise RuntimeError(f"API error {e.code}: {message}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Harmony API at {api_url}. Is `harmony serve` running?"
        ) from e


def init_library(api_url: str) -> dict[str, Any]:
    return _request(api_url, "POST", "/v1/init")


def library_stats(api_url: str) -> dict[str, Any]:
    return _request(api_url, "GET", "/v1/library/stats")


def list_tracks(
    api_url: str,
    *,
    offset: int = 0,
    limit: int = 50,
    status: str | None = None,
) -> dict[str, Any]:
    query = f"?offset={offset}&limit={limit}"
    if status:
        query += f"&status={status}"
    return _request(api_url, "GET", f"/v1/library/tracks{query}")


def get_track(api_url: str, track_id: str) -> dict[str, Any]:
    return _request(api_url, "GET", f"/v1/library/tracks/{track_id}")


def sync_history(api_url: str, *, limit: int = 10) -> dict[str, Any]:
    return _request(api_url, "GET", f"/v1/library/sync?limit={limit}")


def purge_library(
    api_url: str,
    *,
    missing: bool = False,
    removed: bool = False,
    orphans: bool = False,
) -> dict[str, int]:
    payload = _request(
        api_url,
        "POST",
        "/v1/library/purge",
        body={"missing": missing, "removed": removed, "orphans": orphans},
    )
    return {k: int(v) for k, v in payload.items()}


def index_library(
    api_url: str,
    *,
    paths: list[str | Path] | None = None,
    full_rescan: bool = False,
    embed: bool = True,
    prune: bool = False,
    reembed: bool = False,
    async_: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "full_rescan": full_rescan,
        "embed": embed,
        "prune": prune,
        "reembed": reembed,
        "async": async_,
    }
    if paths:
        body["paths"] = [str(p) for p in paths]
    return _request(api_url, "POST", "/v1/index", body=body)


def index_job_status(api_url: str, job_id: str) -> dict[str, Any]:
    return _request(api_url, "GET", f"/v1/index/jobs/{job_id}")


def wait_for_index_job(
    api_url: str,
    job_id: str,
    *,
    poll_interval: float = 1.0,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Poll job status until completed or failed."""
    while True:
        status = index_job_status(api_url, job_id)
        if on_progress is not None:
            on_progress(status)
        state = status.get("status")
        if state in ("completed", "failed"):
            if state == "failed":
                raise RuntimeError(status.get("error") or "Index job failed")
            return status
        time.sleep(poll_interval)


def search_text(api_url: str, query: str, *, k: int = 50) -> dict[str, Any]:
    return _request(
        api_url,
        "POST",
        "/v1/search/text",
        body={"query": query, "k": k},
    )


def search_track(api_url: str, track_id: str, *, k: int = 50) -> dict[str, Any]:
    return _request(
        api_url,
        "POST",
        "/v1/search/track",
        body={"track_id": track_id, "k": k},
    )


def ready(api_url: str) -> dict[str, Any]:
    return _request(api_url, "GET", "/v1/ready", timeout=5)
