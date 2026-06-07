"""Aggregate chunk-level search hits to track-level scores."""

from __future__ import annotations

from collections import defaultdict


def aggregate_to_tracks(
    chunk_hits: list[tuple[str, str, float]],
    *,
    strategy: str = "max",
) -> list[tuple[str, float, str, float]]:
    """Aggregate chunk hits to track scores.

    Args:
        chunk_hits: list of (track_id, chunk_id, score)
        strategy: max | mean_top3 | sum_topk

    Returns:
        list of (track_id, track_score, best_chunk_id, best_chunk_score)
        sorted by track_score descending.
    """
    by_track: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for track_id, chunk_id, score in chunk_hits:
        by_track[track_id].append((chunk_id, score))

    results: list[tuple[str, float, str, float]] = []
    for track_id, chunks in by_track.items():
        chunks.sort(key=lambda x: x[1], reverse=True)
        best_chunk_id, best_score = chunks[0]

        if strategy == "max":
            track_score = best_score
        elif strategy == "mean_top3":
            top = [s for _, s in chunks[:3]]
            track_score = sum(top) / len(top)
        elif strategy == "sum_topk":
            top = [s for _, s in chunks[:3]]
            track_score = sum(top)
        else:
            track_score = best_score

        results.append((track_id, track_score, best_chunk_id, best_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
