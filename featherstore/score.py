"""Group scoring — assign and track numeric quality scores for feature groups."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _scores_path(store_path: str) -> Path:
    return Path(store_path) / "_featherstore" / "scores.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_scores(store_path: str) -> dict[str, Any]:
    path = _scores_path(store_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_scores(store_path: str, scores: dict[str, Any]) -> None:
    path = _scores_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)


def set_score(
    store_path: str,
    group: str,
    score: float,
    metric: str = "quality",
    note: str | None = None,
) -> dict[str, Any]:
    """Assign a numeric score to a group under a named metric."""
    if not isinstance(score, (int, float)):
        raise TypeError(f"score must be numeric, got {type(score).__name__}")
    scores = load_scores(store_path)
    entry = scores.setdefault(group, {})
    entry[metric] = {
        "score": float(score),
        "note": note,
        "updated_at": _now_iso(),
    }
    save_scores(store_path, scores)
    return entry[metric]


def remove_score(store_path: str, group: str, metric: str = "quality") -> bool:
    """Remove a specific metric score from a group. Returns True if removed."""
    scores = load_scores(store_path)
    if group not in scores or metric not in scores[group]:
        return False
    del scores[group][metric]
    if not scores[group]:
        del scores[group]
    save_scores(store_path, scores)
    return True


def get_scores(store_path: str, group: str) -> dict[str, Any]:
    """Return all metric scores for a group."""
    return load_scores(store_path).get(group, {})


def list_top_groups(
    store_path: str, metric: str = "quality", n: int = 10
) -> list[dict[str, Any]]:
    """Return the top-n groups ranked by a given metric score."""
    scores = load_scores(store_path)
    ranked = [
        {"group": g, "score": v[metric]["score"], "metric": metric}
        for g, v in scores.items()
        if metric in v
    ]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:n]
