"""Freshness tracking: record when a group was last updated and check staleness."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


def _freshness_path(store_path: str) -> Path:
    return Path(store_path) / "_freshness.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_freshness(store_path: str) -> dict:
    path = _freshness_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_freshness(store_path: str, data: dict) -> None:
    path = _freshness_path(store_path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def record_freshness(store_path: str, group: str) -> dict:
    """Record the current timestamp as the last-updated time for a group."""
    data = load_freshness(store_path)
    entry = {"last_updated": _now_iso()}
    data[group] = entry
    save_freshness(store_path, data)
    return entry


def get_freshness(store_path: str, group: str) -> dict | None:
    """Return the freshness entry for a group, or None if not recorded."""
    data = load_freshness(store_path)
    return data.get(group)


def is_stale(store_path: str, group: str, max_age_seconds: float) -> bool:
    """Return True if the group has not been updated within max_age_seconds."""
    entry = get_freshness(store_path, group)
    if entry is None:
        return True
    last_updated = datetime.fromisoformat(entry["last_updated"])
    age = datetime.now(timezone.utc) - last_updated
    return age.total_seconds() > max_age_seconds


def remove_freshness(store_path: str, group: str) -> bool:
    """Remove freshness record for a group. Returns True if it existed."""
    data = load_freshness(store_path)
    if group not in data:
        return False
    del data[group]
    save_freshness(store_path, data)
    return True
