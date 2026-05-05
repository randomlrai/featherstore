"""Watchlist: track groups of interest for monitoring or review."""

import json
from datetime import datetime, timezone
from pathlib import Path


def _watchlist_path(store_path: str) -> Path:
    return Path(store_path) / "_featherstore_watchlist.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_watchlist(store_path: str) -> dict:
    """Load the watchlist from disk. Returns empty dict if missing."""
    path = _watchlist_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_watchlist(store_path: str, watchlist: dict) -> None:
    """Persist the watchlist to disk."""
    path = _watchlist_path(store_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(watchlist, f, indent=2)


def add_to_watchlist(store_path: str, group: str, reason: str = "", tags: list = None) -> dict:
    """Add a group to the watchlist. Returns the created entry."""
    watchlist = load_watchlist(store_path)
    entry = {
        "group": group,
        "reason": reason,
        "tags": tags or [],
        "added_at": _now_iso(),
    }
    watchlist[group] = entry
    save_watchlist(store_path, watchlist)
    return entry


def remove_from_watchlist(store_path: str, group: str) -> bool:
    """Remove a group from the watchlist. Returns True if it was present."""
    watchlist = load_watchlist(store_path)
    if group not in watchlist:
        return False
    del watchlist[group]
    save_watchlist(store_path, watchlist)
    return True


def get_watchlist_entry(store_path: str, group: str) -> dict | None:
    """Return the watchlist entry for a group, or None if not watched."""
    return load_watchlist(store_path).get(group)


def list_watched(store_path: str) -> list[dict]:
    """Return all watchlist entries sorted by added_at."""
    watchlist = load_watchlist(store_path)
    return sorted(watchlist.values(), key=lambda e: e["added_at"])


def is_watched(store_path: str, group: str) -> bool:
    """Return True if the group is on the watchlist."""
    return group in load_watchlist(store_path)
