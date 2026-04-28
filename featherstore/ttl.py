"""TTL (time-to-live) management for feature groups in FeatherStore."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


def _ttl_path(store_path: str) -> Path:
    return Path(store_path) / "_ttl.json"


def load_ttl(store_path: str) -> dict:
    path = _ttl_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_ttl(store_path: str, ttl_data: dict) -> None:
    path = _ttl_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ttl_data, f, indent=2)


def set_ttl(store_path: str, group: str, expires_in_seconds: int) -> dict:
    """Set a TTL for a group. Returns the TTL entry."""
    if expires_in_seconds <= 0:
        raise ValueError("expires_in_seconds must be a positive integer")
    ttl_data = load_ttl(store_path)
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(seconds=expires_in_seconds)).isoformat()
    entry = {
        "group": group,
        "expires_at": expires_at,
        "set_at": now.isoformat(),
        "expires_in_seconds": expires_in_seconds,
    }
    ttl_data[group] = entry
    save_ttl(store_path, ttl_data)
    return entry


def remove_ttl(store_path: str, group: str) -> bool:
    """Remove TTL for a group. Returns True if it existed."""
    ttl_data = load_ttl(store_path)
    if group not in ttl_data:
        return False
    del ttl_data[group]
    save_ttl(store_path, ttl_data)
    return True


def is_expired(store_path: str, group: str) -> Optional[bool]:
    """Return True if expired, False if not, None if no TTL is set."""
    ttl_data = load_ttl(store_path)
    if group not in ttl_data:
        return None
    expires_at = datetime.fromisoformat(ttl_data[group]["expires_at"])
    return datetime.now(timezone.utc) >= expires_at


def list_expired(store_path: str) -> list[str]:
    """Return list of group names whose TTL has expired."""
    ttl_data = load_ttl(store_path)
    now = datetime.now(timezone.utc)
    return [
        group
        for group, entry in ttl_data.items()
        if datetime.fromisoformat(entry["expires_at"]) <= now
    ]


def get_ttl(store_path: str, group: str) -> Optional[dict]:
    """Return the TTL entry for a group, or None if not set."""
    ttl_data = load_ttl(store_path)
    return ttl_data.get(group)
