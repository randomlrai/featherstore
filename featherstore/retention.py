"""Retention policy management for FeatherStore groups."""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


def _retention_path(store_path: str) -> Path:
    return Path(store_path) / ".featherstore" / "retention.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def load_retention(store_path: str) -> dict:
    path = _retention_path(store_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_retention(store_path: str, data: dict) -> None:
    path = _retention_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def set_retention(store_path: str, group: str, days: int) -> dict:
    """Set a retention policy (in days) for a group."""
    if days <= 0:
        raise ValueError("Retention days must be a positive integer.")
    data = load_retention(store_path)
    expires_at = (datetime.utcnow() + timedelta(days=days)).isoformat()
    data[group] = {
        "days": days,
        "set_at": _now_iso(),
        "expires_at": expires_at,
    }
    save_retention(store_path, data)
    return data[group]


def remove_retention(store_path: str, group: str) -> bool:
    """Remove retention policy for a group. Returns True if it existed."""
    data = load_retention(store_path)
    if group not in data:
        return False
    del data[group]
    save_retention(store_path, data)
    return True


def get_retention(store_path: str, group: str) -> Optional[dict]:
    """Return retention policy for a group, or None if not set."""
    return load_retention(store_path).get(group)


def is_expired(store_path: str, group: str) -> bool:
    """Return True if the group's retention period has elapsed."""
    policy = get_retention(store_path, group)
    if policy is None:
        return False
    expires_at = datetime.fromisoformat(policy["expires_at"])
    return datetime.utcnow() >= expires_at


def list_expired(store_path: str) -> list:
    """Return list of group names whose retention period has elapsed."""
    data = load_retention(store_path)
    now = datetime.utcnow()
    return [
        group
        for group, policy in data.items()
        if datetime.fromisoformat(policy["expires_at"]) <= now
    ]
