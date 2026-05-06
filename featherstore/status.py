"""Group status tracking for FeatherStore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

VALID_STATUSES = {"active", "deprecated", "experimental", "archived", "draft"}


def _status_path(store_path: str) -> Path:
    return Path(store_path) / ".featherstore" / "status.json"


def load_statuses(store_path: str) -> Dict[str, Dict]:
    path = _status_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_statuses(store_path: str, statuses: Dict[str, Dict]) -> None:
    path = _status_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(statuses, f, indent=2)


def set_status(store_path: str, group: str, status: str, reason: Optional[str] = None) -> Dict:
    """Set the status of a group."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {sorted(VALID_STATUSES)}")
    statuses = load_statuses(store_path)
    entry = {"status": status}
    if reason is not None:
        entry["reason"] = reason
    statuses[group] = entry
    save_statuses(store_path, statuses)
    return entry


def remove_status(store_path: str, group: str) -> bool:
    """Remove status entry for a group. Returns True if removed, False if not found."""
    statuses = load_statuses(store_path)
    if group not in statuses:
        return False
    del statuses[group]
    save_statuses(store_path, statuses)
    return True


def get_status(store_path: str, group: str) -> Optional[Dict]:
    """Return the status entry for a group, or None if not set."""
    statuses = load_statuses(store_path)
    return statuses.get(group)


def list_by_status(store_path: str, status: str) -> list:
    """Return all group names with the given status."""
    statuses = load_statuses(store_path)
    return [g for g, entry in statuses.items() if entry.get("status") == status]
