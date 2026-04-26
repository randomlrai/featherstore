"""Scheduled refresh / recompute support for feature groups."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _schedule_path(store_path: str) -> Path:
    return Path(store_path) / ".featherstore" / "schedules.json"


def load_schedules(store_path: str) -> dict[str, Any]:
    """Return all saved schedules, or an empty dict if none exist."""
    path = _schedule_path(store_path)
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


def save_schedules(store_path: str, schedules: dict[str, Any]) -> None:
    """Persist the schedules mapping to disk."""
    path = _schedule_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(schedules, fh, indent=2)


def register_schedule(
    store_path: str,
    group: str,
    cron: str,
    description: str = "",
) -> dict[str, Any]:
    """Register or update a cron schedule for *group*.

    Parameters
    ----------
    store_path:
        Root directory of the FeatherStore.
    group:
        Feature group name to schedule.
    cron:
        Cron expression, e.g. ``"0 3 * * *"`` for daily at 03:00.
    description:
        Optional human-readable description of the schedule.

    Returns
    -------
    dict
        The newly created / updated schedule entry.
    """
    schedules = load_schedules(store_path)
    entry: dict[str, Any] = {
        "group": group,
        "cron": cron,
        "description": description,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "last_run": None,
    }
    schedules[group] = entry
    save_schedules(store_path, schedules)
    return entry


def remove_schedule(store_path: str, group: str) -> bool:
    """Remove the schedule for *group*. Returns True if it existed."""
    schedules = load_schedules(store_path)
    if group not in schedules:
        return False
    del schedules[group]
    save_schedules(store_path, schedules)
    return True


def mark_run(store_path: str, group: str) -> None:
    """Update *last_run* timestamp for *group*'s schedule."""
    schedules = load_schedules(store_path)
    if group not in schedules:
        raise KeyError(f"No schedule registered for group '{group}'.")
    schedules[group]["last_run"] = datetime.now(timezone.utc).isoformat()
    save_schedules(store_path, schedules)


def list_schedules(store_path: str) -> list[dict[str, Any]]:
    """Return all registered schedules as a list."""
    return list(load_schedules(store_path).values())
