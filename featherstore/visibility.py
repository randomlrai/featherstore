"""Visibility control for feature groups (public, private, internal)."""

import json
from pathlib import Path
from datetime import datetime, timezone

VALID_LEVELS = {"public", "private", "internal"}


def _visibility_path(store_path: str) -> Path:
    return Path(store_path) / "_visibility.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_visibility(store_path: str) -> dict:
    path = _visibility_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_visibility(store_path: str, data: dict) -> None:
    path = _visibility_path(store_path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def set_visibility(store_path: str, group: str, level: str, note: str = "") -> dict:
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid visibility level '{level}'. Must be one of {VALID_LEVELS}.")
    data = load_visibility(store_path)
    data[group] = {
        "level": level,
        "note": note,
        "updated_at": _now_iso(),
    }
    save_visibility(store_path, data)
    return data[group]


def remove_visibility(store_path: str, group: str) -> bool:
    data = load_visibility(store_path)
    if group not in data:
        return False
    del data[group]
    save_visibility(store_path, data)
    return True


def get_visibility(store_path: str, group: str) -> dict | None:
    data = load_visibility(store_path)
    return data.get(group)


def list_by_visibility(store_path: str, level: str) -> list[str]:
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid visibility level '{level}'. Must be one of {VALID_LEVELS}.")
    data = load_visibility(store_path)
    return [g for g, v in data.items() if v.get("level") == level]
