"""Sensitivity classification for feature groups."""

import json
from datetime import datetime, timezone
from pathlib import Path

VALID_LEVELS = {"public", "internal", "confidential", "restricted"}


def _sensitivity_path(store_path: str) -> Path:
    return Path(store_path) / ".featherstore" / "sensitivity.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_sensitivity(store_path: str) -> dict:
    path = _sensitivity_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_sensitivity(store_path: str, data: dict) -> None:
    path = _sensitivity_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def set_sensitivity(store_path: str, group: str, level: str, note: str = "") -> dict:
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid sensitivity level '{level}'. Must be one of: {sorted(VALID_LEVELS)}")
    data = load_sensitivity(store_path)
    data[group] = {
        "level": level,
        "note": note,
        "updated_at": _now_iso(),
    }
    save_sensitivity(store_path, data)
    return data[group]


def remove_sensitivity(store_path: str, group: str) -> bool:
    data = load_sensitivity(store_path)
    if group not in data:
        return False
    del data[group]
    save_sensitivity(store_path, data)
    return True


def get_sensitivity(store_path: str, group: str) -> dict | None:
    data = load_sensitivity(store_path)
    return data.get(group)


def list_by_sensitivity_level(store_path: str, level: str) -> list[str]:
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid sensitivity level '{level}'. Must be one of: {sorted(VALID_LEVELS)}")
    data = load_sensitivity(store_path)
    return [group for group, meta in data.items() if meta.get("level") == level]
