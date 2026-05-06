"""Maturity rating module for featherstore groups.

Allows users to assign a maturity level (e.g. experimental, beta, stable, deprecated)
to feature groups, with optional notes and timestamps.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

VALID_LEVELS = {"experimental", "beta", "stable", "deprecated"}


def _maturity_path(store_path: str) -> Path:
    return Path(store_path) / "_featherstore" / "maturity.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_maturity(store_path: str) -> dict:
    path = _maturity_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_maturity(store_path: str, data: dict) -> None:
    path = _maturity_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def set_maturity(store_path: str, group: str, level: str, note: str = "") -> dict:
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid maturity level '{level}'. Must be one of: {sorted(VALID_LEVELS)}")
    data = load_maturity(store_path)
    data[group] = {
        "level": level,
        "note": note,
        "updated_at": _now_iso(),
    }
    save_maturity(store_path, data)
    return data[group]


def remove_maturity(store_path: str, group: str) -> bool:
    data = load_maturity(store_path)
    if group not in data:
        return False
    del data[group]
    save_maturity(store_path, data)
    return True


def get_maturity(store_path: str, group: str) -> dict | None:
    data = load_maturity(store_path)
    return data.get(group)


def list_by_level(store_path: str, level: str) -> list[str]:
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid maturity level '{level}'. Must be one of: {sorted(VALID_LEVELS)}")
    data = load_maturity(store_path)
    return [group for group, meta in data.items() if meta.get("level") == level]
