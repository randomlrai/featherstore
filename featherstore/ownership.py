"""Ownership tracking for feature groups."""

import json
from datetime import datetime, timezone
from pathlib import Path


def _ownership_path(store_path: str) -> Path:
    return Path(store_path) / "_ownership.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_ownership(store_path: str) -> dict:
    path = _ownership_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_ownership(store_path: str, data: dict) -> None:
    path = _ownership_path(store_path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def set_owner(store_path: str, group: str, owner: str, team: str | None = None, email: str | None = None) -> dict:
    data = load_ownership(store_path)
    data[group] = {
        "owner": owner,
        "team": team,
        "email": email,
        "set_at": _now_iso(),
    }
    save_ownership(store_path, data)
    return data[group]


def remove_owner(store_path: str, group: str) -> bool:
    data = load_ownership(store_path)
    if group not in data:
        return False
    del data[group]
    save_ownership(store_path, data)
    return True


def get_owner(store_path: str, group: str) -> dict | None:
    data = load_ownership(store_path)
    return data.get(group)


def list_by_owner(store_path: str, owner: str) -> list[str]:
    data = load_ownership(store_path)
    return [g for g, meta in data.items() if meta.get("owner") == owner]


def list_by_team(store_path: str, team: str) -> list[str]:
    data = load_ownership(store_path)
    return [g for g, meta in data.items() if meta.get("team") == team]
