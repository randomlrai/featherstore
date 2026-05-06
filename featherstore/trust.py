"""Trust scoring for feature groups — track reliability/confidence levels."""

import json
from pathlib import Path
from datetime import datetime, timezone

VALID_LEVELS = ("untrusted", "experimental", "provisional", "trusted", "verified")


def _trust_path(store_path: str) -> Path:
    return Path(store_path) / "_featherstore" / "trust.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_trust(store_path: str) -> dict:
    path = _trust_path(store_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_trust(store_path: str, trust: dict) -> None:
    path = _trust_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(trust, f, indent=2)


def set_trust(store_path: str, group: str, level: str, note: str = "") -> dict:
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid trust level '{level}'. Must be one of: {VALID_LEVELS}")
    trust = load_trust(store_path)
    trust[group] = {
        "level": level,
        "note": note,
        "updated_at": _now_iso(),
    }
    save_trust(store_path, trust)
    return trust[group]


def remove_trust(store_path: str, group: str) -> bool:
    trust = load_trust(store_path)
    if group not in trust:
        return False
    del trust[group]
    save_trust(store_path, trust)
    return True


def get_trust(store_path: str, group: str) -> dict | None:
    trust = load_trust(store_path)
    return trust.get(group)


def list_by_trust_level(store_path: str, level: str) -> list[str]:
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid trust level '{level}'. Must be one of: {VALID_LEVELS}")
    trust = load_trust(store_path)
    return [g for g, meta in trust.items() if meta.get("level") == level]
