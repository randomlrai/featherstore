"""Badge system for tagging groups with visual status indicators."""

import json
from pathlib import Path
from datetime import datetime, timezone

VALID_BADGES = {"gold", "silver", "bronze", "experimental", "stable", "deprecated", "review"}


def _badges_path(store_path: str) -> Path:
    return Path(store_path) / ".featherstore" / "badges.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_badges(store_path: str) -> dict:
    path = _badges_path(store_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_badges(store_path: str, badges: dict) -> None:
    path = _badges_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(badges, indent=2))


def award_badge(store_path: str, group: str, badge: str, awarded_by: str = "system") -> dict:
    if badge not in VALID_BADGES:
        raise ValueError(f"Invalid badge '{badge}'. Must be one of: {sorted(VALID_BADGES)}")
    badges = load_badges(store_path)
    entry = badges.get(group, {"group": group, "badges": []})
    if badge not in entry["badges"]:
        entry["badges"].append(badge)
    entry["last_awarded"] = _now_iso()
    entry["awarded_by"] = awarded_by
    badges[group] = entry
    save_badges(store_path, badges)
    return entry


def revoke_badge(store_path: str, group: str, badge: str) -> dict:
    badges = load_badges(store_path)
    if group not in badges:
        raise KeyError(f"Group '{group}' has no badges.")
    entry = badges[group]
    entry["badges"] = [b for b in entry["badges"] if b != badge]
    badges[group] = entry
    save_badges(store_path, badges)
    return entry


def get_badges(store_path: str, group: str) -> list:
    badges = load_badges(store_path)
    return badges.get(group, {}).get("badges", [])


def list_groups_with_badge(store_path: str, badge: str) -> list:
    badges = load_badges(store_path)
    return [group for group, entry in badges.items() if badge in entry.get("badges", [])]


def clear_badges(store_path: str, group: str) -> None:
    badges = load_badges(store_path)
    badges.pop(group, None)
    save_badges(store_path, badges)
