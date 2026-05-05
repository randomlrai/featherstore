"""Feature flag support for FeatherStore groups."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _flags_path(store_path: str) -> Path:
    return Path(store_path) / "_flags.json"


def load_flags(store_path: str) -> Dict[str, Dict[str, Any]]:
    path = _flags_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_flags(store_path: str, flags: Dict[str, Dict[str, Any]]) -> None:
    path = _flags_path(store_path)
    with open(path, "w") as f:
        json.dump(flags, f, indent=2)


def set_flag(store_path: str, group: str, flag: str, value: Any = True) -> Dict[str, Any]:
    """Set a named flag on a group. Value defaults to True (boolean toggle)."""
    flags = load_flags(store_path)
    if group not in flags:
        flags[group] = {}
    flags[group][flag] = value
    save_flags(store_path, flags)
    return flags[group]


def remove_flag(store_path: str, group: str, flag: str) -> None:
    """Remove a specific flag from a group. No-op if not present."""
    flags = load_flags(store_path)
    if group in flags and flag in flags[group]:
        del flags[group][flag]
        if not flags[group]:
            del flags[group]
        save_flags(store_path, flags)


def get_flags(store_path: str, group: str) -> Dict[str, Any]:
    """Return all flags set on a group."""
    flags = load_flags(store_path)
    return dict(flags.get(group, {}))


def is_flagged(store_path: str, group: str, flag: str) -> bool:
    """Return True if the given flag is set (and truthy) on the group."""
    flags = load_flags(store_path)
    return bool(flags.get(group, {}).get(flag, False))


def list_flagged_groups(store_path: str, flag: str) -> list:
    """Return all group names that have the specified flag set to a truthy value."""
    flags = load_flags(store_path)
    return [group for group, fmap in flags.items() if fmap.get(flag)]


def clear_flags(store_path: str, group: str) -> None:
    """Remove all flags from a group."""
    flags = load_flags(store_path)
    if group in flags:
        del flags[group]
        save_flags(store_path, flags)
