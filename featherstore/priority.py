"""Priority management for feature groups in FeatherStore."""

import json
from pathlib import Path
from typing import Dict, List, Optional

VALID_PRIORITIES = {"critical", "high", "medium", "low"}


def _priority_path(store_path: str) -> Path:
    return Path(store_path) / "_priority.json"


def load_priorities(store_path: str) -> Dict[str, str]:
    """Load priority assignments from disk. Returns empty dict if missing."""
    path = _priority_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_priorities(store_path: str, priorities: Dict[str, str]) -> None:
    """Persist priority assignments to disk."""
    path = _priority_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(priorities, f, indent=2)


def set_priority(store_path: str, group: str, priority: str) -> Dict[str, str]:
    """Assign a priority level to a group. Returns updated priorities."""
    priority = priority.lower()
    if priority not in VALID_PRIORITIES:
        raise ValueError(
            f"Invalid priority '{priority}'. Must be one of: {sorted(VALID_PRIORITIES)}"
        )
    priorities = load_priorities(store_path)
    priorities[group] = priority
    save_priorities(store_path, priorities)
    return priorities


def remove_priority(store_path: str, group: str) -> Dict[str, str]:
    """Remove priority assignment for a group. Returns updated priorities."""
    priorities = load_priorities(store_path)
    priorities.pop(group, None)
    save_priorities(store_path, priorities)
    return priorities


def get_priority(store_path: str, group: str) -> Optional[str]:
    """Get the priority level for a group, or None if not set."""
    return load_priorities(store_path).get(group)


def list_by_priority(store_path: str, priority: str) -> List[str]:
    """Return all groups with the given priority level."""
    priority = priority.lower()
    priorities = load_priorities(store_path)
    return [g for g, p in priorities.items() if p == priority]


def get_priority_order(store_path: str) -> List[Dict[str, str]]:
    """Return all groups sorted by priority (critical first)."""
    order = ["critical", "high", "medium", "low"]
    priorities = load_priorities(store_path)
    return [
        {"group": g, "priority": p}
        for level in order
        for g, p in priorities.items()
        if p == level
    ]
