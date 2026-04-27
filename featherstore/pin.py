"""Feature pinning: lock a group to a specific version for reproducibility."""

import json
from pathlib import Path
from datetime import datetime, timezone


def _pins_path(store_path: str) -> Path:
    return Path(store_path) / ".featherstore" / "pins.json"


def load_pins(store_path: str) -> dict:
    """Load all pinned groups. Returns empty dict if no pins file exists."""
    path = _pins_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_pins(store_path: str, pins: dict) -> None:
    """Persist pins to disk."""
    path = _pins_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pins, f, indent=2)


def pin_group(store_path: str, group: str, version: str, note: str = "") -> dict:
    """Pin a group to a specific version.

    Args:
        store_path: Root path of the store.
        group: Name of the feature group.
        version: Version string to pin to.
        note: Optional human-readable note.

    Returns:
        The pin metadata dict.
    """
    pins = load_pins(store_path)
    entry = {
        "group": group,
        "version": version,
        "note": note,
        "pinned_at": datetime.now(timezone.utc).isoformat(),
    }
    pins[group] = entry
    save_pins(store_path, pins)
    return entry


def unpin_group(store_path: str, group: str) -> bool:
    """Remove a pin for a group. Returns True if removed, False if not found."""
    pins = load_pins(store_path)
    if group not in pins:
        return False
    del pins[group]
    save_pins(store_path, pins)
    return True


def get_pin(store_path: str, group: str) -> dict | None:
    """Return the pin metadata for a group, or None if not pinned."""
    pins = load_pins(store_path)
    return pins.get(group)


def list_pins(store_path: str) -> list[dict]:
    """Return all pinned groups as a list of metadata dicts."""
    pins = load_pins(store_path)
    return list(pins.values())
