"""Access control for feature groups — track read/write permissions per group."""

import json
from pathlib import Path
from datetime import datetime, timezone


def _access_path(store_path: str) -> Path:
    return Path(store_path) / "_access.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_access(store_path: str) -> dict:
    path = _access_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_access(store_path: str, data: dict) -> None:
    path = _access_path(store_path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def set_access(store_path: str, group: str, read: list[str], write: list[str], note: str = "") -> dict:
    """Set read/write access lists for a group. Roles/users are arbitrary strings."""
    data = load_access(store_path)
    entry = {
        "read": sorted(set(read)),
        "write": sorted(set(write)),
        "note": note,
        "updated_at": _now_iso(),
    }
    data[group] = entry
    save_access(store_path, data)
    return entry


def remove_access(store_path: str, group: str) -> bool:
    data = load_access(store_path)
    if group not in data:
        return False
    del data[group]
    save_access(store_path, data)
    return True


def get_access(store_path: str, group: str) -> dict | None:
    return load_access(store_path).get(group)


def can_read(store_path: str, group: str, principal: str) -> bool:
    """Return True if principal is in the read list (or '*' wildcard present)."""
    entry = get_access(store_path, group)
    if entry is None:
        return True  # no ACL set — open by default
    return "*" in entry["read"] or principal in entry["read"]


def can_write(store_path: str, group: str, principal: str) -> bool:
    """Return True if principal is in the write list (or '*' wildcard present)."""
    entry = get_access(store_path, group)
    if entry is None:
        return True  # no ACL set — open by default
    return "*" in entry["write"] or principal in entry["write"]


def list_by_principal(store_path: str, principal: str) -> dict:
    """Return all groups where the principal has any access."""
    data = load_access(store_path)
    result = {}
    for group, entry in data.items():
        readable = "*" in entry["read"] or principal in entry["read"]
        writable = "*" in entry["write"] or principal in entry["write"]
        if readable or writable:
            result[group] = {"read": readable, "write": writable}
    return result
