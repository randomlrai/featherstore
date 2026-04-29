"""Bookmark support for featherstore — save named references to groups."""

import json
from pathlib import Path
from datetime import datetime, timezone


def _bookmarks_path(store_path: str) -> Path:
    return Path(store_path) / ".featherstore" / "bookmarks.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_bookmarks(store_path: str) -> dict:
    """Load all bookmarks from disk. Returns empty dict if missing."""
    path = _bookmarks_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_bookmarks(store_path: str, bookmarks: dict) -> None:
    """Persist bookmarks to disk."""
    path = _bookmarks_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(bookmarks, f, indent=2)


def add_bookmark(store_path: str, name: str, group: str, note: str = "") -> dict:
    """Create or overwrite a named bookmark pointing to a group."""
    bookmarks = load_bookmarks(store_path)
    entry = {
        "group": group,
        "note": note,
        "created_at": _now_iso(),
    }
    bookmarks[name] = entry
    save_bookmarks(store_path, bookmarks)
    return entry


def remove_bookmark(store_path: str, name: str) -> bool:
    """Remove a bookmark by name. Returns True if it existed."""
    bookmarks = load_bookmarks(store_path)
    if name not in bookmarks:
        return False
    del bookmarks[name]
    save_bookmarks(store_path, bookmarks)
    return True


def get_bookmark(store_path: str, name: str) -> dict | None:
    """Retrieve a single bookmark by name, or None if not found."""
    return load_bookmarks(store_path).get(name)


def list_bookmarks(store_path: str) -> list[dict]:
    """Return all bookmarks as a list of dicts with the bookmark name included."""
    bookmarks = load_bookmarks(store_path)
    return [
        {"name": name, **entry}
        for name, entry in bookmarks.items()
    ]
