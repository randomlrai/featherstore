"""Thread-style comments on feature groups."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _comments_path(store_path: str) -> Path:
    return Path(store_path) / "_comments.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_comments(store_path: str) -> dict:
    path = _comments_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_comments(store_path: str, data: dict) -> None:
    path = _comments_path(store_path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def add_comment(
    store_path: str,
    group: str,
    text: str,
    author: Optional[str] = None,
) -> dict:
    data = load_comments(store_path)
    if group not in data:
        data[group] = []
    entry = {
        "id": str(uuid.uuid4()),
        "text": text,
        "author": author,
        "created_at": _now_iso(),
    }
    data[group].append(entry)
    save_comments(store_path, data)
    return entry


def remove_comment(store_path: str, group: str, comment_id: str) -> bool:
    data = load_comments(store_path)
    if group not in data:
        return False
    before = len(data[group])
    data[group] = [c for c in data[group] if c["id"] != comment_id]
    if len(data[group]) == before:
        return False
    save_comments(store_path, data)
    return True


def get_comments(store_path: str, group: str) -> list:
    data = load_comments(store_path)
    return data.get(group, [])


def clear_comments(store_path: str, group: str) -> None:
    data = load_comments(store_path)
    if group in data:
        del data[group]
        save_comments(store_path, data)
