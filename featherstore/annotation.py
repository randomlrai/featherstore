"""Group annotation support: attach free-form notes/comments to feature groups."""

import json
from pathlib import Path
from typing import Dict, Optional


def _annotations_path(store_path: str) -> Path:
    return Path(store_path) / "_annotations.json"


def load_annotations(store_path: str) -> Dict[str, Dict]:
    """Load all annotations from disk. Returns empty dict if file missing."""
    path = _annotations_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_annotations(store_path: str, annotations: Dict[str, Dict]) -> None:
    """Persist annotations dict to disk."""
    path = _annotations_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)


def set_annotation(store_path: str, group: str, note: str, author: Optional[str] = None) -> Dict:
    """Set or overwrite the annotation for a group."""
    annotations = load_annotations(store_path)
    entry = {"note": note, "author": author}
    annotations[group] = entry
    save_annotations(store_path, annotations)
    return entry


def get_annotation(store_path: str, group: str) -> Optional[Dict]:
    """Return the annotation for a group, or None if not set."""
    annotations = load_annotations(store_path)
    return annotations.get(group)


def remove_annotation(store_path: str, group: str) -> bool:
    """Remove annotation for a group. Returns True if it existed."""
    annotations = load_annotations(store_path)
    if group not in annotations:
        return False
    del annotations[group]
    save_annotations(store_path, annotations)
    return True


def list_annotations(store_path: str) -> Dict[str, Dict]:
    """Return all group annotations."""
    return load_annotations(store_path)
