"""Tag management for FeatherStore feature groups."""

import json
from pathlib import Path
from typing import Dict, List, Optional

TAGS_FILE = "tags.json"


def _tags_path(store_path: str) -> Path:
    return Path(store_path) / TAGS_FILE


def load_tags(store_path: str) -> Dict[str, List[str]]:
    """Load the tags mapping from disk. Returns empty dict if not found."""
    path = _tags_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_tags(store_path: str, tags: Dict[str, List[str]]) -> None:
    """Persist the tags mapping to disk."""
    path = _tags_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(tags, f, indent=2)


def add_tag(store_path: str, group: str, tag: str) -> None:
    """Add a tag to a feature group. Duplicate tags are ignored."""
    tags = load_tags(store_path)
    group_tags = tags.get(group, [])
    if tag not in group_tags:
        group_tags.append(tag)
    tags[group] = group_tags
    save_tags(store_path, tags)


def remove_tag(store_path: str, group: str, tag: str) -> None:
    """Remove a tag from a feature group. No-op if tag doesn't exist."""
    tags = load_tags(store_path)
    group_tags = tags.get(group, [])
    tags[group] = [t for t in group_tags if t != tag]
    save_tags(store_path, tags)


def get_tags(store_path: str, group: str) -> List[str]:
    """Return all tags for a given feature group."""
    tags = load_tags(store_path)
    return tags.get(group, [])


def find_groups_by_tag(store_path: str, tag: str) -> List[str]:
    """Return all feature groups that have a specific tag."""
    tags = load_tags(store_path)
    return [group for group, group_tags in tags.items() if tag in group_tags]


def clear_tags(store_path: str, group: str) -> None:
    """Remove all tags from a feature group."""
    tags = load_tags(store_path)
    tags.pop(group, None)
    save_tags(store_path, tags)
