"""Category management for feature groups."""

import json
from pathlib import Path
from typing import Dict, List, Optional


def _categories_path(store_path: str) -> Path:
    return Path(store_path) / "_categories.json"


def load_categories(store_path: str) -> Dict[str, str]:
    """Load category assignments from disk. Returns empty dict if missing."""
    path = _categories_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_categories(store_path: str, categories: Dict[str, str]) -> None:
    """Persist category assignments to disk."""
    path = _categories_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(categories, f, indent=2)


def set_category(store_path: str, group: str, category: str) -> None:
    """Assign a category to a group, overwriting any existing assignment."""
    if not category or not category.strip():
        raise ValueError("category must be a non-empty string")
    categories = load_categories(store_path)
    categories[group] = category.strip()
    save_categories(store_path, categories)


def remove_category(store_path: str, group: str) -> bool:
    """Remove the category assignment for a group. Returns True if removed."""
    categories = load_categories(store_path)
    if group not in categories:
        return False
    del categories[group]
    save_categories(store_path, categories)
    return True


def get_category(store_path: str, group: str) -> Optional[str]:
    """Return the category for a group, or None if unset."""
    return load_categories(store_path).get(group)


def list_by_category(store_path: str, category: str) -> List[str]:
    """Return all groups assigned to the given category."""
    categories = load_categories(store_path)
    return [g for g, c in categories.items() if c == category]


def all_categories(store_path: str) -> List[str]:
    """Return a sorted list of distinct category names in use."""
    categories = load_categories(store_path)
    return sorted(set(categories.values()))
