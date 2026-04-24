"""Search and discovery utilities for FeatherStore."""

from typing import Dict, List, Optional
from featherstore.tags import find_groups_by_tag, get_tags


def search_catalog(
    catalog: Dict,
    store_path: str,
    tag: Optional[str] = None,
    name_contains: Optional[str] = None,
) -> List[Dict]:
    """
    Search the feature catalog by tag and/or name substring.

    Args:
        catalog: The catalog dict loaded from FeatherStore._catalog.
        store_path: Path to the store root (for tag lookups).
        tag: Optional tag to filter by.
        name_contains: Optional substring to match against group names.

    Returns:
        List of matching catalog entry dicts with 'group' key added.
    """
    tagged_groups = None
    if tag is not None:
        tagged_groups = set(find_groups_by_tag(store_path, tag))

    results = []
    for group, meta in catalog.items():
        if tagged_groups is not None and group not in tagged_groups:
            continue
        if name_contains is not None and name_contains.lower() not in group.lower():
            continue
        entry = dict(meta)
        entry["group"] = group
        entry["tags"] = get_tags(store_path, group)
        results.append(entry)

    return results


def list_groups(catalog: Dict, store_path: str) -> List[Dict]:
    """Return all feature groups with their metadata and tags."""
    return search_catalog(catalog, store_path)
