"""Dependency tracking for feature groups in FeatherStore."""

import json
from pathlib import Path
from typing import Dict, List, Optional


def _deps_path(store_path: str) -> Path:
    return Path(store_path) / ".featherstore" / "dependencies.json"


def load_dependencies(store_path: str) -> Dict[str, List[str]]:
    """Load the dependency graph from disk. Returns empty dict if missing."""
    path = _deps_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_dependencies(store_path: str, deps: Dict[str, List[str]]) -> None:
    """Persist the dependency graph to disk."""
    path = _deps_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(deps, f, indent=2)


def add_dependency(store_path: str, group: str, depends_on: str) -> None:
    """Record that `group` depends on `depends_on`."""
    deps = load_dependencies(store_path)
    current = deps.get(group, [])
    if depends_on not in current:
        current.append(depends_on)
    deps[group] = current
    save_dependencies(store_path, deps)


def remove_dependency(store_path: str, group: str, depends_on: str) -> None:
    """Remove a specific dependency from a group."""
    deps = load_dependencies(store_path)
    current = deps.get(group, [])
    deps[group] = [d for d in current if d != depends_on]
    save_dependencies(store_path, deps)


def get_dependencies(store_path: str, group: str) -> List[str]:
    """Return list of groups that `group` directly depends on."""
    deps = load_dependencies(store_path)
    return deps.get(group, [])


def get_dependents(store_path: str, group: str) -> List[str]:
    """Return list of groups that directly depend on `group`."""
    deps = load_dependencies(store_path)
    return [g for g, parents in deps.items() if group in parents]


def get_full_upstream(store_path: str, group: str, _visited: Optional[set] = None) -> List[str]:
    """Return all transitive upstream dependencies of `group` (DFS)."""
    if _visited is None:
        _visited = set()
    result = []
    for dep in get_dependencies(store_path, group):
        if dep not in _visited:
            _visited.add(dep)
            result.append(dep)
            result.extend(get_full_upstream(store_path, dep, _visited))
    return result


def delete_group_dependencies(store_path: str, group: str) -> None:
    """Remove all dependency entries for a group (as dependent or dependency)."""
    deps = load_dependencies(store_path)
    deps.pop(group, None)
    for g in list(deps.keys()):
        deps[g] = [d for d in deps[g] if d != group]
    save_dependencies(store_path, deps)
