"""DependencyMixin — integrates dependency tracking into FeatherStore."""

from typing import List
from featherstore.dependency import (
    add_dependency,
    remove_dependency,
    get_dependencies,
    get_dependents,
    get_full_upstream,
    delete_group_dependencies,
)


class DependencyMixin:
    """Mixin that adds dependency graph methods to FeatherStore."""

    def add_dependency(self, group: str, depends_on: str) -> None:
        """Declare that `group` depends on `depends_on`.

        Args:
            group: The feature group that has the dependency.
            depends_on: The feature group being depended upon.
        """
        if group not in self.list_groups():
            raise KeyError(f"Group '{group}' not found in store.")
        if depends_on not in self.list_groups():
            raise KeyError(f"Group '{depends_on}' not found in store.")
        if group == depends_on:
            raise ValueError("A group cannot depend on itself.")
        add_dependency(self.path, group, depends_on)

    def remove_dependency(self, group: str, depends_on: str) -> None:
        """Remove a declared dependency between two groups."""
        remove_dependency(self.path, group, depends_on)

    def get_dependencies(self, group: str) -> List[str]:
        """Return direct dependencies of `group`."""
        return get_dependencies(self.path, group)

    def get_dependents(self, group: str) -> List[str]:
        """Return groups that directly depend on `group`."""
        return get_dependents(self.path, group)

    def get_full_upstream(self, group: str) -> List[str]:
        """Return all transitive upstream dependencies of `group`."""
        return get_full_upstream(self.path, group)

    def delete_group_dependencies(self, group: str) -> None:
        """Remove all dependency records involving `group`."""
        delete_group_dependencies(self.path, group)
