"""CategoryMixin — integrates category management into FeatherStore."""

from typing import List, Optional
from featherstore.category import (
    set_category,
    remove_category,
    get_category,
    list_by_category,
    all_categories,
)


class CategoryMixin:
    """Mixin that adds category management methods to FeatherStore."""

    def set_category(self, group: str, category: str) -> None:
        """Assign *category* to *group*.

        Parameters
        ----------
        group:
            Name of the feature group.
        category:
            Category label to assign (e.g. ``"raw"``, ``"processed"``).
        """
        set_category(self.path, group, category)

    def remove_category(self, group: str) -> bool:
        """Remove the category assignment for *group*.

        Returns ``True`` if the entry existed and was removed.
        """
        return remove_category(self.path, group)

    def get_category(self, group: str) -> Optional[str]:
        """Return the category assigned to *group*, or ``None``."""
        return get_category(self.path, group)

    def list_by_category(self, category: str) -> List[str]:
        """Return all groups that belong to *category*."""
        return list_by_category(self.path, category)

    def all_categories(self) -> List[str]:
        """Return a sorted list of all distinct categories in use."""
        return all_categories(self.path)
