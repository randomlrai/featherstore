"""VisibilityMixin for FeatherStore — exposes visibility control on the store object."""

from featherstore.visibility import (
    set_visibility,
    remove_visibility,
    get_visibility,
    list_by_visibility,
    VALID_LEVELS,
)


class VisibilityMixin:
    """Mixin that adds visibility management methods to FeatherStore."""

    def set_visibility(self, group: str, level: str, note: str = "") -> dict:
        """Set the visibility level for a group.

        Args:
            group: Name of the feature group.
            level: One of 'public', 'private', 'internal'.
            note: Optional human-readable note.

        Returns:
            The visibility entry dict.
        """
        return set_visibility(self.store_path, group, level, note=note)

    def remove_visibility(self, group: str) -> bool:
        """Remove the visibility setting for a group.

        Returns True if removed, False if not found.
        """
        return remove_visibility(self.store_path, group)

    def get_visibility(self, group: str) -> dict | None:
        """Return the visibility entry for a group, or None."""
        return get_visibility(self.store_path, group)

    def list_by_visibility(self, level: str) -> list[str]:
        """Return all group names with the given visibility level."""
        return list_by_visibility(self.store_path, level)

    def is_public(self, group: str) -> bool:
        """Return True if the group's visibility is 'public'."""
        entry = get_visibility(self.store_path, group)
        return entry is not None and entry.get("level") == "public"

    def is_private(self, group: str) -> bool:
        """Return True if the group's visibility is 'private'."""
        entry = get_visibility(self.store_path, group)
        return entry is not None and entry.get("level") == "private"
