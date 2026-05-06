"""SensitivityMixin for FeatherStore — classify groups by data sensitivity."""

from featherstore.sensitivity import (
    set_sensitivity,
    remove_sensitivity,
    get_sensitivity,
    list_by_sensitivity_level,
    VALID_LEVELS,
)


class SensitivityMixin:
    """Mixin that adds sensitivity classification methods to FeatherStore."""

    def set_sensitivity(self, group: str, level: str, note: str = "") -> dict:
        """Classify a group with a sensitivity level.

        Args:
            group: Feature group name.
            level: One of 'public', 'internal', 'confidential', 'restricted'.
            note: Optional explanation or compliance note.

        Returns:
            Metadata dict for the entry.
        """
        return set_sensitivity(self.path, group, level, note=note)

    def remove_sensitivity(self, group: str) -> bool:
        """Remove sensitivity classification from a group.

        Returns True if removed, False if not found.
        """
        return remove_sensitivity(self.path, group)

    def get_sensitivity(self, group: str) -> dict | None:
        """Return sensitivity metadata for a group, or None if unclassified."""
        return get_sensitivity(self.path, group)

    def list_by_sensitivity(self, level: str) -> list[str]:
        """Return all groups classified at the given sensitivity level."""
        return list_by_sensitivity_level(self.path, level)

    @property
    def sensitivity_levels(self) -> set:
        """Return the set of valid sensitivity levels."""
        return VALID_LEVELS
