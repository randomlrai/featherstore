"""TrustMixin — integrates trust level management into FeatherStore."""

from featherstore.trust import (
    set_trust,
    remove_trust,
    get_trust,
    list_by_trust_level,
    VALID_LEVELS,
)


class TrustMixin:
    """Mixin that adds trust-level tracking to FeatherStore."""

    def set_trust(self, group: str, level: str, note: str = "") -> dict:
        """Assign a trust level to a feature group.

        Args:
            group: Feature group name.
            level: One of 'untrusted', 'experimental', 'provisional', 'trusted', 'verified'.
            note: Optional free-text justification.

        Returns:
            The trust record dict.
        """
        return set_trust(self.path, group, level, note=note)

    def remove_trust(self, group: str) -> bool:
        """Remove the trust entry for a group. Returns True if removed."""
        return remove_trust(self.path, group)

    def get_trust(self, group: str) -> dict | None:
        """Return the trust record for a group, or None if not set."""
        return get_trust(self.path, group)

    def list_by_trust_level(self, level: str) -> list[str]:
        """Return all group names that have the given trust level."""
        return list_by_trust_level(self.path, level)

    @property
    def trust_levels(self) -> tuple:
        """Return the ordered tuple of valid trust levels."""
        return VALID_LEVELS
