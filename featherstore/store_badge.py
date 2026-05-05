"""BadgeMixin — integrates badge system into FeatherStore."""

from featherstore.badge import (
    award_badge,
    revoke_badge,
    get_badges,
    list_groups_with_badge,
    clear_badges,
)


class BadgeMixin:
    def award_badge(self, group: str, badge: str, awarded_by: str = "system") -> dict:
        """Award a badge to a group."""
        return award_badge(self.store_path, group, badge, awarded_by=awarded_by)

    def revoke_badge(self, group: str, badge: str) -> dict:
        """Remove a specific badge from a group."""
        return revoke_badge(self.store_path, group, badge)

    def get_badges(self, group: str) -> list:
        """Return all badges held by a group."""
        return get_badges(self.store_path, group)

    def list_groups_with_badge(self, badge: str) -> list:
        """Return all groups that hold a given badge."""
        return list_groups_with_badge(self.store_path, badge)

    def clear_badges(self, group: str) -> None:
        """Remove all badges from a group."""
        clear_badges(self.store_path, group)
