"""OwnershipMixin for FeatherStore — owner assignment and lookup."""

from featherstore.ownership import (
    set_owner,
    remove_owner,
    get_owner,
    list_by_owner,
    list_by_team,
)


class OwnershipMixin:
    def set_owner(
        self,
        group: str,
        owner: str,
        team: str | None = None,
        email: str | None = None,
    ) -> dict:
        """Assign an owner (and optionally a team/email) to a group."""
        return set_owner(self.store_path, group, owner, team=team, email=email)

    def remove_owner(self, group: str) -> bool:
        """Remove ownership record for a group. Returns True if it existed."""
        return remove_owner(self.store_path, group)

    def get_owner(self, group: str) -> dict | None:
        """Return ownership metadata for a group, or None if not set."""
        return get_owner(self.store_path, group)

    def list_by_owner(self, owner: str) -> list[str]:
        """Return all groups owned by the given owner."""
        return list_by_owner(self.store_path, owner)

    def list_by_team(self, team: str) -> list[str]:
        """Return all groups belonging to the given team."""
        return list_by_team(self.store_path, team)
