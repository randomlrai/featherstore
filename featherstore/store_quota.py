"""QuotaMixin — integrates disk quota management into FeatherStore."""

from featherstore.quota import (
    set_quota,
    remove_quota,
    get_quota,
    check_quota,
    enforce_quota,
    load_quotas,
)


class QuotaMixin:
    """Mixin that adds quota management methods to FeatherStore."""

    def set_quota(self, group: str, max_bytes: int) -> dict:
        """Set a disk quota in bytes for a group."""
        return set_quota(self.store_path, group, max_bytes)

    def remove_quota(self, group: str) -> bool:
        """Remove the quota for a group. Returns True if removed."""
        return remove_quota(self.store_path, group)

    def get_quota(self, group: str):
        """Return quota info for a group, or None if not set."""
        return get_quota(self.store_path, group)

    def check_quota(self, group: str) -> dict:
        """Return a usage report dict for the group's quota."""
        return check_quota(self.store_path, group)

    def enforce_quota(self, group: str) -> None:
        """Raise QuotaExceededError if the group exceeds its quota."""
        enforce_quota(self.store_path, group)

    def list_quotas(self) -> dict:
        """Return all configured quotas."""
        return load_quotas(self.store_path)
