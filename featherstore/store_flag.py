"""FlagMixin — integrates feature flags into FeatherStore."""

from featherstore.flag import (
    set_flag,
    remove_flag,
    get_flags,
    is_flagged,
    list_flagged_groups,
    clear_flags,
)
from typing import Any, Dict, List, Optional


class FlagMixin:
    """Mixin that adds feature-flag methods to FeatherStore."""

    def set_flag(self, group: str, flag: str, value: Any = True) -> Dict[str, Any]:
        """Set *flag* on *group* to *value* (default True)."""
        return set_flag(self.store_path, group, flag, value)

    def remove_flag(self, group: str, flag: str) -> None:
        """Remove *flag* from *group*. No-op if absent."""
        remove_flag(self.store_path, group, flag)

    def get_flags(self, group: str) -> Dict[str, Any]:
        """Return a dict of all flags set on *group*."""
        return get_flags(self.store_path, group)

    def is_flagged(self, group: str, flag: str) -> bool:
        """Return True if *flag* is set (and truthy) on *group*."""
        return is_flagged(self.store_path, group, flag)

    def list_flagged_groups(self, flag: str) -> List[str]:
        """Return all groups that carry a truthy value for *flag*."""
        return list_flagged_groups(self.store_path, flag)

    def clear_flags(self, group: str) -> None:
        """Remove every flag from *group*."""
        clear_flags(self.store_path, group)
