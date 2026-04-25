"""ProfileMixin — adds .profile() and .get_profile() to FeatherStore."""

from __future__ import annotations

from typing import Any

from featherstore.profile import load_profile


class ProfileMixin:
    """Mixin that exposes profiling helpers on FeatherStore."""

    def get_profile(self, group: str) -> dict[str, Any]:
        """Return the stored profile for *group*, or {} if none exists."""
        return load_profile(self.path, group)

    def profile(self, group: str) -> dict[str, Any]:
        """Re-compute and persist a fresh profile for *group*.

        Raises
        ------
        KeyError
            If *group* has not been saved to the store yet.
        """
        if group not in self._catalog:
            raise KeyError(f"Group '{group}' not found in store.")

        df = self.load(group)
        from featherstore.profile import record_profile
        return record_profile(self.path, group, df)
