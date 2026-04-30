"""LabelMixin — integrates group labeling into FeatherStore."""

from __future__ import annotations

from typing import Any

from featherstore.label import (
    clear_labels,
    find_by_label,
    get_labels,
    remove_label,
    set_label,
)


class LabelMixin:
    """Mixin that adds label management methods to FeatherStore."""

    def set_label(self, group: str, key: str, value: Any) -> dict[str, Any]:
        """Attach or update a label *key*/*value* pair on *group*."""
        return set_label(self.store_path, group, key, value)

    def remove_label(self, group: str, key: str) -> dict[str, Any]:
        """Remove label *key* from *group*. No-op if absent."""
        return remove_label(self.store_path, group, key)

    def get_labels(self, group: str) -> dict[str, Any]:
        """Return all labels attached to *group*."""
        return get_labels(self.store_path, group)

    def clear_labels(self, group: str) -> None:
        """Remove every label from *group*."""
        clear_labels(self.store_path, group)

    def find_by_label(
        self, key: str, value: Any | None = None
    ) -> list[str]:
        """Return sorted list of group names that carry *key* (optionally matching *value*)."""
        return find_by_label(self.store_path, key, value)
