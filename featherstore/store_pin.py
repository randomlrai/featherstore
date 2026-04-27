"""PinMixin — adds pin/unpin group management to FeatherStore."""

from __future__ import annotations

from typing import List

from featherstore.pin import (
    load_pins,
    pin_group,
    unpin_group,
    list_pinned_groups,
    is_pinned,
)


class PinMixin:
    """Mixin that exposes pin-related methods on FeatherStore.

    Pinned groups are protected from accidental deletion or overwrite,
    and can be quickly enumerated for pipeline bootstrapping.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pin(self, group: str, note: str = "") -> dict:
        """Pin *group* so it is flagged as protected / important.

        Parameters
        ----------
        group:
            Name of the feature group to pin.
        note:
            Optional human-readable reason for pinning.

        Returns
        -------
        dict
            The pin metadata entry that was recorded.

        Raises
        ------
        KeyError
            If *group* does not exist in the catalog.
        """
        if group not in self._catalog:
            raise KeyError(f"Group '{group}' not found in store.")

        entry = pin_group(self.store_path, group, note=note)
        return entry

    def unpin(self, group: str) -> bool:
        """Remove the pin from *group*.

        Parameters
        ----------
        group:
            Name of the feature group to unpin.

        Returns
        -------
        bool
            ``True`` if the pin was removed, ``False`` if the group was not
            pinned in the first place.
        """
        return unpin_group(self.store_path, group)

    def is_pinned(self, group: str) -> bool:
        """Return ``True`` if *group* is currently pinned."""
        return is_pinned(self.store_path, group)

    def list_pinned(self) -> List[str]:
        """Return a list of all currently pinned group names."""
        return list_pinned_groups(self.store_path)

    def pinned_info(self, group: str) -> dict | None:
        """Return the pin metadata for *group*, or ``None`` if not pinned.

        The returned dict contains at minimum:
        - ``group``  – group name
        - ``pinned_at`` – ISO-8601 timestamp
        - ``note``   – optional annotation
        """
        pins = load_pins(self.store_path)
        return pins.get(group)
