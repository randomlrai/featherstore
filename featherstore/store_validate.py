"""Mixin that adds schema validation capabilities to FeatherStore."""

from __future__ import annotations

from typing import Any

import pandas as pd

from featherstore.validate import (
    load_schema,
    record_schema,
    validate_schema,
)


class ValidateMixin:
    """Mixin for FeatherStore — provides get_schema, validate, and pin_schema."""

    def get_schema(self, group: str) -> dict[str, Any]:
        """Return the persisted schema for *group*, or an empty dict if none."""
        return load_schema(self.path, group)

    def pin_schema(self, group: str) -> dict[str, Any]:
        """Capture and save the current schema of *group*.

        Raises KeyError if the group does not exist in the catalog.
        """
        if group not in self._catalog:
            raise KeyError(f"Group '{group}' not found in store.")
        df = self.load(group)
        return record_schema(self.path, group, df)

    def validate(self, group: str, df: pd.DataFrame) -> list[str]:
        """Validate *df* against the pinned schema for *group*.

        Returns a list of violation strings.  An empty list means the
        DataFrame is compatible with the stored schema.
        """
        expected = load_schema(self.path, group)
        return validate_schema(df, expected)
