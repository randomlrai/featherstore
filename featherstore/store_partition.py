"""PartitionMixin — adds partitioning capabilities to FeatherStore."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from featherstore.partition import (
    load_partition_meta,
    partition_dataframe,
    load_partition,
)


class PartitionMixin:
    """Mixin that adds partition_save / partition_load to FeatherStore."""

    def partition_save(
        self,
        df: pd.DataFrame,
        group: str,
        column: str,
    ) -> dict[str, str]:
        """Partition *df* by *column* and persist each partition under *group*.

        Returns a mapping of partition value -> filename.
        """
        partition_map = partition_dataframe(
            df=df,
            column=column,
            store_path=self.store_path,
            group=group,
        )
        # Register group in catalog so search / list_groups picks it up
        if group not in self._catalog:
            self._catalog[group] = {}
        self._catalog[group]["partitioned_by"] = column
        self._save_catalog()
        return partition_map

    def partition_load(
        self,
        group: str,
        value: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load partition(s) for *group*. Pass *value* to load a single slice."""
        return load_partition(
            store_path=self.store_path,
            group=group,
            value=value,
        )

    def partition_info(self, group: str) -> dict:
        """Return partition metadata for *group*."""
        return load_partition_meta(self.store_path, group)
