"""MergeMixin — adds merge() to FeatherStore."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from featherstore.merge import merge_groups, merge_on_index


class MergeMixin:
    """Mixin that provides merge() on top of FeatherStore."""

    def merge(
        self,
        groups: List[str],
        on: Optional[List[str]] = None,
        how: str = "inner",
        use_index: bool = False,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load and merge multiple stored feature groups.

        Parameters
        ----------
        groups:
            Names of the feature groups to merge.
        on:
            Column(s) to join on (ignored when *use_index* is True).
        how:
            Join strategy – 'inner', 'left', 'right', or 'outer'.
        use_index:
            When True, merge by DataFrame index instead of columns.
        columns:
            Optional column subset to load from each group *before* merging.
            Join key columns are always included automatically.

        Returns
        -------
        pd.DataFrame
            Merged feature set.
        """
        if len(groups) < 2:
            raise ValueError("At least two groups are required for a merge.")

        frames: List[pd.DataFrame] = []
        for name in groups:
            load_cols = None
            if columns is not None and on is not None:
                # Always keep join keys so the merge can proceed
                load_cols = list(set(columns) | set(on))
            elif columns is not None:
                load_cols = columns
            frames.append(self.load(name, columns=load_cols))

        if use_index:
            return merge_on_index(frames, how=how)
        return merge_groups(frames, on=on, how=how)
