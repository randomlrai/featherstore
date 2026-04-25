"""CompareMixin — adds compare() to FeatherStore."""

from __future__ import annotations

from typing import Optional

from featherstore.compare import compare_groups, compare_numeric_stats


class CompareMixin:
    """Mixin that provides group-comparison helpers on FeatherStore."""

    def compare(
        self,
        group_a: str,
        group_b: str,
        version_a: Optional[str] = None,
        version_b: Optional[str] = None,
    ) -> dict:
        """Compare two feature groups (or two versions of the same group).

        Args:
            group_a: Name of the first group.
            group_b: Name of the second group (may equal group_a).
            version_a: Optional version tag for group_a.
            version_b: Optional version tag for group_b.

        Returns:
            Structured comparison report dict (see compare.compare_groups).
        """
        df_a = self.load(group_a, version=version_a)
        df_b = self.load(group_b, version=version_b)

        label_a = group_a if version_a is None else f"{group_a}@{version_a}"
        label_b = group_b if version_b is None else f"{group_b}@{version_b}"

        return compare_groups(df_a, df_b, label_a=label_a, label_b=label_b)

    def compare_stats(
        self,
        group_a: str,
        group_b: str,
        columns: Optional[list[str]] = None,
    ) -> object:
        """Return side-by-side numeric describe() for two groups.

        Args:
            group_a: Name of the first group.
            group_b: Name of the second group.
            columns: Optional column filter.

        Returns:
            pandas DataFrame with suffixed describe() statistics.
        """
        df_a = self.load(group_a)
        df_b = self.load(group_b)
        return compare_numeric_stats(df_a, df_b, columns=columns)
