"""ScoreMixin — integrates group scoring into FeatherStore."""

from __future__ import annotations

from typing import Any

from featherstore.score import (
    get_scores,
    list_top_groups,
    remove_score,
    set_score,
)


class ScoreMixin:
    """Mixin that adds scoring capabilities to FeatherStore."""

    def set_score(
        self,
        group: str,
        score: float,
        metric: str = "quality",
        note: str | None = None,
    ) -> dict[str, Any]:
        """Assign a numeric score to *group* under *metric*.

        Parameters
        ----------
        group:  Feature group name.
        score:  Numeric value (higher is better by convention).
        metric: Named dimension for the score (default ``"quality"``).
        note:   Optional free-text annotation stored alongside the score.

        Returns
        -------
        dict with ``score``, ``note``, and ``updated_at`` fields.
        """
        return set_score(self.path, group, score, metric=metric, note=note)

    def remove_score(
        self, group: str, metric: str = "quality"
    ) -> bool:
        """Remove a metric score from *group*. Returns ``True`` if it existed."""
        return remove_score(self.path, group, metric=metric)

    def get_scores(self, group: str) -> dict[str, Any]:
        """Return all metric scores recorded for *group*."""
        return get_scores(self.path, group)

    def top_groups(
        self, metric: str = "quality", n: int = 10
    ) -> list[dict[str, Any]]:
        """Return the top-*n* groups ranked by *metric* score (descending)."""
        return list_top_groups(self.path, metric=metric, n=n)
