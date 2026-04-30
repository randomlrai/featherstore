"""AlertMixin: integrates alert rules into FeatherStore."""

from featherstore.alert import (
    add_alert,
    remove_alert,
    load_alerts,
    evaluate_all_alerts,
)
from featherstore.stats import load_stats


class AlertMixin:
    def add_alert(
        self,
        group: str,
        metric: str,
        op: str,
        threshold: float,
        label: str | None = None,
    ) -> dict:
        """Register a threshold alert on a stat metric for a group."""
        return add_alert(self.store_path, group, metric, op, threshold, label)

    def remove_alert(self, label: str) -> bool:
        """Remove an alert by its label. Returns True if removed."""
        return remove_alert(self.store_path, label)

    def list_alerts(self, group: str | None = None) -> list[dict]:
        """List all alerts, optionally filtered by group."""
        alerts = load_alerts(self.store_path)
        entries = list(alerts.values())
        if group is not None:
            entries = [a for a in entries if a["group"] == group]
        return entries

    def check_alerts(self, group: str) -> list[dict]:
        """Evaluate all alerts for a group against its current stats.

        Returns a list of result dicts with 'fired' bool and context.
        Raises KeyError if group has no recorded stats.
        """
        all_stats = load_stats(self.store_path)
        if group not in all_stats:
            raise KeyError(f"No stats recorded for group '{group}'.")
        stats = all_stats[group]
        return evaluate_all_alerts(self.store_path, group, stats)
