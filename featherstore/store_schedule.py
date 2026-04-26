"""ScheduleMixin — integrates schedule management into FeatherStore."""

from __future__ import annotations

from typing import Any

from featherstore.schedule import (
    list_schedules,
    mark_run,
    register_schedule,
    remove_schedule,
)


class ScheduleMixin:
    """Mixin that adds schedule-management methods to FeatherStore."""

    # `self.path` is expected to be set by FeatherStore.__init__

    def add_schedule(
        self,
        group: str,
        cron: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Register a cron schedule for *group*.

        Parameters
        ----------
        group:
            Feature group name.
        cron:
            Cron expression string.
        description:
            Optional description.

        Returns
        -------
        dict
            The created schedule entry.
        """
        return register_schedule(self.path, group, cron, description=description)

    def remove_schedule(self, group: str) -> bool:
        """Remove the schedule for *group*. Returns True if it existed."""
        return remove_schedule(self.path, group)

    def mark_schedule_run(self, group: str) -> None:
        """Record that *group*'s scheduled job has just run."""
        mark_run(self.path, group)

    def list_schedules(self) -> list[dict[str, Any]]:
        """Return all registered schedules."""
        return list_schedules(self.path)
