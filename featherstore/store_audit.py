"""AuditMixin — integrates audit logging into FeatherStore."""

from __future__ import annotations

from typing import Any

from featherstore.audit import (
    clear_audit_log,
    get_audit_history,
    record_event,
)


class AuditMixin:
    """Mixin that adds audit-log capabilities to FeatherStore."""

    # `self.path` is expected to be set by FeatherStore.__init__

    def log_event(
        self,
        group: str,
        operation: str,
        user: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Manually record an audit event for *group*."""
        return record_event(
            self.path,
            group=group,
            operation=operation,
            user=user,
            details=details,
        )

    def audit_history(
        self,
        group: str | None = None,
        operation: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return audit log entries, optionally filtered by group or operation."""
        return get_audit_history(self.path, group=group, operation=operation)

    def clear_audit(self) -> None:
        """Wipe the entire audit log for this store."""
        clear_audit_log(self.path)
