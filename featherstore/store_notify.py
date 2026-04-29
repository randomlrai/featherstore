"""NotifyMixin — integrates notification hooks into FeatherStore."""

from typing import Callable, Dict, List, Optional

from featherstore.notify import (
    fire_hooks,
    list_hooks,
    register_hook,
    remove_hook,
)


class NotifyMixin:
    """Mixin that adds hook registration and firing to FeatherStore."""

    # Optional in-process callable registry (label -> callable)
    _hook_registry: Dict[str, Callable] = {}

    def add_hook(self, event: str, label: str, callback_ref: str) -> Dict:
        """Register a persistent hook for a store event.

        Args:
            event: One of 'on_save', 'on_load', 'on_delete', or custom.
            label: Unique name for this hook.
            callback_ref: Dotted import path, e.g. 'mymodule.alert'.

        Returns:
            The hook entry dict.
        """
        return register_hook(self.store_path, event, label, callback_ref)

    def remove_hook(self, event: str, label: str) -> bool:
        """Remove a registered hook. Returns True if it existed."""
        return remove_hook(self.store_path, event, label)

    def list_hooks(self, event: Optional[str] = None) -> Dict[str, List[Dict]]:
        """List all hooks, optionally filtered by event name."""
        return list_hooks(self.store_path, event=event)

    def fire(self, event: str, context: Optional[Dict] = None) -> List[str]:
        """Manually fire hooks for a given event.

        Args:
            event: The event name to fire.
            context: Optional dict passed to each callback.

        Returns:
            Labels of hooks that were successfully fired.
        """
        return fire_hooks(
            self.store_path,
            event,
            context=context,
            registry=self._hook_registry,
        )

    def register_callable(self, callback_ref: str, fn: Callable) -> None:
        """Register an in-process callable for a given ref (useful in tests)."""
        self._hook_registry[callback_ref] = fn
