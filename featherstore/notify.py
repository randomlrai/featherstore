"""Notification hooks for FeatherStore events."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional


def _hooks_path(store_path: str) -> Path:
    return Path(store_path) / "_hooks.json"


def load_hooks(store_path: str) -> Dict[str, List[Dict]]:
    path = _hooks_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_hooks(store_path: str, hooks: Dict[str, List[Dict]]) -> None:
    path = _hooks_path(store_path)
    with open(path, "w") as f:
        json.dump(hooks, f, indent=2)


def register_hook(store_path: str, event: str, label: str, callback_ref: str) -> Dict:
    """Register a named hook for a given event type.

    Args:
        store_path: Root path of the store.
        event: Event name, e.g. 'on_save', 'on_load', 'on_delete'.
        label: Human-readable label for this hook.
        callback_ref: Dotted import path to the callback, e.g. 'mymodule.my_fn'.

    Returns:
        The registered hook entry.
    """
    hooks = load_hooks(store_path)
    entry = {"label": label, "callback_ref": callback_ref}
    hooks.setdefault(event, [])
    hooks[event] = [h for h in hooks[event] if h["label"] != label]
    hooks[event].append(entry)
    save_hooks(store_path, hooks)
    return entry


def remove_hook(store_path: str, event: str, label: str) -> bool:
    """Remove a named hook. Returns True if removed, False if not found."""
    hooks = load_hooks(store_path)
    original = hooks.get(event, [])
    filtered = [h for h in original if h["label"] != label]
    if len(filtered) == len(original):
        return False
    hooks[event] = filtered
    save_hooks(store_path, hooks)
    return True


def list_hooks(store_path: str, event: Optional[str] = None) -> Dict[str, List[Dict]]:
    """List all hooks, optionally filtered by event."""
    hooks = load_hooks(store_path)
    if event is not None:
        return {event: hooks.get(event, [])}
    return hooks


def fire_hooks(
    store_path: str,
    event: str,
    context: Optional[Dict] = None,
    registry: Optional[Dict[str, Callable]] = None,
) -> List[str]:
    """Fire all registered hooks for an event.

    Args:
        store_path: Root path of the store.
        event: Event name to fire.
        context: Dict passed to each callback.
        registry: Optional mapping of callback_ref -> callable (for testing).

    Returns:
        List of labels of hooks that were fired.
    """
    import importlib

    hooks = load_hooks(store_path)
    fired = []
    for entry in hooks.get(event, []):
        ref = entry["callback_ref"]
        try:
            if registry and ref in registry:
                fn = registry[ref]
            else:
                module_path, fn_name = ref.rsplit(".", 1)
                module = importlib.import_module(module_path)
                fn = getattr(module, fn_name)
            fn(event=event, context=context or {})
            fired.append(entry["label"])
        except Exception:
            pass
    return fired
