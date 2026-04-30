"""Alert rules: define threshold-based alerts on group stats."""

import json
from pathlib import Path
from typing import Any

_VALID_OPS = {">", ">=", "<", "<=", "==", "!="}


def _alerts_path(store_path: str) -> Path:
    return Path(store_path) / "_alerts.json"


def load_alerts(store_path: str) -> dict:
    path = _alerts_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_alerts(store_path: str, alerts: dict) -> None:
    path = _alerts_path(store_path)
    with open(path, "w") as f:
        json.dump(alerts, f, indent=2)


def add_alert(
    store_path: str,
    group: str,
    metric: str,
    op: str,
    threshold: float,
    label: str | None = None,
) -> dict:
    if op not in _VALID_OPS:
        raise ValueError(f"Invalid operator '{op}'. Must be one of {_VALID_OPS}.")
    alerts = load_alerts(store_path)
    entry = {
        "group": group,
        "metric": metric,
        "op": op,
        "threshold": threshold,
        "label": label or f"{group}.{metric}{op}{threshold}",
    }
    key = entry["label"]
    alerts[key] = entry
    save_alerts(store_path, alerts)
    return entry


def remove_alert(store_path: str, label: str) -> bool:
    alerts = load_alerts(store_path)
    if label not in alerts:
        return False
    del alerts[label]
    save_alerts(store_path, alerts)
    return True


def evaluate_alert(alert: dict, stats: dict) -> dict:
    """Evaluate a single alert against a stats dict. Returns result dict."""
    metric = alert["metric"]
    op = alert["op"]
    threshold = alert["threshold"]
    value = stats.get(metric)
    if value is None:
        return {"label": alert["label"], "fired": False, "reason": f"metric '{metric}' not found"}
    ops = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    fired = ops[op](value, threshold)
    return {
        "label": alert["label"],
        "fired": fired,
        "value": value,
        "threshold": threshold,
        "op": op,
        "metric": metric,
    }


def evaluate_all_alerts(store_path: str, group: str, stats: dict) -> list[dict]:
    alerts = load_alerts(store_path)
    results = []
    for alert in alerts.values():
        if alert["group"] == group:
            results.append(evaluate_alert(alert, stats))
    return results
