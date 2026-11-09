"""Decide whether a finished warehouse stage may be followed by another.

This module only reads metrics. It does not construct an environment,
change the active world, or start training.
"""

from __future__ import annotations

import math
from typing import Mapping


def advancement_decision(metrics: Mapping[str, object], rules: Mapping[str, object]) -> dict:
    """Return whether measured performance meets the configured gates."""

    reasons = []
    episodes = int(metrics.get("episodes") or 0)
    min_episodes = int(rules.get("min_episodes", 20))
    if episodes < min_episodes:
        reasons.append(f"episodes {episodes} < {min_episodes}")
    success = _rate(metrics, "success_rate")
    collision = _rate(metrics, "collision_rate")
    faults = int(metrics.get("safety_failures", 0))
    min_success = float(rules.get("min_success_rate", 0.8))
    max_collision = float(rules.get("max_collision_rate", 0.1))
    max_faults = int(rules.get("max_safety_failures", 0))
    if success is None or success < min_success:
        reasons.append(f"success rate {success} < {min_success}")
    if collision is None or collision > max_collision:
        reasons.append(f"collision rate {collision} > {max_collision}")
    if faults > max_faults:
        reasons.append(f"safety failures {faults} > {max_faults}")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "metrics_used": {
            "episodes": episodes,
            "success_rate": success,
            "collision_rate": collision,
            "safety_failures": faults,
        },
        "changed_stage": False,
    }


def _rate(metrics: Mapping[str, object], key: str):
    value = metrics.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value
