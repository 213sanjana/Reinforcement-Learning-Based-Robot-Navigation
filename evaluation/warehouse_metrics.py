"""Aggregate warehouse episode records. Empty input is an error, not a zero score."""

from __future__ import annotations

import math
from typing import Sequence


def _values(episodes: Sequence[dict], key: str, *, successes_only: bool = False) -> list[float]:
    values = []
    for episode in episodes:
        if successes_only and not episode.get("success"):
            continue
        value = episode.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _mean(values: list[float]):
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_episodes(episodes: Sequence[dict]) -> dict:
    if not episodes:
        raise ValueError("No episodes to aggregate")
    count = len(episodes)
    successes = sum(bool(episode.get("success")) for episode in episodes)
    collisions = sum(bool(episode.get("collision")) for episode in episodes)
    timeouts = sum(bool(episode.get("timeout")) for episode in episodes)
    return {
        "episodes": count,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "success_rate": successes / count,
        "collision_rate": collisions / count,
        "timeout_rate": timeouts / count,
        "mean_path_length_m": _mean(_values(episodes, "path_length_m")),
        "mean_duration_s": _mean(_values(episodes, "duration_s")),
        "mean_time_to_goal_s": _mean(_values(episodes, "time_to_goal_s", successes_only=True)),
        "mean_linear_speed_mps": _mean(_values(episodes, "mean_linear_speed_mps")),
        "mean_angular_speed_radps": _mean(_values(episodes, "mean_angular_speed_radps")),
        "mean_min_clearance_m": _mean(_values(episodes, "min_clearance_m")),
        "mean_rotations": _mean(_values(episodes, "rotations")),
        "mean_terminal_goal_distance_m": _mean(_values(episodes, "goal_distance")),
        "mean_episode_length": _mean(_values(episodes, "episode_length")),
        "mean_return": _mean(_values(episodes, "episode_return")),
    }
