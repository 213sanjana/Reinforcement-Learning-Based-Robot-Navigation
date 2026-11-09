"""Episode metrics for Jackal navigation.

Success rate uses every episode in the denominator. A timeout is not a
success. A collision count of zero is not success by itself. Classical
records stay labeled as classical records.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


class MetricError(ValueError):
    """A metric record contradicts the evaluation rules."""


@dataclass(frozen=True)
class JackalEpisodeRecord:
    scenario_id: str
    start_pose: tuple
    goal_pose: tuple
    map_id: str
    policy_id: str
    algorithm_family: str
    seed: int
    duration_s: float
    steps: int
    goal_completed: bool
    collision_count: int
    timeout: bool
    final_goal_distance: float
    path_length: float
    straight_line_distance: float
    path_efficiency: float | None
    average_linear_velocity: float
    average_angular_velocity: float
    safety_intervention_count: int
    invalid_observation_count: int
    invalid_action_count: int
    split: str
    termination: str


def path_efficiency(straight_line_distance, path_length):
    """Straight-line distance divided by the odometry path.

    A zero path has no defined efficiency. The value is computed for
    failures as well as successes and is not itself a success flag.
    """

    straight = float(straight_line_distance)
    path = float(path_length)
    if not math.isfinite(straight) or not math.isfinite(path):
        raise MetricError("Distances must be finite")
    if straight < 0 or path < 0:
        raise MetricError("Distances must be nonnegative")
    if path == 0.0:
        return None
    return straight / path


def _pose(values, label):
    if len(values) < 2:
        raise MetricError(f"{label} needs x and y")
    pose = tuple(float(v) for v in values[:3]) if len(values) >= 3 else tuple(float(v) for v in values[:2])
    if not all(math.isfinite(v) for v in pose):
        raise MetricError(f"{label} must be finite")
    return pose


def validate_episode(record, goal_tolerance=0.30):
    """Raise unless the record follows the success, timeout, and path rules."""

    if not isinstance(record, JackalEpisodeRecord):
        raise MetricError("Expected a JackalEpisodeRecord")
    if record.split not in {"fixed", "unseen"}:
        raise MetricError("split must be fixed or unseen")
    if record.algorithm_family not in {"reinforcement_learning", "classical"}:
        raise MetricError("Unknown algorithm family")
    if record.algorithm_family == "classical" and "rl" in record.policy_id.lower():
        raise MetricError("A classical policy_id must not be labeled as RL")
    if record.goal_completed and record.timeout:
        raise MetricError("A timeout cannot be marked successful")
    if record.goal_completed and record.collision_count > 0:
        raise MetricError("A colliding episode cannot be marked successful")
    if record.goal_completed and record.termination != "GOAL_REACHED":
        raise MetricError("A success must terminate as GOAL_REACHED")
    if record.timeout and record.termination != "TIMEOUT":
        raise MetricError("A timeout record must terminate as TIMEOUT")
    if record.goal_completed and float(record.final_goal_distance) > float(goal_tolerance):
        raise MetricError("Success requires the final distance to be inside tolerance")
    if record.steps < 0 or record.collision_count < 0:
        raise MetricError("Counts must be nonnegative")
    if record.safety_intervention_count < 0:
        raise MetricError("Intervention count must be nonnegative")
    if record.invalid_observation_count < 0 or record.invalid_action_count < 0:
        raise MetricError("Invalid sample counts must be nonnegative")
    if record.duration_s < 0 or not math.isfinite(record.duration_s):
        raise MetricError("Duration must be finite and nonnegative")
    if record.final_goal_distance < 0 or not math.isfinite(record.final_goal_distance):
        raise MetricError("Final goal distance must be finite and nonnegative")
    for name in ("average_linear_velocity", "average_angular_velocity"):
        value = getattr(record, name)
        if not math.isfinite(value):
            raise MetricError(f"{name} must be finite")
    _pose(record.start_pose, "start_pose")
    _pose(record.goal_pose, "goal_pose")
    expected = path_efficiency(record.straight_line_distance, record.path_length)
    if expected is None:
        if record.path_efficiency is not None:
            raise MetricError("Zero path length has undefined efficiency")
    elif record.path_efficiency is None or not math.isclose(
        record.path_efficiency, expected, rel_tol=0.0, abs_tol=1e-9
    ):
        raise MetricError("path_efficiency must equal straight_line / path_length")
    if not isinstance(record.seed, int) or isinstance(record.seed, bool):
        raise MetricError("seed must be an integer")
    if not record.scenario_id or not record.map_id or not record.policy_id:
        raise MetricError("Identifiers must be non-empty")
    return record


def result_label(record):
    """Human-readable family label. Classical success is not RL success."""

    validate_episode(record)
    if record.algorithm_family == "classical":
        return "classical baseline result; not a reinforcement-learning result"
    return "reinforcement-learning result"


def summarize(records, split=None, goal_tolerance=0.30):
    """Aggregate a non-empty set of episodes.

    Failed episodes stay in the denominator. ``mean_time_to_goal_s`` averages
    only episodes that reached the goal.
    """

    rows = [validate_episode(record, goal_tolerance) for record in records]
    if split is not None:
        if split not in {"fixed", "unseen"}:
            raise MetricError("split must be fixed or unseen")
        rows = [record for record in rows if record.split == split]
    if not rows:
        raise MetricError("No episodes to summarize")

    successes = [record for record in rows if record.goal_completed]
    collisions = [record for record in rows if record.collision_count > 0]
    defined = [record.path_efficiency for record in rows if record.path_efficiency is not None]
    successful_defined = [
        record.path_efficiency for record in successes if record.path_efficiency is not None
    ]
    count = len(rows)
    return {
        "episodes": count,
        "successes": len(successes),
        "success_rate": len(successes) / count,
        "collision_episodes": len(collisions),
        "collision_rate": len(collisions) / count,
        "timeouts": sum(1 for record in rows if record.timeout),
        "timeout_rate": sum(1 for record in rows if record.timeout) / count,
        "mean_time_to_goal_s": (
            sum(record.duration_s for record in successes) / len(successes)
            if successes else None
        ),
        "mean_path_efficiency": (sum(defined) / len(defined)) if defined else None,
        "mean_successful_path_efficiency": (
            sum(successful_defined) / len(successful_defined)
            if successful_defined else None
        ),
        "split": split if split is not None else "all",
        "note": (
            "Timeouts are failures and remain in the success-rate denominator. "
            "A collision count of zero is not success."
        ),
    }


def record_to_dict(record):
    validate_episode(record)
    payload = asdict(record)
    payload["result_label"] = result_label(record)
    return payload
