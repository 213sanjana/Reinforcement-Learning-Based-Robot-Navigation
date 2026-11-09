"""Episode termination decisions for warehouse PPO.

Infrastructure failures are fatal to a training run. A goal, a collision,
or a step limit ends only the episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


FATAL_REASONS = frozenset(
    {
        "stale_sensors",
        "simulator_timeout",
        "communication_failure",
        "invalid_observation",
        "safety_fault",
        "ownership_lost",
        "controller_inactive",
    }
)


@dataclass(frozen=True)
class EpisodeLimits:
    goal_tolerance_m: float = 0.30
    max_episode_steps: int = 600
    sensor_timeout_s: float = 5.0
    control_frequency_hz: float = 5.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.goal_tolerance_m) or self.goal_tolerance_m <= 0:
            raise ValueError("goal_tolerance_m must be positive and finite")
        if not math.isfinite(self.sensor_timeout_s) or self.sensor_timeout_s <= 0:
            raise ValueError("sensor_timeout_s must be positive and finite")
        if not math.isfinite(self.control_frequency_hz) or self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive and finite")
        if isinstance(self.max_episode_steps, bool) or not isinstance(self.max_episode_steps, int):
            raise ValueError("max_episode_steps must be an integer")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")

    @property
    def control_dt_s(self) -> float:
        return 1.0 / self.control_frequency_hz

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> "EpisodeLimits":
        payload = {} if payload is None else payload
        aliases = {
            "goal_tolerance_m": payload.get("goal_tolerance_m", payload.get("goal_tolerance", 0.30)),
            "max_episode_steps": payload.get("max_episode_steps", 600),
            "sensor_timeout_s": payload.get("sensor_timeout_s", payload.get("sensor_timeout", 5.0)),
            "control_frequency_hz": payload.get("control_frequency_hz", 5.0),
        }
        return cls(**aliases)


def decide_episode(
    *,
    goal_distance: float,
    limits: EpisodeLimits,
    collision: bool,
    episode_steps: int,
    safety_fault: bool = False,
    sensor_failure: bool = False,
    simulator_failure: bool = False,
    communication_failure: bool = False,
    invalid_observation: bool = False,
    command_reason: str | None = None,
) -> tuple[bool, bool, str | None, bool]:
    """Return terminated, truncated, reason, and whether training must stop."""

    goal_distance = float(goal_distance)
    if not math.isfinite(goal_distance) or goal_distance < 0:
        raise ValueError("goal_distance must be nonnegative and finite")
    if episode_steps < 0:
        raise ValueError("episode_steps cannot be negative")

    if collision:
        return True, False, "collision", False
    if safety_fault:
        return True, False, "safety_fault", True
    if command_reason is not None:
        terminated, truncated, reason, fatal = classify_rejection(command_reason)
        return terminated, truncated, reason, fatal
    if sensor_failure or invalid_observation:
        reason = "invalid_observation" if invalid_observation and not sensor_failure else "stale_sensors"
        return False, True, reason, True
    if simulator_failure:
        return False, True, "simulator_timeout", True
    if communication_failure:
        return False, True, "communication_failure", True
    if goal_distance <= limits.goal_tolerance_m:
        return True, False, "goal_reached", False
    if episode_steps >= limits.max_episode_steps:
        return False, True, "time_limit", False
    return False, False, None, False


def classify_rejection(reason: str) -> tuple[bool, bool, str, bool]:
    """Map an existing safety-layer rejection onto an episode outcome."""

    text = "" if reason is None else str(reason)
    lowered = text.lower()
    if "emergency" in lowered:
        return True, False, "safety_fault", True
    if "collision" in lowered:
        return True, False, "collision", False
    if "stale" in lowered or "sensor" in lowered:
        return False, True, "stale_sensors", True
    if "simulator" in lowered or "clock" in lowered:
        return False, True, "simulator_timeout", True
    if "pilot owns" in lowered or "ownership" in lowered:
        return True, False, "ownership_lost", True
    if "controller" in lowered:
        return True, False, "controller_inactive", True
    if "publish" in lowered or "unavailable" in lowered:
        return False, True, "communication_failure", True
    if "interval" in lowered or text == "":
        return False, True, "command_rejected", False
    return False, True, "safety_rejected", False


@dataclass
class EpisodeTracker:
    """Accumulate path and reward diagnostics for one episode."""

    positions: list[tuple[float, float]] = field(default_factory=list)
    linear_samples: list[float] = field(default_factory=list)
    angular_samples: list[float] = field(default_factory=list)
    clearances: list[float] = field(default_factory=list)
    reward_sum: float = 0.0
    yaw_travel: float = 0.0
    path_length_m: float = 0.0
    duration_s: float = 0.0
    steps: int = 0
    components: dict = field(default_factory=dict)

    def begin(self, x: float, y: float) -> None:
        self.positions = [(float(x), float(y))]
        self.linear_samples = []
        self.angular_samples = []
        self.clearances = []
        self.reward_sum = 0.0
        self.yaw_travel = 0.0
        self.path_length_m = 0.0
        self.duration_s = 0.0
        self.steps = 0
        self.components = {}

    def record(
        self,
        *,
        x: float,
        y: float,
        yaw_delta: float,
        linear: float,
        angular: float,
        clearance: float | None,
        reward: float,
        dt: float,
        reward_info: Mapping[str, object],
    ) -> None:
        previous = self.positions[-1]
        self.positions.append((float(x), float(y)))
        self.path_length_m += math.hypot(float(x) - previous[0], float(y) - previous[1])
        self.yaw_travel += abs(float(yaw_delta))
        self.linear_samples.append(abs(float(linear)))
        self.angular_samples.append(abs(float(angular)))
        if clearance is not None and math.isfinite(clearance):
            self.clearances.append(float(clearance))
        self.reward_sum += float(reward)
        self.duration_s += float(dt)
        self.steps += 1
        for key, value in reward_info.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                self.components[key] = self.components.get(key, 0.0) + float(value)

    def metrics(self, *, reason: str | None, goal_distance: float) -> dict:
        success = reason == "goal_reached"
        collision = reason == "collision"
        timeout = reason == "time_limit"
        count = max(self.steps, 1)
        return {
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "termination_reason": reason,
            "episode_return": self.reward_sum,
            "episode_length": self.steps,
            "path_length_m": self.path_length_m,
            "duration_s": self.duration_s,
            "time_to_goal_s": self.duration_s if success else None,
            "mean_linear_speed_mps": sum(self.linear_samples) / count if self.linear_samples else 0.0,
            "mean_angular_speed_radps": sum(self.angular_samples) / count if self.angular_samples else 0.0,
            "min_clearance_m": min(self.clearances) if self.clearances else None,
            "rotations": self.yaw_travel / (2.0 * math.pi),
            "goal_distance": float(goal_distance),
            "reward_components": dict(self.components),
        }
