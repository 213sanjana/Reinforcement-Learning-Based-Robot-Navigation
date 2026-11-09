"""Offline coordinator for one Jackal goal episode.

The manager selects a policy, maps and safety-checks the action, and stops
commanding in a terminal state. It does not publish ROS velocity, load a
checkpoint, or step a simulator.

``CommandDecision.linear`` and ``angular`` are meters per second and radians
per second. ``normalized_action`` is the policy output in [-1, 1]. The live
``command_action`` path expects the normalized pair and maps it again, so
the physical fields must not be passed back into that function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.safety_monitor import SafetyMonitor

from navigation.jackal_episode_metrics import JackalEpisodeRecord, path_efficiency
from navigation.jackal_navigation_state import (
    TERMINAL_STATES,
    NavState,
    NavigationStateError,
)
from navigation.jackal_policy import PolicyError, validate_action


@dataclass(frozen=True)
class GoalSpec:
    x: float
    y: float
    yaw: float | None
    tolerance_m: float
    timeout_s: float
    max_steps: int
    scenario_id: str
    map_id: str
    split: str
    seed: int

    def __post_init__(self):
        numbers = (self.x, self.y, self.tolerance_m, self.timeout_s)
        if self.yaw is not None:
            numbers = numbers + (self.yaw,)
        if not all(math.isfinite(float(v)) for v in numbers):
            raise NavigationStateError("Goal values must be finite")
        if self.tolerance_m < 0 or self.timeout_s <= 0 or self.max_steps < 1:
            raise NavigationStateError("Goal limits must be positive")
        if self.split not in {"fixed", "unseen"}:
            raise NavigationStateError("split must be fixed or unseen")
        if not self.scenario_id or not self.map_id:
            raise NavigationStateError("Goal identifiers must be non-empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise NavigationStateError("seed must be an integer")


@dataclass(frozen=True)
class CommandDecision:
    state: NavState
    source: str
    commanding: bool
    linear: float
    angular: float
    normalized_action: tuple | None
    detail: str


class NavigationManager:
    """Run the goal state machine around an already constructed policy."""

    def __init__(self, policy, mapper=None, safety=None, start_pose=(0.0, 0.0, 0.0)):
        self.policy = policy
        self.mapper = mapper if mapper is not None else ActionMapper()
        self.safety = safety if safety is not None else SafetyMonitor()
        self.start_pose = tuple(float(v) for v in start_pose)
        self.state = NavState.IDLE
        self.goal = None
        self.steps = 0
        self.path_length = 0.0
        self._previous_xy = None
        self._linear_sum = 0.0
        self._angular_sum = 0.0
        self._velocity_samples = 0
        self._final_distance = None
        self.record = None
        self.invalid_observations = 0
        self.invalid_actions = 0
        self.map_invocations = 0
        if not hasattr(policy, "predict") or not hasattr(policy, "reset"):
            raise NavigationStateError("Policy must provide reset and predict")
        role = getattr(policy, "role", "")
        if role not in {"rl_policy", "classical_baseline"}:
            raise NavigationStateError("Policy role must be rl_policy or classical_baseline")

    def reset(self):
        self.state = NavState.IDLE
        self.goal = None
        self.steps = 0
        self.path_length = 0.0
        self._previous_xy = None
        self._linear_sum = 0.0
        self._angular_sum = 0.0
        self._velocity_samples = 0
        self._final_distance = None
        self.record = None
        self.invalid_observations = 0
        self.invalid_actions = 0
        self.map_invocations = 0
        self.mapper.reset()
        self.policy.reset()
        return self.state

    def set_goal(self, goal):
        if self.state in TERMINAL_STATES:
            raise NavigationStateError(
                f"Cannot replace a goal after {self.state.value}"
            )
        if not isinstance(goal, GoalSpec):
            raise NavigationStateError("goal must be a GoalSpec")
        self.goal = goal
        self.state = NavState.INITIALIZING
        return self.state

    def emergency_stop(self):
        if self.state is NavState.IDLE:
            raise NavigationStateError("Cannot emergency-stop before a goal is set")
        self.safety.emergency_stop()
        self.mapper.emergency_stop()
        if self.state not in TERMINAL_STATES:
            self.state = NavState.EMERGENCY_STOP
            self._freeze(self._final_distance if self._final_distance is not None else math.inf)
        return self._decision("emergency_stop", False, 0.0, 0.0, None, "Emergency stop")

    def step(
        self,
        observation,
        lidar_ranges,
        lidar_angles,
        scan_age,
        odom_age,
        goal_distance,
        collision,
        dt,
        pose_xy,
        elapsed_s,
    ):
        """Advance one control cycle and return the command that may be sent.

        ``commanding`` is false in every terminal state. The linear and
        angular fields are physical velocities after the mapper and the
        safety monitor, or zeros when the episode must hold.
        """

        if self.state in TERMINAL_STATES or self.state is NavState.IDLE:
            return self._decision("hold", False, 0.0, 0.0, None, self.state.value)

        if self.goal is None or self.state not in {NavState.INITIALIZING, NavState.NAVIGATING}:
            raise NavigationStateError(f"Unexpected state {self.state}")

        distance = float(goal_distance)
        elapsed = float(elapsed_s)
        self._final_distance = distance
        self._extend_path(pose_xy)

        if collision:
            return self._enter(NavState.COLLISION, distance, "Collision")

        if distance <= self.goal.tolerance_m:
            return self._enter(NavState.GOAL_REACHED, distance, "Goal reached")

        if elapsed >= self.goal.timeout_s or self.steps >= self.goal.max_steps:
            return self._enter(NavState.TIMEOUT, distance, "Timeout")

        if scan_age > self.safety.sensor_timeout or odom_age > self.safety.sensor_timeout:
            self.invalid_observations += 1
            return self._enter(NavState.SENSOR_ERROR, distance, "Sensor age exceeded the timeout")

        if observation is None or not getattr(observation, "valid", False):
            self.invalid_observations += 1
            return self._enter(NavState.SENSOR_ERROR, distance, "Observation invalid")

        self.state = NavState.NAVIGATING
        self.steps += 1

        try:
            action = validate_action(
                self.policy.predict(observation.vector, deterministic=True)
            )
        except (PolicyError, ValueError, TypeError) as exc:
            self.invalid_actions += 1
            return self._enter(NavState.POLICY_ERROR, distance, str(exc))

        try:
            command = self.mapper.map_action(action, dt)
            self.map_invocations += 1
        except (ValueError, TypeError) as exc:
            self.invalid_actions += 1
            return self._enter(NavState.POLICY_ERROR, distance, str(exc))

        safety = self.safety.evaluate(
            linear=command.linear,
            angular=command.angular,
            lidar_ranges=lidar_ranges,
            lidar_angles=lidar_angles,
            scan_age=scan_age,
            odom_age=odom_age,
        )
        if not safety.safe:
            self.mapper.emergency_stop()
            return self._decision(
                "safety_intervention",
                False,
                0.0,
                0.0,
                tuple(float(v) for v in action),
                safety.reason,
            )

        if safety.reason != "Command permitted":
            from jackal_rl_navigation.action_mapper import VelocityCommand

            self.mapper.previous = VelocityCommand(float(safety.linear), float(safety.angular))
            source = "safety_intervention"
            detail = safety.reason
        else:
            source = self.policy.role
            detail = "Command permitted"

        self._linear_sum += float(safety.linear)
        self._angular_sum += float(safety.angular)
        self._velocity_samples += 1
        return self._decision(
            source,
            True,
            float(safety.linear),
            float(safety.angular),
            tuple(float(v) for v in action),
            detail,
        )

    def _extend_path(self, pose_xy):
        x, y = float(pose_xy[0]), float(pose_xy[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise NavigationStateError("Pose must be finite")
        if self._previous_xy is not None:
            self.path_length += math.hypot(x - self._previous_xy[0], y - self._previous_xy[1])
        self._previous_xy = (x, y)

    def _enter(self, state, distance, detail):
        self.mapper.emergency_stop()
        self.state = state
        self._freeze(distance)
        return self._decision("hold", False, 0.0, 0.0, None, detail)

    def _freeze(self, distance):
        if self.goal is None or self.record is not None:
            return
        straight = math.hypot(self.goal.x - self.start_pose[0], self.goal.y - self.start_pose[1])
        samples = self._velocity_samples
        family = getattr(self.policy, "algorithm_family", None)
        if family is None:
            family = (
                "classical" if self.policy.role == "classical_baseline"
                else "reinforcement_learning"
            )
        final_distance = float(distance)
        if not math.isfinite(final_distance):
            final_distance = straight
        self.record = JackalEpisodeRecord(
            scenario_id=self.goal.scenario_id,
            start_pose=self.start_pose,
            goal_pose=(self.goal.x, self.goal.y, 0.0 if self.goal.yaw is None else self.goal.yaw),
            map_id=self.goal.map_id,
            policy_id=type(self.policy).__name__,
            algorithm_family=family,
            seed=self.goal.seed,
            duration_s=0.0,
            steps=self.steps,
            goal_completed=self.state is NavState.GOAL_REACHED,
            collision_count=1 if self.state is NavState.COLLISION else 0,
            timeout=self.state is NavState.TIMEOUT,
            final_goal_distance=final_distance,
            path_length=self.path_length,
            straight_line_distance=straight,
            path_efficiency=path_efficiency(straight, self.path_length),
            average_linear_velocity=(self._linear_sum / samples) if samples else 0.0,
            average_angular_velocity=(self._angular_sum / samples) if samples else 0.0,
            safety_intervention_count=int(self.safety.intervention_count),
            invalid_observation_count=self.invalid_observations,
            invalid_action_count=self.invalid_actions,
            split=self.goal.split,
            termination=self.state.value,
        )

    def finish(self, duration_s):
        """Stamp the wall-clock duration onto a terminal record."""

        if self.record is None or self.state not in TERMINAL_STATES:
            raise NavigationStateError("finish requires a terminal episode")
        duration = float(duration_s)
        if not math.isfinite(duration) or duration < 0:
            raise NavigationStateError("duration must be finite and nonnegative")
        payload = self.record
        self.record = JackalEpisodeRecord(
            scenario_id=payload.scenario_id,
            start_pose=payload.start_pose,
            goal_pose=payload.goal_pose,
            map_id=payload.map_id,
            policy_id=payload.policy_id,
            algorithm_family=payload.algorithm_family,
            seed=payload.seed,
            duration_s=duration,
            steps=payload.steps,
            goal_completed=payload.goal_completed,
            collision_count=payload.collision_count,
            timeout=payload.timeout,
            final_goal_distance=payload.final_goal_distance,
            path_length=payload.path_length,
            straight_line_distance=payload.straight_line_distance,
            path_efficiency=payload.path_efficiency,
            average_linear_velocity=payload.average_linear_velocity,
            average_angular_velocity=payload.average_angular_velocity,
            safety_intervention_count=payload.safety_intervention_count,
            invalid_observation_count=payload.invalid_observation_count,
            invalid_action_count=payload.invalid_action_count,
            split=payload.split,
            termination=payload.termination,
        )
        return self.record

    def _decision(self, source, commanding, linear, angular, action, detail):
        return CommandDecision(
            state=self.state,
            source=source,
            commanding=bool(commanding) and self.state not in TERMINAL_STATES,
            linear=float(linear),
            angular=float(angular),
            normalized_action=action,
            detail=detail,
        )
