"""One-publisher goal cycle for the classical Jackal pilot.

The cycle asks the goal manager to map a normalized action once, then hands
the resulting physical velocity to the caller-supplied publisher. It does
not construct a ROS publisher, load a checkpoint, or call a trainer.
"""

from __future__ import annotations

from navigation.jackal_classical_baseline import ClassicalGoalBaseline
from navigation.jackal_command_contract import (
    accept_normalized_action,
    assert_physical_command,
)
from navigation.jackal_navigation_manager import NavigationManager
from navigation.jackal_navigation_state import TERMINAL_STATES, NavState
from navigation.jackal_policy import PolicyError


class PilotError(RuntimeError):
    """The pilot was asked to take a second command path."""


class GoalPilot:
    """Own the manager and exactly one velocity publisher callback."""

    def __init__(self, manager, publish_velocity):
        if not isinstance(manager, NavigationManager):
            raise PilotError("GoalPilot requires a NavigationManager")
        if not callable(publish_velocity):
            raise PilotError("A single publish callback is required")
        if getattr(manager.policy, "algorithm_family", None) != "classical":
            raise PilotError("This pilot selects the classical baseline only")
        self.manager = manager
        self._publish_velocity = publish_velocity
        self.publish_count = 0
        self.last_map_delta = 0
        self.last_decision = None

    @classmethod
    def classical(cls, mapper, safety, publish_velocity, start_pose=(0.0, 0.0, 0.0)):
        """Bind the classical baseline to an existing mapper, safety check, and publisher."""

        policy = ClassicalGoalBaseline()
        manager = NavigationManager(
            policy,
            mapper=mapper,
            safety=safety,
            start_pose=start_pose,
        )
        return cls(manager, publish_velocity)

    def cycle(
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
        """Map at most once and publish the resulting velocity once."""

        before = self.manager.map_invocations
        decision = self.manager.step(
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
        )
        self.last_map_delta = self.manager.map_invocations - before
        if self.last_map_delta > 1:
            raise PilotError("Action mapping ran more than once in one cycle")
        if decision.normalized_action is not None:
            accept_normalized_action(decision.normalized_action)
        assert_physical_command(
            decision.linear,
            decision.angular,
            self.manager.mapper,
        )
        if decision.commanding and self.last_map_delta != 1:
            raise PilotError("A motion command must come from one mapping")
        published = self._publish_velocity(decision.linear, decision.angular)
        self.publish_count += 1
        if self.manager.record is not None and self.manager.state in TERMINAL_STATES:
            if self.manager.record.duration_s == 0.0 and float(elapsed_s) > 0.0:
                self.manager.finish(elapsed_s)
        self.last_decision = decision
        return decision, published

    @property
    def record(self):
        return self.manager.record

    @property
    def state(self):
        return self.manager.state


def refuse_checkpoint_controller(controller):
    """The pilot accepts one named controller and no weight file."""

    text = str(controller)
    if text.endswith(".zip") or "initialized_ppo" in text or "jackal_6g1" in text:
        raise PolicyError("Refusing to load a checkpoint")
    if text != "classical_baseline":
        raise PolicyError(
            "Pilot controller must be classical_baseline. Checkpoints are not loaded."
        )
    return text
