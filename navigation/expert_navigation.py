
"""Reusable A* expert planning and guarded navigation core.

This module does not create ROS nodes or velocity publishers.
"""

from dataclasses import dataclass
import math
import time

from navigation.guarded_navigation import GuardedNavigation
from navigation.live_astar_executor import PurePursuitAdapter
from planners.waypoint_controller import WaypointController


@dataclass(frozen=True)
class ExpertStatus:
    status: str
    reason: str
    linear: float = 0.0
    angular: float = 0.0
    progress: float = 0.0


class ExpertNavigator:
    """Plan and calculate safe navigation commands for one episode."""

    def __init__(
        self,
        planner,
        controller_type="pure-pursuit",
        navigation_timeout=120.0,
    ):
        if controller_type not in ("waypoint", "pure-pursuit"):
            raise ValueError("Invalid controller type")

        if not math.isfinite(navigation_timeout) or navigation_timeout <= 0:
            raise ValueError("Invalid navigation timeout")

        self.planner = planner
        self.controller_type = controller_type
        self.navigation_timeout = float(navigation_timeout)

        self.path = None
        self.controller = None
        self.guard = None
        self.goal = None
        self.start = None
        self.terminal_status = None

    @staticmethod
    def _validate_point(point):
        if len(point) != 2:
            raise ValueError("Expected (x, y) world coordinates")

        result = tuple(float(v) for v in point)

        if not all(math.isfinite(v) for v in result):
            raise ValueError("Non-finite world coordinates")

        return result

    def plan(self, start, goal):
        """Create a fresh A* plan and controller for an episode."""

        start = self._validate_point(start)
        goal = self._validate_point(goal)

        if self.guard is not None and self.terminal_status is None:
            raise RuntimeError("Previous episode has not terminated")

        path = self.planner.plan(start, goal)

        if not path:
            self.terminal_status = "PLANNING_FAILED"
            self.path = None
            self.controller = None
            self.guard = None
            return None

        if not self.planner.validate_path(path):
            self.terminal_status = "INVALID_PATH"
            self.path = None
            self.controller = None
            self.guard = None
            return None

        self.start = start
        self.goal = goal
        self.path = list(path)

        if self.controller_type == "pure-pursuit":
            self.controller = PurePursuitAdapter(self.path)
        else:
            self.controller = WaypointController(self.path)

        self.guard = GuardedNavigation(
            path=self.path,
            controller=self.controller,
            stall_timeout=30.0,
            navigation_timeout=self.navigation_timeout,
        )

        self.terminal_status = None
        return self.path

    def evaluate(
        self,
        pose,
        sensor_ages,
        collision_detected=False,
        now=None,
    ):
        """Return a guarded decision; never publish a velocity command."""

        if self.terminal_status is not None:
            return ExpertStatus(
                self.terminal_status,
                "Episode already terminated",
            )

        if self.guard is None:
            return ExpertStatus(
                "NOT_INITIALIZED",
                "No valid expert path",
            )

        if now is None:
            now = time.monotonic()

        decision = self.guard.evaluate(
            pose=pose,
            sensor_ages=sensor_ages,
            collision_detected=collision_detected,
            now=now,
        )

        if decision.status != "RUNNING":
            self.terminal_status = decision.status

        return ExpertStatus(
            status=decision.status,
            reason=decision.reason,
            linear=decision.linear,
            angular=decision.angular,
            progress=decision.progress,
        )

    def terminate(self, reason="STOPPED"):
        """Mark an episode terminal; caller must stop the robot."""
        self.terminal_status = str(reason)
        return ExpertStatus(self.terminal_status, "Episode terminated")
