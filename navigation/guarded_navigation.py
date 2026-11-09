"""Guarded A* navigation decisions without ROS velocity publishing."""

import math
from dataclasses import dataclass

from navigation.navigation_safety import evaluate_navigation_safety
from navigation.route_progress import RouteProgressWatchdog


@dataclass(frozen=True)
class NavigationDecision:
    status: str
    reason: str
    linear: float = 0.0
    angular: float = 0.0
    progress: float = 0.0


def velocity_to_action(linear, angular, mapper):
    """Convert desired velocities to normalized ActionMapper inputs."""

    if not all(math.isfinite(v) for v in (linear, angular)):
        raise ValueError("Nonfinite velocity command")

    if linear >= 0:
        linear_action = linear / mapper.max_forward
    else:
        linear_action = linear / mapper.max_reverse

    angular_action = angular / mapper.max_angular

    return (
        max(-1.0, min(1.0, linear_action)),
        max(-1.0, min(1.0, angular_action)),
    )


class GuardedNavigation:
    """Combine waypoint tracking with independent safety checks."""

    def __init__(
        self,
        path,
        controller,
        stall_timeout=12.0,
        navigation_timeout=120.0,
    ):
        self.path = path
        self.controller = controller

        self.route_watchdog = RouteProgressWatchdog(
            path,
            stall_timeout=stall_timeout,
            min_progress=0.10,
        )

        self.navigation_timeout = float(navigation_timeout)

        if self.navigation_timeout <= 0:
            raise ValueError("Navigation timeout must be positive")

        self.started_at = None
        self.terminal_status = None

    def evaluate(
        self,
        pose,
        sensor_ages,
        collision_detected,
        now,
    ):
        """Calculate a guarded navigation decision."""

        if self.terminal_status is not None:
            return NavigationDecision(
                self.terminal_status,
                "Navigation already terminated",
            )

        safety = evaluate_navigation_safety(
            pose=pose,
            path=self.path,
            sensor_ages=sensor_ages,
            collision_detected=collision_detected,
        )

        if not safety.safe:
            self.terminal_status = "SAFETY_STOP"

            return NavigationDecision(
                "SAFETY_STOP",
                safety.reason,
            )

        if not math.isfinite(now):
            self.terminal_status = "SAFETY_STOP"

            return NavigationDecision(
                "SAFETY_STOP",
                "Invalid navigation time",
            )

        if self.started_at is None:
            self.started_at = now
            self.route_watchdog.reset(pose, now)

        if now < self.started_at:
            self.terminal_status = "SAFETY_STOP"

            return NavigationDecision(
                "SAFETY_STOP",
                "Navigation time moved backwards",
            )

        if now - self.started_at > self.navigation_timeout:
            self.terminal_status = "TIMEOUT"

            return NavigationDecision(
                "TIMEOUT",
                "Navigation exceeded time limit",
            )

        command = self.controller.compute(*pose)

        if command.finished:
            self.terminal_status = "GOAL_REACHED"

            return NavigationDecision(
                "GOAL_REACHED",
                "Waypoint controller reports goal reached",
            )

        safe, reason, progress = self.route_watchdog.evaluate(
            pose,
            now,
        )

        if not safe:
            self.terminal_status = "STALLED"

            return NavigationDecision(
                "STALLED",
                reason,
                progress=progress,
            )

        return NavigationDecision(
            "RUNNING",
            "Navigation permitted",
            linear=max(0.0, command.linear),
            angular=command.angular,
            progress=progress,
        )
