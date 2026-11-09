
import math
import time
from dataclasses import dataclass

from ros_integration.jackal_lidar_safety import JackalLidarSafety
from ros_integration.jackal_turn_safety import JackalTurnSafety


@dataclass(frozen=True)
class CellMoveResult:
    status: str
    action: int
    distance: float
    heading_error_deg: float


class JackalCellController:
    HEADINGS = {
        0: math.pi / 2,
        1: -math.pi / 2,
        2: math.pi,
        3: 0.0,
    }

    def __init__(self, robot, cell_size=0.5, clearance=1.5):
        self.robot = robot
        self.cell_size = cell_size
        self.forward_safety = JackalLidarSafety(
            stop_distance=clearance
        )
        self.turn_safety = JackalTurnSafety(
            clearance=clearance
        )

    @staticmethod
    def angle_error(target, current):
        return math.atan2(
            math.sin(target - current),
            math.cos(target - current),
        )

    def execute(self, action, timeout=60.0):
        if action not in self.HEADINGS:
            raise ValueError(f"Invalid GridWorld action: {action}")

        target = self.HEADINGS[action]
        deadline = time.monotonic() + timeout
        status = "TIMEOUT"
        progress = 0.0
        heading_error = float("nan")

        try:
            # Phase 1: Turn to the absolute GridWorld heading.
            while time.monotonic() < deadline:
                obs = self.robot.observe(timeout=2.0)
                if obs is None:
                    status = "SENSOR_TIMEOUT"
                    break

                heading_error = self.angle_error(
                    target, obs.heading
                )

                if abs(heading_error) <= math.radians(4):
                    status = "HEADING_REACHED"
                    break

                safe, reason = self.turn_safety.check(
                    obs.lidar_ranges, obs.lidar_angles
                )
                if not safe:
                    status = f"TURN_BLOCKED_{reason}"
                    break

                angular = min(
                    0.35,
                    max(0.18, 0.9 * abs(heading_error))
                )

                self.robot.command(
                    0.0,
                    math.copysign(angular, heading_error)
                )
                self.robot.spin_for(0.08)

            self.robot.stop()

            if status != "HEADING_REACHED":
                return CellMoveResult(
                    status, action, progress,
                    math.degrees(heading_error)
                )

            # Phase 2: Travel one cell in the target direction.
            start = self.robot.observe(timeout=2.0)
            if start is None:
                return CellMoveResult(
                    "SENSOR_TIMEOUT", action, 0.0, 0.0
                )

            start_x = start.position_x
            start_y = start.position_y
            status = "TIMEOUT"

            while time.monotonic() < deadline:
                obs = self.robot.observe(timeout=2.0)
                if obs is None:
                    status = "SENSOR_TIMEOUT"
                    break

                dx = obs.position_x - start_x
                dy = obs.position_y - start_y

                progress = (
                    dx * math.cos(target)
                    + dy * math.sin(target)
                )

                heading_error = self.angle_error(
                    target, obs.heading
                )

                remaining = self.cell_size - progress

                if remaining <= 0.025:
                    status = "TARGET_REACHED"
                    break

                if abs(heading_error) > math.radians(12):
                    status = "HEADING_DRIFT"
                    break

                if not self.forward_safety.is_forward_safe(
                    obs.lidar_ranges, obs.lidar_angles
                ):
                    status = "OBSTACLE_STOP"
                    break

                speed = min(
                    0.15,
                    max(0.06, 0.6 * remaining)
                )

                angular = max(
                    -0.15,
                    min(0.15, 0.8 * heading_error)
                )

                self.robot.command(speed, angular)
                self.robot.spin_for(0.08)

            return CellMoveResult(
                status,
                action,
                progress,
                math.degrees(heading_error)
            )

        finally:
            self.robot.stop()
