
import math
import time
from dataclasses import dataclass

from ros_integration.jackal_lidar_safety import JackalLidarSafety
from ros_integration.jackal_turn_safety import JackalTurnSafety


@dataclass(frozen=True)
class CellCenterResult:
    status: str
    distance_remaining: float
    final_x: float
    final_y: float


class JackalCellCenterController:
    def __init__(self, robot, grid, clearance=1.5):
        self.robot = robot
        self.grid = grid
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
            math.cos(target - current)
        )

    def execute(self, target_row, target_col, timeout=120.0):
        target_x, target_y = self.grid.grid_to_world(
            target_row, target_col
        )

        deadline = time.monotonic() + timeout
        status = "TIMEOUT"
        last = None

        try:
            while time.monotonic() < deadline:
                obs = self.robot.observe(timeout=2.0)

                if obs is None:
                    status = "SENSOR_TIMEOUT"
                    break

                last = obs

                dx = target_x - obs.position_x
                dy = target_y - obs.position_y
                distance = math.hypot(dx, dy)

                if distance <= 0.04:
                    status = "TARGET_REACHED"
                    break

                target_heading = math.atan2(dy, dx)
                error = self.angle_error(
                    target_heading, obs.heading
                )

                if abs(error) > math.radians(8):
                    safe, reason = self.turn_safety.check(
                        obs.lidar_ranges,
                        obs.lidar_angles
                    )

                    if not safe:
                        status = f"TURN_BLOCKED_{reason}"
                        break

                    angular = min(
                        0.35,
                        max(0.18, 0.9 * abs(error))
                    )

                    self.robot.command(
                        0.0,
                        math.copysign(angular, error)
                    )

                else:
                    if not self.forward_safety.is_forward_safe(
                        obs.lidar_ranges,
                        obs.lidar_angles
                    ):
                        status = "OBSTACLE_STOP"
                        break

                    linear = min(
                        0.15,
                        max(0.05, 0.6 * distance)
                    )

                    angular = max(
                        -0.15,
                        min(0.15, 0.8 * error)
                    )

                    self.robot.command(linear, angular)

                self.robot.spin_for(0.08)

            if last is None:
                return CellCenterResult(
                    status, float("inf"),
                    float("nan"), float("nan")
                )

            remaining = math.hypot(
                target_x - last.position_x,
                target_y - last.position_y
            )

            return CellCenterResult(
                status,
                remaining,
                last.position_x,
                last.position_y
            )

        finally:
            self.robot.stop()
