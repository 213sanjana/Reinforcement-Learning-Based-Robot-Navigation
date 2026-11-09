"""Bounded, opt-in A* navigation for the Gazebo Jackal.

Default behavior is read-only.
Movement requires the explicit --enable-motion flag.
"""

import argparse
import math
import time

import rclpy

from navigation.guarded_ros_executor import GuardedROSExecutor
from navigation.guarded_navigation import velocity_to_action




class PurePursuitAdapter:
    """Expose the command interface required by GuardedNavigation."""

    def __init__(self, path):
        from planners.pure_pursuit_controller import PurePursuitController

        self.impl = PurePursuitController(
            path,
            lookahead_distance=0.45,
            max_linear=0.25,
            max_angular=0.60,
            goal_tolerance=0.15,
        )

    def compute(self, x, y, yaw):
        from types import SimpleNamespace

        command = self.impl.compute(x, y, yaw)

        return SimpleNamespace(
            linear=command.linear,
            angular=command.angular,
            finished=command.reached_goal,
            waypoint_index=command.waypoint_index,
        )

class LiveAStarExecutor(GuardedROSExecutor):

    def __init__(self, goal=(3.0, 0.0), enable_motion=False, inflation_radius=0.45, controller_type="waypoint", safe_start_recovery=False):
        super().__init__(goal=goal, inflation_radius=inflation_radius)
        if controller_type not in ("waypoint", "pure-pursuit"):
            raise ValueError("Unknown controller type")

        self.controller_type = controller_type
        self.safe_start_recovery = bool(safe_start_recovery)
        self.enable_motion = bool(enable_motion)
        self.commands_accepted = 0
        self.commands_rejected = 0
        self.command_interval = 0.1
        self.last_action_attempt = None

    def step(self):
        pose = self.get_pose()

        if pose is None or self.node.scan is None:
            if self.enable_motion and self.commands_accepted > 0:
                return "SENSOR_LOSS"
            return "WAITING_FOR_SENSORS"

        if self.controller is None:
            try:
                initialized = self.initialize_plan()

            except RuntimeError as exc:
                if not self.safe_start_recovery:
                    print(f"PLANNING_FAILED: {exc}", flush=True)
                    return "PLANNING_FAILED"

                from planners.safe_start_recovery import plan_with_safe_start
                from planners.waypoint_controller import WaypointController

                pose = self.get_pose()

                if pose is None:
                    return "WAITING_FOR_PLAN"

                recovered = plan_with_safe_start(
                    self.planner.world_path,
                    (pose[0], pose[1]),
                    self.goal,
                    inflation_radius=self.planner.inflation_radius,
                )

                if recovered is None:
                    print(
                        "PLANNING_FAILED: No safe recovery route",
                        flush=True,
                    )
                    return "PLANNING_FAILED"

                self.path = recovered
                self.controller = WaypointController(recovered)
                initialized = True

                print(
                    f"SAFE_START_RECOVERY: {len(recovered)} waypoints",
                    flush=True,
                )

            if not initialized:
                return "WAITING_FOR_PLAN"

            if self.controller_type == "pure-pursuit":
                self.controller = PurePursuitAdapter(self.path)
                print(
                    "Controller: PURE_PURSUIT",
                    flush=True,
                )
            else:
                print(
                    "Controller: WAYPOINT",
                    flush=True,
                )

        if self.guard is None:
            from navigation.guarded_navigation import GuardedNavigation

            self.guard = GuardedNavigation(
                path=self.path,
                controller=self.controller,
                stall_timeout=12.0,
                navigation_timeout=120.0,
            )

        decision = self.guard.evaluate(
            pose=pose,
            sensor_ages=self.node._sensor_ages(),
            collision_detected=self.collision_monitor.collision_detected,
            now=time.monotonic(),
        )

        if decision.status != "RUNNING":
            print(
                f"STOP: {decision.status} — {decision.reason}",
                flush=True,
            )
            return decision.status

        action = velocity_to_action(
            decision.linear,
            decision.angular,
            self.node.action_mapper,
        )

        if self.enable_motion:
            now = time.monotonic()

            if (
                self.last_action_attempt is not None
                and now - self.last_action_attempt
                < self.command_interval
            ):
                return "RUNNING"

            self.last_action_attempt = now
            accepted = self.node.command_action(action)

            if accepted:
                self.commands_accepted += 1
            else:
                self.commands_rejected += 1

                # Fail closed: never assume rejection is harmless.
                # The caller must stop and investigate the reason.
                return "COMMAND_REJECTED"

        now = time.monotonic()

        if now - self.last_log >= 1.0:
            distance = math.hypot(
                pose[0] - self.goal[0],
                pose[1] - self.goal[1],
            )

            print(
                f"MODE={'LIVE' if self.enable_motion else 'DRY_RUN'} "
                f"POSE=({pose[0]:.3f},{pose[1]:.3f}) "
                f"GOAL_DISTANCE={distance:.3f} "
                f"ROUTE_PROGRESS={decision.progress:.3f} "
                f"ACTION=({action[0]:.3f},{action[1]:.3f}) "
                f"ACCEPTED={self.commands_accepted} "
                f"REJECTED={self.commands_rejected}",
                flush=True,
            )

            self.last_log = now

        return "RUNNING"

    def run(self, duration=5.0):
        print(
            "Movement:",
            "ENABLED" if self.enable_motion else "DISABLED",
            flush=True,
        )

        deadline = time.monotonic() + duration
        final_status = "TIME_LIMIT"

        try:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(
                    self.node,
                    timeout_sec=0.05,
                )

                status = self.step()

                if status not in (
                    "RUNNING",
                    "WAITING_FOR_SENSORS",
                    "WAITING_FOR_PLAN",
                ):
                    final_status = status
                    break

        except KeyboardInterrupt:
            final_status = "INTERRUPTED"

        finally:
            if self.enable_motion:
                # Always publish a zero-velocity command.
                self.node.stop()

            print("FINAL_STATUS:", final_status)
            print("COMMANDS_ACCEPTED:", self.commands_accepted)
            print("COMMANDS_REJECTED:", self.commands_rejected)

            self.node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Explicitly authorize robot movement",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
    )


    parser.add_argument(
        "--inflation-radius",
        type=float,
        default=0.45,
        help="A* obstacle inflation radius in metres",
    )

    parser.add_argument(
        "--controller",
        choices=("waypoint", "pure-pursuit"),
        default="waypoint",
        help="Navigation path-following controller",
    )

    parser.add_argument(
        "--safe-start-recovery",
        action="store_true",
        help="Attempt checked recovery when normal A* planning fails",
    )

    args = parser.parse_args()

    if not math.isfinite(args.duration) or not 0 < args.duration <= 120:
        parser.error("Duration must be between 0 and 120 seconds")

    rclpy.init()


    if (
        not math.isfinite(args.inflation_radius)
        or args.inflation_radius <= 0
    ):
        parser.error(
            "--inflation-radius must be a finite positive number"
        )

    executor = LiveAStarExecutor(
        enable_motion=args.enable_motion,
        inflation_radius=args.inflation_radius,
        controller_type=args.controller,
        safe_start_recovery=args.safe_start_recovery,
    )

    executor.run(duration=args.duration)


if __name__ == "__main__":
    main()
