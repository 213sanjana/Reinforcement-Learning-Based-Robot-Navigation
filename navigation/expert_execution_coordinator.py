
"""Coordinate guarded expert decisions and existing robot command recording."""

import math

from navigation.live_astar_executor import velocity_to_action


class ExpertExecutionCoordinator:
    """Execute one guarded expert decision using the existing publisher."""

    def __init__(self, robot, navigator, recording_bridge):
        self.robot = robot
        self.navigator = navigator
        self.bridge = recording_bridge
        self.accepted = 0
        self.rejected = 0
        self.last_status = None

    def step(
        self,
        pose,
        sensor_ages,
        collision_detected=False,
        now=None,
    ):
        decision = self.navigator.evaluate(
            pose=pose,
            sensor_ages=sensor_ages,
            collision_detected=collision_detected,
            now=now,
        )

        self.last_status = decision.status

        if decision.status != "RUNNING":
            self.robot.stop()
            return decision.status

        if not all(
            math.isfinite(float(v))
            for v in (decision.linear, decision.angular)
        ):
            self.robot.stop()
            self.navigator.terminate("INVALID_COMMAND")
            self.last_status = "INVALID_COMMAND"
            return self.last_status

        action = velocity_to_action(
            decision.linear,
            decision.angular,
            self.robot.action_mapper,
        )

        accepted = self.bridge.execute(action, pose)

        if not accepted:
            # An empty reason means the command interval has not elapsed.
            # That is not a rejected motion command.
            if getattr(self.robot, "last_safety_reason", None) == "":
                return "RUNNING"

            self.rejected += 1
            self.robot.stop()
            self.navigator.terminate("COMMAND_REJECTED")
            self.last_status = "COMMAND_REJECTED"
            return self.last_status

        self.accepted += 1
        return "RUNNING"
