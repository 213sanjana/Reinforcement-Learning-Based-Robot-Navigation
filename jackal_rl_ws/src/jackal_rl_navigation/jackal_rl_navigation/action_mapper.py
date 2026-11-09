"""Map normalized continuous actions to bounded Jackal velocities."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VelocityCommand:
    linear: float
    angular: float


class ActionMapper:
    """Convert policy actions into acceleration-limited velocity commands."""

    def __init__(
        self,
        max_forward=0.30,
        max_reverse=0.15,
        max_angular=0.60,
        linear_acceleration=0.30,
        angular_acceleration=0.60,
        min_command_interval=0.05,
    ):
        parameters = (
            max_forward,
            max_reverse,
            max_angular,
            linear_acceleration,
            angular_acceleration,
            min_command_interval,
        )

        if not all(math.isfinite(v) and v > 0 for v in parameters):
            raise ValueError("All mapper parameters must be positive and finite")

        self.max_forward = float(max_forward)
        self.max_reverse = float(max_reverse)
        self.max_angular = float(max_angular)

        self.linear_acceleration = float(linear_acceleration)
        self.angular_acceleration = float(angular_acceleration)
        self.min_command_interval = float(min_command_interval)

        self.previous = VelocityCommand(0.0, 0.0)

    def reset(self):
        """Reset mapper state after a confirmed robot stop."""
        self.previous = VelocityCommand(0.0, 0.0)

    def emergency_stop(self):
        """Return zero immediately, bypassing acceleration limiting."""
        self.reset()
        return self.previous

    def map_action(self, action, dt):
        """Map an action in [-1, 1]^2 into safe bounded velocities."""

        values = np.asarray(action, dtype=np.float64)

        if values.shape != (2,):
            raise ValueError("Action must contain exactly two values")

        if not np.all(np.isfinite(values)):
            raise ValueError("Action values must be finite")

        if not np.all((-1.0 <= values) & (values <= 1.0)):
            raise ValueError("Action values must be within [-1, 1]")

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")

        if dt < self.min_command_interval:
            raise ValueError("Command interval is shorter than configured minimum")

        linear_action = float(values[0])
        angular_action = float(values[1])

        if linear_action >= 0:
            target_linear = linear_action * self.max_forward
        else:
            target_linear = linear_action * self.max_reverse

        target_angular = angular_action * self.max_angular

        linear_delta = self.linear_acceleration * dt
        angular_delta = self.angular_acceleration * dt

        linear = self.previous.linear + np.clip(
            target_linear - self.previous.linear,
            -linear_delta,
            linear_delta,
        )

        angular = self.previous.angular + np.clip(
            target_angular - self.previous.angular,
            -angular_delta,
            angular_delta,
        )

        linear = float(np.clip(
            linear,
            -self.max_reverse,
            self.max_forward,
        ))

        angular = float(np.clip(
            angular,
            -self.max_angular,
            self.max_angular,
        ))

        self.previous = VelocityCommand(linear, angular)

        return self.previous


if __name__ == "__main__":
    mapper = ActionMapper()

    print("========== ACTION MAPPER TEST ==========")

    for step in range(5):
        command = mapper.map_action([1.0, 0.5], dt=0.1)

        print(
            f"Step {step + 1}: "
            f"linear={command.linear:.3f} m/s, "
            f"angular={command.angular:.3f} rad/s"
        )

    stopped = mapper.emergency_stop()

    print(
        f"Emergency stop: "
        f"linear={stopped.linear:.3f}, "
        f"angular={stopped.angular:.3f}"
    )

    print("========== TEST COMPLETE ==========")
