"""Reward calculation for Jackal RL navigation."""

import math


class NavigationReward:
    """Calculate progress-based navigation rewards."""

    def __init__(
        self,
        progress_scale=10.0,
        step_penalty=0.01,
    ):
        self.progress_scale = float(progress_scale)
        self.step_penalty = float(step_penalty)

        if not math.isfinite(self.progress_scale):
            raise ValueError("progress_scale must be finite")

        if not math.isfinite(self.step_penalty):
            raise ValueError("step_penalty must be finite")

        if self.progress_scale < 0 or self.step_penalty < 0:
            raise ValueError("Reward parameters must be nonnegative")

    def calculate(self, previous_distance, current_distance):
        """Return reward and diagnostics for one navigation step."""

        previous_distance = float(previous_distance)
        current_distance = float(current_distance)

        if not (
            math.isfinite(previous_distance)
            and math.isfinite(current_distance)
        ):
            raise ValueError("Goal distances must be finite")

        if previous_distance < 0 or current_distance < 0:
            raise ValueError("Goal distances cannot be negative")

        progress = previous_distance - current_distance

        progress_reward = self.progress_scale * progress

        reward = progress_reward - self.step_penalty

        info = {
            "previous_distance": previous_distance,
            "current_distance": current_distance,
            "progress": progress,
            "progress_reward": progress_reward,
            "step_penalty": self.step_penalty,
        }

        return float(reward), info
