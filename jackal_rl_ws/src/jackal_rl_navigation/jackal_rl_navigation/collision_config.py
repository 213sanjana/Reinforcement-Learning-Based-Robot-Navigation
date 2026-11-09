"""Configuration for physical collision monitoring."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CollisionConfig:
    """Settings for collision-aware RL episodes."""

    enabled: bool = False
    topic: str = "/jackal_rl/obstacle_contacts"
    robot_name: str = "j100_0000"
    collision_penalty: float = -10.0

    def __post_init__(self):
        """Validate collision monitoring settings."""
        import math

        if not self.topic.startswith("/"):
            raise ValueError(
                "Collision topic must be an absolute ROS topic"
            )

        if not self.robot_name:
            raise ValueError("Robot name cannot be empty")

        if not math.isfinite(self.collision_penalty):
            raise ValueError("Collision penalty must be finite")

        if self.collision_penalty > 0:
            raise ValueError(
                "Collision penalty cannot be positive"
            )
