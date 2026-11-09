import numpy as np
from ros_integration.jackal_turn_safety import JackalTurnSafety


class JackalGeometryTurnSafety:
    """Conservative LiDAR check for an in-place rotation.

    Assumes the sensor's minimum-range blind zone has been
    independently verified clear before use.
    """

    def __init__(self, robot_radius=0.366, margin=0.15):
        self.required_clearance = robot_radius + margin
        self.validator = JackalTurnSafety(
            clearance=self.required_clearance,
            sectors=36,
            max_invalid_fraction=0.05,
        )

    def check(self, ranges, angles):
        return self.validator.check(ranges, angles)
