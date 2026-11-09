"""Unit tests for the pure-pursuit path follower."""

import math

from planners.pure_pursuit_controller import PurePursuitController


def test_curve_keeps_cruise_speed_until_the_final_approach():
    path = [(-1.0, -1.5), (-0.95, -0.95), (-0.95, -0.45), (3.0, 1.5)]
    controller = PurePursuitController(path, max_linear=0.25)

    command = controller.compute(-1.0, -1.5, 0.38)

    assert command.reached_goal is False
    assert command.linear >= 0.15
    assert abs(command.angular) <= 0.60


def test_goal_tolerance_stops_the_robot():
    controller = PurePursuitController([(0.0, 0.0), (2.0, 0.0)])

    command = controller.compute(1.90, 0.0, 0.0)

    assert command.reached_goal is True
    assert command.linear == 0.0
    assert command.angular == 0.0


def test_target_behind_the_robot_turns_without_driving_forward():
    controller = PurePursuitController([(0.0, 0.0), (2.0, 0.0)])

    command = controller.compute(0.0, 0.0, math.pi)

    assert command.linear == 0.0
    assert command.angular < 0.0
    assert abs(command.angular) <= controller.max_angular
