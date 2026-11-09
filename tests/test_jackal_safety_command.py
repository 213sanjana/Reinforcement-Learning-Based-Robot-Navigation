"""Mocked command-path test. Skips when rclpy is not installed.

The publisher is a Mock. The test does not start a node or contact Gazebo.
"""

import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("rclpy")

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.robot_interface import JackalRLInterface
from jackal_rl_navigation.safety_monitor import SafetyMonitor


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=True)
def test_robot_interface_publishes_the_reduced_command_to_the_mock(mock_ok):
    interface = object.__new__(JackalRLInterface)
    interface.safety_monitor = SafetyMonitor()
    interface.action_mapper = ActionMapper()
    interface.last_safety_reason = ""
    interface.last_command_time = None
    interface.last_publish_time = None
    interface.expert_publish_callback = None
    interface.publisher = Mock()
    interface.get_logger = Mock()
    interface._context = Mock()
    angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    ranges = np.full(720, 5.0)
    side = np.abs(
        np.arctan2(np.sin(angles - math.pi / 2), np.cos(angles - math.pi / 2))
    ) < 0.2
    ranges[side] = 0.40
    interface.scan = SimpleNamespace(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
    )
    interface._sensor_ages = lambda: (0.1, 0.1)

    assert interface.command_action([1.0, 1.0]) is True
    message = interface.publisher.publish.call_args.args[0]
    assert message.linear.x > 0.0
    assert message.angular.z == pytest.approx(0.0)
    assert interface.action_mapper.previous.angular == pytest.approx(0.0)
    assert "Rotation clearance" in interface.last_safety_reason
    assert mock_ok.called
