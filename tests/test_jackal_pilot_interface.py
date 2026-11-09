"""Interface binding for the goal pilot. The publisher is a Mock.

Importing rclpy here does not start a node or Gazebo. The module skips
when rclpy is absent.
"""

import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("rclpy")

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.robot_interface import JackalRLInterface
from jackal_rl_navigation.safety_monitor import SafetyMonitor

from navigation.jackal_navigation_state import NavState


def _odom(x=0.0, y=0.0):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="odom"),
        child_frame_id="base_link",
        pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )),
        twist=SimpleNamespace(twist=SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        )),
    )


def _interface(goal_distance_ranges=8.0):
    interface = object.__new__(JackalRLInterface)
    interface.action_mapper = ActionMapper()
    interface.safety_monitor = SafetyMonitor()
    interface.observation_builder = ObservationBuilder()
    interface.goal_x = 4.0
    interface.goal_y = 0.0
    interface.goal_frame = "odom"
    interface.last_command_time = None
    interface.last_publish_time = None
    interface.last_safety_reason = ""
    interface.expert_publish_callback = None
    interface._goal_pilot = None
    interface._pilot_started = None
    interface.collision_monitor = SimpleNamespace(
        collision_detected=False,
        collision_pairs=[],
        messages_received=0,
        valid_messages=0,
        malformed_messages=0,
        subscription=object(),
        topic="/jackal_rl/obstacle_contacts",
    )
    interface.collision_init_error = None
    interface.collision_topic = "/jackal_rl/obstacle_contacts"
    interface._collision_latched = False
    interface._collision_contact_details = []
    interface.publisher = Mock()
    interface.get_logger = Mock()
    interface._context = Mock()
    angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    interface.scan = SimpleNamespace(
        ranges=np.full(720, goal_distance_ranges),
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
    )
    interface.odom = _odom()
    interface._sensor_ages = lambda: (0.1, 0.1)
    return interface


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=True)
def test_interface_pilot_maps_once_and_blocks_the_other_command_path(mock_ok):
    interface = _interface()
    interface.configure_goal_pilot("classical_baseline", start_pose=(0.0, 0.0, 0.0))
    interface.set_navigation_goal(
        4.0, 0.0, scenario_id="world1_open", map_id="jackal_demo_open",
        split="fixed", seed=1, max_steps=10,
    )
    decision = interface.step_navigation(elapsed_s=0.1)
    assert decision.commanding is True
    assert decision.source == "classical_baseline"
    assert interface._goal_pilot.last_map_delta == 1
    assert interface._goal_pilot.manager.map_invocations == 1
    message = interface.publisher.publish.call_args.args[0]
    assert message.linear.x == pytest.approx(decision.linear)
    assert message.linear.x == pytest.approx(0.015)
    assert message.angular.z == pytest.approx(0.0)
    assert interface.publisher is interface._goal_pilot._publish_velocity.__self__.publisher
    with pytest.raises(RuntimeError):
        interface.command_action([1.0, 0.0])
    assert interface._goal_pilot.manager.map_invocations == 1
    assert mock_ok.called


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=True)
def test_interface_goal_stop_and_emergency_publish_zero(mock_ok):
    interface = _interface()
    interface.configure_goal_pilot("classical_baseline")
    interface.set_navigation_goal(
        0.2, 0.0, scenario_id="world1_open", map_id="jackal_demo_open",
        split="fixed", seed=2,
    )
    reached = interface.step_navigation(elapsed_s=0.2)
    assert reached.state is NavState.GOAL_REACHED
    assert interface._goal_pilot.last_map_delta == 0
    message = interface.publisher.publish.call_args.args[0]
    assert message.linear.x == 0.0
    assert message.angular.z == 0.0
    record = interface._goal_pilot.record
    assert record.algorithm_family == "classical"
    assert record.goal_completed is True
    again = interface.step_navigation(elapsed_s=0.3)
    assert again.commanding is False
    assert interface._goal_pilot.manager.map_invocations == 0

    moving = _interface()
    moving.configure_goal_pilot("classical_baseline")
    moving.set_navigation_goal(
        4.0, 0.0, scenario_id="world1_open", map_id="jackal_demo_open",
        split="fixed", seed=3, max_steps=10,
    )
    moving.step_navigation(elapsed_s=0.1)
    moving.emergency_stop()
    assert moving._goal_pilot.state is NavState.EMERGENCY_STOP
    stopped = moving.step_navigation(elapsed_s=0.4)
    assert stopped.linear == 0.0 and stopped.angular == 0.0
    assert moving._goal_pilot.manager.map_invocations == 1
    assert mock_ok.called
