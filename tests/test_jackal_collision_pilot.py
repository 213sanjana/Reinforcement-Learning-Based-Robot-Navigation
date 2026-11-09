"""Offline collision latch for the classical Jackal goal pilot.

These tests build contact messages and call the monitor callback directly.
They do not start a ROS node, Gazebo, or a velocity publisher.
"""

import ast
import math
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("ros_gz_interfaces")

from ros_gz_interfaces.msg import Contact, Contacts, Entity

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.collision_filter import is_robot_obstacle_collision
from jackal_rl_navigation.collision_monitor import CollisionMonitor
from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.robot_interface import JackalRLInterface
from jackal_rl_navigation.safety_monitor import SafetyMonitor

from navigation.jackal_episode_metrics import validate_episode
from navigation.jackal_navigation_state import NavState
from navigation.jackal_policy import PolicyError


ROOT = Path(__file__).resolve().parents[1]
INTERFACE_SOURCE = (
    ROOT
    / "jackal_rl_ws/src/jackal_rl_navigation/jackal_rl_navigation/robot_interface.py"
)
MONITOR_SOURCE = (
    ROOT
    / "jackal_rl_ws/src/jackal_rl_navigation/jackal_rl_navigation/collision_monitor.py"
)
LAUNCH_SOURCE = (
    ROOT
    / "jackal_rl_ws/src/jackal_rl_bringup/launch/jackal_demo_sim.launch.py"
)
ROBOT = "j100_0000::base_link::base_link_fixed_joint_lump__chassis_link_collision"
WHEEL = "j100_0000::front_left_wheel_link::front_left_wheel_link_collision"
GROUNDED = "ground_plane::ground_link::collision"
WALL = "wall_north::link::collision"
OTHER_ROBOT = "j100_0000/robot::base_link::base_link_fixed_joint_lump__chassis_link_collision"


def _contacts(*pairs):
    message = Contacts()
    entries = []
    for first, second in pairs:
        contact = Contact()
        contact.collision1 = Entity(name=first)
        contact.collision2 = Entity(name=second)
        entries.append(contact)
    message.contacts = entries
    return message


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
    interface.publisher = Mock()
    interface.get_logger = Mock()
    interface._context = Mock()
    interface.create_subscription = Mock(return_value="contact-subscription")
    angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    interface.scan = SimpleNamespace(
        ranges=np.full(720, goal_distance_ranges),
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
    )
    interface.odom = _odom()
    interface._sensor_ages = lambda: (0.1, 0.1)
    interface._initialize_collision_monitor()
    return interface


def _pilot(interface):
    interface.configure_goal_pilot("classical_baseline", start_pose=(0.0, 0.0, 0.0))
    interface.set_navigation_goal(
        4.0, 0.0, scenario_id="world1_open", map_id="jackal_demo_open",
        split="fixed", seed=1, max_steps=10,
    )
    return interface


def _published(interface):
    velocities = []
    for call in interface.publisher.publish.call_args_list:
        message = call.args[0]
        velocities.append((message.linear.x, message.angular.z))
    return velocities


def _monitor():
    node = Mock()
    node.create_subscription = Mock(return_value="contact-subscription")
    seen = Mock()
    monitor = CollisionMonitor(node, on_collision=seen)
    return monitor, seen, node


def test_filter_accepts_only_robot_obstacle_pairs():
    assert is_robot_obstacle_collision(ROBOT, GROUNDED) is True
    assert is_robot_obstacle_collision(WHEEL, GROUNDED) is False
    assert is_robot_obstacle_collision(OTHER_ROBOT, GROUNDED) is True
    assert is_robot_obstacle_collision(ROBOT, WALL) is True
    assert is_robot_obstacle_collision(WALL, OTHER_ROBOT) is True
    assert is_robot_obstacle_collision(ROBOT, WHEEL) is False
    assert is_robot_obstacle_collision(WALL, "wall_south::link::collision") is False
    assert is_robot_obstacle_collision(ROBOT, "") is True


def test_contact_callback_detects_an_external_collision():
    monitor, seen, node = _monitor()
    args = node.create_subscription.call_args.args
    assert args[0] is Contacts
    assert args[1] == "/jackal_rl/obstacle_contacts"
    assert args[2] == monitor._contact_callback
    monitor._contact_callback(_contacts((WHEEL, GROUNDED), (ROBOT, WALL)))
    assert monitor.collision_detected is True
    assert monitor.collision_pairs == [(ROBOT, WALL)]
    assert seen.call_count == 1
    status = seen.call_args.args[0]
    assert status["collision_detected"] is True
    assert status["valid_contact_messages"] == 1


def test_ground_only_contacts_do_not_trigger_collision():
    monitor, seen, _node = _monitor()
    with patch(
        "jackal_rl_navigation.collision_monitor.is_robot_obstacle_collision",
        wraps=is_robot_obstacle_collision,
    ) as used:
        monitor._contact_callback(_contacts((WHEEL, GROUNDED), (WHEEL, GROUNDED)))
        monitor._contact_callback(_contacts((ROBOT, WHEEL)))
    assert used.call_count == 3
    assert monitor.collision_detected is False
    assert monitor.collision_pairs == []
    assert seen.call_count == 0
    assert monitor.valid_messages == 2


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=True)
def test_collision_publishes_zero_and_stays_latched(mock_ok):
    interface = _pilot(_interface())
    moving = interface.step_navigation(elapsed_s=0.1)
    assert moving.commanding is True
    assert moving.linear > 0.0
    before = len(_published(interface))

    interface.collision_monitor._contact_callback(_contacts((ROBOT, WALL)))
    stopped = _published(interface)[before:]
    assert stopped
    assert stopped[-1] == (0.0, 0.0)
    assert interface._collision_latched is True
    assert interface._goal_pilot.state is NavState.NAVIGATING

    interface.last_command_time = time.monotonic()
    decision = interface.step_navigation(False, elapsed_s=0.2)
    assert decision is not None
    assert decision.state is NavState.COLLISION
    assert decision.commanding is False
    assert decision.linear == 0.0 and decision.angular == 0.0
    record = interface._goal_pilot.record
    validate_episode(record)
    assert record.termination == "COLLISION"
    assert record.collision_count == 1
    assert record.goal_completed is False
    assert "collision_pairs" not in record.__dataclass_fields__
    assert interface.collision_contact_details == [(ROBOT, WALL)]
    assert interface._goal_pilot.manager.collision_contact_details == ((ROBOT, WALL),)

    interface.collision_monitor._contact_callback(_contacts((WHEEL, GROUNDED)))
    again = interface.step_navigation(elapsed_s=0.4)
    assert again.state is NavState.COLLISION
    assert again.linear == 0.0 and again.angular == 0.0
    assert interface._collision_latched is True
    assert interface.collision_monitor.collision_detected is True

    interface._publish_velocity(0.3, 0.4)
    assert _published(interface)[-1] == (0.0, 0.0)
    with pytest.raises(RuntimeError, match="during an episode"):
        interface.reset_collision_latch()
    assert interface._collision_latched is True

    interface.collision_monitor.reset()
    assert interface._collision_latched is True
    held = interface.step_navigation(elapsed_s=0.5)
    assert held.linear == 0.0 and held.angular == 0.0
    interface._publish_velocity(0.2, 0.2)
    assert _published(interface)[-1] == (0.0, 0.0)
    assert all(speed == (0.0, 0.0) for speed in _published(interface)[before:])

    interface._goal_pilot.manager.reset()
    interface.reset_collision_latch()
    assert interface._collision_latched is False
    assert interface.collision_monitor.collision_detected is False
    assert interface.collision_contact_details == []
    interface.set_navigation_goal(
        4.0, 0.0, scenario_id="world1_open", map_id="jackal_demo_open",
        split="fixed", seed=2, max_steps=10,
    )
    interface.publisher.publish.reset_mock()
    resumed = interface.step_navigation(elapsed_s=0.1)
    assert resumed.commanding is True
    assert resumed.linear > 0.0
    assert mock_ok.called


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=True)
def test_step_navigation_reads_the_monitor_when_collision_argument_is_false(mock_ok):
    interface = _pilot(_interface())
    interface.collision_monitor.collision_detected = True
    interface.collision_monitor.collision_pairs = [(ROBOT, WALL)]
    decision = interface.step_navigation(False, elapsed_s=0.2)
    assert decision.state is NavState.COLLISION
    assert decision.commanding is False
    assert _published(interface)[-1] == (0.0, 0.0)
    assert interface._goal_pilot.record.termination == "COLLISION"
    assert interface._goal_pilot.record.collision_count == 1
    assert interface.collision_contact_details == [(ROBOT, WALL)]
    assert mock_ok.called


def test_empty_is_benign_but_malformed_contacts_latch_without_crashing():
    monitor, seen, _node = _monitor()
    empty = Contacts()
    malformed = [
        None,
        SimpleNamespace(),
        SimpleNamespace(contacts=None),
        SimpleNamespace(contacts="not-a-sequence"),
        SimpleNamespace(contacts=[SimpleNamespace()]),
        SimpleNamespace(contacts=[SimpleNamespace(
            collision1=SimpleNamespace(name=ROBOT),
            collision2=SimpleNamespace(),
        )]),
    ]
    monitor._contact_callback(empty)
    for message in malformed:
        monitor._contact_callback(message)
    assert monitor.collision_detected is True
    assert monitor.collision_pairs == []
    assert seen.call_count > 0
    assert monitor.valid_messages == 3
    assert monitor.malformed_messages == 4
    assert monitor.malformed_contacts == 2

    mixed = SimpleNamespace(contacts=[
        SimpleNamespace(),
        SimpleNamespace(
            collision1=SimpleNamespace(name=ROBOT),
            collision2=SimpleNamespace(name=WALL),
        ),
    ])
    monitor._contact_callback(mixed)
    assert monitor.collision_detected is True
    assert monitor.collision_pairs == [(ROBOT, WALL)]
    assert monitor.malformed_contacts == 3
    assert seen.call_count == 8


def test_missing_monitor_initialization_prevents_goal_loop_startup():
    interface = _interface()
    interface.collision_monitor = None
    interface.collision_init_error = "ImportError: ros_gz_interfaces/msg/Contacts"
    with pytest.raises(RuntimeError, match="Goal loop refused"):
        interface.configure_goal_pilot("classical_baseline")
    assert interface._goal_pilot is None

    broken = _interface()
    with patch.object(
        broken,
        "_import_collision_dependencies",
        side_effect=ImportError("ros_gz_interfaces.msg.Contacts"),
    ):
        broken._initialize_collision_monitor()
    assert broken.collision_monitor is None
    assert "Contacts" in broken.collision_init_error
    with pytest.raises(RuntimeError, match="Goal loop refused"):
        broken.configure_goal_pilot("classical_baseline")
    assert broken._goal_pilot is None

    from jackal_rl_navigation.collision_config import CollisionConfig

    unsubscribed = _interface()

    class NoSubscription:
        def __init__(self, *args, **kwargs):
            self.subscription = None
            self.topic = "/jackal_rl/obstacle_contacts"

    with patch.object(
        unsubscribed,
        "_import_collision_dependencies",
        return_value=(Contacts, CollisionConfig, NoSubscription),
    ):
        unsubscribed._initialize_collision_monitor()
    assert unsubscribed.collision_monitor is None
    assert "subscription" in unsubscribed.collision_init_error
    with pytest.raises(RuntimeError, match="Goal loop refused"):
        unsubscribed.configure_goal_pilot("classical_baseline")
    assert unsubscribed._goal_pilot is None


def test_goal_loop_requires_the_monitor_before_it_can_step():
    source = INTERFACE_SOURCE.read_text(encoding="utf-8")
    init = source.split("def __init__", 1)[1].split("\n    def ", 1)[0]
    assert "_initialize_collision_monitor()" in init
    main = source.split("def main(", 1)[1]
    assert main.index("_require_collision_monitor()") < main.index("step_navigation()")
    configure = source.split("def configure_goal_pilot", 1)[1].split("\n    def ", 1)[0]
    assert configure.index("_require_collision_monitor()") < configure.index("GoalPilot.classical")


def test_preflight_does_not_treat_silence_or_a_launch_bridge_as_collision_free():
    interface = _interface()
    launch = LAUNCH_SOURCE.read_text(encoding="utf-8")
    assert "/jackal_rl/obstacle_contacts" in launch
    assert "ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts" in launch
    quiet = interface.collision_preflight()
    assert quiet["initialized"] is True
    assert quiet["topic"] == "/jackal_rl/obstacle_contacts"
    assert quiet["message_type"] == "ros_gz_interfaces/msg/Contacts"
    assert quiet["receiving_valid_contacts"] is False
    assert quiet["subscription_created"] is True
    assert quiet["collision_free_operation"] is not True
    assert quiet["bridge_declaration_used"] is False
    assert "not evidence of collision-free operation" in quiet["interpretation"]
    assert "does not mean the contact topic is active" in quiet["interpretation"]
    interface_text = INTERFACE_SOURCE.read_text(encoding="utf-8")
    preflight_node = next(
        node for node in ast.walk(ast.parse(interface_text))
        if isinstance(node, ast.FunctionDef) and node.name == "collision_preflight"
    )
    preflight_source = ast.get_source_segment(interface_text, preflight_node)
    assert "jackal_demo_sim.launch" not in preflight_source
    assert "read_text" not in preflight_source

    interface.collision_monitor._contact_callback(Contacts())
    empty = interface.collision_preflight()
    assert empty["receiving_valid_contacts"] is True
    assert empty["obstacle_contact_observed"] is False
    assert empty["collision_free_operation"] is not True
    assert "not proof of collision-free operation" in empty["interpretation"]

    interface.collision_monitor._contact_callback(_contacts((WHEEL, GROUNDED)))
    ground = interface.collision_preflight()
    assert ground["stream"] == "contacts_received_no_obstacle_collision"
    assert ground["collision_latched"] is False
    assert ground["collision_free_operation"] is not True

    interface.collision_monitor = None
    interface.collision_init_error = "ImportError: Contacts"
    missing = interface.collision_preflight()
    assert missing["initialized"] is False
    assert missing["receiving_valid_contacts"] is False
    assert missing["collision_free_operation"] is not True
    assert "goal loop must not start" in missing["interpretation"].lower()


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=True)
def test_watchdog_lidar_safety_and_emergency_stop_still_hold(mock_ok):
    fresh = _interface()
    fresh._watchdog_callback()
    assert fresh.publisher.publish.call_count == 0
    assert fresh._collision_latched is False

    fresh._sensor_ages = lambda: (1.0, 0.1)
    fresh._watchdog_callback()
    assert _published(fresh)[-1] == (0.0, 0.0)
    assert fresh._collision_latched is False
    assert fresh.collision_monitor.collision_detected is False

    lidar = _interface()
    angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    ranges = np.full(720, 5.0)
    side = np.abs(
        np.arctan2(np.sin(angles - math.pi / 2), np.cos(angles - math.pi / 2))
    ) < 0.2
    ranges[side] = 0.40
    lidar.scan = SimpleNamespace(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
    )
    lidar.collision_monitor._contact_callback(_contacts((WHEEL, GROUNDED)))
    assert lidar.command_action([1.0, 1.0]) is True
    message = lidar.publisher.publish.call_args.args[0]
    assert message.linear.x > 0.0
    assert message.angular.z == pytest.approx(0.0)
    assert "Rotation clearance" in lidar.last_safety_reason
    assert lidar._collision_latched is False

    stopped = _pilot(_interface())
    stopped.step_navigation(elapsed_s=0.1)
    stopped.emergency_stop()
    assert stopped._goal_pilot.state is NavState.EMERGENCY_STOP
    assert stopped._goal_pilot.record.termination == "EMERGENCY_STOP"
    assert stopped._goal_pilot.record.collision_count == 0
    assert stopped._collision_latched is False
    held = stopped.step_navigation(elapsed_s=0.4)
    assert held.linear == 0.0 and held.angular == 0.0
    assert mock_ok.called


def test_classical_pilot_rejects_checkpoint_controllers():
    interface = _interface()
    with pytest.raises(PolicyError):
        interface.configure_goal_pilot(
            "artifacts/ppo/jackal_6g1/run_20261022_230534/initialized_ppo.zip"
        )
    assert interface._goal_pilot is None
    with pytest.raises(PolicyError):
        interface.configure_goal_pilot("ppo")
    assert interface._goal_pilot is None
    interface.configure_goal_pilot("classical_baseline")
    assert interface._goal_pilot.manager.policy.algorithm_family == "classical"


def test_only_one_velocity_publisher_exists():
    source = INTERFACE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    publishers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_publisher"
    ]
    assert len(publishers) == 1
    monitor_source = MONITOR_SOURCE.read_text(encoding="utf-8")
    assert "create_publisher" not in monitor_source
    assert "cmd_vel" not in monitor_source

    interface = _interface()
    assert interface.collision_monitor.subscription == "contact-subscription"
    interface.configure_goal_pilot("classical_baseline")
    assert interface.publisher is interface._goal_pilot._publish_velocity.__self__.publisher
    with pytest.raises(RuntimeError, match="owns velocity commands"):
        interface.command_action([1.0, 0.0])


def test_collision_latch_reset_is_allowed_only_between_episodes():
    monitor, _seen, _node = _monitor()
    monitor._contact_callback(_contacts((ROBOT, WALL)))
    assert monitor.collision_detected is True
    monitor._contact_callback(_contacts((WHEEL, GROUNDED)))
    assert monitor.collision_detected is True
    monitor.reset()
    assert monitor.collision_detected is False
    assert monitor.collision_pairs == []
    assert monitor.messages_received == 0
    assert monitor.valid_messages == 0


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=True)
def test_malformed_contact_latches_interface_and_only_requests_mock_stop(mock_ok):
    """Offline: malformed data follows the existing safety callback/latch."""
    interface = _pilot(_interface())
    interface.collision_monitor._contact_callback(None)
    assert interface._collision_latched
    assert interface.collision_monitor.collision_detected
    assert all(pair == (0.0, 0.0) for pair in _published(interface))
    interface.collision_monitor._contact_callback(_contacts((WHEEL, GROUNDED)))
    assert interface._collision_latched
