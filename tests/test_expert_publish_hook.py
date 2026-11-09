
from unittest.mock import Mock, patch

from jackal_rl_navigation.robot_interface import JackalRLInterface


def make_interface():
    interface = object.__new__(JackalRLInterface)

    # Node.__init__ is intentionally bypassed for isolated tests.
    # Provide the minimal ROS context attribute required by Node.context.
    interface._context = Mock()
    interface.publisher = Mock()
    interface.last_publish_time = None
    interface.expert_publish_callback = Mock()

    interface.get_clock = Mock()
    interface.get_clock.return_value.now.return_value.nanoseconds = 123456

    interface.get_logger = Mock()

    return interface


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=True)
def test_successful_publish(mock_ok):
    interface = make_interface()

    assert interface._publish_velocity(0.1, -0.2) is True

    interface.publisher.publish.assert_called_once()
    interface.expert_publish_callback.assert_called_once_with(
        0.1, -0.2, 123456
    )


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=True)
def test_callback_failure_does_not_cancel_publish(mock_ok):
    interface = make_interface()

    interface.expert_publish_callback.side_effect = RuntimeError(
        "Recorder unavailable"
    )

    assert interface._publish_velocity(0.1, 0.0) is True

    interface.publisher.publish.assert_called_once()


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=True)
def test_publish_failure_is_reported(mock_ok):
    interface = make_interface()

    interface.publisher.publish.side_effect = RuntimeError(
        "Publisher unavailable"
    )

    assert interface._publish_velocity(0.1, 0.0) is False

    interface.expert_publish_callback.assert_not_called()


@patch("jackal_rl_navigation.robot_interface.rclpy.ok", return_value=False)
def test_inactive_ros_context_does_not_publish(mock_ok):
    interface = make_interface()

    assert interface._publish_velocity(0.1, 0.0) is not True

    interface.publisher.publish.assert_not_called()
