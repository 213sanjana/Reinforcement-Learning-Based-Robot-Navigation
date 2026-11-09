
import subprocess

from navigation import expert_collection_preflight as preflight


def completed(stdout="", returncode=0):
    return subprocess.CompletedProcess(
        ["ros2"], returncode, stdout, ""
    )


def topic_output(message_type, publishers, subscribers):
    return (
        f"Type: {message_type}\n"
        f"Publisher count: {publishers}\n"
        f"Subscription count: {subscribers}\n"
    )


def test_valid_odometry_topic(monkeypatch):
    monkeypatch.setattr(
        preflight,
        "run_command",
        lambda command, timeout=12: completed(
            topic_output("nav_msgs/msg/Odometry", 1, 0)
        ),
    )

    result = preflight.inspect_topic("/test/odom")

    assert result["publishers"] == 1
    assert result["type"] == "nav_msgs/msg/Odometry"


def test_missing_topic(monkeypatch):
    monkeypatch.setattr(
        preflight,
        "run_command",
        lambda command, timeout=12: completed(
            "", returncode=1
        ),
    )

    assert preflight.inspect_topic("/missing") is None


def test_malformed_topic_output(monkeypatch):
    monkeypatch.setattr(
        preflight,
        "run_command",
        lambda command, timeout=12: completed(
            "Unexpected ROS output"
        ),
    )

    assert preflight.inspect_topic("/test") is None


def test_colored_active_controllers(monkeypatch):
    output = (
        "\x1b[92mjoint_state_broadcaster\x1b[0m "
        "joint_state_broadcaster/JointStateBroadcaster "
        "\x1b[92mactive\x1b[0m\n"
        "\x1b[92mplatform_velocity_controller\x1b[0m "
        "diff_drive_controller/DiffDriveController "
        "\x1b[92mactive\x1b[0m\n"
    )

    monkeypatch.setattr(
        preflight,
        "run_command",
        lambda command, timeout=12: completed(output),
    )

    assert preflight.inspect_controllers() is True


def test_inactive_controller_rejected(monkeypatch):
    output = (
        "joint_state_broadcaster "
        "joint_state_broadcaster/JointStateBroadcaster active\n"
        "platform_velocity_controller "
        "diff_drive_controller/DiffDriveController inactive\n"
    )

    monkeypatch.setattr(
        preflight,
        "run_command",
        lambda command, timeout=12: completed(output),
    )

    assert preflight.inspect_controllers() is False


def test_missing_controller_rejected(monkeypatch):
    output = (
        "joint_state_broadcaster "
        "joint_state_broadcaster/JointStateBroadcaster active\n"
    )

    monkeypatch.setattr(
        preflight,
        "run_command",
        lambda command, timeout=12: completed(output),
    )

    assert preflight.inspect_controllers() is False


def test_controller_command_failure(monkeypatch):
    monkeypatch.setattr(
        preflight,
        "run_command",
        lambda command, timeout=12: completed(
            "", returncode=1
        ),
    )

    assert preflight.inspect_controllers() is False


def test_preflight_all_healthy(monkeypatch):
    def fake_topic(topic):
        for name, (expected_topic, message_type) in preflight.TOPICS.items():
            if topic == expected_topic:
                return {
                    "type": message_type,
                    "publishers": 0 if name == "velocity" else 1,
                    "subscribers": 1 if name == "velocity" else 0,
                }
        return None

    monkeypatch.setattr(preflight, "inspect_topic", fake_topic)
    monkeypatch.setattr(preflight, "inspect_controllers", lambda: True)

    result = preflight.run_preflight()

    assert result.passed is True
    assert all(result.checks.values())


def test_competing_velocity_publisher_rejected(monkeypatch):
    def fake_topic(topic):
        for name, (expected_topic, message_type) in preflight.TOPICS.items():
            if topic == expected_topic:
                return {
                    "type": message_type,
                    "publishers": 1,
                    "subscribers": 1,
                }
        return None

    monkeypatch.setattr(preflight, "inspect_topic", fake_topic)
    monkeypatch.setattr(preflight, "inspect_controllers", lambda: True)

    result = preflight.run_preflight()

    assert result.passed is False
    assert result.checks["velocity_no_competing_publisher"] is False


def test_missing_collision_publisher_rejected(monkeypatch):
    def fake_topic(topic):
        for name, (expected_topic, message_type) in preflight.TOPICS.items():
            if topic == expected_topic:
                return {
                    "type": message_type,
                    "publishers": 0 if name in ("collision", "velocity") else 1,
                    "subscribers": 1,
                }
        return None

    monkeypatch.setattr(preflight, "inspect_topic", fake_topic)
    monkeypatch.setattr(preflight, "inspect_controllers", lambda: True)

    result = preflight.run_preflight()

    assert result.passed is False
    assert result.checks["collision_publisher"] is False


def test_inactive_controllers_fail_preflight(monkeypatch):
    def fake_topic(topic):
        for name, (expected_topic, message_type) in preflight.TOPICS.items():
            if topic == expected_topic:
                return {
                    "type": message_type,
                    "publishers": 0 if name == "velocity" else 1,
                    "subscribers": 1,
                }
        return None

    monkeypatch.setattr(preflight, "inspect_topic", fake_topic)
    monkeypatch.setattr(preflight, "inspect_controllers", lambda: False)

    result = preflight.run_preflight()

    assert result.passed is False
    assert result.checks["controllers_active"] is False
