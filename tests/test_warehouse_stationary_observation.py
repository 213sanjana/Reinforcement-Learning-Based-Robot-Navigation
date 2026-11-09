"""Offline tests for the stationary observation path. No Gazebo and no training."""

import ast
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.warehouse_evaluate import evaluate
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_env import JackalWarehouseRLEnv
from jackal_rl_navigation.warehouse_observation import (
    COMMAND_TOPIC,
    FRESHNESS_S,
    RosObservationTransport,
    StationaryObservationSession,
    controllers_dict,
    require_active_controllers,
)
from training.warehouse_ppo_train import train


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "jackal_rl_ws" / "src" / "jackal_rl_navigation" / "jackal_rl_navigation"
GOAL = {"start": [0.0, 0.0, 0.0], "goal": [1.5, 0.0]}
WHEEL = "j100_0000::front_left_wheel_link::front_left_wheel_link_collision"
FLOOR = (
    "aws_robomaker_warehouse_GroundB_01_001::"
    "aws_robomaker_warehouse_GroundB_01::ground_link::collision"
)
CHASSIS = "j100_0000::base_link::base_link_fixed_joint_lump__chassis_link_collision"


def _config():
    config = load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml")
    config["execution"] = "stationary_observation"
    return config


def _sample(**overrides):
    now = time.monotonic()
    values = {
        "ranges": np.full(900, 5.0, dtype=np.float32),
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "linear_x": 0.0,
        "linear_y": 0.0,
        "angular_z": 0.0,
        "scan_stamp_ns": 2_000_000_000,
        "previous_scan_stamp_ns": 1_000_000_000,
        "odom_stamp_ns": 2_000_000_000,
        "previous_odom_stamp_ns": 1_000_000_000,
        "scan_received": now,
        "odom_received": now,
        "scan_advanced_at": now,
        "odom_advanced_at": now,
    }
    values.update(overrides)
    from jackal_rl_navigation.warehouse_observation import SensorSample

    return SensorSample(**values)


def _active_controllers():
    return {
        "controller": [
            {"name": "joint_state_broadcaster", "state": "active"},
            {"name": "platform_velocity_controller", "state": "active"},
        ]
    }


class RecordingProbe:
    """Stand-in transport that records every call, including ones that must not happen."""

    def __init__(self, sample=None, controllers=None, pairs=None):
        self.sample = _sample() if sample is None else sample
        self.controllers = _active_controllers() if controllers is None else controllers
        self.pairs = [] if pairs is None else pairs
        self.emergency = False
        self.collision_latched = False
        self.error = None
        self.calls = []
        self.publications = []
        self.gazebo_calls = []

    def list_controllers(self):
        self.calls.append("list_controllers")
        if self.error == "controllers":
            return None
        return self.controllers

    def wait_for_advancing_sensors(self, timeout_s, freshness_s):
        self.calls.append(("wait", timeout_s, freshness_s))
        if self.error == "timeout":
            raise TimeoutError("sensors timed out")
        return self.sample

    def contact_pairs(self):
        self.calls.append("contacts")
        return list(self.pairs)

    def emergency_active(self):
        return self.emergency

    def publish(self, *args):
        self.publications.append(args)

    def set_pose(self, *args, **kwargs):
        self.gazebo_calls.append("set_pose")

    def pause(self, *args, **kwargs):
        self.gazebo_calls.append("pause")

    def step(self, *args, **kwargs):
        self.gazebo_calls.append("step")

    def stop(self, *args, **kwargs):
        self.publications.append((COMMAND_TOPIC, 0.0, 0.0))
        self.gazebo_calls.append("stop")


def _session(probe, **overrides):
    config = _config()
    config.update(overrides)
    return StationaryObservationSession(config, transport=probe), config


def _assert_silent(probe):
    assert probe.publications == []
    assert probe.gazebo_calls == []


def _reset(session, probe):
    observation, info = session.reset(seed=0, scenario={"goal": (1.5, 0.0)})
    _assert_silent(probe)
    return observation, info


def test_configured_external_target_is_explicit():
    config = _config()
    assert config["world_name"] == "industrial-warehouse"
    assert config["gazebo_partition"] == "jackal_rl_external"
    assert config["namespace"] == "j100_0000"
    assert config["lidar_topic"] == "/j100_0000/sensors/lidar3d_0/scan"
    assert config["odom_topic"] == "/j100_0000/platform/odom"


def test_partition_is_not_defaulted(tmp_path):
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        "\n".join([
            "namespace: j100_0000",
            "active_stage: open",
            "stages:",
            "  open:",
            "    world_name: jackal_demo_open",
            "    scenarios:",
            "      - {id: a, split: train, start: [0.0, 0.0, 0.0], goal: [1.0, 0.0]}",
            "      - {id: b, split: validation, start: [0.0, 0.2, 0.0], goal: [1.0, 0.4]}",
            "      - {id: c, split: test, start: [-0.2, 0.0, 0.0], goal: [1.2, -0.2]}",
            "",
        ]),
        encoding="utf-8",
    )
    ppo = tmp_path / "ppo.yaml"
    ppo.write_text(
        "\n".join([
            "execution: stationary_observation",
            "stage_id: open",
            f"environment_config: {environment}",
            "reward: {}",
            "",
        ]),
        encoding="utf-8",
    )
    config = load_warehouse_config(ppo)
    assert config["world_name"] == "jackal_demo_open"
    assert "gazebo_partition" not in config
    assert config.get("gazebo_partition") != "jackal_rl"
    with pytest.raises(RuntimeError, match="industrial-warehouse"):
        StationaryObservationSession(config, transport=RecordingProbe())


@pytest.mark.parametrize("world,partition", [
    ("jackal_obstacle_test", "jackal_rl_external"),
    ("industrial-warehouse", "jackal_rl"),
    ("industrial-warehouse", None),
    (None, None),
])
def test_wrong_world_or_partition_fails_before_any_call(world, partition):
    probe = RecordingProbe()
    config = _config()
    config["world_name"] = world
    if partition is None:
        config.pop("gazebo_partition", None)
    else:
        config["gazebo_partition"] = partition
    with pytest.raises(RuntimeError, match="industrial-warehouse"):
        StationaryObservationSession(config, transport=probe)
    assert probe.calls == []
    _assert_silent(probe)


def test_reset_returns_finite_observation_without_commands():
    probe = RecordingProbe(pairs=[(WHEEL, FLOOR)])
    session, config = _session(probe)
    observation, info = _reset(session, probe)
    assert observation.shape == (40,)
    assert observation.dtype == np.float32
    assert bool(np.all(np.isfinite(observation)))
    assert observation[0] == pytest.approx(5.0 / 12.0)
    assert observation[36] == pytest.approx(1.5 / 20.0)
    assert observation[37] == pytest.approx(0.0)
    assert observation[38] == pytest.approx(0.0)
    assert observation[39] == pytest.approx(0.0)
    assert info["goal_distance"] == pytest.approx(1.5)
    assert info["pose_xy"] == (0.0, 0.0)
    wait = next(call for call in probe.calls if isinstance(call, tuple))
    assert wait[1] == config["limits"].sensor_timeout_s
    assert wait[2] == FRESHNESS_S == 0.5
    assert not hasattr(probe, "controller_ready")
    probe.sample = _sample(yaw=math.pi / 2, linear_x=0.01)
    turned, _info = _reset(session, probe)
    assert turned[36] == pytest.approx(1.5 / 20.0)
    assert turned[37] == pytest.approx(-0.5)
    assert turned[38] == pytest.approx(0.01 / 0.30)
    assert session.collision_latched is False
    assert session.emergency_latched is False


def test_missing_controller_ready_attribute_is_not_readiness():
    class Robot:
        pass

    robot = Robot()
    assert not hasattr(robot, "controller_ready")
    probe = RecordingProbe(controllers=robot)
    session, _config = _session(probe)
    with pytest.raises(RuntimeError, match="missing"):
        session.reset(seed=0, scenario={"goal": (1.5, 0.0)})
    _assert_silent(probe)

    probe.controller_ready = True
    probe.controllers = {
        "controller_ready": True,
        "controller": [
            {"name": "joint_state_broadcaster", "state": "active"},
            {"name": "platform_velocity_controller", "state": "inactive"},
        ],
    }
    with pytest.raises(RuntimeError, match="not active"):
        session.reset(seed=0, scenario={"goal": (1.5, 0.0)})
    _assert_silent(probe)


@pytest.mark.parametrize("overrides,message", [
    ({"scan_received": None}, "stale"),
    ({"previous_odom_stamp_ns": 2_000_000_000}, "not advancing"),
    ({"linear_x": 0.2}, "not stationary"),
    ({"angular_z": 0.2}, "not stationary"),
    ({"ranges": np.full(36, np.nan, dtype=np.float32)}, "valid observation"),
])
def test_bad_sensors_and_motion_fail_closed(overrides, message):
    now = time.monotonic()
    if "scan_received" in overrides:
        overrides["scan_received"] = now - 1.0
        overrides["scan_advanced_at"] = now - 1.0
    probe = RecordingProbe(sample=_sample(**overrides))
    session, _config = _session(probe)
    with pytest.raises(RuntimeError, match=message):
        session.reset(seed=0, scenario={"goal": (1.5, 0.0)})
    _assert_silent(probe)


def test_sensor_timeout_fails_closed():
    probe = RecordingProbe()
    probe.error = "timeout"
    session, _config = _session(probe)
    with pytest.raises(RuntimeError, match="timed out"):
        session.reset(seed=0, scenario={"goal": (1.5, 0.0)})
    _assert_silent(probe)


def test_safety_fault_latches_and_a_later_benign_contact_cannot_clear_it():
    probe = RecordingProbe(pairs=[(CHASSIS, "shelf::link::collision")])
    session, _config = _session(probe)
    with pytest.raises(RuntimeError, match="Collision latch"):
        session.reset(seed=0, scenario={"goal": (1.5, 0.0)})
    assert session.collision_latched is True
    probe.pairs = [(WHEEL, FLOOR)]
    probe.collision_latched = False
    with pytest.raises(RuntimeError, match="Collision latch"):
        session.reset(seed=1, scenario={"goal": (1.5, 0.0)})
    assert session.collision_latched is True
    _assert_silent(probe)
    assert not hasattr(session, "reset_collision_latch")
    assert not hasattr(session, "clear_emergency")


def test_emergency_latch_is_not_cleared_by_reset():
    probe = RecordingProbe()
    probe.emergency = True
    session, _config = _session(probe)
    with pytest.raises(RuntimeError, match="Emergency stop"):
        session.reset(seed=0, scenario={"goal": (1.5, 0.0)})
    assert session.emergency_latched is True
    probe.emergency = False
    with pytest.raises(RuntimeError, match="Emergency stop"):
        session.reset(seed=1, scenario={"goal": (1.5, 0.0)})
    assert session.emergency_latched is True
    _assert_silent(probe)


def test_environment_reset_is_silent_and_step_stays_blocked():
    probe = RecordingProbe()
    config = _config()
    env = JackalWarehouseRLEnv(config, execution="stationary_observation", transport=probe)
    assert env.execution == "stationary_observation"
    assert env.plant is None
    observation, info = env.reset(seed=1, options=GOAL)
    assert observation.shape == (40,) and observation.dtype == np.float32
    assert bool(np.all(np.isfinite(observation)))
    assert info["execution"] == "stationary_observation"
    assert info["world_name"] == "industrial-warehouse"
    with pytest.raises(RuntimeError, match="blocked"):
        env.step(np.zeros(2, dtype=np.float32))
    with pytest.raises(RuntimeError, match="blocked"):
        env.session.step(np.zeros(2, dtype=np.float32))
    env.close()
    _assert_silent(probe)
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
    _assert_silent(probe)


def test_passed_observation_session_is_not_forced_onto_the_gazebo_path():
    probe = RecordingProbe()
    session, config = _session(probe)
    env = JackalWarehouseRLEnv(config, session=session)
    assert env.execution == "stationary_observation"
    assert env.session is session
    with pytest.raises(RuntimeError, match="authorize_live"):
        JackalWarehouseRLEnv(config, session=session, authorize_live=True)
    with pytest.raises(RuntimeError, match="authorize_live"):
        JackalWarehouseRLEnv(
            config,
            execution="stationary_observation",
            authorize_live=True,
            transport=probe,
        )
    _assert_silent(probe)


def test_training_and_inference_stay_blocked(tmp_path):
    config = _config()
    output = tmp_path / "run"
    with pytest.raises(SystemExit, match="stationary observation"):
        train(config=config, output_dir=output, authorize_live=True)
    assert not output.exists()
    with pytest.raises(SystemExit, match="stationary observation"):
        evaluate(
            config,
            tmp_path / "checkpoint",
            episodes=1,
            seed=0,
            authorize_live=True,
            policy=lambda _observation, _info: np.zeros(2, dtype=np.float32),
        )
    assert not (tmp_path / "checkpoint").exists()


def test_subscriber_transport_has_no_command_path(monkeypatch):
    monkeypatch.setenv("IGN_PARTITION", "jackal_rl_external")
    monkeypatch.setenv("GZ_PARTITION", "jackal_rl_external")
    client = _Client(_controller_response("active"))
    node = _FakeNode(client)
    transport = RosObservationTransport(
        _config(),
        node=node,
        message_types=_message_types(),
    )
    transport._on_scan(_scan(1))
    transport._on_scan(_scan(2))
    transport._on_odom(_odom(1))
    transport._on_odom(_odom(2))
    spins = []
    sample = transport.wait_for_advancing_sensors(5.0, FRESHNESS_S, spin=lambda _timeout: spins.append("spin"))
    listed = transport.list_controllers(spin=lambda _timeout: spins.append("spin"))
    require_active_controllers(listed)
    session = StationaryObservationSession(_config(), transport=transport)
    observation, info = session.reset(seed=0, scenario={"goal": (1.5, 0.0)})
    assert sample.x == 0.0
    assert observation.shape == (40,) and bool(np.all(np.isfinite(observation)))
    assert info["gazebo_partition"] == "jackal_rl_external"
    assert spins == []
    names = [call if isinstance(call, str) else call[0] for call in node.calls]
    assert names.count("create_subscription") == 3
    subscriptions = [call[1] for call in node.calls if call[0] == "create_subscription"]
    assert COMMAND_TOPIC not in subscriptions
    assert ("create_client", "/j100_0000/controller_manager/list_controllers") in node.calls
    assert ("get_publishers_info_by_topic", COMMAND_TOPIC) in node.calls
    for forbidden in ("create_publisher", "create_timer", "set_pose", "pause", "step", "stop"):
        assert forbidden not in names
    transport._on_contacts(SimpleNamespace(contacts=[_contact(CHASSIS, "shelf::link::collision")]))
    assert transport.collision_latched is True
    transport._on_contacts(SimpleNamespace(contacts=[_contact(WHEEL, FLOOR)]))
    assert transport.collision_latched is True


def test_wrong_process_partition_does_not_create_a_node(monkeypatch):
    monkeypatch.setenv("IGN_PARTITION", "jackal_rl")
    monkeypatch.setenv("GZ_PARTITION", "jackal_rl")

    def factory():
        raise AssertionError("node created")

    with pytest.raises(RuntimeError, match="jackal_rl_external"):
        StationaryObservationSession(
            _config(),
            connect=True,
            node_factory=factory,
            message_types=_message_types(),
        )


def test_inactive_service_response_fails_closed():
    payload = controllers_dict(_controller_response("inactive"))
    with pytest.raises(RuntimeError, match="not active"):
        require_active_controllers(payload)
    with pytest.raises(RuntimeError, match="missing"):
        controllers_dict(None)
    with pytest.raises(RuntimeError, match="missing"):
        require_active_controllers(SimpleNamespace(controller_ready=True))


def test_existing_command_publisher_blocks_reset():
    probe = RecordingProbe()
    probe.command_publishers = lambda: ["jackal_rl_interface"]
    session, _config = _session(probe)
    with pytest.raises(RuntimeError, match="command publisher"):
        session.reset(seed=0, scenario={"goal": (1.5, 0.0)})
    assert "list_controllers" not in probe.calls
    _assert_silent(probe)


def test_live_connection_requests_sensor_qos(monkeypatch):
    monkeypatch.setenv("IGN_PARTITION", "jackal_rl_external")
    monkeypatch.setenv("GZ_PARTITION", "jackal_rl_external")
    sentinel = object()
    node = _FakeNode(_Client(_controller_response("active")))

    def qos():
        return {"scan": sentinel, "odom": sentinel, "contacts": 10}

    monkeypatch.setattr(
        "jackal_rl_navigation.warehouse_observation.live_subscription_qos",
        qos,
    )
    StationaryObservationSession(
        _config(),
        connect=True,
        node_factory=lambda: node,
        message_types=_message_types(),
    )
    subscriptions = [call for call in node.calls if call[0] == "create_subscription"]
    assert [call[2] for call in subscriptions] == [sentinel, sentinel, 10]
    assert "qos_profile_sensor_data" in (
        PACKAGE / "warehouse_observation.py"
    ).read_text(encoding="utf-8")
    assert "create_publisher" not in [call if isinstance(call, str) else call[0] for call in node.calls]


def test_unconnected_reset_does_not_create_a_node():
    session = StationaryObservationSession(_config())
    assert session.transport is None
    with pytest.raises(RuntimeError, match="command interface"):
        session.reset(seed=0, scenario={"goal": (1.5, 0.0)})


def test_observation_module_has_no_command_or_gazebo_control_calls():
    source = (PACKAGE / "warehouse_observation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {
        "create_publisher",
        "create_timer",
        "set_pose",
        "pause",
        "stop",
        "reset_emergency",
        "reset_collision_latch",
        "gazebo_process_env",
    }
    forbidden_modules = {
        "jackal_rl_navigation.gazebo_controller",
        "jackal_rl_navigation.gazebo_env",
        "jackal_rl_navigation.robot_interface",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            assert name not in forbidden_calls
    assert "os.environ[" not in source
    assert "create_publisher" not in source
    assert "controller_ready" not in source


def test_active_reset_remains_a_separate_control_path():
    controller = (PACKAGE / "gazebo_controller.py").read_text(encoding="utf-8")
    assert 'JACKAL_GAZEBO_PARTITION = "jackal_rl"' in controller
    assert 'world="jackal_obstacle_test"' in controller
    names = _reset_attributes(PACKAGE / "gazebo_env.py", "JackalGazeboEnv")
    for required in ("pause", "stop", "set_pose", "step", "reset_emergency"):
        assert required in names


def _reset_attributes(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "reset":
                    return {
                        child.attr
                        for child in ast.walk(item)
                        if isinstance(child, ast.Attribute)
                    }
    raise AssertionError(f"{class_name}.reset was not found")


class _FakeNode:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def create_publisher(self, *args, **kwargs):
        self.calls.append("create_publisher")

    def create_timer(self, *args, **kwargs):
        self.calls.append("create_timer")

    def create_subscription(self, message_type, topic, callback, qos):
        self.calls.append(("create_subscription", topic, qos))
        return object()

    def get_publishers_info_by_topic(self, topic):
        self.calls.append(("get_publishers_info_by_topic", topic))
        return []

    def create_client(self, message_type, service):
        self.calls.append(("create_client", service))
        return self.client

    def set_pose(self, *args, **kwargs):
        self.calls.append("set_pose")

    def pause(self, *args, **kwargs):
        self.calls.append("pause")

    def step(self, *args, **kwargs):
        self.calls.append("step")

    def stop(self, *args, **kwargs):
        self.calls.append("stop")


class _Client:
    def __init__(self, response):
        self.response = response

    def wait_for_service(self, timeout_sec):
        return True

    def call_async(self, request):
        return _Future(self.response)


class _Future:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


def _message_types():
    class Service:
        class Request:
            pass

    return {"scan": object, "odom": object, "contacts": object, "controllers": Service}


def _controller_response(state):
    return SimpleNamespace(controller=[
        SimpleNamespace(name="joint_state_broadcaster", state="active"),
        SimpleNamespace(name="platform_velocity_controller", state=state),
    ])


def _stamp(sec):
    return SimpleNamespace(sec=sec, nanosec=0)


def _scan(sec):
    return SimpleNamespace(header=SimpleNamespace(stamp=_stamp(sec), frame_id="lidar"), ranges=[5.0] * 36)


def _odom(sec):
    orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    pose = SimpleNamespace(position=position, orientation=orientation)
    twist = SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0, z=0.0), angular=SimpleNamespace(z=0.0))
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(sec), frame_id="odom"),
        pose=SimpleNamespace(pose=pose),
        twist=SimpleNamespace(twist=twist),
    )


def _contact(first, second):
    return SimpleNamespace(
        collision1=SimpleNamespace(name=first),
        collision2=SimpleNamespace(name=second),
    )
