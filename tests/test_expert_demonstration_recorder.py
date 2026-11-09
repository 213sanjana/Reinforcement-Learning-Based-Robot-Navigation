"""Integrity tests for the Phase 6E.1 expert demonstration recorder."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigation.expert_demonstration_recorder import (
    ExpertDemonstrationRecorder,
    _OPEN_RECORDERS,
    _write_npz,
    load_dataset,
    load_episode,
)
from navigation.expert_demonstration_schema import (
    SCHEMA_VERSION,
    DemonstrationEpisode,
    episode_from_expert_result,
    inspect_policy_spaces,
    jackal_policy_spaces,
)


GAZEBO_ENV = (
    ROOT
    / "jackal_rl_ws/src/jackal_rl_navigation/jackal_rl_navigation/gazebo_env.py"
)
ROBOT_INTERFACE = (
    ROOT
    / "jackal_rl_ws/src/jackal_rl_navigation/jackal_rl_navigation/robot_interface.py"
)

SMOKE_SPECS = (
    (0, "low_speed_single", "GOAL_REACHED", 5, (2.5, -1.5)),
    (1, "reachable_north", "GOAL_REACHED", 5, (2.5, 1.5)),
    (2, "reachable_around", "GOAL_REACHED", 8, (3.0, 1.5)),
    (3, "low_speed_single", "GOAL_REACHED", 4, (2.5, -1.5)),
    (4, "reachable_north", "GOAL_REACHED", 4, (2.5, 1.5)),
    (5, "reachable_around", "GOAL_REACHED", 6, (3.0, 1.5)),
    (6, "unreachable_goal", "PLANNING_FAILED", 0, (1.5, 0.0)),
    (7, "invalid_start", "PLANNING_FAILED", 0, (2.0, 0.0)),
    (8, "collision_probe", "SAFETY_STOP", 3, (1.5, 0.0)),
    (9, "timeout_short", "TIME_LIMIT", 2, (2.5, -1.5)),
)


def make_episode(
    episode_id=0,
    scenario_id="low_speed_single",
    termination_reason="GOAL_REACHED",
    steps=4,
    goal=(2.5, -1.5),
    action=0.4,
    linear=0.12,
):
    goal = np.asarray(goal, dtype=np.float64)
    if steps == 0:
        return DemonstrationEpisode(
            episode_id=episode_id,
            scenario_id=scenario_id,
            termination_reason=termination_reason,
            goal=goal,
            observations=np.zeros((0, 40), dtype=np.float32),
            actions=np.zeros((0, 2), dtype=np.float32),
            requested_actions=np.zeros((0, 2), dtype=np.float32),
            applied_velocities=np.zeros((0, 2), dtype=np.float32),
            poses=np.zeros((0, 3), dtype=np.float64),
            simulation_timestamps_ns=np.zeros((0,), dtype=np.int64),
        )

    observations = np.zeros((steps, 40), dtype=np.float32)
    observations[:, :36] = 0.5
    observations[:, 36] = 0.2
    observations[:, 37] = 0.1
    actions = np.zeros((steps, 2), dtype=np.float32)
    actions[:, 0] = action
    actions[:, 1] = episode_id * 0.01
    requested = np.array(actions, copy=True)
    requested[:, 0] = min(1.0, action + 0.05)
    velocities = np.zeros((steps, 2), dtype=np.float32)
    velocities[:, 0] = linear
    velocities[:, 1] = episode_id * 0.005
    poses = np.zeros((steps, 3), dtype=np.float64)
    poses[:, 0] = np.linspace(0.0, 0.05 * steps, steps)
    timestamps = np.arange(steps, dtype=np.int64) * 100_000_000 + 1_000
    return DemonstrationEpisode(
        episode_id=episode_id,
        scenario_id=scenario_id,
        termination_reason=termination_reason,
        goal=goal,
        observations=observations,
        actions=actions,
        requested_actions=requested,
        applied_velocities=velocities,
        poses=poses,
        simulation_timestamps_ns=timestamps,
    )


def smoke_episodes():
    return [
        make_episode(episode_id, scenario_id, reason, steps, goal)
        for episode_id, scenario_id, reason, steps, goal in SMOKE_SPECS
    ]


def _record_prefix(output, storage):
    observation_space, action_space = jackal_policy_spaces()
    recorder = ExpertDemonstrationRecorder(
        output,
        observation_space,
        action_space,
        storage=storage,
    )
    for episode in smoke_episodes()[:6]:
        recorder.record(episode)
    (Path(output) / "incomplete.partial").write_bytes(b"partial")


class PolicyEnv:
    def __init__(self):
        self.observation_space, self.action_space = jackal_policy_spaces()


def test_inspected_spaces_match_the_gazebo_environment():
    source = GAZEBO_ENV.read_text()
    robot_source = ROBOT_INTERFACE.read_text()
    observation_space, action_space = inspect_policy_spaces(PolicyEnv())

    assert observation_space.shape == (40,)
    assert observation_space.dtype == np.float32
    assert action_space.shape == (2,)
    assert np.all(action_space.low == -1.0)
    assert np.all(action_space.high == 1.0)
    assert "shape=(40,)" in source
    assert "low=-np.inf" in source
    assert "high=np.inf" in source
    assert "shape=(2,)" in source
    assert "low=-1.0" in source
    assert "high=1.0" in source
    assert 'declare_parameter("namespace", "j100_0000")' in robot_source
    assert 'declare_parameter("lidar_bins", 36)' in robot_source

    wrong = spaces.Box(low=0.0, high=1.0, shape=(40,), dtype=np.float32)
    with pytest.raises(ValueError):
        inspect_policy_spaces(SimpleNamespace(
            observation_space=wrong,
            action_space=action_space,
        ))


def test_successful_and_failed_episodes_round_trip(tmp_path):
    observation_space, action_space = jackal_policy_spaces()
    with ExpertDemonstrationRecorder(
        tmp_path / "dataset",
        observation_space,
        action_space,
    ) as recorder:
        success_path = recorder.record(make_episode())
        failure = make_episode(
            episode_id=1,
            scenario_id="collision_probe",
            termination_reason="SAFETY_STOP",
            steps=2,
            goal=(1.5, 0.0),
        )
        failure_path = recorder.record(failure)
        empty = make_episode(
            episode_id=2,
            scenario_id="unreachable_goal",
            termination_reason="PLANNING_FAILED",
            steps=0,
            goal=(1.5, 0.0),
        )
        recorder.record(empty)

    success = load_episode(success_path)
    failure_loaded = load_episode(failure_path)
    loaded = load_dataset(tmp_path / "dataset")

    assert success["success"] is True
    assert success["observations"].shape == (4, 40)
    assert success["actions"].shape == (4, 2)
    assert success["applied_velocities"].shape == (4, 2)
    assert success["poses"].shape == (4, 3)
    assert success["simulation_timestamps_ns"].shape == (4,)
    assert success["scenario_id"] == "low_speed_single"
    assert success["termination_reason"] == "GOAL_REACHED"
    np.testing.assert_allclose(success["goal"], [2.5, -1.5])
    assert not np.allclose(success["actions"], success["requested_actions"])
    assert failure_loaded["success"] is False
    assert failure_loaded["termination_reason"] == "SAFETY_STOP"
    assert loaded[2]["observations"].shape == (0, 40)
    assert loaded[2]["termination_reason"] == "PLANNING_FAILED"

    schema = json.loads((tmp_path / "dataset" / "schema.json").read_text())
    assert schema["schema_version"] == SCHEMA_VERSION
    assert schema["training_action_key"] == "actions"
    assert schema["applied_velocity_key"] == "applied_velocities"
    assert schema["observation_normalization"]["lidar_bins"] == 36
    assert schema["robot_configuration"]["namespace"] == "j100_0000"
    assert schema["robot_configuration"]["action_mapper"]["max_forward_m_s"] == 0.3
    assert schema["retains_failed_episodes"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        ("nan", "nonfinite"),
        ("action", "action space"),
        ("velocity", "forward limit"),
        ("time", "monotonic"),
        ("rows", "aligned"),
        ("empty_success", "at least one"),
    ),
)
def test_invalid_episode_is_rejected_without_a_file(tmp_path, mutate, message):
    episode = make_episode()
    if mutate == "nan":
        episode.observations[0, 0] = np.nan
    elif mutate == "action":
        episode.actions[0, 0] = 1.5
    elif mutate == "velocity":
        episode.applied_velocities[0, 0] = 1.0
    elif mutate == "time":
        episode.simulation_timestamps_ns[1] = 1
    elif mutate == "rows":
        episode.actions = episode.actions[:1]
    elif mutate == "empty_success":
        episode = make_episode(steps=0)

    observation_space, action_space = jackal_policy_spaces()
    with ExpertDemonstrationRecorder(
        tmp_path / "dataset",
        observation_space,
        action_space,
    ) as recorder:
        with pytest.raises(ValueError, match=message):
            recorder.record(episode)

    assert list((tmp_path / "dataset").glob("episode_*")) == []


def test_existing_episode_is_not_overwritten(tmp_path):
    observation_space, action_space = jackal_policy_spaces()
    output = tmp_path / "dataset"
    with ExpertDemonstrationRecorder(
        output,
        observation_space,
        action_space,
    ) as recorder:
        path = recorder.record(make_episode())
        original = path.read_bytes()
        with pytest.raises(FileExistsError):
            recorder.record(make_episode(action=0.2))

    assert path.read_bytes() == original


def test_interrupted_recording_keeps_completed_episodes(tmp_path):
    observation_space, action_space = jackal_policy_spaces()
    output = tmp_path / "dataset"
    recorder = ExpertDemonstrationRecorder(
        output,
        observation_space,
        action_space,
    )
    recorder.record(make_episode())
    recorder.record(make_episode(
        episode_id=1,
        scenario_id="unreachable_goal",
        termination_reason="PLANNING_FAILED",
        steps=0,
        goal=(1.5, 0.0),
    ))
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["episodes"] = manifest["episodes"][:1]
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "orphan.partial").write_bytes(b"partial")
    (output / "recorder.lock").write_text(json.dumps({"pid": 99999999}))
    recorder._closed = True
    _OPEN_RECORDERS.discard(id(recorder))

    resumed = ExpertDemonstrationRecorder(
        output,
        observation_space,
        action_space,
    )
    try:
        assert resumed.recovered_interrupt
        assert not list(output.glob("*.partial"))
        loaded = load_dataset(output)
        assert [item["episode_id"] for item in loaded] == [0, 1]
        assert loaded[1]["success"] is False
        resumed.record(make_episode(episode_id=2, scenario_id="reachable_north"))
    finally:
        resumed.close()


def test_schema_mismatch_is_refused(tmp_path):
    observation_space, action_space = jackal_policy_spaces()
    output = tmp_path / "dataset"
    with ExpertDemonstrationRecorder(
        output,
        observation_space,
        action_space,
    ):
        pass

    schema_path = output / "schema.json"
    schema = json.loads(schema_path.read_text())
    schema["schema_version"] = "6e.0.0"
    schema_path.write_text(json.dumps(schema))

    with pytest.raises(RuntimeError, match="schema"):
        ExpertDemonstrationRecorder(
            output,
            observation_space,
            action_space,
        )


def test_tampered_episode_fails_integrity_check(tmp_path):
    observation_space, action_space = jackal_policy_spaces()
    with ExpertDemonstrationRecorder(
        tmp_path / "dataset",
        observation_space,
        action_space,
    ) as recorder:
        path = recorder.record(make_episode())

    payload = load_episode(path)
    payload["observations"] = np.array(payload["observations"], copy=True)
    payload["observations"][0, 0] = 0.25
    tampered = tmp_path / "tampered.npz"
    _write_npz(tampered, payload)

    with pytest.raises(ValueError, match="Integrity"):
        load_episode(tampered)


def test_expert_result_conversion_preserves_alignment():
    transition = SimpleNamespace(
        observation=np.full(40, 0.25, dtype=np.float32),
        action=np.array([0.5, -0.2], dtype=np.float32),
        applied_action=np.array([0.4, -0.1], dtype=np.float32),
        published_linear=0.12,
        published_angular=-0.06,
        simulation_time_ns=2500,
        pose=(0.2, -1.5, 0.1),
    )
    result = SimpleNamespace(
        status="GOAL_REACHED",
        transitions=[transition],
        accepted_commands=1,
        rejected_commands=0,
    )
    episode = episode_from_expert_result(
        result,
        episode_id=3,
        scenario_id="reachable_around",
        goal=(3.0, 1.5),
    )

    assert episode.observations.shape == (1, 40)
    np.testing.assert_allclose(episode.actions[0], [0.4, -0.1])
    np.testing.assert_allclose(episode.requested_actions[0], [0.5, -0.2])
    np.testing.assert_allclose(episode.applied_velocities[0], [0.12, -0.06])
    assert int(episode.simulation_timestamps_ns[0]) == 2500

    rejected = SimpleNamespace(
        status="GOAL_REACHED",
        transitions=[transition],
        accepted_commands=1,
        rejected_commands=1,
    )
    failed = episode_from_expert_result(
        rejected,
        episode_id=4,
        scenario_id="reachable_around",
        goal=(3.0, 1.5),
    )
    assert failed.termination_reason == "REJECTED_COMMANDS"

    planning_failure = episode_from_expert_result(
        SimpleNamespace(
            status="PLANNING_FAILED",
            transitions=[],
            accepted_commands=0,
            rejected_commands=0,
        ),
        episode_id=5,
        scenario_id="invalid_start",
        goal=(2.0, 0.0),
    )
    assert planning_failure.observations.shape == (0, 40)
    assert planning_failure.termination_reason == "PLANNING_FAILED"


@pytest.mark.parametrize("storage", ("npz", "hdf5"))
def test_ten_episode_recording_smoke(tmp_path, storage):
    output = tmp_path / "smoke"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(output), storage],
        cwd=str(ROOT),
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2, completed.stderr
    assert json.loads((output / "session.json").read_text())["state"] == "open"
    assert (output / "incomplete.partial").is_file()
    assert len(list(output.glob("episode_*"))) == 6

    observation_space, action_space = jackal_policy_spaces()
    recorder = ExpertDemonstrationRecorder(
        output,
        observation_space,
        action_space,
        storage=storage,
    )
    try:
        assert recorder.recovered_interrupt
        assert not list(output.glob("*.partial"))
        for episode in smoke_episodes()[6:]:
            recorder.record(episode)
        with pytest.raises(FileExistsError):
            recorder.record(smoke_episodes()[0])
    finally:
        recorder.close()

    loaded = load_dataset(output)
    assert len(loaded) == 10
    assert [episode["episode_id"] for episode in loaded] == list(range(10))
    assert sum(episode["success"] for episode in loaded) == 6
    assert [episode["termination_reason"] for episode in loaded] == [
        spec[2] for spec in SMOKE_SPECS
    ]

    schema = json.loads((output / "schema.json").read_text())
    assert schema["schema_version"] == SCHEMA_VERSION
    assert schema["storage"] == storage
    assert schema["observation_space"]["shape"] == [40]
    assert schema["action_space"]["low"] == [-1.0, -1.0]
    assert schema["observation_normalization"]["lidar_max_range_m"] == 12.0
    assert (output / "session.json").read_text()
    assert json.loads((output / "session.json").read_text())["state"] == "closed"
    assert not (output / "recorder.lock").exists()

    for episode, spec in zip(loaded, SMOKE_SPECS):
        steps = spec[3]
        assert episode["scenario_id"] == spec[1]
        assert episode["observations"].shape == (steps, 40)
        assert episode["actions"].shape[0] == steps
        assert episode["requested_actions"].shape[0] == steps
        assert episode["applied_velocities"].shape[0] == steps
        assert episode["poses"].shape[0] == steps
        assert episode["simulation_timestamps_ns"].shape[0] == steps
        np.testing.assert_allclose(episode["goal"], spec[4])
        if steps:
            assert float(episode["actions"][0, 1]) == pytest.approx(spec[0] * 0.01)


if __name__ == "__main__":
    _record_prefix(sys.argv[1], sys.argv[2])
    os._exit(2)
