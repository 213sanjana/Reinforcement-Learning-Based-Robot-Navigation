"""Phase 6F.3 closed-loop evaluation tests.

These tests use the saved checkpoint and scripted environments. They do not
start Gazebo or publish a velocity command.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from navigation.jackal_bc_data import ObservationNormalizer
from navigation.jackal_bc_evaluation import (
    ASTAR_NAME,
    POLICY_NAME,
    allocate_output,
    bound_action,
    evaluation_scenarios,
    execute,
    infer_action,
    load_config,
    path_length,
    plan_scenarios,
    run_astar_episode,
    run_closed_loop_episode,
    save_evaluation,
    training_snapshot,
    validate_observation,
    verify_artifacts,
)
from navigation.jackal_bc_policy import JackalBCPolicy
from navigation.jackal_bc_train import file_sha256


ROOT = Path(__file__).resolve().parents[1]
EVAL_CONFIG = ROOT / "configs" / "jackal_behavior_cloning_6f3.yaml"
TRAINING_DIR = ROOT / "artifacts" / "behavior_cloning" / "jackal_6f2"
TEST_SPLIT = ROOT / "artifacts" / "expert_dataset" / "v6e.2.0" / "test"
WORLD = ROOT / "jackal_rl_ws" / "worlds" / "jackal_obstacle_test_collision.sdf"


class FakeRobot:
    def __init__(self):
        self.stopped = 0
        self.expert_publish_callback = None
        self.last_safety_reason = ""
        self.odom = None
        self.scan = object()
        self.command_topic = "/j100_0000/platform/cmd_vel_unstamped"
        self.distance = 2.5
        self.fail_stop = False

    def stop(self):
        if self.fail_stop:
            raise RuntimeError("stop failed")
        self.stopped += 1
        callback = getattr(self, "expert_publish_callback", None)
        if callback is not None:
            callback(0.0, 0.0, 0)

    def _goal_relative(self):
        return (float(self.distance), 0.0)


class ScriptedEnv:
    """Episode double with the Gymnasium reset/step/stop surface."""

    def __init__(self, behavior="success"):
        self.behavior = behavior
        self.robot = FakeRobot()
        self.goal_tolerance = 0.30
        self.max_episode_steps = 300
        self.previous_goal_distance = None
        self.actions = []
        self.resets = []
        self.closed = False
        self.observation_space = None
        self.action_space = None
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        self.resets.append({"seed": seed, "options": options})
        self._steps = 0
        start = tuple(options["start"])
        self.robot.distance = 2.5
        self.previous_goal_distance = 2.5
        if self.behavior == "bad_shape":
            return np.zeros(39, dtype=np.float32), {}
        if self.behavior == "nan_observation":
            observation = np.zeros(40, dtype=np.float32)
            observation[0] = np.nan
            return observation, {"robot_pose": (start[0], start[1], start[2])}
        return self._observation(2.5), {
            "robot_pose": (float(start[0]), float(start[1]), float(start[2])),
            "robot_world_pose": (
                float(start[0]),
                float(start[1]),
                0.3,
                float(start[2]),
            ),
        }

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self._steps += 1
        if self.behavior == "sensor_loss":
            raise TimeoutError("No fresh LiDAR and odometry after Gazebo step")
        start = self.resets[-1]["options"]["start"]
        goal = self.resets[-1]["options"]["goal"]
        if self.behavior == "success":
            distance = 0.2
            pose = (float(goal[0]), float(goal[1]), 0.0)
            flags = dict(
                goal_reached=True,
                collision_detected=False,
                time_limit_reached=False,
                termination_reason="goal_reached",
                terminated=True,
                truncated=False,
            )
        elif self.behavior == "near_miss":
            distance = 0.1
            pose = (float(goal[0]), float(goal[1]), 0.0)
            flags = dict(
                goal_reached=False,
                collision_detected=False,
                time_limit_reached=True,
                termination_reason="time_limit",
                terminated=False,
                truncated=True,
            )
        elif self.behavior == "collision":
            distance = 1.0
            pose = (float(start[0]) + 0.4, float(start[1]), float(start[2]))
            flags = dict(
                goal_reached=False,
                collision_detected=True,
                time_limit_reached=False,
                termination_reason="collision",
                terminated=True,
                truncated=False,
            )
        elif self.behavior == "horizon":
            distance = max(0.4, 2.5 - 0.1 * self._steps)
            pose = (float(start[0]) + 0.1 * self._steps, float(start[1]), 0.0)
            done = self._steps >= int(self.max_episode_steps)
            flags = dict(
                goal_reached=False,
                collision_detected=False,
                time_limit_reached=done,
                termination_reason="time_limit" if done else None,
                terminated=False,
                truncated=done,
            )
        else:
            raise RuntimeError(f"Unknown behavior {self.behavior}")
        self.robot.distance = distance
        info = {
            "robot_pose": pose,
            "current_distance": distance,
            "progress": 0.1,
            "action_accepted": True,
            "odom_stamp": float(self._steps),
            "scan_stamp": float(self._steps),
            **{key: value for key, value in flags.items() if key not in ("terminated", "truncated")},
        }
        return (
            self._observation(distance),
            1.25,
            flags["terminated"],
            flags["truncated"],
            info,
        )

    def close(self):
        self.closed = True
        self.robot.stop()

    @staticmethod
    def _observation(distance):
        observation = np.zeros(40, dtype=np.float32)
        observation[36] = np.float32(distance / 20.0)
        return observation


class FakeRunner:
    def __init__(self, status="GOAL_REACHED"):
        self.robot = FakeRobot()
        self.env = None
        self.status = status
        self.prepared = []
        self.durations = []
        self.closed = False

    def prepare(self, start, goal):
        self.prepared.append((tuple(start), tuple(goal)))

    def run(self, duration):
        self.durations.append(float(duration))
        start, goal = self.prepared[-1]
        from types import SimpleNamespace

        return SimpleNamespace(
            status=self.status,
            executed_poses=[
                (start[0], start[1], start[2]),
                (goal[0], goal[1], 0.0),
            ],
            planned_path=[(start[0], start[1]), (goal[0], goal[1])],
            accepted_commands=4,
            rejected_commands=0,
            frame_alignment=None,
        )

    def close(self):
        self.closed = True
        self.robot.stop()


class Preflight:
    def __init__(self, passed=True):
        self.passed = passed
        self.checks = {"velocity_no_competing_publisher": passed}


def _config():
    return load_config(EVAL_CONFIG)


def _action(policy, observation, action_space=None):
    del policy, observation, action_space
    return np.asarray([0.25, -0.1], dtype=np.float32)


def _open_scenario():
    return evaluation_scenarios(5)[0]


def test_checkpoint_and_preprocessing_match_training_pins():
    config = _config()
    before = training_snapshot(TRAINING_DIR)
    loaded = verify_artifacts(config)
    policy = loaded["policy"]

    assert loaded["checkpoint_sha256"] == config["expected_checkpoint_sha256"]
    assert loaded["normalizer"].observation_sha256 == config["expected_observation_sha256"]
    assert int(loaded["normalizer"].count) == 78834
    assert loaded["normalizer"].source_split == "train"
    assert loaded["payload"]["action_normalization"] == "identity"
    assert loaded["payload"]["trained"] is True
    assert policy.normalizer_is_ready
    assert not policy.training
    assert all(not parameter.requires_grad for parameter in policy.parameters())
    assert training_snapshot(TRAINING_DIR) == before


def test_statistics_are_not_refit_on_evaluation(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("evaluation refit normalization statistics")

    monkeypatch.setattr(
        ObservationNormalizer,
        "from_observations",
        _forbidden,
    )
    verify_artifacts(_config())


def test_observation_schema_mismatch_is_rejected():
    with pytest.raises(ValueError, match="40"):
        validate_observation(np.zeros(39, dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_observation(np.full(40, np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_observation(np.full(40, np.inf, dtype=np.float32))


def test_inference_is_deterministic_and_uses_raw_observations():
    loaded = verify_artifacts(_config())
    policy = loaded["policy"]
    observation = np.linspace(0.1, 0.9, 40, dtype=np.float32)
    seen = {}

    def _capture(observations):
        seen["value"] = observations.detach().cpu().numpy().copy()
        return JackalBCPolicy._forward_batch(policy, observations)

    policy._forward_batch = _capture
    first = infer_action(policy, observation)
    second = infer_action(policy, observation)

    assert first.shape == (2,)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert np.all(first <= 1.0) and np.all(first >= -1.0)
    assert np.allclose(seen["value"], observation)
    normalized = loaded["normalizer"].transform(observation)
    assert not np.allclose(seen["value"], normalized)


def test_action_bounds_reject_nonfinite_and_out_of_range_values():
    inside = bound_action([1.0, -1.0])
    assert np.array_equal(inside, np.asarray([1.0, -1.0], dtype=np.float32))
    clipped = bound_action([1.0 + 1e-6, 0.0])
    assert clipped[0] == np.float32(1.0)
    with pytest.raises(ValueError, match="outside"):
        bound_action([1.0 + 1e-4, 0.0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        bound_action([np.nan, 0.0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        bound_action([np.inf, 0.0])
    with pytest.raises(ValueError, match="shape"):
        bound_action([0.0, 0.0, 0.0])


def test_environment_action_is_not_scaled_a_second_time():
    env = ScriptedEnv("success")
    scenario = _open_scenario()
    record, steps = run_closed_loop_episode(
        env,
        policy=None,
        scenario=scenario,
        trial=0,
        seed=642,
        predict=_action,
    )

    assert len(env.actions) == 1
    assert np.allclose(env.actions[0], [0.25, -0.1])
    assert np.allclose(
        [steps[0]["linear_action"], steps[0]["angular_action"]],
        [0.25, -0.1],
    )
    assert record["success"] is True
    assert record["collision"] is False
    assert record["timeout"] is False
    assert record["termination_reason"] == "goal_reached"
    assert record["return"] == pytest.approx(1.25)
    assert record["control_steps"] == 1
    assert record["stop_verified"] is True
    assert env.robot.stopped >= 1
    assert env.resets[0]["seed"] == 642
    assert tuple(env.resets[0]["options"]["start"]) == scenario.start
    assert tuple(env.resets[0]["options"]["goal"]) == scenario.goal


def test_goal_distance_alone_is_not_counted_as_success():
    env = ScriptedEnv("near_miss")
    record, _steps = run_closed_loop_episode(
        env,
        policy=None,
        scenario=_open_scenario(),
        trial=0,
        seed=642,
        predict=_action,
    )

    assert record["final_goal_distance_m"] == pytest.approx(0.1)
    assert record["final_goal_distance_m"] < 0.30
    assert record["success"] is False
    assert record["timeout"] is True
    assert record["termination_reason"] == "time_limit"


def test_collision_timeout_and_invalid_outputs_stop_the_robot():
    collision_env = ScriptedEnv("collision")
    collision, _steps = run_closed_loop_episode(
        collision_env,
        policy=None,
        scenario=_open_scenario(),
        trial=0,
        seed=1,
        predict=_action,
    )
    assert collision["collision"] is True
    assert collision["success"] is False
    assert collision["timeout"] is False
    assert collision["flags_overlap"] is False
    assert collision_env.robot.stopped >= 1

    timeout_scenario = evaluation_scenarios(5)[-1]
    horizon_env = ScriptedEnv("horizon")
    timeout, steps = run_closed_loop_episode(
        horizon_env,
        policy=None,
        scenario=timeout_scenario,
        trial=0,
        seed=1,
        predict=_action,
    )
    assert horizon_env.max_episode_steps == 300
    assert timeout["timeout"] is True
    assert timeout["success"] is False
    assert timeout["collision"] is False
    assert timeout["control_steps"] == 5
    assert len(steps) == 5
    expected = path_length([
        (0.0, -1.5, 0.0),
        (0.1, -1.5, 0.0),
        (0.2, -1.5, 0.0),
        (0.3, -1.5, 0.0),
        (0.4, -1.5, 0.0),
        (0.5, -1.5, 0.0),
    ])
    assert timeout["path_length_m"] == pytest.approx(expected)
    assert timeout["path_length_m"] == pytest.approx(0.5)

    nan_env = ScriptedEnv("nan_observation")
    invalid, _steps = run_closed_loop_episode(
        nan_env,
        policy=None,
        scenario=_open_scenario(),
        trial=0,
        seed=1,
        predict=_action,
    )
    assert invalid["invalid_observations"] == 1
    assert invalid["safety_stop"] is True
    assert nan_env.actions == []
    assert nan_env.robot.stopped >= 1

    shape_env = ScriptedEnv("bad_shape")
    mismatched, _steps = run_closed_loop_episode(
        shape_env,
        policy=None,
        scenario=_open_scenario(),
        trial=0,
        seed=1,
        predict=_action,
    )
    assert mismatched["termination_reason"] == "invalid_observation"
    assert shape_env.actions == []

    def _nan_action(policy, observation, action_space=None):
        del policy, observation, action_space
        return np.asarray([np.nan, 0.0], dtype=np.float32)

    action_env = ScriptedEnv("success")
    rejected, _steps = run_closed_loop_episode(
        action_env,
        policy=None,
        scenario=_open_scenario(),
        trial=0,
        seed=1,
        predict=_nan_action,
    )
    assert rejected["invalid_actions"] == 1
    assert rejected["safety_stop"] is True
    assert action_env.actions == []
    assert action_env.robot.stopped >= 1


def test_cleanup_failure_is_recorded_and_attempt_still_stops():
    env = ScriptedEnv("success")
    env.robot.fail_stop = True
    record, _steps = run_closed_loop_episode(
        env,
        policy=None,
        scenario=_open_scenario(),
        trial=0,
        seed=1,
        predict=_action,
    )

    assert record["stop_verified"] is False
    assert record["termination_reason"] == "cleanup_failed"
    assert record["completed"] is False
    assert "stop failed" in record["error"]


def test_sensor_loss_is_not_counted_as_the_episode_horizon():
    env = ScriptedEnv("sensor_loss")
    record, _steps = run_closed_loop_episode(
        env,
        policy=None,
        scenario=_open_scenario(),
        trial=0,
        seed=1,
        predict=_action,
    )

    assert record["termination_reason"] == "interface_failure"
    assert record["timeout"] is False
    assert record["safety_stop"] is True
    assert env.robot.stopped >= 1


def test_bc_and_astar_use_the_same_scenario_definitions():
    scenarios = evaluation_scenarios(5)
    plans = plan_scenarios(scenarios, WORLD, 0.45)
    identifiers = [scenario.scenario_id for scenario in scenarios]

    assert identifiers == [
        "open_south",
        "obstacle_around",
        "obstacle_west_east",
        "obstacle_south_north",
        "timeout_open_south",
    ]
    assert sum(scenario.category == "obstacle" for scenario in scenarios) == 3
    assert scenarios[0].smoke is True
    assert scenarios[0].repeats == 2
    assert all(plans[scenario.scenario_id]["planning_only"] for scenario in scenarios)
    assert all(plans[scenario.scenario_id]["reachable"] for scenario in scenarios)
    assert plans["open_south"]["planned_length_m"] == pytest.approx(2.53, abs=0.05)

    runner = FakeRunner("TIME_LIMIT")
    reached_goal = scenarios[0]
    record = run_astar_episode(runner, reached_goal, 0, seed=642)
    assert runner.prepared == [(reached_goal.start, reached_goal.goal)]
    assert runner.durations == [reached_goal.astar_duration_s]
    assert record["success"] is False
    assert record["timeout"] is True
    assert record["final_goal_distance_m"] == pytest.approx(0.0)
    assert record["policy"] == ASTAR_NAME
    assert runner.robot.stopped >= 1

    collision_runner = FakeRunner("SAFETY_STOP")
    collision = run_astar_episode(collision_runner, scenarios[1], 0, seed=642)
    assert collision["collision"] is True
    assert collision["success"] is False
    assert collision["start"] == list(scenarios[1].start)
    assert collision["goal"] == list(scenarios[1].goal)


def test_evaluation_output_does_not_change_training_artifacts(tmp_path, monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("evaluation refit normalization statistics")

    monkeypatch.setattr(ObservationNormalizer, "from_observations", _forbidden)
    before = training_snapshot(TRAINING_DIR)
    test_stamp = file_sha256(TEST_SPLIT / "manifest.json")
    env = ScriptedEnv("success")
    runner = FakeRunner()
    summary = execute(
        _config(),
        "all",
        output_dir=tmp_path,
        preflight=lambda: Preflight(True),
        publisher_count=lambda: 0,
        environment_factory=lambda: env,
        astar_runner_factory=lambda: runner,
    )

    assert training_snapshot(TRAINING_DIR) == before
    assert file_sha256(TEST_SPLIT / "manifest.json") == test_stamp
    assert summary["gazebo_backend"] is False
    assert summary["training_artifacts_unchanged"] is True
    assert summary["read_test"] is False
    assert "not a Gazebo navigation result" in Path(summary["report"]).read_text()
    bc = [episode for episode in summary["episodes"] if episode["policy"] == POLICY_NAME]
    astar = [episode for episode in summary["episodes"] if episode["policy"] == ASTAR_NAME]
    assert [(episode["scenario_id"], episode["trial"]) for episode in bc] == [
        (episode["scenario_id"], episode["trial"]) for episode in astar
    ]
    assert [tuple(episode["start"]) for episode in bc] == [
        tuple(episode["start"]) for episode in astar
    ]
    assert summary["aggregates"][POLICY_NAME]["success"] == {
        "count": 6,
        "attempted": 6,
        "rate": 1.0,
    }
    assert (tmp_path / "episodes.csv").is_file()
    assert (tmp_path / "behavior_cloning_open_south_t0_steps.npz").is_file()
    assert (tmp_path / "comparison.json").is_file()
    with pytest.raises(FileExistsError):
        save_evaluation(tmp_path, summary, [], [], {})


def test_unsafe_smoke_blocks_later_scenarios(tmp_path):
    env = ScriptedEnv("nan_observation")
    summary = execute(
        _config(),
        "bc",
        output_dir=tmp_path,
        preflight=lambda: Preflight(True),
        publisher_count=lambda: 0,
        environment_factory=lambda: env,
        astar_runner_factory=lambda: FakeRunner(),
    )

    assert len(env.resets) == 1
    assert env.actions == []
    assert summary["bc_smoke_safe"] is False
    assert summary["aggregates"][POLICY_NAME]["attempted_episodes"] == 1
    assert "were not started" in summary["blocked_reason"]


def test_competing_publisher_blocks_before_reset(tmp_path):
    env = ScriptedEnv("success")
    summary = execute(
        _config(),
        "smoke",
        output_dir=tmp_path,
        preflight=lambda: Preflight(True),
        publisher_count=lambda: 2,
        environment_factory=lambda: env,
    )

    assert env.resets == []
    assert summary["live_executed"] is False
    assert "publisher" in summary["blocked_reason"]


def test_failed_preflight_does_not_construct_a_controller(tmp_path):
    created = []

    def _factory():
        created.append(ScriptedEnv("success"))
        return created[-1]

    summary = execute(
        _config(),
        "all",
        output_dir=tmp_path,
        preflight=lambda: Preflight(False),
        publisher_count=lambda: 0,
        environment_factory=_factory,
        astar_runner_factory=FakeRunner,
    )

    assert created == []
    assert summary["episodes"] == []
    assert summary["live_executed"] is False


def test_allocate_output_does_not_reuse_a_non_empty_directory(tmp_path):
    occupied = tmp_path / "jackal_6f3"
    occupied.mkdir()
    (occupied / "summary.json").write_text("{}\n", encoding="utf-8")
    created = allocate_output(occupied)

    assert created != occupied
    assert created.parent == occupied
    assert created.name.startswith("run_")


def test_config_rejects_the_held_out_test_split():
    config = _config()
    config["read_test"] = True
    with pytest.raises(ValueError, match="test split"):
        verify_artifacts(config)
