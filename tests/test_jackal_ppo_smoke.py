"""Unit tests for the Phase 6G.2 PPO smoke test.

The scripted environment is not Gazebo. It checks configuration, checkpoint
loading, optimization accounting, and shutdown. Live Jackal execution is the
separate smoke command.
"""

import copy
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from navigation.expert_demonstration_schema import jackal_policy_spaces
from navigation.jackal_ppo_train import (
    OFFICIAL_BC_CHECKPOINT,
    OFFICIAL_BC_SHA256,
    OFFICIAL_CHECKPOINT,
    OFFICIAL_CHECKPOINT_SHA256,
    SmokeFailure,
    assess_topic_readiness,
    load_config,
    parse_topic_info,
    run_smoke,
    validate_config,
)
from navigation.jackal_bc_ppo_transfer import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "jackal_ppo_smoke_6g2.yaml"


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class _Robot:
    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1


class ScriptedJackalEnv(gym.Env):
    """Finite stand-in with the Jackal observation and action spaces."""

    metadata = {"render_modes": []}

    def __init__(self, episode_length=30, fail_at=None, fail_with=None):
        super().__init__()
        self.observation_space, self.action_space = jackal_policy_spaces()
        self.max_episode_steps = 300
        self.episode_length = int(episode_length)
        self.fail_at = fail_at
        self.fail_with = fail_with
        self.robot = _Robot()
        self.actions = []
        self.resets = 0
        self.step_calls = 0
        self.elapsed = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.elapsed = 0
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        observation[36] = 2.5
        return observation, {"options": options}

    def step(self, action):
        self.step_calls += 1
        if self.fail_at is not None and self.step_calls == self.fail_at:
            raise self.fail_with
        action = np.asarray(action, dtype=np.float32)
        self.actions.append(action.copy())
        self.elapsed += 1
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        observation[36] = max(0.0, 2.5 - 0.01 * self.elapsed)
        reward = float(action[0]) * 0.1 - 0.01
        finished = self.elapsed >= self.episode_length
        if finished:
            self.elapsed = 0
        info = {
            "goal_reached": False,
            "collision_detected": False,
            "time_limit_reached": finished,
            "termination_reason": "time_limit" if finished else None,
        }
        return observation, reward, False, finished, info

    def close(self):
        self.closed = True
        self.robot.stop()


def _config():
    return load_config(CONFIG_PATH)


def _run(tmp_path, env, config=None):
    return run_smoke(
        config or _config(),
        tmp_path / "run",
        environment_factory=lambda: env,
        handle_signals=False,
    )


def test_configuration_rejects_unsafe_values():
    config = _config()
    assert config["initialization"] == "bc"
    assert config["runtime"]["total_timesteps"] == 1024
    assert config["ppo"]["n_steps"] == 128
    assert config["ppo"]["n_epochs"] == 2
    assert config["ppo"]["checkpoint_n_epochs"] == 4
    cases = [
        ("ppo", "n_envs", 2, "n_envs"),
        ("ppo", "use_sde", True, "use_sde"),
        ("ppo", "log_std_init", 1.0, "log_std_init"),
        ("runtime", "total_timesteps", 1000, "multiple"),
        ("runtime", "device", "cuda", "cpu"),
    ]
    for section, key, value, message in cases:
        bad = copy.deepcopy(config)
        bad[section][key] = value
        with pytest.raises(ValueError, match=message):
            validate_config(bad)


def test_scratch_initialization_is_refused(tmp_path):
    config = _config()
    config["initialization"] = "scratch"
    with pytest.raises(SmokeFailure, match="scratch"):
        run_smoke(config, tmp_path / "run", environment_factory=ScriptedJackalEnv, handle_signals=False)
    assert not (tmp_path / "run").exists()


def test_checkpoint_hash_mismatch_does_not_load_or_train(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "navigation.jackal_ppo_train.file_sha256",
        lambda path: "0" * 64,
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("PPO.load was called after a hash mismatch")

    monkeypatch.setattr("navigation.jackal_ppo_train.PPO.load", fail_load)
    with pytest.raises(SmokeFailure, match="hash mismatch"):
        run_smoke(
            _config(),
            tmp_path / "run",
            environment_factory=ScriptedJackalEnv,
            handle_signals=False,
        )
    assert not (tmp_path / "run").exists()


def test_topic_readiness_rejects_a_second_commander():
    counts = parse_topic_info("Publisher count: 1\nSubscription count: 2\n")
    assert counts == (1, 2)
    with pytest.raises(SmokeFailure, match="publisher"):
        assess_topic_readiness((1, 1), (1, 1), (1, 1))
    with pytest.raises(SmokeFailure, match="lidar3d_0"):
        assess_topic_readiness((0, 0), (1, 1), (0, 1))
    assess_topic_readiness((1, 1), (1, 1), (0, 1))


def test_output_cannot_replace_phase_6g1_or_behavior_cloning(tmp_path):
    with pytest.raises(SmokeFailure, match="Refusing to write"):
        run_smoke(
            _config(),
            ROOT / "artifacts" / "ppo" / "jackal_6g1" / "run_smoke",
            environment_factory=ScriptedJackalEnv,
            handle_signals=False,
        )


def test_scripted_smoke_optimizes_reloads_and_preserves_sources(tmp_path):
    before = {
        "ppo": file_sha256(OFFICIAL_CHECKPOINT),
        "bc": file_sha256(OFFICIAL_BC_CHECKPOINT),
    }
    assert before["ppo"] == OFFICIAL_CHECKPOINT_SHA256
    assert before["bc"] == OFFICIAL_BC_SHA256
    config = _config()
    config["runtime"]["total_timesteps"] = 256
    config["runtime"]["verbose"] = 0
    config["runtime"]["wall_clock_timeout_s"] = 180
    config["checkpointing"]["intermediate_timesteps"] = 128
    env = ScriptedJackalEnv(episode_length=30)
    summary = _run(tmp_path, env, config)

    assert summary["status"] == "passed", summary
    assert summary["environment_timesteps"] == 256
    assert summary["rollouts_completed"] == 2
    assert summary["n_updates"] == 4
    assert summary["gradient_steps"] == 8
    assert summary["losses_finite"] is True
    assert summary["gradients_finite"] is True
    assert summary["optimization_passed"] is True
    assert summary["critic_requires_grad"] is True
    assert summary["normalizer_buffers_unchanged"] is True
    assert summary["actor_changed"] or summary["critic_changed"] or summary["log_std_changed"]
    assert summary["completed_episodes"] == 8
    assert summary["partial_episodes"] == 1
    assert summary["raw_actions_outside_box"] > 0
    assert summary["clipped_actions_outside_box"] == 0
    assert summary["reload"]["passed"] is True
    assert summary["protected_artifacts_unchanged"] is True
    assert summary["initial_checkpoint_sha256"] == OFFICIAL_CHECKPOINT_SHA256
    assert summary["final_checkpoint_sha256"] != OFFICIAL_CHECKPOINT_SHA256
    assert env.closed is True
    assert env.robot.stops >= 1
    output = Path(summary["output_directory"])
    assert (output / "intermediate_ppo.zip").is_file()
    assert (output / "ppo_smoke.zip").is_file()
    assert (output / "episodes.csv").is_file()
    assert (output / "optimization.json").is_file()

    partial = [
        episode for episode in summary_episodes(output)
        if episode["status"] == "partial"
    ]
    assert len(partial) == 1
    assert partial[0]["length"] == 16
    assert partial[0]["success"] is False
    assert partial[0]["timeout"] is False
    completed = [
        episode for episode in summary_episodes(output)
        if episode["status"] == "completed"
    ]
    assert completed
    assert all(episode["termination_reason"] == "time_limit" for episode in completed)
    assert all(episode["timeout"] is True for episode in completed)

    with (output / "steps.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 256
    moderate = []
    for row in rows:
        assert row["action_finite"] == "True"
        assert row["action_in_bounds"] == "True"
        assert row["observation_finite"] == "True"
        assert row["reward_finite"] == "True"
        raw = np.array([
            float(row["raw_action_linear"]),
            float(row["raw_action_angular"]),
        ])
        clipped = np.array([
            float(row["action_linear"]),
            float(row["action_angular"]),
        ])
        assert np.allclose(clipped, np.clip(raw, -1.0, 1.0), atol=1e-5, rtol=0.0)
        if np.any((np.abs(raw) > 1.2) & (np.abs(raw) < 4.0)):
            moderate.append((raw, clipped))
    assert moderate
    raw, clipped = moderate[0]
    assert np.max(np.abs(clipped - np.tanh(raw))) > 1e-3
    assert file_sha256(OFFICIAL_CHECKPOINT) == before["ppo"]
    assert file_sha256(OFFICIAL_BC_CHECKPOINT) == before["bc"]


def test_simulator_failure_stops_without_retry(tmp_path):
    before = file_sha256(OFFICIAL_CHECKPOINT)
    env = ScriptedJackalEnv(
        fail_at=1,
        fail_with=TimeoutError("No fresh LiDAR and odometry after Gazebo step"),
    )
    summary = _run(tmp_path, env)
    assert summary["status"] == "failed"
    assert summary["failure_kind"] == "simulator_failure"
    assert summary["environment_timesteps"] == 0
    assert env.step_calls == 1
    assert env.resets == 1
    assert env.robot.stops >= 1
    assert env.closed is True
    assert (Path(summary["output_directory"]) / "failure.json").is_file()
    assert file_sha256(OFFICIAL_CHECKPOINT) == before


def test_interrupt_records_partial_episode_and_stops(tmp_path):
    env = ScriptedJackalEnv(fail_at=3, fail_with=KeyboardInterrupt())
    summary = _run(tmp_path, env)
    assert summary["status"] == "interrupted"
    assert summary["failure_kind"] == "interrupted"
    assert env.step_calls == 3
    assert env.robot.stops >= 1
    episodes = summary_episodes(Path(summary["output_directory"]))
    assert len(episodes) == 1
    assert episodes[0]["status"] == "partial"
    assert episodes[0]["length"] == 2
    assert episodes[0]["success"] is False
    assert summary["final_checkpoint"] is None


def summary_episodes(output):
    import json
    return json.loads((output / "episodes.json").read_text(encoding="utf-8"))
