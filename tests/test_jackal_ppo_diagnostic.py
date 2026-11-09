"""Unit tests for the Phase 6G.3 diagnostic PPO comparison.

The scripted environment is not Gazebo. Live navigation results come only
from the diagnostic command against the simulator.
"""

import copy
import csv
import inspect
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO

from navigation.expert_demonstration_schema import jackal_policy_spaces
from navigation.jackal_bc_evaluation import evaluation_scenarios
from navigation.jackal_bc_ppo_transfer import file_sha256
from navigation.jackal_ppo_diagnostic import (
    COMPARABILITY_NOTES,
    OFFICIAL_BC_CHECKPOINT,
    OFFICIAL_BC_SHA256,
    OFFICIAL_CHECKPOINT,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_SMOKE_CHECKPOINT,
    OFFICIAL_SMOKE_SHA256,
    SELECTION_RULE,
    VALIDATION_SCENARIOS,
    DiagnosticFailure,
    create_model,
    load_config,
    remaining_timesteps,
    resolve_validation_scenarios,
    run_diagnostic,
    selection_key,
    validate_config,
    validate_runner,
)
from navigation.jackal_ppo_diagnostic_report import write_comparison


ROOT = Path(__file__).resolve().parents[1]
BC_CONFIG = ROOT / "configs" / "jackal_ppo_diagnostic_6g3_bc.yaml"
SCRATCH_CONFIG = ROOT / "configs" / "jackal_ppo_diagnostic_6g3_scratch.yaml"


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
    """Finite stand-in. This is not a Gazebo result."""

    metadata = {"render_modes": []}

    def __init__(self, episode_length=30, fail_at=None, fail_with=None, nan_at=None):
        super().__init__()
        self.observation_space, self.action_space = jackal_policy_spaces()
        self.max_episode_steps = 300
        self.episode_length = int(episode_length)
        self.fail_at = fail_at
        self.fail_with = fail_with
        self.nan_at = nan_at
        self.robot = _Robot()
        self.actions = []
        self.reset_options = []
        self.resets = 0
        self.step_calls = 0
        self.elapsed = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.elapsed = 0
        self.reset_options.append(dict(options or {}))
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
        reward = float("nan") if self.nan_at == self.step_calls else float(action[0]) * 0.1 - 0.01
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
        self.robot.stop()


def _shrink(config, steps, interval):
    config = copy.deepcopy(config)
    config["runtime"]["total_timesteps"] = steps
    config["runtime"]["wall_clock_timeout_s"] = 180
    config["runtime"]["verbose"] = 0
    config["budget"]["total_timesteps"] = steps
    config["checkpointing"]["interval_timesteps"] = interval
    config["evaluation"]["interval_timesteps"] = interval
    return validate_config(config)


def _run(tmp_path, env, config, session_timesteps=None, resume=False):
    return run_diagnostic(
        config,
        tmp_path,
        environment_factory=lambda: env,
        handle_signals=False,
        session_timesteps=session_timesteps,
        resume=resume,
    )


@pytest.mark.parametrize("failure_stage", [None, "reset", "control_and_fresh_sensors"])
@pytest.mark.parametrize("sensor_evidence", ["fresh", "missing", "nonfinite"])
def test_validate_runner_is_bounded_and_never_trains(monkeypatch, failure_stage, sensor_evidence):
    import navigation.jackal_ppo_diagnostic as diagnostic

    class ValidationEnv(ScriptedJackalEnv):
        closed = False

        def reset(self, **kwargs):
            if failure_stage == "reset":
                raise TimeoutError("reset sensors unavailable")
            return super().reset(**kwargs)

        def close(self):
            self.closed = True
            super().close()

        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            if sensor_evidence != "missing":
                info.update(scan_stamp=1.1, odom_stamp=1.1)
            if sensor_evidence == "nonfinite":
                info["scan_stamp"] = float("nan")
            return observation, reward, terminated, truncated, info

    env = ValidationEnv(
        fail_at=1 if failure_stage == "control_and_fresh_sensors" else None,
        fail_with=TimeoutError("world control timed out"),
    )
    model, _report = create_model(load_config(BC_CONFIG))
    initial_steps, initial_updates = model.num_timesteps, model._n_updates
    initial_parameters = {k: v.clone() for k, v in model.policy.state_dict().items()}

    def forbidden(*args, **kwargs):
        pytest.fail("Validation entered a training or full-evaluation path")

    monkeypatch.setattr(PPO, "learn", forbidden)
    monkeypatch.setattr(PPO, "train", forbidden)
    monkeypatch.setattr(diagnostic, "evaluate_policy", forbidden)
    monkeypatch.setattr(diagnostic, "run_diagnostic", forbidden)
    monkeypatch.setattr(diagnostic, "create_model", lambda config: (model, {}))
    summary = validate_runner(load_config(BC_CONFIG), environment_factory=lambda: env)
    expected_stage = failure_stage or ("complete" if sensor_evidence == "fresh" else "control_and_fresh_sensors")
    assert summary["status"] == ("passed" if expected_stage == "complete" else "failed")
    assert summary["stage"] == expected_stage
    if failure_stage:
        assert summary["error"]["type"] == "TimeoutError"
    elif sensor_evidence != "fresh":
        assert summary["error"]["type"] == "DiagnosticFailure"
        assert "finite scan_stamp" in summary["error"]["message"]
    assert env.closed and env.robot.stops >= 1
    assert env.resets <= 1 and env.step_calls <= 1
    assert all(np.array_equal(action, np.zeros(2)) for action in env.actions)
    assert model.num_timesteps == initial_steps
    assert model._n_updates == initial_updates
    assert all(torch.equal(v, initial_parameters[k]) for k, v in model.policy.state_dict().items())
    assert summary["trained_timesteps"] == summary["ppo_updates"] == 0


def test_validate_cli_does_not_dispatch_training(monkeypatch, capsys):
    import navigation.jackal_ppo_diagnostic as diagnostic

    def forbidden(*args, **kwargs):
        pytest.fail("validate CLI dispatched training")

    monkeypatch.setattr(diagnostic, "run_diagnostic", forbidden)
    monkeypatch.setattr(diagnostic, "validate_runner", lambda config: {"status": "passed"})
    assert diagnostic.main(["validate", "--config", str(BC_CONFIG)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_official_configurations_match_and_record_the_epoch_discrepancy():
    bc = load_config(BC_CONFIG)
    scratch = load_config(SCRATCH_CONFIG)
    assert bc["initialization"] == "bc"
    assert scratch["initialization"] == "scratch"
    assert bc["ppo"] == scratch["ppo"]
    assert bc["runtime"] == scratch["runtime"]
    assert bc["evaluation"] == scratch["evaluation"]
    assert bc["training_scenario"] == scratch["training_scenario"]
    assert bc["runtime"]["total_timesteps"] == 49920
    assert bc["runtime"]["seed"] == 642
    assert bc["ppo"]["n_epochs"] == 4
    assert "checkpoint_n_epochs" not in bc["ppo"]
    assert bc["output"]["directory"] != scratch["output"]["directory"]
    assert tuple(bc["comparability"]) == COMPARABILITY_NOTES
    assert bc["evaluation"]["selection_rule"] == SELECTION_RULE
    scenarios = resolve_validation_scenarios(bc)
    assert [scenario.scenario_id for scenario in scenarios] == list(VALIDATION_SCENARIOS)
    catalog = {scenario.scenario_id: scenario for scenario in evaluation_scenarios(5)}
    for scenario in scenarios:
        assert scenario.start == catalog[scenario.scenario_id].start
        assert scenario.goal == catalog[scenario.scenario_id].goal
        assert scenario.max_episode_steps in (None, 300)
    assert "LiveExpertEpisode" not in inspect.getsource(
        __import__("navigation.jackal_ppo_diagnostic", fromlist=["evaluate_policy"]).evaluate_policy
    )


def test_configuration_rejects_unsafe_or_inconsistent_values():
    config = load_config(BC_CONFIG)
    cases = [
        ("ppo", "n_envs", 2, "n_envs"),
        ("ppo", "n_epochs", 2, "n_epochs"),
        ("ppo", "use_sde", True, "use_sde"),
        ("ppo", "log_std_init", -0.5, "log_std_init"),
        ("runtime", "total_timesteps", 50000, "50000"),
        ("runtime", "device", "cuda", "cpu"),
        ("initialization", None, "scratch", "scratch"),
    ]
    for section, key, value, message in cases:
        bad = copy.deepcopy(config)
        if section == "initialization":
            bad["initialization"] = value
        else:
            bad[section][key] = value
        with pytest.raises(ValueError, match=message):
            validate_config(bad)
    bad = copy.deepcopy(config)
    bad["ppo"]["checkpoint_n_epochs"] = 4
    with pytest.raises(ValueError, match="override"):
        validate_config(bad)


def test_loaded_smoke_checkpoint_used_two_epochs_and_phase_6g1_stores_four():
    initialized = PPO.load(str(OFFICIAL_CHECKPOINT), device="cpu")
    smoke = PPO.load(str(OFFICIAL_SMOKE_CHECKPOINT), device="cpu")
    try:
        assert int(initialized.n_epochs) == 4
        assert int(initialized.num_timesteps) == 0
        assert int(smoke.n_epochs) == 2
        assert int(smoke.num_timesteps) == 1024
        assert int(smoke._n_updates) == 16
    finally:
        if initialized.get_env() is not None:
            initialized.get_env().close()
        if smoke.get_env() is not None:
            smoke.get_env().close()
    smoke_config = yaml.safe_load(Path(ROOT / "configs" / "jackal_ppo_smoke_6g2.yaml").read_text())
    assert smoke_config["ppo"]["n_epochs"] == 2
    assert smoke_config["ppo"]["checkpoint_n_epochs"] == 4


def test_remaining_timesteps_do_not_add_another_budget():
    assert remaining_timesteps(0, 49920, 128) == 49920
    assert remaining_timesteps(10240, 49920, 128) == 39680
    with pytest.raises(ValueError):
        remaining_timesteps(100, 49920, 128)
    completed = 128
    # Stable-Baselines3 adds the learn argument to the current count when
    # reset_num_timesteps is false. The remaining count lands on the original total.
    assert completed + remaining_timesteps(completed, 256, 128) == 256


def test_selection_prefers_success_then_return_then_earlier_checkpoint():
    low = {"success_rate": 0.0, "mean_return": 5.0, "timesteps": 0}
    high = {"success_rate": 0.5, "mean_return": -1.0, "timesteps": 10240}
    later = {"success_rate": 0.0, "mean_return": 5.0, "timesteps": 20480}
    assert selection_key(high) > selection_key(low)
    assert selection_key(low) > selection_key(later)


def test_hash_mismatch_does_not_construct_a_model(tmp_path, monkeypatch):
    def reject(path):
        return "0" * 64

    monkeypatch.setattr("navigation.jackal_ppo_diagnostic.file_sha256", reject)

    class Disallowed(PPO):
        def __init__(self, *args, **kwargs):
            raise AssertionError("PPO was constructed after a hash mismatch")

        @classmethod
        def load(cls, *args, **kwargs):
            raise AssertionError("PPO.load was called after a hash mismatch")

    monkeypatch.setattr("navigation.jackal_ppo_diagnostic.PPO", Disallowed)
    with pytest.raises(DiagnosticFailure, match="hash mismatch"):
        run_diagnostic(
            _shrink(load_config(BC_CONFIG), 256, 128),
            tmp_path / "run",
            environment_factory=ScriptedJackalEnv,
            handle_signals=False,
        )
    assert not (tmp_path / "run").exists()


def test_scratch_initialization_does_not_copy_actor_weights():
    report_model, report = create_model(load_config(SCRATCH_CONFIG))
    try:
        assert report["training_model_source"] == "framework_initialization"
        assert report["loaded_training_weights_from_checkpoint"] is False
        assert report["actor_matches_bc"] is False
        assert "action_net.weight" not in report["actor_parameters_equal_to_6g1"]
        assert report["normalizer_matches_bc"] is True
        assert report["log_std"] == [0.0, 0.0]
        assert tuple(report_model.observation_space.shape) == (40,)
        assert tuple(report_model.action_space.low) == (-1.0, -1.0)
        assert tuple(report_model.action_space.high) == (1.0, 1.0)
        assert report_model.policy.squash_output is False
    finally:
        if report_model.get_env() is not None:
            report_model.get_env().close()


def test_bc_initialization_matches_the_transferred_actor():
    model, report = create_model(load_config(BC_CONFIG))
    try:
        assert report["method"] == "PPO.load"
        assert report["actor_matches_bc"] is True
        assert report["constructed_new_weights"] is False
        assert int(model.n_epochs) == 4
        assert int(model.num_timesteps) == 0
    finally:
        if model.get_env() is not None:
            model.get_env().close()


def test_scripted_bc_run_accounts_for_episodes_actions_and_reload(tmp_path):
    before = {
        "ppo": file_sha256(OFFICIAL_CHECKPOINT),
        "bc": file_sha256(OFFICIAL_BC_CHECKPOINT),
        "smoke": file_sha256(OFFICIAL_SMOKE_CHECKPOINT),
    }
    env = ScriptedJackalEnv(episode_length=30)
    config = _shrink(load_config(BC_CONFIG), 256, 128)
    summary = _run(tmp_path / "run", env, config)
    assert summary["status"] == "completed", summary
    assert summary["trained_timesteps"] == 256
    assert summary["rollouts_completed"] == 2
    assert summary["n_updates"] == 8
    assert summary["gradient_steps"] == 16
    assert summary["n_epochs"] == 4
    assert summary["successes"] == 0
    assert summary["timeouts"] == 8
    assert summary["completed_episodes"] == 8
    assert summary["partial_episodes"] == 2
    assert summary["navigation_success_is_not_required"] is True
    assert summary["losses_finite"] is True
    assert summary["normalizer_buffers_unchanged"] is True
    assert summary["reload"]["passed"] is True
    assert summary["protected_artifacts_unchanged"] is True
    assert summary["initial_validation_success_rate"] == 0.0
    assert summary["final_validation_success_rate"] == 0.0
    output = Path(summary["output_directory"])
    episodes = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    partial = [episode for episode in episodes if episode["status"] == "partial"]
    assert {episode["interruption"] for episode in partial} == {"evaluation_reset", "training_budget"}
    assert all(episode["timeout"] is False and episode["success"] is False for episode in partial)
    completed = [episode for episode in episodes if episode["status"] == "completed"]
    assert all(episode["timeout"] is True and episode["termination_reason"] == "time_limit" for episode in completed)
    with (output / "steps.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 256
    moderate = []
    for row in rows:
        assert row["action_finite"] == "True"
        assert row["action_in_bounds"] == "True"
        assert row["reward_finite"] == "True"
        raw = np.array([float(row["raw_action_linear"]), float(row["raw_action_angular"])])
        clipped = np.array([float(row["action_linear"]), float(row["action_angular"])])
        assert np.allclose(clipped, np.clip(raw, -1.0, 1.0), atol=1e-5, rtol=0.0)
        assert np.max(np.abs(clipped)) <= 1.0
        if np.any((np.abs(raw) > 1.2) & (np.abs(raw) < 4.0)):
            moderate.append((raw, clipped))
    assert moderate
    raw, clipped = moderate[0]
    assert np.max(np.abs(clipped - np.tanh(raw))) > 1e-3
    evaluations = json.loads((output / "evaluations.json").read_text(encoding="utf-8"))["records"]
    assert [int(record["timesteps"]) for record in evaluations] == [0, 128, 256]
    assert {episode["scenario_id"] for episode in evaluations[0]["episodes"]} == set(VALIDATION_SCENARIOS)
    assert all(episode["path_length_odom_m"] is None for episode in evaluations[0]["episodes"])
    assert (output / "selected_ppo.zip").is_file()
    assert (output / "final_ppo.zip").is_file()
    assert (output / "checkpoints" / "step_00000000.zip").is_file()
    assert env.robot.stops >= 1
    starts = [tuple(item["start"]) for item in env.reset_options if "start" in item]
    assert (0.0, -1.5, 0.0) in starts
    assert (-1.0, -1.5, 0.0) in starts
    assert file_sha256(OFFICIAL_CHECKPOINT) == before["ppo"]
    assert file_sha256(OFFICIAL_BC_CHECKPOINT) == before["bc"]
    assert file_sha256(OFFICIAL_SMOKE_CHECKPOINT) == before["smoke"]


def test_resume_continues_to_the_original_budget_without_duplicating_episodes(tmp_path):
    env = ScriptedJackalEnv(episode_length=4)
    config = _shrink(load_config(BC_CONFIG), 256, 128)
    output = tmp_path / "run"
    first = _run(output, env, config, session_timesteps=128)
    assert first["status"] == "incomplete"
    assert first["trained_timesteps"] == 128
    assert first["stop_reason"] == "session_limit"
    episodes = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    episodes.append({
        "episode_index": 999,
        "scenario_id": "open_south",
        "seed": 642,
        "status": "partial",
        "episode_return": 0.0,
        "length": 3,
        "success": False,
        "collision": False,
        "timeout": False,
        "termination_reason": "",
        "interruption": "synthetic_orphan",
        "environment_steps_completed": 999,
        "start": [0.0, -1.5, 0.0],
        "goal": [2.5, -1.5],
    })
    (output / "episodes.json").write_text(json.dumps(episodes), encoding="utf-8")
    with (output / "steps.csv").open("a", encoding="utf-8") as handle:
        handle.write("999,0,0,0,0,0,True,True,0,True,True,False,False,\n")
    steps_before = env.step_calls
    second = _run(output, env, config, resume=True)
    assert second["status"] == "completed", second
    assert second["trained_timesteps"] == 256
    assert second["resumed"] is True
    assert "rollout buffer" in second["resume_limitations"]
    resumed_episodes = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    assert all(int(episode["environment_steps_completed"]) <= 256 for episode in resumed_episodes)
    assert all(episode["episode_index"] != 999 for episode in resumed_episodes)
    indexes = [episode["episode_index"] for episode in resumed_episodes]
    assert len(indexes) == len(set(indexes))
    orphan = json.loads((output / "orphaned_episodes.json").read_text(encoding="utf-8"))
    assert any(episode["episode_index"] == 999 for episode in orphan)
    third_calls = env.step_calls
    third = _run(output, env, config, resume=True)
    assert third["status"] == "completed"
    assert third["resume_noop"] is True
    assert third["trained_timesteps"] == 256
    assert env.step_calls == third_calls
    assert env.step_calls > steps_before


def test_resume_archives_logs_newer_than_the_saved_checkpoint(tmp_path):
    config = _shrink(load_config(BC_CONFIG), 384, 256)
    output = tmp_path / "lagging_checkpoint"
    first = _run(output, ScriptedJackalEnv(
        episode_length=4, fail_at=150, fail_with=TimeoutError("simulator lost"),
    ), config)
    assert first["status"] == "failed"
    assert first["trained_timesteps"] == 128
    assert not (output / "checkpoints" / "step_00000128.zip").exists()
    second = _run(output, ScriptedJackalEnv(episode_length=4), config, resume=True)
    assert second["status"] == "completed", second
    assert second["trained_timesteps"] == 384
    with (output / "steps.csv").open() as handle:
        steps = [int(row["step_index"]) for row in csv.DictReader(handle)]
    assert steps == list(range(1, 385))
    updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert [row["timesteps"] for row in updates] == [128, 256, 384]
    assert [row["optimizer_steps"] for row in updates] == [8, 16, 24]
    assert (output / "orphaned_updates.jsonl").is_file()
    assert second["orphaned_steps"] >= 128


def test_resume_rejects_a_different_initialization(tmp_path):
    env = ScriptedJackalEnv(episode_length=4)
    output = tmp_path / "run"
    first = _run(output, env, _shrink(load_config(SCRATCH_CONFIG), 256, 128), session_timesteps=128)
    assert first["initialization"] == "scratch"
    assert first["initialization_report"]["loaded_training_weights_from_checkpoint"] is False

    def forbidden():
        raise AssertionError("wrong-experiment resume started the environment")

    with pytest.raises(DiagnosticFailure, match="different"):
        run_diagnostic(
            _shrink(load_config(BC_CONFIG), 256, 128),
            output,
            environment_factory=forbidden,
            handle_signals=False,
            resume=True,
        )


def test_output_cannot_replace_prior_runs(tmp_path):
    config = _shrink(load_config(BC_CONFIG), 256, 128)
    with pytest.raises(DiagnosticFailure, match="Refusing to write"):
        run_diagnostic(
            config,
            ROOT / "artifacts" / "ppo" / "jackal_6g1" / "phase6g3_should_not_exist",
            environment_factory=ScriptedJackalEnv,
            handle_signals=False,
        )
    assert not (ROOT / "artifacts" / "ppo" / "jackal_6g1" / "phase6g3_should_not_exist").exists()


def test_interrupt_and_nonfinite_reward_stop_without_counting_a_timeout(tmp_path):
    interrupted = ScriptedJackalEnv(episode_length=100, fail_at=404, fail_with=KeyboardInterrupt())
    summary = _run(tmp_path / "interrupt", interrupted, _shrink(load_config(BC_CONFIG), 256, 128))
    assert summary["status"] == "interrupted"
    assert interrupted.robot.stops >= 1
    output = Path(summary["output_directory"])
    episodes = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    orphan_path = output / "orphaned_episodes.json"
    if orphan_path.is_file():
        episodes.extend(json.loads(orphan_path.read_text(encoding="utf-8")))
    partial = [episode for episode in episodes if episode["status"] == "partial"]
    assert partial
    assert all(episode["timeout"] is False for episode in partial)
    assert summary["orphaned_episodes"] >= 1 or summary["partial_episodes"] >= 1

    poisoned = ScriptedJackalEnv(episode_length=30, nan_at=121)
    failed = _run(tmp_path / "nan", poisoned, _shrink(load_config(BC_CONFIG), 256, 128))
    assert failed["status"] == "failed"
    assert failed["failure_kind"] == "nonfinite_transition"
    assert poisoned.robot.stops >= 1
    failed_episodes = json.loads((Path(failed["output_directory"]) / "episodes.json").read_text(encoding="utf-8"))
    assert all(episode["timeout"] is False for episode in failed_episodes if episode["status"] == "partial")


@pytest.mark.parametrize("failing_stage", ["reset", "predict", "step"])
def test_evaluation_preserves_completed_and_incomplete_progress_and_time(tmp_path, monkeypatch, failing_stage):
    import navigation.jackal_ppo_diagnostic as diagnostic

    ledger = tmp_path / "evaluation_episodes.jsonl"

    class FailingEvaluationEnv(ScriptedJackalEnv):
        def reset(self, **kwargs):
            if self.resets == 1:
                # The first episode must be durable BEFORE the next reset.
                first = json.loads(ledger.read_text().splitlines()[0])
                assert first["status"] == "completed" and first["length"] == 2
                if failing_stage == "reset":
                    raise TimeoutError("reset failed")
            return super().reset(**kwargs)

    class ZeroPolicy:
        calls = 0

        def predict(self, observation, deterministic):
            self.calls += 1
            if self.calls == 4 and failing_stage == "predict":
                raise TimeoutError("predict failed")
            return np.zeros(2, dtype=np.float32), None

    env = FailingEvaluationEnv(episode_length=2,
        fail_at=4 if failing_stage == "step" else None,
        fail_with=TimeoutError("step failed"))
    state = {"evaluated_timesteps": set(), "session_started_at": None,
             "session_eval_s": 0.0, "evaluations": []}
    ticks = iter(float(i) for i in range(100))
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError):
        diagnostic._maybe_evaluate(ZeroPolicy(), env, tmp_path, load_config(BC_CONFIG), state, 0, False)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 2
    failed = rows[1]
    assert failed["status"] == "incomplete" and failed["complete"] is False
    assert failed["scenario_id"] != rows[0]["scenario_id"]
    assert failed["completed_steps"] == (0 if failing_stage == "reset" else 1)
    assert failed["failing_stage"] == failing_stage
    assert failed["error"]["type"] == "TimeoutError"
    assert state["session_eval_s"] > 0
    assert not state["evaluated_timesteps"] and not state["evaluations"]
    assert env.robot.stops >= 2


def test_simulator_failure_does_not_retry(tmp_path):
    env = ScriptedJackalEnv(
        episode_length=30,
        fail_at=1,
        fail_with=TimeoutError("No fresh LiDAR and odometry after Gazebo step"),
    )
    summary = _run(tmp_path / "fail", env, _shrink(load_config(BC_CONFIG), 256, 128))
    assert summary["status"] == "failed"
    assert summary["failure_kind"] == "simulator_failure"
    assert summary["trained_timesteps"] == 0
    assert env.step_calls == 1
    assert env.robot.stops >= 1
    assert (Path(summary["output_directory"]) / "failure.json").is_file()


def test_comparison_report_uses_measured_zeros_without_a_significance_claim(tmp_path):
    bc = _synthetic_run(tmp_path / "bc", "bc_initialized", "bc")
    scratch = _synthetic_run(tmp_path / "scratch", "scratch", "scratch")
    summary = write_comparison(bc, scratch, tmp_path / "comparison")
    text = (tmp_path / "comparison" / "comparison_report.md").read_text(encoding="utf-8")
    assert "final fixed-validation success rate is zero" in text
    assert "does not establish statistical significance" in text
    assert "learned successful navigation" not in text
    assert (tmp_path / "comparison" / "validation_success.png").is_file()
    assert (tmp_path / "comparison" / "episode_series.csv").is_file()
    assert summary["source_hashes"]["phase_6g1"]["unchanged"] is True
    assert summary["source_hashes"]["phase_6f2"]["unchanged"] is True
    assert summary["source_hashes"]["phase_6g2"]["unchanged"] is True


def _synthetic_run(path, experiment, initialization):
    path.mkdir(parents=True)
    episodes = []
    for index, timestep in enumerate((30, 60, 90), start=1):
        episodes.append({
            "episode_index": index,
            "scenario_id": "open_south",
            "status": "completed",
            "episode_return": -1.0,
            "success": False,
            "collision": False,
            "timeout": True,
            "environment_steps_completed": timestep,
        })
    episodes.append({
        "episode_index": 4,
        "scenario_id": "open_south",
        "status": "partial",
        "episode_return": -0.2,
        "success": False,
        "collision": False,
        "timeout": False,
        "interruption": "training_budget",
        "environment_steps_completed": 100,
    })
    evaluations = []
    for timestep in (0, 128):
        evaluations.append({
            "timesteps": timestep,
            "complete": True,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "timeout_rate": 1.0,
            "mean_return": -1.0,
            "median_return": -1.0,
            "episodes": [
                {
                    "scenario_id": scenario,
                    "success": False,
                    "collision": False,
                    "timeout": True,
                    "episode_return": -1.0,
                    "length": 30,
                    "elapsed_s": 1.0,
                    "path_length_odom_m": None,
                    "status": "completed",
                }
                for scenario in VALIDATION_SCENARIOS
            ],
        })
    (path / "episodes.json").write_text(json.dumps(episodes), encoding="utf-8")
    (path / "evaluations.json").write_text(json.dumps({"records": evaluations}), encoding="utf-8")
    (path / "updates.jsonl").write_text(json.dumps({
        "timesteps": 128,
        "metrics": {
            "train/policy_gradient_loss": -0.01,
            "train/value_loss": 0.2,
            "train/entropy_loss": -2.8,
            "train/approx_kl": 0.001,
            "train/explained_variance": -0.1,
        },
        "learning_rate": 3e-5,
        "log_std": [0.0, 0.0],
        "std": [1.0, 1.0],
    }) + "\n", encoding="utf-8")
    (path / "summary.json").write_text(json.dumps({
        "status": "completed",
        "experiment": experiment,
        "initialization": initialization,
        "trained_timesteps": 128,
        "requested_timesteps": 128,
        "rollouts_completed": 1,
        "n_updates": 4,
        "gradient_steps": 8,
        "training_wall_time_s": 10.0,
        "evaluation_wall_time_s": 4.0,
        "completed_episodes": 3,
        "partial_episodes": 1,
        "successes": 0,
        "collisions": 0,
        "timeouts": 3,
        "uncommitted_steps": 0,
        "orphaned_steps": 0,
        "losses_finite": True,
        "gradients_finite": True,
        "action_summary": {"saturated_action_fraction": 0.25},
    }), encoding="utf-8")
    return path
