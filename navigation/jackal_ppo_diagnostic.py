"""Phase 6G.3 diagnostic comparison of BC-initialized PPO and PPO from scratch.

The two configurations share the Jackal environment, PPO hyperparameters,
and fixed validation scenarios. BC initialization loads the verified Phase
6G.1 checkpoint. Scratch initialization constructs a new policy and value
network, then installs only the frozen observation-normalizer buffers.

Resume restores the policy, optimizer, and timestep count. It does not
restore the rollout buffer or the NumPy, PyTorch, or Python generators, so a
resumed run is not bit-for-bit identical to an uninterrupted run.

Gazebo must already be running. This process creates one robot interface.

    # One reset and one zero-action control/sensor check, without training:
    python -m navigation.jackal_ppo_diagnostic validate --config configs/jackal_ppo_diagnostic_6g3_bc.yaml

    python -m navigation.jackal_ppo_diagnostic train --config configs/jackal_ppo_diagnostic_6g3_bc.yaml
    python -m navigation.jackal_ppo_diagnostic train --config configs/jackal_ppo_diagnostic_6g3_scratch.yaml
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

_ROOT = Path(__file__).resolve().parents[1]
_JACKAL_SRC = _ROOT / "jackal_rl_ws" / "src" / "jackal_rl_navigation"
for _path in (str(_JACKAL_SRC), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agents.jackal_pretrained_ppo_policy import (  # noqa: E402
    JackalPolicyContractEnv,
    JackalPretrainedPPOPolicy,
    check_sb3_version,
)
from navigation.expert_demonstration_schema import validate_jackal_policy_spaces  # noqa: E402
from navigation.expert_evaluation_report import path_length  # noqa: E402
from navigation.jackal_bc_evaluation import evaluation_scenarios  # noqa: E402
from navigation.jackal_bc_policy import load_checkpoint  # noqa: E402
from navigation.jackal_bc_ppo_transfer import (  # noqa: E402
    _seed_everything,
    _transfer_pairs,
    file_sha256,
    tensor_sha256,
)
from navigation.jackal_ppo_train import (  # noqa: E402
    FixedScenarioWrapper,
    OptimizationFailure,
    SmokeFailure,
    StopOnFailureWrapper,
    TimeLimitInfoWrapper,
    _buffer_rows,
    _check_environment,
    _compare_parameters,
    _error_payload,
    _flush_logger,
    _fsync,
    _make_jackal_env,
    _parameter_rows,
    _plain,
    _robot,
    _same_number,
    _save_checkpoint,
    _schedule_value,
    _shutdown,
    _stop_robot,
    _train_metrics,
    _versions,
    _write_json,
    _write_text,
    audit_loaded_policy,
    preflight_simulator,
    threading_is_main,
)


PHASE = "6g.3"
OFFICIAL_CHECKPOINT = (
    _ROOT / "artifacts" / "ppo" / "jackal_6g1" / "run_20261022_230534" / "initialized_ppo.zip"
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "dff19be924f920490915d8f625d44f32da79452caaf3f7698160d2b8fc287750"
)
OFFICIAL_BC_CHECKPOINT = _ROOT / "artifacts" / "behavior_cloning" / "jackal_6f2" / "best.pt"
OFFICIAL_BC_SHA256 = "1cc42601ae918057bc01fd18f3cfa3975bb9a390e9c2d08715a1b05f0ff515c9"
OFFICIAL_SMOKE_CHECKPOINT = (
    _ROOT / "artifacts" / "ppo" / "jackal_6g2" / "run_20261022_233150" / "ppo_smoke.zip"
)
OFFICIAL_SMOKE_SHA256 = "38038dc1566398c9365c0eceb86a69a2c54ea307b884631570269746589a195c"
OFFICIAL_ARCHITECTURE = _ROOT / "artifacts" / "behavior_cloning" / "jackal_6f2" / "architecture.json"
VALIDATION_SCENARIOS = (
    "open_south",
    "obstacle_around",
    "obstacle_west_east",
    "obstacle_south_north",
)
SELECTION_RULE = (
    "Select the checkpoint with the highest fixed-validation success rate. "
    "Break ties with the higher mean episode return, then the earlier checkpoint. "
    "Use only complete validation runs. Keep the final checkpoint as a separate file."
)
COMPARABILITY_NOTES = (
    "Both approaches use one JackalGazeboEnv, the 40-value observation, and clipped diagonal-Gaussian actions.",
    "Total environment timesteps are 49920 because 50000 is not divisible by n_steps=128.",
    "n_epochs is 4, matching the Phase 6G.1 checkpoint. The Phase 6G.2 smoke test overrode this to 2 and its saved zip stores 2.",
    "Scratch initialization installs the frozen BC observation normalizer buffers and does not copy actor or critic weights.",
    "Training uses the open_south scenario only. There is no curriculum.",
    "Validation uses one episode of open_south, obstacle_around, obstacle_west_east, and obstacle_south_north.",
    "timeout_open_south is excluded because its 5-step horizon forces a timeout.",
    "Evaluation is deterministic. Training samples the diagonal Gaussian and Stable-Baselines3 clips to [-1, 1].",
    "Evaluation resets the single robot and records the open training episode as partial, not as a timeout.",
    "The matched seed is 642. Gazebo physics and sensor timing are not controlled by that seed.",
    "Checkpoint selection uses validation success rate, then mean return, then the earlier checkpoint.",
    "Resume does not restore the rollout buffer or the random-number generator.",
)
RESUME_LIMITATIONS = (
    "The saved checkpoint restores the policy, value network, optimizer state, "
    "timestep count, and Stable-Baselines3 update count. It does not restore the "
    "on-policy rollout buffer or the Python, NumPy, or PyTorch random generators. "
    "Gazebo physics is also not restored. Resumed training continues toward the "
    "original timestep budget, but it is not bit-for-bit identical to an uninterrupted run. "
    "Steps collected after the last completed update are archived and are not part of the restored policy."
)
FORBIDDEN_OUTPUT_ROOTS = (
    _ROOT / "artifacts" / "ppo" / "jackal_6g1",
    _ROOT / "artifacts" / "ppo" / "jackal_6g2",
    _ROOT / "artifacts" / "behavior_cloning",
)
STEP_COLUMNS = (
    "step_index",
    "episode_index",
    "action_linear",
    "action_angular",
    "raw_action_linear",
    "raw_action_angular",
    "action_finite",
    "action_in_bounds",
    "reward",
    "reward_finite",
    "observation_finite",
    "terminated",
    "truncated",
    "termination_reason",
)
EPISODE_COLUMNS = (
    "episode_index",
    "scenario_id",
    "seed",
    "status",
    "episode_return",
    "length",
    "success",
    "collision",
    "timeout",
    "termination_reason",
    "interruption",
    "environment_steps_completed",
    "start",
    "goal",
)
REQUIRED_METRICS = (
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/loss",
    "train/std",
)


class DiagnosticFailure(RuntimeError):
    """The diagnostic run cannot continue safely."""


def load_config(path):
    """Load and validate a Phase 6G.3 YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = validate_config(config)
    config["config_path"] = str(Path(path).resolve())
    return config


def validate_config(config):
    """Raise unless ``config`` describes one Phase 6G.3 diagnostic run."""

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    config = json.loads(json.dumps(config))
    if config.get("phase") != PHASE:
        raise ValueError("Configuration phase must be 6g.3")
    initialization = config.get("initialization")
    experiment = config.get("experiment")
    if initialization == "bc" and experiment != "bc_initialized":
        raise ValueError("BC initialization uses experiment bc_initialized")
    if initialization == "scratch" and experiment != "scratch":
        raise ValueError("Scratch initialization uses experiment scratch")
    if initialization not in ("bc", "scratch"):
        raise ValueError("initialization must be 'bc' or 'scratch'")
    checkpoint = _resolve(config.get("checkpoint"))
    bc_checkpoint = _resolve(config.get("bc_checkpoint"))
    smoke_checkpoint = _resolve(config.get("smoke_checkpoint"))
    architecture = _resolve(config.get("architecture"))
    if checkpoint != OFFICIAL_CHECKPOINT.resolve():
        raise ValueError("checkpoint must be the Phase 6G.1 initialized PPO zip")
    if bc_checkpoint != OFFICIAL_BC_CHECKPOINT.resolve():
        raise ValueError("bc_checkpoint must be the frozen Phase 6F.2 best.pt")
    if smoke_checkpoint != OFFICIAL_SMOKE_CHECKPOINT.resolve():
        raise ValueError("smoke_checkpoint must be the Phase 6G.2 smoke zip")
    if architecture != OFFICIAL_ARCHITECTURE.resolve():
        raise ValueError("architecture must be the Phase 6F.2 architecture file")
    if config.get("expected_checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("expected_checkpoint_sha256 does not match Phase 6G.1")
    if config.get("expected_bc_sha256") != OFFICIAL_BC_SHA256:
        raise ValueError("expected_bc_sha256 does not match Phase 6F.2")
    if config.get("expected_smoke_sha256") != OFFICIAL_SMOKE_SHA256:
        raise ValueError("expected_smoke_sha256 does not match Phase 6G.2")
    config["checkpoint"] = str(checkpoint)
    config["bc_checkpoint"] = str(bc_checkpoint)
    config["smoke_checkpoint"] = str(smoke_checkpoint)
    config["architecture"] = str(architecture)

    output = _mapping(config, "output")
    if not isinstance(output.get("directory"), str) or not output["directory"]:
        raise ValueError("Output directory is required")
    runtime = _mapping(config, "runtime")
    _nonnegative_int(runtime.get("seed"), "seed")
    if runtime.get("device") != "cpu":
        raise ValueError("Phase 6G.3 diagnostic training runs on cpu")
    total = _positive_int(runtime.get("total_timesteps"), "total_timesteps")
    if total == 50000:
        raise ValueError("50000 is not a multiple of n_steps")
    timeout = _positive_number(runtime.get("wall_clock_timeout_s"), "wall_clock_timeout_s")
    if timeout > 86400:
        raise ValueError("wall_clock_timeout_s must be at most 86400 seconds")
    if runtime.get("verbose") not in (0, 1):
        raise ValueError("verbose must be 0 or 1")
    budget = _mapping(config, "budget")
    if int(budget.get("nominal_timesteps")) != 50000:
        raise ValueError("nominal_timesteps must stay 50000")
    if int(budget.get("total_timesteps")) != total:
        raise ValueError("budget.total_timesteps must equal runtime.total_timesteps")
    if not isinstance(budget.get("reason"), str) or "128" not in budget["reason"]:
        raise ValueError("budget.reason must explain the 128-step multiple")

    ppo = _mapping(config, "ppo")
    if "checkpoint_n_epochs" in ppo:
        raise ValueError("Phase 6G.3 does not override n_epochs")
    if int(ppo.get("n_epochs")) != 4:
        raise ValueError("n_epochs must be 4 for both approaches")
    note = ppo.get("n_epochs_note")
    if not isinstance(note, str) or "n_epochs=4" not in note or "n_epochs=2" not in note:
        raise ValueError("n_epochs_note must describe the Phase 6G.2 override")
    _positive_int(ppo.get("n_steps"), "n_steps")
    _positive_int(ppo.get("batch_size"), "batch_size")
    if int(ppo["n_envs"]) != 1:
        raise ValueError("n_envs must be 1. One Jackal controller is allowed.")
    if int(ppo["n_steps"]) % int(ppo["batch_size"]) != 0:
        raise ValueError("batch_size must divide n_steps")
    if total % int(ppo["n_steps"]) != 0:
        raise ValueError("total_timesteps must be a multiple of n_steps")
    _positive_number(ppo.get("learning_rate"), "learning_rate")
    _unit_interval(ppo.get("gamma"), "gamma")
    _unit_interval(ppo.get("gae_lambda"), "gae_lambda")
    _unit_interval(ppo.get("clip_range"), "clip_range")
    _nonnegative_number(ppo.get("ent_coef"), "ent_coef")
    _positive_number(ppo.get("vf_coef"), "vf_coef")
    _positive_number(ppo.get("max_grad_norm"), "max_grad_norm")
    if ppo.get("use_sde") is not False:
        raise ValueError("use_sde must be false")
    if ppo.get("normalize_advantage") is not True:
        raise ValueError("normalize_advantage must be true")
    if float(ppo.get("log_std_init")) != 0.0:
        raise ValueError("log_std_init must stay 0.0")

    environment = _mapping(config, "environment")
    if int(environment.get("expected_max_episode_steps")) != 300:
        raise ValueError("expected_max_episode_steps must remain 300")
    _positive_number(environment.get("sensor_timeout_s"), "sensor_timeout_s")
    if environment.get("collision_monitoring") is not True:
        raise ValueError("collision_monitoring must stay enabled")
    training = _mapping(config, "training_scenario")
    if training.get("scenario_id") != "open_south":
        raise ValueError("training scenario must be open_south")
    _matching_scenario(training, _scenario_catalog()["open_south"])

    checkpointing = _mapping(config, "checkpointing")
    interval = _positive_int(checkpointing.get("interval_timesteps"), "interval_timesteps")
    evaluation = _mapping(config, "evaluation")
    if int(evaluation.get("interval_timesteps")) != interval:
        raise ValueError("evaluation and checkpoint intervals must match")
    if interval % int(ppo["n_steps"]) != 0 or interval >= total:
        raise ValueError("interval_timesteps must be a multiple of n_steps and before the final budget")
    if evaluation.get("deterministic") is not True:
        raise ValueError("validation evaluation must be deterministic")
    if int(evaluation.get("episodes_per_scenario")) != 1:
        raise ValueError("episodes_per_scenario must be 1 for this diagnostic")
    if evaluation.get("metric") != "success_rate":
        raise ValueError("validation metric must be success_rate")
    if evaluation.get("selection_rule") != SELECTION_RULE:
        raise ValueError("selection_rule must be the predeclared Phase 6G.3 rule")
    if list(evaluation.get("scenarios") or []) != list(VALIDATION_SCENARIOS):
        raise ValueError("validation scenarios must be the four Phase 6F.3 navigation routes")
    excluded = evaluation.get("excluded_scenarios")
    if (
        not isinstance(excluded, list)
        or len(excluded) != 1
        or excluded[0].get("id") != "timeout_open_south"
        or "5-step" not in str(excluded[0].get("reason"))
    ):
        raise ValueError("timeout_open_south must stay excluded")
    if list(config.get("comparability") or []) != list(COMPARABILITY_NOTES):
        raise ValueError("comparability notes must match the predeclared diagnostic notes")
    resolve_validation_scenarios(config)
    return config


def resolve_validation_scenarios(config):
    """Return the fixed navigation scenarios named by ``config``."""

    catalog = _scenario_catalog()
    selected = []
    for name in config["evaluation"]["scenarios"]:
        if name not in catalog:
            raise ValueError(f"Unknown validation scenario {name}")
        scenario = catalog[name]
        if scenario.max_episode_steps not in (None, 300):
            raise ValueError(f"{name} does not use the 300-step navigation horizon")
        selected.append(scenario)
    return tuple(selected)


def remaining_timesteps(completed, total, n_steps):
    """Return the steps still required to reach ``total``.

    Stable-Baselines3 adds this value to the loaded timestep count when
    ``reset_num_timesteps`` is false. Passing ``total`` again would train
    past the original budget.
    """

    completed = _nonnegative_int(completed, "completed")
    total = _positive_int(total, "total")
    n_steps = _positive_int(n_steps, "n_steps")
    if completed > total:
        raise ValueError("completed timesteps exceed the original budget")
    if completed % n_steps != 0 or total % n_steps != 0:
        raise ValueError("timesteps must fall on rollout boundaries")
    return total - completed


def selection_key(record):
    """Order complete validation records. Higher is better."""

    return (
        float(record["success_rate"]),
        float(record["mean_return"]),
        -int(record["timesteps"]),
    )


def verify_source_hashes(config):
    """Check source checkpoints without loading them into a training model."""

    checks = (
        (config["checkpoint"], OFFICIAL_CHECKPOINT_SHA256, "Phase 6G.1"),
        (config["bc_checkpoint"], OFFICIAL_BC_SHA256, "Phase 6F.2"),
        (config["smoke_checkpoint"], OFFICIAL_SMOKE_SHA256, "Phase 6G.2"),
    )
    hashes = {}
    for path, expected, label in checks:
        digest = file_sha256(path)
        if digest != expected:
            raise DiagnosticFailure(
                f"{label} hash mismatch for {path}. Training was not started."
            )
        hashes[str(Path(path).resolve())] = digest
    return hashes


def run_diagnostic(
    config,
    output_directory,
    environment_factory=None,
    handle_signals=True,
    session_timesteps=None,
    resume=False,
):
    """Train one Phase 6G.3 approach and return its summary."""

    config = validate_config(config)
    output = _resolve_output(output_directory)
    _reject_output(output)
    verify_source_hashes(config)
    if resume:
        if not output.is_dir():
            raise DiagnosticFailure(f"Resume directory does not exist: {output}")
        manifest = _read_json(output / "run_manifest.json")
        _check_manifest(manifest, config)
        if (output / "summary.json").is_file():
            archive = output / f"summary_before_resume_{int(_read_json(output / 'summary.json').get('trained_timesteps') or 0):08d}.json"
            if not archive.exists():
                _write_json(archive, _read_json(output / "summary.json"))
    else:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        manifest = _fresh_manifest(config, output)

    state = _new_state(config, manifest)
    state["resume"] = bool(resume)
    model = None
    env = None
    previous_handler = None
    try:
        if resume:
            _load_resume_state(output, state)
        else:
            _write_fresh_files(output, config, manifest)
        if handle_signals and threading_is_main():
            previous_handler = signal.signal(signal.SIGINT, _sigint_handler(state))
        state["session_started_at"] = time.monotonic()
        if resume:
            model = _load_resume_model(output, config, state)
            # Logs can be newer than the last saved policy. Archive against
            # the verified checkpoint, not the latest logged update.
            _reconcile_logs(output, int(model.num_timesteps))
            _load_resume_state(output, state)
            state["last_trained_timesteps"] = int(model.num_timesteps)
            state["gradient_steps"] = int(model._n_updates) * (
                int(config["ppo"]["n_steps"]) // int(config["ppo"]["batch_size"])
            )
        else:
            model, init_report = create_model(config)
            _write_json(output / "initialization_report.json", init_report)
            state["initialization_report"] = init_report
        if environment_factory is None:
            preflight_simulator()
            environment_factory = lambda: _make_jackal_env(config)  # noqa: E731
        env = _build_training_env(environment_factory(), config)
        _check_environment(env, config)
        controller = getattr(env.unwrapped, "controller", None)
        if controller is not None and hasattr(controller, "diagnostics_path"):
            controller.diagnostics_path = output / "control_attempts.jsonl"
        previous_env = model.get_env()
        model.set_env(env)
        if previous_env is not None and previous_env is not model.get_env():
            previous_env.close()
        if not resume:
            model.set_random_seed(int(config["runtime"]["seed"]))
        verbose = int(config["runtime"]["verbose"])
        logs = output / "sb3_logs" / f"session_{_timestamp()}"
        logs.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(logs), ["stdout", "csv"] if verbose else ["csv"]))
        _ensure_logs(output, state)
        if not (output / "initial_parameters.json").is_file():
            state["initial_parameters"] = _parameter_rows(model.policy)
            state["initial_buffers"] = _buffer_rows(model.policy)
            _write_json(output / "initial_parameters.json", state["initial_parameters"])
            _write_json(output / "initial_buffers.json", state["initial_buffers"])
        else:
            state["initial_parameters"] = _read_json(output / "initial_parameters.json")
            state["initial_buffers"] = _read_json(output / "initial_buffers.json")
        _append_event(output, "resume" if resume else "start", int(model.num_timesteps), None)
        _maybe_evaluate(model, env, output, config, state, int(model.num_timesteps), restoring=True)
        _save_step_checkpoint(model, output, config, state)
        callback = _DiagnosticCallback(state, config, output)
        _install_diagnostics(model, state, output, config)
        learn_steps = _session_budget(config, int(model.num_timesteps), session_timesteps)
        state["session_timesteps"] = learn_steps
        if learn_steps > 0:
            model.learn(
                total_timesteps=int(learn_steps),
                callback=callback,
                log_interval=1,
                reset_num_timesteps=not resume,
                progress_bar=False,
            )
            _flush_logger(model)
        else:
            state["resume_noop"] = True
        _finish_session(model, env, output, config, state)
    except KeyboardInterrupt as exc:
        state["status"] = "interrupted"
        state["failure_kind"] = "interrupted"
        state["error"] = _error_payload(exc)
    except Exception as exc:
        state["status"] = "failed"
        state["failure_kind"] = _failure_kind(exc)
        state["error"] = _error_payload(exc)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)
        callback_obj = state.get("callback")
        if callback_obj is not None:
            callback_obj.close_open_episode(_interruption_for(state))
        state["shutdown"] = _shutdown(model, env)
        _finalize_reports(model, output, config, state)
    return _public_summary(output, state)


def validate_runner(config, environment_factory=None):
    """Check initialization, one reset and one zero-action step; never learn.

    Uses the training wrappers and their bounded control/sensor waits. This
    is a live transport smoke check, not a fixed-scenario policy evaluation.
    No experiment directory, checkpoint or training log is written.
    """
    model = env = None
    stage = "configuration"
    result = {"status": "failed", "trained_timesteps": 0, "ppo_updates": 0}
    started = time.monotonic()
    try:
        config = validate_config(config)
        stage = "source_hashes"
        verify_source_hashes(config)
        stage = "initialization"
        model, report = create_model(config)
        result["initialization_report"] = report
        if environment_factory is None:
            stage = "preflight"
            preflight_simulator()
            environment_factory = lambda: _make_jackal_env(config)
        stage = "environment"
        env = environment_factory()
        env = _build_training_env(env, config)
        _check_environment(env, config)
        stage = "reset"
        observation, _info = env.reset(seed=int(config["runtime"]["seed"]))
        if not env.observation_space.contains(observation) or not np.all(np.isfinite(observation)):
            raise DiagnosticFailure("Reset returned an invalid or non-finite observation")
        stage = "control_and_fresh_sensors"
        observation, reward, _terminated, _truncated, info = env.step(np.zeros(2, dtype=np.float32))
        if not env.observation_space.contains(observation) or not np.all(np.isfinite(observation)) or not np.isfinite(reward):
            raise DiagnosticFailure("Zero-action step returned an invalid transition")
        # The live environment only returns these after both sensor stamps
        # advance past its pre-action boundary. Missing evidence must fail.
        for name in ("scan_stamp", "odom_stamp"):
            value = info.get(name)
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
                raise DiagnosticFailure(f"Zero-action step lacks a finite {name}")
        result.update(status="passed", observation_shape=list(np.shape(observation)),
                      scan_stamp=info.get("scan_stamp"), odom_stamp=info.get("odom_stamp"))
        stage = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = _error_payload(exc)
    finally:
        result["stage"] = stage
        result["shutdown"] = _shutdown(model, env)
        if result["shutdown"]["errors"]:
            result["status"] = "failed"
            if stage == "complete":
                result["stage"] = "shutdown"
        result["elapsed_s"] = time.monotonic() - started
    return result


def create_model(config):
    """Build the BC-initialized or scratch model without stepping Gazebo."""

    config = validate_config(config)
    verify_source_hashes(config)
    check_sb3_version()
    if config["initialization"] == "bc":
        return _load_bc_initialized_model(config)
    if config["initialization"] != "scratch":
        raise DiagnosticFailure("Unknown initialization")
    return _build_scratch_model(config)


def evaluate_policy(model, env, scenarios, seed, deterministic=True, episodes_per_scenario=1, episode_sink=None):
    """Run the learned policy on fixed scenarios. This does not call an expert planner."""

    if hasattr(env, "num_envs"):
        raise DiagnosticFailure("Evaluate the single environment, not a vector wrapper")
    episodes = []
    for scenario in scenarios:
        for trial in range(int(episodes_per_scenario)):
            episodes.append(_evaluate_episode(
                model,
                env,
                scenario,
                seed=int(seed),
                trial=trial,
                deterministic=bool(deterministic),
                episode_sink=episode_sink,
            ))
    completed = [episode for episode in episodes if episode["status"] == "completed"]
    returns = [float(episode["episode_return"]) for episode in completed]
    lengths = [int(episode["length"]) for episode in completed]
    elapsed = [float(episode["elapsed_s"]) for episode in completed if episode["elapsed_s"] is not None]
    paths = [
        float(episode["path_length_odom_m"])
        for episode in completed
        if episode["path_length_odom_m"] is not None
    ]
    count = len(completed)
    return {
        "episodes": episodes,
        "complete": bool(episodes) and len(completed) == len(episodes),
        "attempted": len(episodes),
        "completed": count,
        "successes": sum(bool(episode["success"]) for episode in completed),
        "collisions": sum(bool(episode["collision"]) for episode in completed),
        "timeouts": sum(bool(episode["timeout"]) for episode in completed),
        "success_rate": None if count == 0 else sum(bool(episode["success"]) for episode in completed) / count,
        "collision_rate": None if count == 0 else sum(bool(episode["collision"]) for episode in completed) / count,
        "timeout_rate": None if count == 0 else sum(bool(episode["timeout"]) for episode in completed) / count,
        "mean_return": None if not returns else float(np.mean(returns)),
        "median_return": _median(returns),
        "mean_length": None if not lengths else float(np.mean(lengths)),
        "mean_elapsed_s": None if not elapsed else float(np.mean(elapsed)),
        "mean_path_length_odom_m": None if not paths else float(np.mean(paths)),
        "deterministic": bool(deterministic),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output-directory")
    train.add_argument("--resume", action="store_true")
    validate = commands.add_parser(
        "validate", help="Verify initialization, one reset and one zero-action step; no PPO training",
    )
    validate.add_argument("--config", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--bc", required=True)
    compare.add_argument("--scratch", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "compare":
        from navigation.jackal_ppo_diagnostic_report import write_comparison

        summary = write_comparison(args.bc, args.scratch, args.output)
        print(json.dumps({"output_directory": summary["output_directory"]}, indent=2))
        return 0
    if args.command == "validate":
        try:
            summary = validate_runner(load_config(args.config))
        except Exception as exc:
            print(json.dumps({"status": "failed", "stage": "configuration", "error": _error_payload(exc)}, indent=2))
            return 1
        print(json.dumps(summary, indent=2))
        return 0 if summary["status"] == "passed" else 1
    config = load_config(args.config)
    if args.resume:
        if not args.output_directory:
            print("Resume requires --output-directory", file=sys.stderr)
            return 1
        output = Path(args.output_directory)
    elif args.output_directory:
        output = Path(args.output_directory)
    else:
        output = Path(config["output"]["directory"]) / f"run_{_timestamp()}"
    try:
        summary = run_diagnostic(config, output, resume=bool(args.resume))
    except (DiagnosticFailure, SmokeFailure, ValueError, FileExistsError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": summary.get("status"),
        "failure_kind": summary.get("failure_kind"),
        "trained_timesteps": summary.get("trained_timesteps"),
        "output_directory": summary.get("output_directory"),
        "final_checkpoint_sha256": summary.get("final_checkpoint_sha256"),
    }, indent=2))
    return _exit_code(summary.get("status"))


def _load_bc_initialized_model(config):
    """Load the Phase 6G.1 zip. A hash failure never reaches this function."""

    model = PPO.load(config["checkpoint"], device="cpu")
    audit_loaded_policy(model)
    _assert_hyperparameters(model, config)
    _assert_architecture(model, config)
    bc_policy, _payload = load_checkpoint(config["bc_checkpoint"], device="cpu")
    equal, differ, normalizer_differ = _pair_split(model, bc_policy)
    if differ or normalizer_differ or not equal:
        raise DiagnosticFailure(
            "Loaded actor does not match the frozen BC checkpoint. "
            "A new random model was not created."
        )
    report = {
        "method": "PPO.load",
        "training_model_source": "phase_6g1_checkpoint",
        "loaded_source_checkpoint": True,
        "constructed_new_weights": False,
        "source_checkpoint": config["checkpoint"],
        "source_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "actor_matches_bc": True,
        "normalizer_matches_bc": True,
        "actor_matches_6g1": True,
    }
    del bc_policy
    return model, report


def _build_scratch_model(config):
    """Construct a new PPO model and install normalizer buffers only."""

    _seed_everything(int(config["runtime"]["seed"]), use_cuda=False)
    architecture = _network_config(config)
    contract = JackalPolicyContractEnv()
    model = PPO(
        JackalPretrainedPPOPolicy,
        contract,
        policy_kwargs={
            "network_config": architecture,
            "log_std_init": float(config["ppo"]["log_std_init"]),
        },
        learning_rate=float(config["ppo"]["learning_rate"]),
        n_steps=int(config["ppo"]["n_steps"]),
        batch_size=int(config["ppo"]["batch_size"]),
        n_epochs=int(config["ppo"]["n_epochs"]),
        gamma=float(config["ppo"]["gamma"]),
        gae_lambda=float(config["ppo"]["gae_lambda"]),
        clip_range=float(config["ppo"]["clip_range"]),
        ent_coef=float(config["ppo"]["ent_coef"]),
        vf_coef=float(config["ppo"]["vf_coef"]),
        max_grad_norm=float(config["ppo"]["max_grad_norm"]),
        normalize_advantage=True,
        use_sde=False,
        seed=int(config["runtime"]["seed"]),
        device="cpu",
        verbose=0,
    )
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.policy.named_parameters()
    }
    bc_policy, _payload = load_checkpoint(config["bc_checkpoint"], device="cpu")
    if bc_policy.network_config() != architecture:
        raise DiagnosticFailure("BC network config does not match the architecture file")
    _install_normalizer_buffers(model.policy, bc_policy)
    for name, parameter in model.policy.named_parameters():
        if not torch.equal(parameter.detach().cpu(), before[name]):
            raise DiagnosticFailure(f"Normalizer installation changed parameter {name}")
    audit_loaded_policy(model)
    _assert_hyperparameters(model, config)
    equal, differ, normalizer_differ = _pair_split(model, bc_policy)
    if normalizer_differ:
        raise DiagnosticFailure("Scratch normalizer buffers do not match the BC checkpoint")
    if "action_net.weight" not in differ or not differ:
        raise DiagnosticFailure("Scratch actor matches the BC actor. Training was not started.")
    reference = PPO.load(config["checkpoint"], device="cpu")
    try:
        actor_match = _matching_parameter_names(model.policy, reference.policy, _is_actor_parameter)
        critic_match = _matching_parameter_names(model.policy, reference.policy, _is_critic_parameter)
        actor_names = [name for name, _parameter in model.policy.named_parameters() if _is_actor_parameter(name)]
        if len(actor_match) == len(actor_names) or "action_net.weight" in actor_match:
            raise DiagnosticFailure(
                "Scratch actor matches the Phase 6G.1 checkpoint. Training was not started."
            )
    finally:
        if reference.get_env() is not None:
            reference.get_env().close()
        del reference
    report = {
        "method": "PPO",
        "training_model_source": "framework_initialization",
        "loaded_source_checkpoint": False,
        "loaded_training_weights_from_checkpoint": False,
        "constructed_new_weights": True,
        "actor_matches_bc": False,
        "actor_weight_tensors_equal_to_bc": equal,
        "actor_weight_tensors_different_from_bc": differ,
        "normalizer_matches_bc": True,
        "actor_parameters_equal_to_6g1": actor_match,
        "critic_parameters_equal_to_6g1": critic_match,
        "log_std": [float(value) for value in model.policy.log_std.detach().cpu().tolist()],
        "seed": int(config["runtime"]["seed"]),
    }
    del bc_policy
    return model, report


def _install_normalizer_buffers(policy, bc_policy):
    for trunk in (
        policy.pi_features_extractor.trunk,
        policy.vf_features_extractor.trunk,
    ):
        with torch.no_grad():
            trunk.obs_mean.copy_(bc_policy.obs_mean.detach().to(device=trunk.obs_mean.device))
            trunk.obs_std.copy_(bc_policy.obs_std.detach().to(device=trunk.obs_std.device))
            trunk.normalizer_count.copy_(
                bc_policy.normalizer_count.detach().to(device=trunk.normalizer_count.device)
            )
        if not trunk.normalizer_is_ready:
            raise DiagnosticFailure("Observation normalizer was not installed")


def _pair_split(model, bc_policy):
    equal = []
    differ = []
    normalizer_differ = []
    for pair in _transfer_pairs(bc_policy, model.policy):
        same = torch.equal(pair["source_tensor"].detach().cpu(), pair["destination"].detach().cpu())
        if pair["group"] == "observation_normalizer":
            if not same:
                normalizer_differ.append(pair["target"])
        elif same:
            equal.append(pair["target"])
        else:
            differ.append(pair["target"])
    return equal, differ, normalizer_differ


def _matching_parameter_names(left, right, predicate):
    matches = []
    right_state = right.state_dict()
    for name, parameter in left.named_parameters():
        if not predicate(name):
            continue
        if torch.equal(parameter.detach().cpu(), right_state[name].detach().cpu()):
            matches.append(name)
    return matches


def _is_actor_parameter(name):
    return name.startswith("pi_features_extractor") or name.startswith("action_net") or name == "log_std"


def _is_critic_parameter(name):
    return name.startswith("vf_features_extractor") or name.startswith("value_net")


_GOAL_DISTANCE_ATOL_M = 1e-5


def _nonnegative_finite(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _geometric_goal_distance(env, goal_odom):
    """Euclidean odometry distance from the current pose to ``goal_odom``."""

    point = _odom_xy(env)
    if point is None or goal_odom is None:
        return None
    try:
        gx = float(goal_odom[0])
        gy = float(goal_odom[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(gx) or not math.isfinite(gy):
        return None
    return float(math.hypot(point[0] - gx, point[1] - gy))


def _agreed_goal_distance(reported, geometric):
    """Return the reported distance only when it matches the odometry geometry."""

    reported = _nonnegative_finite(reported)
    geometric = _nonnegative_finite(geometric)
    if reported is None or geometric is None:
        return None, False
    if abs(reported - geometric) > _GOAL_DISTANCE_ATOL_M:
        return None, False
    return reported, True


def _reset_goal_distance(env, reset_info):
    reported = getattr(getattr(env, "unwrapped", env), "previous_goal_distance", None)
    goal_odom = reset_info.get("goal_odom") if isinstance(reset_info, dict) else None
    return _agreed_goal_distance(reported, _geometric_goal_distance(env, goal_odom))


def _terminal_goal_distance(env, info, reset_info):
    reported = info.get("current_distance") if isinstance(info, dict) else None
    goal_odom = reset_info.get("goal_odom") if isinstance(reset_info, dict) else None
    return _agreed_goal_distance(reported, _geometric_goal_distance(env, goal_odom))


def _episode_distance_fields(points, length, status, start_distance, final_distance, start_consistent, final_consistent):
    """Measured goal distances and odometry coverage for one completed attempt.

    ``path_length_odom_m`` remains the available-point length. Efficiency is
    recorded only when reset and terminal distances agree with ``goal_odom``
    and every reset/step pose was sampled. ``start_world_pose`` is unchanged
    and is not a measured reset distance.
    """

    measured_path = path_length(points)
    odom_count = len(points)
    distances_ok = (
        start_distance is not None
        and final_distance is not None
        and start_consistent
        and final_consistent
    )
    coverage = bool(
        status == "completed"
        and length > 0
        and odom_count == length + 1
        and distances_ok
        and measured_path is not None
        and measured_path > 0
    )
    net = None
    efficiency = None
    if distances_ok:
        net = float(start_distance - final_distance)
        if coverage and measured_path is not None and math.isfinite(net):
            efficiency = float(net / measured_path)
            if not math.isfinite(efficiency):
                efficiency = None
    return {
        "path_length_odom_m": measured_path,
        "path_length_frame": None if measured_path is None else "odom",
        "start_goal_distance_m": None if not distances_ok else float(start_distance),
        "final_goal_distance_m": None if not distances_ok else float(final_distance),
        "goal_distance_frame": "odom" if distances_ok else None,
        "goal_distance_consistent": bool(distances_ok),
        "odom_point_count": odom_count,
        "odom_coverage_complete": coverage,
        "net_goal_progress_m": net if distances_ok else None,
        "path_efficiency": efficiency,
    }


def _evaluate_episode(model, env, scenario, seed, trial, deterministic, episode_sink=None):
    started = time.monotonic()
    stage = "reset"
    info = {}
    points = []
    total_reward = 0.0
    length = 0
    saturated = 0
    abs_sum = 0.0
    outcome = {
        "goal_reached": False,
        "collision_detected": False,
        "time_limit_reached": False,
        "termination_reason": None,
    }
    status = "failed"
    limit = int(getattr(env.unwrapped, "max_episode_steps", 300))
    reset_info = {}
    start_distance = None
    final_distance = None
    start_consistent = False
    final_consistent = False
    try:
        observation, info = env.reset(seed=int(seed), options={
            "start": tuple(scenario.start),
            "goal": tuple(scenario.goal),
        })
        reset_info = info if isinstance(info, dict) else {}
        start_distance, start_consistent = _reset_goal_distance(env, reset_info)
        point = _odom_xy(env)
        if point is not None:
            points.append(point)
        while True:
            stage = "predict"
            action, _state = model.predict(observation, deterministic=bool(deterministic))
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape != (2,) or not np.all(np.isfinite(action)):
                raise DiagnosticFailure("Evaluation action is not a finite 2-vector")
            if np.any(action < -1.0) or np.any(action > 1.0):
                raise DiagnosticFailure("Evaluation action left [-1, 1]")
            if np.max(np.abs(action)) >= 0.99:
                saturated += 1
            abs_sum += float(np.mean(np.abs(action)))
            stage = "step"
            observation, reward, terminated, truncated, info = env.step(action)
            stage = "transition"
            observation = np.asarray(observation, dtype=np.float64).reshape(-1)
            if observation.shape != (40,) or not np.all(np.isfinite(observation)) or not np.isfinite(reward):
                raise DiagnosticFailure("Evaluation produced a non-finite transition")
            total_reward += float(reward)
            length += 1
            point = _odom_xy(env)
            if point is not None:
                points.append(point)
            outcome = info
            if terminated or truncated:
                status = "completed"
                final_distance, final_consistent = _terminal_goal_distance(env, info, reset_info)
                break
            if length > limit:
                raise DiagnosticFailure("Evaluation episode exceeded the environment timeout")
    except BaseException as exc:
        if episode_sink is not None:
            episode_sink({
                "scenario_id": scenario.scenario_id, "trial": int(trial),
                "seed": int(seed), "status": "incomplete", "complete": False,
                "length": length, "completed_steps": length,
                "episode_return": total_reward,
                "elapsed_s": time.monotonic() - started,
                "failing_stage": stage, "error": _error_payload(exc),
            })
        raise
    finally:
        _stop_robot(env)
    elapsed = time.monotonic() - started
    completed = status == "completed"
    distance_fields = _episode_distance_fields(
        points, length, status, start_distance, final_distance, start_consistent, final_consistent,
    )
    episode = {
        "scenario_id": scenario.scenario_id,
        "category": scenario.category,
        "trial": int(trial),
        "start": [float(value) for value in scenario.start],
        "goal": [float(value) for value in scenario.goal],
        "status": status,
        "episode_return": total_reward,
        "length": length,
        "elapsed_s": elapsed,
        "success": bool(info.get("goal_reached")) if completed else False,
        "collision": bool(info.get("collision_detected")) if completed else False,
        "timeout": bool(info.get("time_limit_reached")) if completed else False,
        "termination_reason": None if not completed or info.get("termination_reason") is None else str(info.get("termination_reason")),
        **distance_fields,
        "start_world_pose": _plain(info.get("robot_world_pose")) if length == 0 else _plain(outcome.get("robot_world_pose")),
        "saturated_action_fraction": None if length == 0 else saturated / length,
        "mean_abs_action": None if length == 0 else abs_sum / length,
    }
    if episode_sink is not None:
        episode_sink(episode)
    return episode


class _DiagnosticCallback(BaseCallback):
    """Record training transitions and stop on the wall-clock bound."""

    def __init__(self, state, config, output):
        super().__init__(verbose=0)
        self.state = state
        self.config = config
        self.output = output
        self.episode_index = int(state["episode_index"])
        self.episode_open = False
        self.episode_return = 0.0
        self.episode_length = 0
        state["callback"] = self

    def _on_step(self):
        if self.state["stop_requested"]:
            self.state["stop_reason"] = "interrupted"
            return False
        if _clock_exceeded(self.state, self.config):
            self.state["stop_reason"] = "wall_clock_timeout"
            return False
        if not self.episode_open:
            self._start_episode()
        info = self.locals["infos"][0]
        reward = float(np.asarray(self.locals["rewards"]).reshape(-1)[0])
        done = bool(np.asarray(self.locals["dones"]).reshape(-1)[0])
        raw = np.asarray(self.locals["actions"][0], dtype=np.float64).reshape(2)
        clipped = np.asarray(self.locals["clipped_actions"][0], dtype=np.float64).reshape(2)
        observation = np.asarray(self.locals["new_obs"][0], dtype=np.float64).reshape(-1)
        observation_finite = bool(observation.shape == (40,) and np.all(np.isfinite(observation)))
        reward_finite = bool(np.isfinite(reward))
        action_finite = bool(np.all(np.isfinite(clipped)) and np.all(np.isfinite(raw)))
        expected = np.clip(raw, -1.0, 1.0)
        matches_clip = bool(np.allclose(clipped, expected, atol=1e-5, rtol=0.0))
        in_bounds = bool(np.all(clipped >= -1.0) and np.all(clipped <= 1.0))
        self.state["action_rows"] += 1
        self.state["abs_action_sum"] += float(np.mean(np.abs(clipped)))
        if np.max(np.abs(clipped)) >= 0.99:
            self.state["saturated_actions"] += 1
        if np.any(np.abs(raw) > 1.0):
            self.state["raw_outside_box"] += 1
            if len(self.state["clip_examples"]) < 3:
                self.state["clip_examples"].append({
                    "raw": [float(raw[0]), float(raw[1])],
                    "clipped": [float(clipped[0]), float(clipped[1])],
                })
        if not in_bounds:
            self.state["clipped_outside_box"] += 1
        if not matches_clip:
            self.state["action_contract_ok"] = False
        if not (observation_finite and reward_finite and action_finite and in_bounds and matches_clip):
            self.state["stop_reason"] = "nonfinite_transition" if not (
                observation_finite and reward_finite and action_finite
            ) else "action_contract"
            self._write_step(
                raw, clipped, reward, observation_finite, reward_finite,
                action_finite, in_bounds, done, info,
            )
            return False
        self.episode_return += reward
        self.episode_length += 1
        self._write_step(
            raw, clipped, reward, observation_finite, reward_finite,
            action_finite, in_bounds, done, info,
        )
        if done:
            self._finish_episode("completed", info, "")
        return True

    def close_open_episode(self, interruption):
        if self.episode_open and self.episode_length > 0:
            self._finish_episode("partial", {}, interruption)

    def _start_episode(self):
        self.episode_index += 1
        self.episode_open = True
        self.episode_return = 0.0
        self.episode_length = 0

    def _finish_episode(self, status, info, interruption):
        reason = info.get("termination_reason")
        record = {
            "episode_index": self.episode_index,
            "scenario_id": self.config["training_scenario"]["scenario_id"],
            "seed": int(self.config["runtime"]["seed"]),
            "status": status,
            "episode_return": self.episode_return,
            "length": self.episode_length,
            "success": bool(info.get("goal_reached")) if status == "completed" else False,
            "collision": bool(info.get("collision_detected")) if status == "completed" else False,
            "timeout": bool(info.get("time_limit_reached")) if status == "completed" else False,
            "termination_reason": "" if reason is None or status != "completed" else str(reason),
            "interruption": "" if status == "completed" else str(interruption),
            "environment_steps_completed": int(self.num_timesteps),
            "start": list(self.config["training_scenario"]["start"]),
            "goal": list(self.config["training_scenario"]["goal"]),
        }
        self.state["episodes"].append(record)
        self.state["episode_log"].write(_episode_row(record))
        self.episode_open = False
        self.state["episode_index"] = self.episode_index

    def _write_step(
        self, raw, clipped, reward, observation_finite, reward_finite,
        action_finite, in_bounds, done, info,
    ):
        self.state["step_log"].write([
            int(self.num_timesteps),
            self.episode_index,
            float(clipped[0]),
            float(clipped[1]),
            float(raw[0]),
            float(raw[1]),
            action_finite,
            in_bounds,
            reward,
            reward_finite,
            observation_finite,
            bool(info.get("goal_reached") or info.get("collision_detected")) if done else False,
            bool(info.get("time_limit_reached")) if done else False,
            "" if info.get("termination_reason") is None else str(info.get("termination_reason")),
        ])


def _install_diagnostics(model, state, output, config):
    original_train = model.train
    original_step = model.policy.optimizer.step
    expected_steps = int(config["ppo"]["n_epochs"]) * (
        int(config["ppo"]["n_steps"]) // int(config["ppo"]["batch_size"])
    )

    def traced_step(closure=None):
        finite = True
        norms = []
        for parameter in model.policy.parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                finite = False
                break
            norms.append(float(parameter.grad.detach().norm().item()))
        state["gradient_steps"] += 1
        if norms:
            state["grad_norms"].append(max(norms))
        if not finite:
            state["grad_finite"] = False
            raise OptimizationFailure("Non-finite PPO gradient")
        if closure is None:
            return original_step()
        return original_step(closure)

    def traced_train():
        before_steps = int(state["gradient_steps"])
        before_updates = int(model._n_updates)
        original_train()
        delta_steps = int(state["gradient_steps"]) - before_steps
        delta_updates = int(model._n_updates) - before_updates
        metrics = _train_metrics(model)
        log_std = model.policy.log_std.detach().cpu()
        record = {
            "timesteps": int(model.num_timesteps),
            "n_updates": int(model._n_updates),
            "n_updates_this_rollout": delta_updates,
            "optimizer_steps": int(state["gradient_steps"]),
            "optimizer_steps_this_rollout": delta_steps,
            "learning_rate": float(model.policy.optimizer.param_groups[0]["lr"]),
            "log_std": [float(value) for value in log_std.tolist()],
            "std": [float(value) for value in torch.exp(log_std).tolist()],
            "gradient_norm_max": max(state["grad_norms"][-delta_steps:]) if delta_steps and state["grad_norms"] else None,
            "metrics": metrics,
        }
        if (
            delta_steps != expected_steps
            or delta_updates != int(config["ppo"]["n_epochs"])
            or any(key not in metrics or not math.isfinite(metrics[key]) for key in REQUIRED_METRICS)
        ):
            state["nonfinite_loss"] = True
            state["updates"].append(record)
            _append_jsonl(output / "updates.jsonl", record)
            raise OptimizationFailure("PPO update was incomplete or non-finite")
        state["updates"].append(record)
        _append_jsonl(output / "updates.jsonl", record)
        state["last_trained_timesteps"] = int(model.num_timesteps)
        _write_progress(output, state, "running")
        print(
            "update timesteps={timesteps} policy_loss={policy} value_loss={value} kl={kl}".format(
                timesteps=record["timesteps"],
                policy=metrics["train/policy_gradient_loss"],
                value=metrics["train/value_loss"],
                kl=metrics["train/approx_kl"],
            ),
            flush=True,
        )
        target = int(config["runtime"]["total_timesteps"])
        interval = int(config["evaluation"]["interval_timesteps"])
        steps = int(model.num_timesteps)
        if (
            not state["stop_requested"]
            and steps % interval == 0
            and steps < target
            and steps not in state["evaluated_timesteps"]
        ):
            state["callback"].close_open_episode("evaluation_reset")
            _maybe_evaluate(model, model.get_env().envs[0], output, config, state, steps, restoring=True)
            _save_step_checkpoint(model, output, config, state)

    model.train = traced_train
    model.policy.optimizer.step = traced_step


def _maybe_evaluate(model, env, output, config, state, timesteps, restoring):
    timesteps = int(timesteps)
    if timesteps in state["evaluated_timesteps"]:
        if restoring:
            _restore_training_episode(model, config)
        return
    interval = int(config["evaluation"]["interval_timesteps"])
    target = int(config["runtime"]["total_timesteps"])
    due = timesteps == 0 or timesteps == target or timesteps % interval == 0
    if not due:
        return
    if _clock_exceeded(state, config):
        state["stop_reason"] = "wall_clock_timeout"
        return
    started = time.monotonic()
    print(f"evaluation timesteps={timesteps}", flush=True)
    def persist_episode(episode):
        _append_jsonl(output / "evaluation_episodes.jsonl", {
            **episode, "timesteps": timesteps, "experiment": config["experiment"],
            "seed": int(config["runtime"]["seed"]),
        })

    try:
        result = evaluate_policy(
            model, env, resolve_validation_scenarios(config),
            seed=int(config["runtime"]["seed"]), deterministic=True,
            episodes_per_scenario=1, episode_sink=persist_episode,
        )
    finally:
        elapsed = time.monotonic() - started
        state["session_eval_s"] += elapsed
    result.update({
        "timesteps": timesteps,
        "evaluation_wall_time_s": elapsed,
        "phase": PHASE,
        "experiment": config["experiment"],
    })
    state["evaluations"].append(result)
    state["evaluated_timesteps"].add(timesteps)
    _write_json(output / "evaluations.json", {"records": state["evaluations"]})
    if result["complete"]:
        _update_selection(model, output, config, state, result)
    print(
        "evaluation timesteps={timesteps} success_rate={rate} mean_return={ret}".format(
            timesteps=timesteps,
            rate=result["success_rate"],
            ret=result["mean_return"],
        ),
        flush=True,
    )
    if restoring:
        _restore_training_episode(model, config)


def _restore_training_episode(model, config):
    vec_env = model.get_env()
    if vec_env is None:
        return
    observation = vec_env.reset()
    model._last_obs = observation
    model._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    callback = None
    # The next training step starts a new episode. The reset above consumed
    # the evaluation pose and returned the robot to open_south.
    del config


def _update_selection(model, output, config, state, result):
    if result.get("success_rate") is None or result.get("mean_return") is None:
        return
    current = state.get("selection")
    if current is not None and selection_key(result) <= selection_key(current):
        return
    checkpoint = _checkpoint_path(output, int(result["timesteps"]))
    if not checkpoint.is_file():
        _save_step_checkpoint(model, output, config, state)
    if not checkpoint.is_file():
        return
    selected = output / "selected_ppo.zip"
    _copy_file(checkpoint, selected)
    choice = {
        "timesteps": int(result["timesteps"]),
        "success_rate": result["success_rate"],
        "mean_return": result["mean_return"],
        "median_return": result["median_return"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "selected_ppo": str(selected),
        "selected_ppo_sha256": file_sha256(selected),
        "rule": SELECTION_RULE,
        "metric": "success_rate",
    }
    state["selection"] = choice
    _write_json(output / "selection.json", choice)


def _save_step_checkpoint(model, output, config, state):
    timesteps = int(state["last_trained_timesteps"])
    path = _checkpoint_path(output, timesteps)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    _save_checkpoint(model, path)
    digest = file_sha256(path)
    sidecar = {
        "phase": PHASE,
        "experiment": config["experiment"],
        "initialization": config["initialization"],
        "timesteps": timesteps,
        "n_updates": int(model._n_updates),
        "sha256": digest,
        "resume_allowed": True,
        "consistent_rollout_boundary": True,
    }
    _write_json(path.with_suffix(".json"), sidecar)
    index_path = path.parent / "index.json"
    index = _read_json(index_path) if index_path.is_file() else {"files": []}
    index["files"].append({
        "path": str(path),
        "sha256": digest,
        "timesteps": timesteps,
        "experiment": config["experiment"],
        "initialization": config["initialization"],
    })
    _write_json(index_path, index)
    state["checkpoints"].append(sidecar)
    return path


def _load_resume_model(output, config, state):
    checkpoint = _latest_checkpoint(output, config)
    sidecar = _read_json(checkpoint.with_suffix(".json"))
    if sidecar.get("resume_allowed") is not True:
        raise DiagnosticFailure("Checkpoint is not a consistent resume point")
    if sidecar.get("experiment") != config["experiment"] or sidecar.get("initialization") != config["initialization"]:
        raise DiagnosticFailure("Checkpoint belongs to a different initialization strategy")
    digest = file_sha256(checkpoint)
    if digest != sidecar.get("sha256"):
        raise DiagnosticFailure("Checkpoint hash does not match its sidecar")
    model = PPO.load(str(checkpoint), device="cpu")
    if not isinstance(model.policy, JackalPretrainedPPOPolicy):
        raise DiagnosticFailure("Resume checkpoint policy class changed")
    validate_jackal_policy_spaces(model.observation_space, model.action_space)
    _assert_hyperparameters(model, config)
    _assert_architecture(model, config)
    if int(model.num_timesteps) != int(sidecar["timesteps"]) or int(model._n_updates) != int(sidecar["n_updates"]):
        raise DiagnosticFailure("Resume checkpoint timesteps do not match its sidecar")
    if int(model.num_timesteps) % int(config["ppo"]["n_steps"]) != 0:
        raise DiagnosticFailure("Resume checkpoint is not on a rollout boundary")
    if not all(torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
        raise DiagnosticFailure("Resume checkpoint parameters are not finite")
    state["last_trained_timesteps"] = int(model.num_timesteps)
    state["resumed_from"] = str(checkpoint)
    state["resumed_from_sha256"] = digest
    _append_event(output, "loaded_checkpoint", int(model.num_timesteps), digest)
    return model


def _latest_checkpoint(output, config):
    candidates = []
    for path in (output / "checkpoints").glob("step_*.zip"):
        sidecar_path = path.with_suffix(".json")
        if not sidecar_path.is_file():
            continue
        sidecar = _read_json(sidecar_path)
        if sidecar.get("experiment") != config["experiment"] or sidecar.get("resume_allowed") is not True:
            continue
        candidates.append((int(sidecar["timesteps"]), path))
    if not candidates:
        raise DiagnosticFailure("No consistent checkpoint is available for resume")
    _timesteps, path = max(candidates)
    return path


def _finish_session(model, env, output, config, state):
    if state.get("stop_reason"):
        return
    target = int(config["runtime"]["total_timesteps"])
    trained = int(state["last_trained_timesteps"])
    if model is None:
        return
    if int(model.num_timesteps) != trained:
        state["uncommitted_steps"] = int(model.num_timesteps) - trained
        state["status"] = "incomplete"
        state["stop_reason"] = state.get("stop_reason") or "uncommitted_rollout"
        return
    callback = state.get("callback")
    if callback is not None:
        reason = "training_budget" if trained == target else "session_limit"
        callback.close_open_episode(reason)
    if trained == target:
        _maybe_evaluate(model, env, output, config, state, trained, restoring=False)
        _save_step_checkpoint(model, output, config, state)
        final_path = output / "final_ppo.zip"
        step_path = _checkpoint_path(output, trained)
        if not final_path.exists():
            _copy_file(step_path, final_path)
        state["final_checkpoint"] = str(final_path)
        state["final_checkpoint_sha256"] = file_sha256(final_path)
        state["reload"] = _reload_checkpoint(final_path, model, trained)
    elif trained < target:
        _save_step_checkpoint(model, output, config, state)
        state["status"] = "incomplete"
        state["stop_reason"] = "session_limit"


def _finalize_reports(model, output, config, state):
    trained = int(state.get("last_trained_timesteps") or 0)
    if model is not None and int(model.num_timesteps) > trained:
        state["uncommitted_steps"] = int(model.num_timesteps) - trained
    if output.is_dir() and state.get("episodes") is not None:
        _write_json(output / "episodes.json", state["episodes"])
    if output.is_dir():
        _reconcile_logs(output, trained)
        _reload_committed_episodes(output, state)
    if model is not None:
        try:
            state["final_parameters"] = _parameter_rows(model.policy)
            state["final_buffers"] = _buffer_rows(model.policy)
            _write_json(output / f"parameters_step_{trained:08d}.json", state["final_parameters"])
        except Exception as exc:
            state["parameter_error"] = f"{type(exc).__name__}: {exc}"
    for log_name in ("step_log", "episode_log"):
        log = state.get(log_name)
        if log is not None:
            log.close()
    started = state.get("session_started_at")
    if started is not None:
        elapsed = time.monotonic() - started
        state["session_training_s"] = max(0.0, elapsed - float(state["session_eval_s"]))
    state["training_wall_time_s"] = float(state["accumulated_training_s"]) + float(state.get("session_training_s") or 0.0)
    state["evaluation_wall_time_s"] = float(state["accumulated_eval_s"]) + float(state["session_eval_s"])
    if state.get("protected_before") is None and (output / "protected_hashes_before.json").is_file():
        state["protected_before"] = _read_json(output / "protected_hashes_before.json")
    protected_after = _protected_hashes()
    state["protected_after"] = protected_after
    state["protected_unchanged"] = (
        state.get("protected_before") is None or protected_after == state.get("protected_before")
    )
    state["optimization"] = _optimization_report(model, state, config)
    state["action_summary"] = _action_summary(output, state)
    if not state.get("protected_unchanged", False):
        state["status"] = "failed"
        state["failure_kind"] = "protected_artifact_changed"
    elif state["status"] == "running":
        if state.get("stop_reason"):
            state["status"] = _status_for_stop(state["stop_reason"])
            if state["stop_reason"] in ("nonfinite_transition", "action_contract") and not state.get("failure_kind"):
                state["failure_kind"] = state["stop_reason"]
        else:
            state["status"] = _completed_status(state, config)
    if state["status"] == "completed" and not state.get("reload", {}).get("passed", False):
        state["status"] = "failed"
        state["failure_kind"] = "reload"
    if state["status"] == "completed" and not state["optimization"]["passed"]:
        state["status"] = "failed"
        state["failure_kind"] = "optimization"
    _write_reports(output, config, state)


def _completed_status(state, config):
    target = int(config["runtime"]["total_timesteps"])
    if int(state.get("last_trained_timesteps") or 0) != target:
        return "incomplete"
    if state.get("stop_reason"):
        return _status_for_stop(state["stop_reason"])
    if 0 not in state["evaluated_timesteps"] or target not in state["evaluated_timesteps"]:
        state["failure_kind"] = "evaluation"
        return "failed"
    return "completed"


def _optimization_report(model, state, config):
    comparison = _compare_parameters(
        state.get("initial_parameters") or [],
        state.get("final_parameters") or [],
    )
    buffers = _compare_parameters(
        state.get("initial_buffers") or [],
        state.get("final_buffers") or [],
    )
    target = int(config["runtime"]["total_timesteps"])
    n_steps = int(config["ppo"]["n_steps"])
    expected_rollouts = target // n_steps
    expected_updates = expected_rollouts * int(config["ppo"]["n_epochs"])
    expected_gradients = expected_updates * (n_steps // int(config["ppo"]["batch_size"]))
    updates = _read_jsonl(Path(state["output"]) / "updates.jsonl") if state.get("output") else state["updates"]
    losses_finite = bool(updates) and all(
        key in update["metrics"] and math.isfinite(update["metrics"][key])
        for update in updates
        for key in REQUIRED_METRICS
    )
    reached = int(state.get("last_trained_timesteps") or 0) == target
    counts_match = (
        reached
        and len(updates) == expected_rollouts
        and (model is None or int(model._n_updates) == expected_updates)
        and int(updates[-1]["optimizer_steps"]) == expected_gradients
    )
    normalizers_unchanged = bool(buffers["changed"] == 0 and buffers["count"] > 0)
    changed = int(comparison["changed"]) > 0
    passed = bool(
        counts_match
        and changed
        and comparison["critic_requires_grad"]
        and losses_finite
        and state["grad_finite"]
        and state["action_contract_ok"]
        and normalizers_unchanged
        and int(state["clipped_outside_box"]) == 0
        and not state["nonfinite_loss"]
    )
    return {
        "passed": passed,
        "trained_timesteps": int(state.get("last_trained_timesteps") or 0),
        "requested_timesteps": target,
        "rollouts_completed": len(updates),
        "expected_rollouts": expected_rollouts,
        "n_updates": None if model is None else int(model._n_updates),
        "expected_n_updates": expected_updates,
        "gradient_steps": None if not updates else int(updates[-1]["optimizer_steps"]),
        "expected_gradient_steps": expected_gradients,
        "losses_finite": losses_finite,
        "gradients_finite": bool(state["grad_finite"]),
        "actor_changed": bool(comparison["actor_changed"]),
        "critic_changed": bool(comparison["critic_changed"]),
        "log_std_changed": bool(comparison["log_std_changed"]),
        "critic_requires_grad": bool(comparison["critic_requires_grad"]),
        "normalizer_buffers_unchanged": normalizers_unchanged,
        "action_contract_ok": bool(state["action_contract_ok"]),
        "definitions": {
            "trained_timesteps": "Environment steps incorporated by completed PPO updates and stored in a rollout-aligned checkpoint.",
            "rollouts_completed": "Calls to PPO.train, one per n_steps environment steps.",
            "n_updates": "Stable-Baselines3 _n_updates. It increases by n_epochs once per rollout.",
            "gradient_steps": "Optimizer steps. Each rollout performs n_epochs * (n_steps / batch_size) steps.",
            "partial_episode": "An episode that was still open at an evaluation reset, session limit, interruption, or training boundary. It is not a timeout.",
            "timeout": "A completed episode whose environment reported time_limit_reached.",
            "uncommitted_steps": "Steps executed after the last completed update. They are archived and are not in the restored policy.",
        },
        "parameters": comparison,
        "buffers": buffers,
    }


def _write_reports(output, config, state):
    episodes = state.get("episodes") or []
    completed = [episode for episode in episodes if episode["status"] == "completed"]
    partial = [episode for episode in episodes if episode["status"] == "partial"]
    optimization = state.get("optimization") or {}
    evaluations = state.get("evaluations") or []
    summary = {
        "phase": PHASE,
        "experiment": config["experiment"],
        "initialization": config["initialization"],
        "status": state["status"],
        "failure_kind": state.get("failure_kind"),
        "stop_reason": state.get("stop_reason") or None,
        "resumed": bool(state.get("resume")),
        "resumed_from": state.get("resumed_from"),
        "resumed_from_sha256": state.get("resumed_from_sha256"),
        "resume_noop": bool(state.get("resume_noop")),
        "resume_limitations": RESUME_LIMITATIONS,
        "seed": int(config["runtime"]["seed"]),
        "scenario_id": config["training_scenario"]["scenario_id"],
        "start": list(config["training_scenario"]["start"]),
        "goal": list(config["training_scenario"]["goal"]),
        "trained_timesteps": int(state.get("last_trained_timesteps") or 0),
        "requested_timesteps": int(config["runtime"]["total_timesteps"]),
        "nominal_timesteps": 50000,
        "uncommitted_steps": int(state.get("uncommitted_steps") or 0),
        "orphaned_steps": int(state.get("orphaned_steps") or 0),
        "orphaned_episodes": int(state.get("orphaned_episodes") or 0),
        "rollouts_completed": optimization.get("rollouts_completed"),
        "n_updates": optimization.get("n_updates"),
        "gradient_steps": optimization.get("gradient_steps"),
        "n_epochs": int(config["ppo"]["n_epochs"]),
        "n_steps": int(config["ppo"]["n_steps"]),
        "batch_size": int(config["ppo"]["batch_size"]),
        "n_envs": 1,
        "training_wall_time_s": state.get("training_wall_time_s"),
        "evaluation_wall_time_s": state.get("evaluation_wall_time_s"),
        "completed_episodes": len(completed),
        "partial_episodes": len(partial),
        "successes": sum(bool(episode["success"]) for episode in completed),
        "collisions": sum(bool(episode["collision"]) for episode in completed),
        "timeouts": sum(bool(episode["timeout"]) for episode in completed),
        "other_completed_outcomes": sum(
            episode["status"] == "completed" and not episode["success"] and not episode["collision"] and not episode["timeout"]
            for episode in completed
        ),
        "navigation_success_is_not_required": True,
        "goal_success_means_navigation_success": True,
        "initial_checkpoint": config["checkpoint"] if config["initialization"] == "bc" else state.get("initial_checkpoint"),
        "initial_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256 if config["initialization"] == "bc" else state.get("initial_checkpoint_sha256"),
        "bc_checkpoint": config["bc_checkpoint"],
        "bc_checkpoint_sha256": OFFICIAL_BC_SHA256,
        "smoke_checkpoint_sha256": OFFICIAL_SMOKE_SHA256,
        "final_checkpoint": state.get("final_checkpoint"),
        "final_checkpoint_sha256": state.get("final_checkpoint_sha256"),
        "selection": state.get("selection"),
        "initialization_report": state.get("initialization_report"),
        "protected_artifacts_unchanged": bool(state.get("protected_unchanged")),
        "optimization_passed": bool(optimization.get("passed")),
        "actor_changed": optimization.get("actor_changed"),
        "critic_changed": optimization.get("critic_changed"),
        "log_std_changed": optimization.get("log_std_changed"),
        "losses_finite": optimization.get("losses_finite"),
        "gradients_finite": optimization.get("gradients_finite"),
        "normalizer_buffers_unchanged": optimization.get("normalizer_buffers_unchanged"),
        "action_summary": state.get("action_summary"),
        "clip_examples": state.get("clip_examples"),
        "evaluations_recorded": len(evaluations),
        "initial_validation_success_rate": _evaluation_rate(evaluations, 0),
        "final_validation_success_rate": _evaluation_rate(
            evaluations,
            -1 if state.get("last_trained_timesteps") is None else int(state.get("last_trained_timesteps")),
        ),
        "reload": state.get("reload"),
        "shutdown": state.get("shutdown"),
        "error": state.get("error"),
        "output_directory": str(output),
        "comparability": list(COMPARABILITY_NOTES),
        "vectorized_environment": (
            "Stable-Baselines3 DummyVecEnv around one JackalGazeboEnv. "
            "No second command publisher is created."
        ),
    }
    if summary["initialization"] == "scratch":
        step0 = _checkpoint_path(output, 0)
        if step0.is_file():
            summary["initial_checkpoint"] = str(step0)
            summary["initial_checkpoint_sha256"] = file_sha256(step0)
    _write_json(output / "summary.json", summary)
    _write_json(output / "episodes.json", episodes)
    if optimization:
        stored = dict(optimization)
        stored["updates"] = _read_jsonl(output / "updates.jsonl")
        _write_json(output / "optimization.json", stored)
    if state.get("final_parameters") is not None and summary["status"] == "completed":
        final_parameters = output / "final_parameters.json"
        if not final_parameters.exists():
            _write_json(final_parameters, state["final_parameters"])
    if state.get("reload") is not None:
        _write_json(output / "checkpoint_reload.json", state["reload"])
    _write_json(output / "protected_hashes_after.json", state.get("protected_after") or {})
    if state.get("error") is not None:
        _write_json(output / "failure.json", state["error"])
    _write_progress(output, state, summary["status"])


def _evaluation_rate(evaluations, timesteps):
    for record in evaluations:
        if int(record.get("timesteps", -1)) == int(timesteps):
            return record.get("success_rate")
    return None


def _reconcile_logs(output, trained_timesteps):
    """Archive rows collected after the last completed update."""

    trained_timesteps = int(trained_timesteps)
    step_path = output / "steps.csv"
    if step_path.is_file():
        with step_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        keep = [row for row in rows if int(row["step_index"]) <= trained_timesteps]
        orphan = [row for row in rows if int(row["step_index"]) > trained_timesteps]
        if orphan:
            _append_csv_rows(output / "orphaned_steps.csv", STEP_COLUMNS, orphan)
            _rewrite_csv(step_path, STEP_COLUMNS, keep)
    episode_path = output / "episodes.json"
    episodes = _read_json(episode_path) if episode_path.is_file() else []
    if isinstance(episodes, list):
        keep_episodes = [
            episode for episode in episodes
            if int(episode["environment_steps_completed"]) <= trained_timesteps
        ]
        orphan_episodes = [
            episode for episode in episodes
            if int(episode["environment_steps_completed"]) > trained_timesteps
        ]
        if orphan_episodes:
            existing = _read_json(output / "orphaned_episodes.json") if (output / "orphaned_episodes.json").is_file() else []
            _write_json(output / "orphaned_episodes.json", list(existing) + orphan_episodes)
            _write_json(episode_path, keep_episodes)
            _rewrite_csv(output / "episodes.csv", EPISODE_COLUMNS, [_episode_row(episode) for episode in keep_episodes])
    updates = _read_jsonl(output / "updates.jsonl")
    if updates:
        keep_updates = [update for update in updates if int(update["timesteps"]) <= trained_timesteps]
        orphan_updates = [update for update in updates if int(update["timesteps"]) > trained_timesteps]
        if orphan_updates:
            for update in orphan_updates:
                _append_jsonl(output / "orphaned_updates.jsonl", update)
            _rewrite_jsonl(output / "updates.jsonl", keep_updates)
    evaluation_path = output / "evaluations.json"
    if evaluation_path.is_file():
        payload = _read_json(evaluation_path)
        records = list(payload.get("records") or [])
        keep_records = []
        orphan_records = []
        for record in records:
            timesteps = int(record["timesteps"])
            checkpoint = _checkpoint_path(output, timesteps)
            if timesteps <= trained_timesteps and (timesteps == 0 or checkpoint.is_file() or timesteps == trained_timesteps):
                keep_records.append(record)
            else:
                orphan_records.append(record)
        if orphan_records:
            _write_json(output / "orphaned_evaluations.json", {"records": orphan_records})
            _write_json(evaluation_path, {"records": keep_records})


def _reload_committed_episodes(output, state):
    episodes = _read_json(output / "episodes.json") if (output / "episodes.json").is_file() else []
    state["episodes"] = list(episodes or [])
    if state["episodes"]:
        state["episode_index"] = max(int(episode["episode_index"]) for episode in state["episodes"])
    else:
        state["episode_index"] = -1
    orphan_steps = output / "orphaned_steps.csv"
    state["orphaned_steps"] = 0
    if orphan_steps.is_file():
        with orphan_steps.open("r", encoding="utf-8", newline="") as handle:
            state["orphaned_steps"] = sum(1 for _row in csv.DictReader(handle))
    orphan_episodes = _read_json(output / "orphaned_episodes.json") if (output / "orphaned_episodes.json").is_file() else []
    state["orphaned_episodes"] = len(orphan_episodes or [])
    evaluations = _read_json(output / "evaluations.json") if (output / "evaluations.json").is_file() else {"records": []}
    state["evaluations"] = list(evaluations.get("records") or [])
    state["evaluated_timesteps"] = {int(record["timesteps"]) for record in state["evaluations"]}
    if (output / "selection.json").is_file():
        state["selection"] = _read_json(output / "selection.json")


def _load_resume_state(output, state):
    progress = _read_json(output / "progress.json") if (output / "progress.json").is_file() else {}
    state["accumulated_training_s"] = float(progress.get("training_wall_time_s") or 0.0)
    state["accumulated_eval_s"] = float(progress.get("evaluation_wall_time_s") or 0.0)
    state["gradient_steps"] = int(progress.get("gradient_steps") or 0)
    state["last_trained_timesteps"] = int(progress.get("trained_timesteps") or 0)
    updates = _read_jsonl(output / "updates.jsonl")
    if updates:
        state["gradient_steps"] = int(updates[-1]["optimizer_steps"])
        state["last_trained_timesteps"] = int(updates[-1]["timesteps"])
    _reload_committed_episodes(output, state)
    if (output / "initialization_report.json").is_file():
        state["initialization_report"] = _read_json(output / "initialization_report.json")
    state["protected_before"] = _read_json(output / "protected_hashes_before.json")
    state["output"] = str(output)


def _fresh_manifest(config, output):
    manifest = {
        "phase": PHASE,
        "experiment": config["experiment"],
        "initialization": config["initialization"],
        "seed": int(config["runtime"]["seed"]),
        "total_timesteps": int(config["runtime"]["total_timesteps"]),
        "n_epochs": int(config["ppo"]["n_epochs"]),
        "n_steps": int(config["ppo"]["n_steps"]),
        "batch_size": int(config["ppo"]["batch_size"]),
        "learning_rate": float(config["ppo"]["learning_rate"]),
        "config_sha256": config_sha256(config),
        "source_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "bc_checkpoint_sha256": OFFICIAL_BC_SHA256,
        "smoke_checkpoint_sha256": OFFICIAL_SMOKE_SHA256,
        "output_directory": str(output),
        "resume_limitations": RESUME_LIMITATIONS,
    }
    return manifest


def _write_fresh_files(output, config, manifest):
    protected = _protected_hashes()
    _write_json(output / "run_manifest.json", manifest)
    _write_json(output / "config.json", config)
    _write_json(output / "comparability.json", {"notes": list(COMPARABILITY_NOTES)})
    _write_json(output / "framework_versions.json", _versions())
    _write_json(output / "protected_hashes_before.json", protected)
    _write_text(output / "commands.txt", _commands())
    _write_json(output / "resume_limitations.json", {"text": RESUME_LIMITATIONS})
    transport_sources = _JACKAL_SRC / "jackal_rl_navigation"
    _write_json(output / "infrastructure.json", {
        "backend": "persistent_ignition_transport11",
        "ambiguous_multi_step": "fail closed without resending; no automatic recovery",
        "sources_sha256": {
            name: file_sha256(transport_sources / name)
            for name in ("gazebo_controller.py", "gazebo_transport.py", "gazebo_transport_client.cpp", "gazebo_env.py")
        },
    })


def _check_manifest(manifest, config):
    if manifest.get("phase") != PHASE or manifest.get("experiment") != config["experiment"]:
        raise DiagnosticFailure("Run directory belongs to a different experiment")
    if manifest.get("initialization") != config["initialization"]:
        raise DiagnosticFailure("Run directory uses a different initialization strategy")
    if manifest.get("config_sha256") != config_sha256(config):
        raise DiagnosticFailure("Resume configuration does not match the original run")
    if int(manifest.get("total_timesteps")) != int(config["runtime"]["total_timesteps"]):
        raise DiagnosticFailure("Resume would change the original timestep budget")


def config_sha256(config):
    payload = json.dumps(_plain(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _new_state(config, manifest):
    return {
        "status": "running",
        "failure_kind": None,
        "stop_requested": False,
        "sigint_count": 0,
        "stop_reason": "",
        "updates": [],
        "gradient_steps": 0,
        "grad_finite": True,
        "grad_norms": [],
        "nonfinite_loss": False,
        "action_contract_ok": True,
        "action_rows": 0,
        "raw_outside_box": 0,
        "clipped_outside_box": 0,
        "saturated_actions": 0,
        "abs_action_sum": 0.0,
        "clip_examples": [],
        "episodes": [],
        "episode_index": -1,
        "evaluations": [],
        "evaluated_timesteps": set(),
        "selection": None,
        "checkpoints": [],
        "last_trained_timesteps": 0,
        "resume": False,
        "resume_noop": False,
        "accumulated_training_s": 0.0,
        "accumulated_eval_s": 0.0,
        "session_eval_s": 0.0,
        "session_training_s": 0.0,
        "uncommitted_steps": 0,
        "orphaned_steps": 0,
        "orphaned_episodes": 0,
        "callback": None,
        "output": manifest.get("output_directory"),
        "protected_before": None,
        "initialization_report": None,
    }


def _ensure_logs(output, state):
    state["protected_before"] = state.get("protected_before") or _read_json(output / "protected_hashes_before.json")
    state["output"] = str(output)
    state["step_log"] = _CsvLog(output / "steps.csv", STEP_COLUMNS, append=True)
    state["episode_log"] = _CsvLog(output / "episodes.csv", EPISODE_COLUMNS, append=True)
    _reload_committed_episodes(output, state)


def _session_budget(config, completed, session_timesteps):
    remaining = remaining_timesteps(
        completed,
        int(config["runtime"]["total_timesteps"]),
        int(config["ppo"]["n_steps"]),
    )
    if session_timesteps is None:
        return remaining
    session_timesteps = _positive_int(session_timesteps, "session_timesteps")
    if session_timesteps % int(config["ppo"]["n_steps"]) != 0:
        raise ValueError("session_timesteps must be a multiple of n_steps")
    if session_timesteps > remaining:
        raise ValueError("session_timesteps exceeds the remaining budget")
    return session_timesteps


def _build_training_env(env, config):
    scenario = config["training_scenario"]
    env = FixedScenarioWrapper(
        env,
        scenario["start"],
        scenario["goal"],
        scenario["scenario_id"],
        config["runtime"]["seed"],
    )
    env = TimeLimitInfoWrapper(env)
    return StopOnFailureWrapper(env)


def _assert_hyperparameters(model, config):
    ppo = config["ppo"]
    if int(model.n_epochs) != int(ppo["n_epochs"]):
        raise DiagnosticFailure(
            f"Model n_epochs is {model.n_epochs}; Phase 6G.3 requires {ppo['n_epochs']} "
            "and does not apply the Phase 6G.2 override."
        )
    checks = {
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "learning_rate": _schedule_value(model.learning_rate),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "clip_range": _schedule_value(model.clip_range),
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
    }
    for name, actual in checks.items():
        if not _same_number(actual, ppo[name]):
            raise DiagnosticFailure(f"Model {name}={actual} does not match the diagnostic configuration")
    if model.use_sde or not model.normalize_advantage:
        raise DiagnosticFailure("Model exploration settings do not match the diagnostic configuration")


def _assert_architecture(model, config):
    if model.policy.network_config != _network_config(config):
        raise DiagnosticFailure("Policy architecture does not match the Phase 6F.2 network config")


def _network_config(config):
    payload = _read_json(config["architecture"])
    network = payload.get("network_config")
    if not isinstance(network, dict):
        raise DiagnosticFailure("Architecture file has no network_config")
    return network


def _reload_checkpoint(path, model, timesteps):
    report = {"passed": False, "in_process": False, "fresh_process": False}
    loaded = None
    try:
        loaded = PPO.load(str(path), device="cpu")
        if not isinstance(loaded.policy, JackalPretrainedPPOPolicy):
            raise DiagnosticFailure("Reloaded policy class changed")
        validate_jackal_policy_spaces(loaded.observation_space, loaded.action_space)
        if int(loaded.num_timesteps) != int(timesteps) or int(loaded._n_updates) != int(model._n_updates):
            raise DiagnosticFailure("Reloaded timestep or update count changed")
        for name, parameter in model.policy.state_dict().items():
            restored = loaded.policy.state_dict()[name]
            if not torch.equal(parameter.detach().cpu(), restored.detach().cpu()):
                raise DiagnosticFailure(f"Reloaded tensor changed: {name}")
        if not all(torch.isfinite(parameter).all() for parameter in loaded.policy.parameters()):
            raise DiagnosticFailure("Reloaded parameters are not finite")
        report["in_process"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    finally:
        if loaded is not None and loaded.get_env() is not None:
            loaded.get_env().close()
    report["fresh_process"] = _fresh_reload(path, timesteps)
    report["passed"] = bool(report["in_process"] and report["fresh_process"].get("loaded"))
    return report


def _fresh_reload(path, timesteps):
    code = """
import os, sys
from pathlib import Path
root = Path(%(root)r)
src = root / "jackal_rl_ws" / "src" / "jackal_rl_navigation"
sys.path.insert(0, str(root))
sys.path.insert(0, str(src))
from stable_baselines3 import PPO
from agents.jackal_pretrained_ppo_policy import JackalPretrainedPPOPolicy
from navigation.expert_demonstration_schema import validate_jackal_policy_spaces
import torch
model = PPO.load(%(path)r, device="cpu")
validate_jackal_policy_spaces(model.observation_space, model.action_space)
assert isinstance(model.policy, JackalPretrainedPPOPolicy)
assert int(model.num_timesteps) == %(timesteps)d
assert model.policy.squash_output is False
assert tuple(model.policy.log_std.shape) == (2,)
assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
print("RELOAD_OK", int(model.num_timesteps))
""" % {"root": str(_ROOT), "path": str(path), "timesteps": int(timesteps)}
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(_ROOT), str(_ROOT / "jackal_rl_ws" / "src" / "jackal_rl_navigation"), env.get("PYTHONPATH", ""))
        if part
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0 or "RELOAD_OK" not in result.stdout:
        detail = (result.stderr or result.stdout or "").strip()
        return {"loaded": False, "error": detail[-2000:]}
    return {"loaded": True, "timesteps": int(timesteps)}


def _action_summary(output, state):
    path = output / "steps.csv"
    rows = 0
    saturated = 0
    raw_outside = 0
    clipped_outside = 0
    abs_sum = 0.0
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                clipped = np.array([float(row["action_linear"]), float(row["action_angular"])])
                raw = np.array([float(row["raw_action_linear"]), float(row["raw_action_angular"])])
                abs_sum += float(np.mean(np.abs(clipped)))
                if np.max(np.abs(clipped)) >= 0.99:
                    saturated += 1
                if np.any(np.abs(raw) > 1.0):
                    raw_outside += 1
                if np.any(clipped < -1.0) or np.any(clipped > 1.0):
                    clipped_outside += 1
    return {
        "steps": rows,
        "saturated_action_fraction": None if rows == 0 else saturated / rows,
        "mean_abs_action": None if rows == 0 else abs_sum / rows,
        "raw_actions_outside_box": raw_outside,
        "clipped_actions_outside_box": clipped_outside,
        "running_clipped_outside_box": int(state["clipped_outside_box"]),
    }


def _protected_hashes():
    files = [
        OFFICIAL_BC_CHECKPOINT,
        OFFICIAL_BC_CHECKPOINT.with_name("final.pt"),
        OFFICIAL_BC_CHECKPOINT.with_name("preprocessing.json"),
        OFFICIAL_ARCHITECTURE,
        OFFICIAL_CHECKPOINT,
        OFFICIAL_SMOKE_CHECKPOINT,
    ]
    roots = (OFFICIAL_CHECKPOINT.parent, OFFICIAL_SMOKE_CHECKPOINT.parent)
    for root in roots:
        files.extend(path for path in root.rglob("*") if path.is_file())
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return {str(path): file_sha256(path) for path in unique}


def _reject_output(output):
    for root in FORBIDDEN_OUTPUT_ROOTS:
        resolved = root.resolve()
        if output == resolved or resolved in output.parents:
            raise DiagnosticFailure(f"Refusing to write inside {resolved}")
        if output in resolved.parents:
            raise DiagnosticFailure(f"Output directory would contain protected artifacts: {resolved}")


def _clock_exceeded(state, config):
    started = state.get("session_started_at")
    if started is None:
        return False
    elapsed = time.monotonic() - started
    total = float(state["accumulated_training_s"]) + float(state["accumulated_eval_s"]) + elapsed
    return total > float(config["runtime"]["wall_clock_timeout_s"])


def _odom_xy(env):
    robot = _robot(env)
    if robot is None:
        return None
    odom = getattr(robot, "odom", None)
    if odom is None:
        return None
    position = odom.pose.pose.position
    point = (float(position.x), float(position.y))
    if not np.all(np.isfinite(point)):
        return None
    return point


def _scenario_catalog():
    return {scenario.scenario_id: scenario for scenario in evaluation_scenarios(5)}


def _matching_scenario(record, scenario):
    start = record.get("start")
    goal = record.get("goal")
    if list(np.asarray(start, dtype=np.float64)) and len(start) == 3 and len(goal) == 2:
        if np.allclose(np.asarray(start, dtype=np.float64), np.asarray(scenario.start, dtype=np.float64)) and np.allclose(
            np.asarray(goal, dtype=np.float64), np.asarray(scenario.goal, dtype=np.float64)
        ):
            return
    raise ValueError("training scenario coordinates do not match the fixed open_south definition")


def _checkpoint_path(output, timesteps):
    return Path(output) / "checkpoints" / f"step_{int(timesteps):08d}.zip"


def _copy_file(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(Path(source).read_bytes())
    _fsync(temporary)
    os.replace(temporary, destination)


def _append_event(output, event, timesteps, digest):
    _append_jsonl(Path(output) / "run_events.jsonl", {
        "event": event,
        "timesteps": int(timesteps),
        "sha256": digest,
        "time": _timestamp(),
    })


def _write_progress(output, state, status):
    _write_json(output / "progress.json", {
        "status": status,
        "trained_timesteps": int(state.get("last_trained_timesteps") or 0),
        "gradient_steps": int(state.get("gradient_steps") or 0),
        "training_wall_time_s": float(state.get("accumulated_training_s") or 0.0) + float(state.get("session_training_s") or 0.0),
        "evaluation_wall_time_s": float(state.get("accumulated_eval_s") or 0.0) + float(state.get("session_eval_s") or 0.0),
    })


def _interruption_for(state):
    reason = state.get("stop_reason") or state.get("failure_kind") or "stopped"
    if reason == "interrupted":
        return "interrupted"
    if reason == "wall_clock_timeout":
        return "wall_clock_timeout"
    if state.get("status") == "running":
        return "stopped"
    return str(reason)


def _status_for_stop(reason):
    if reason == "interrupted":
        return "interrupted"
    if reason in ("nonfinite_transition", "action_contract"):
        return "failed"
    if reason in ("wall_clock_timeout", "session_limit", "uncommitted_rollout"):
        return "incomplete"
    return "failed"


def _failure_kind(exc):
    if isinstance(exc, (OptimizationFailure, DiagnosticFailure)):
        text = str(exc).lower()
        if "non-finite" in text or "nonfinite" in text:
            return "optimization"
        return "failed"
    text = str(exc).lower()
    if "gazebo" in text or "timed out" in text or "lidar" in text or "odometry" in text:
        return "simulator_failure"
    return "failed"


def _sigint_handler(state):
    def handler(signum, frame):
        state["sigint_count"] += 1
        state["stop_requested"] = True
        if state["sigint_count"] >= 2:
            raise KeyboardInterrupt
    return handler


def _exit_code(status):
    if status == "completed":
        return 0
    if status == "interrupted":
        return 130
    if status == "incomplete":
        return 2
    return 1


def _public_summary(output, state):
    path = Path(output) / "summary.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": state.get("status"), "output_directory": str(output)}


def _commands():
    return "\n".join((
        "source /opt/ros/humble/setup.bash",
        "source jackal_rl_ws/install/setup.bash",
        "ros2 launch jackal_rl_bringup jackal_sim.launch.py",
        "",
        "source /opt/ros/humble/setup.bash",
        "source jackal_rl_ws/install/setup.bash",
        "source .venv/bin/activate",
        "python -m navigation.jackal_ppo_diagnostic train --config configs/jackal_ppo_diagnostic_6g3_bc.yaml",
        "python -m navigation.jackal_ppo_diagnostic train --config configs/jackal_ppo_diagnostic_6g3_scratch.yaml",
        "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_jackal_ppo_diagnostic.py "
        "tests/test_jackal_ppo_smoke.py tests/test_jackal_gazebo_env_contract.py "
        "tests/test_jackal_bc_ppo_transfer.py tests/test_jackal_bc_architecture.py "
        "tests/test_jackal_bc_training.py tests/test_jackal_bc_evaluation.py",
        "",
    ))


def _resolve_output(path):
    return _resolve(path)


def _resolve(path):
    if not path:
        raise ValueError("Path is required")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _ROOT / candidate
    return candidate.resolve()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_plain(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rewrite_jsonl(path, records):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_plain(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _episode_row(record):
    return [
        record["episode_index"],
        record["scenario_id"],
        record["seed"],
        record["status"],
        record["episode_return"],
        record["length"],
        record["success"],
        record["collision"],
        record["timeout"],
        record["termination_reason"],
        record["interruption"],
        record["environment_steps_completed"],
        json.dumps(record["start"]),
        json.dumps(record["goal"]),
    ]


def _append_csv_rows(path, columns, rows):
    path = Path(path)
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(columns)
        for row in rows:
            if isinstance(row, dict):
                writer.writerow([row[column] for column in columns])
            else:
                writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _rewrite_csv(path, columns, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            if isinstance(row, dict):
                writer.writerow([row[column] for column in columns])
            else:
                writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _median(values):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _mapping(config, name):
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _nonnegative_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if float(value) < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _unit_interval(value, name):
    number = _positive_number(value, name)
    if number > 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return number


class _CsvLog:
    def __init__(self, path, columns, append=False):
        self.path = Path(path)
        exists = self.path.is_file() and self.path.stat().st_size > 0
        mode = "a" if append and exists else "w"
        self.handle = self.path.open(mode, encoding="utf-8", newline="")
        self.writer = csv.writer(self.handle)
        if mode == "w":
            self.writer.writerow(columns)
            self.handle.flush()

    def write(self, row):
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        if self.handle is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None


if __name__ == "__main__":
    sys.exit(main())
