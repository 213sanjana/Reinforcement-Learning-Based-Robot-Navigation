"""Bounded PPO smoke test for the Jackal BC-initialized policy.

The command loads the Phase 6G.1 checkpoint, checks its hash and action
contract, and runs one Stable-Baselines3 PPO update cycle in
``JackalGazeboEnv``. It does not modify the behavior-cloning checkpoint
or the Phase 6G.1 initialization.

Gazebo must already be running through
``ros2 launch jackal_rl_bringup jackal_sim.launch.py``. This process
creates one robot interface. Stable-Baselines3 wraps that single
environment in ``DummyVecEnv``; that adapter does not launch a second
controller.

Run from the repository root:

    python -m navigation.jackal_ppo_train --config configs/jackal_ppo_smoke_6g2.yaml
"""

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import platform
import re
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
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.logger import configure
from stable_baselines3.common.preprocessing import preprocess_obs
from torch.distributions import Normal

_ROOT = Path(__file__).resolve().parents[1]
_JACKAL_SRC = _ROOT / "jackal_rl_ws" / "src" / "jackal_rl_navigation"
if _JACKAL_SRC.is_dir() and str(_JACKAL_SRC) not in sys.path:
    sys.path.insert(0, str(_JACKAL_SRC))

from agents.jackal_pretrained_ppo_policy import (  # noqa: E402
    JackalPretrainedPPOPolicy,
    check_sb3_version,
)
from navigation.expert_demonstration_schema import (  # noqa: E402
    validate_jackal_policy_spaces,
)
from navigation.jackal_bc_ppo_transfer import file_sha256, tensor_sha256  # noqa: E402


PHASE = "6g.2"
OFFICIAL_CHECKPOINT = (
    _ROOT / "artifacts" / "ppo" / "jackal_6g1" / "run_20261022_230534" / "initialized_ppo.zip"
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "dff19be924f920490915d8f625d44f32da79452caaf3f7698160d2b8fc287750"
)
OFFICIAL_BC_CHECKPOINT = _ROOT / "artifacts" / "behavior_cloning" / "jackal_6f2" / "best.pt"
OFFICIAL_BC_SHA256 = "1cc42601ae918057bc01fd18f3cfa3975bb9a390e9c2d08715a1b05f0ff515c9"
FORBIDDEN_OUTPUT_ROOTS = (
    _ROOT / "artifacts" / "ppo" / "jackal_6g1",
    _ROOT / "artifacts" / "behavior_cloning",
)
LIDAR_TOPIC = "/j100_0000/sensors/lidar3d_0/scan"
ODOM_TOPIC = "/j100_0000/platform/odom"
COMMAND_TOPIC = "/j100_0000/platform/cmd_vel_unstamped"
GAZEBO_CONTROL_SERVICE = "/world/jackal_obstacle_test/control"
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
    "environment_steps_completed",
    "start",
    "goal",
)


class SmokeFailure(RuntimeError):
    """The smoke test cannot continue safely."""


class OptimizationFailure(SmokeFailure):
    """PPO produced a non-finite loss or gradient."""


def load_config(path):
    """Load and validate a Phase 6G.2 YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config)


def validate_config(config):
    """Raise unless ``config`` describes this bounded smoke test."""

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    config = copy.deepcopy(config)
    if config.get("phase") != PHASE:
        raise ValueError("Configuration phase must be 6g.2")
    if config.get("initialization") not in ("bc", "scratch"):
        raise ValueError(
            "initialization must be 'bc' or 'scratch'. "
            "Phase 6G.2 executes only 'bc'."
        )
    if config["initialization"] == "bc":
        checkpoint = _resolve(config.get("checkpoint"))
        bc_checkpoint = _resolve(config.get("bc_checkpoint"))
        if checkpoint != OFFICIAL_CHECKPOINT.resolve():
            raise ValueError("BC-initialized training must load the Phase 6G.1 checkpoint")
        if bc_checkpoint != OFFICIAL_BC_CHECKPOINT.resolve():
            raise ValueError("BC source must be the frozen Phase 6F.2 best.pt checkpoint")
        if config.get("expected_checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256:
            raise ValueError("expected_checkpoint_sha256 does not match Phase 6G.1")
        if config.get("expected_bc_sha256") != OFFICIAL_BC_SHA256:
            raise ValueError("expected_bc_sha256 does not match Phase 6F.2")
        config["checkpoint"] = str(checkpoint)
        config["bc_checkpoint"] = str(bc_checkpoint)

    output = _mapping(config, "output")
    if not output.get("directory"):
        raise ValueError("Output directory is required")
    runtime = _mapping(config, "runtime")
    _nonnegative_int(runtime.get("seed"), "seed")
    if runtime.get("device") != "cpu":
        raise ValueError("Phase 6G.2 smoke training runs on cpu")
    _positive_int(runtime.get("total_timesteps"), "total_timesteps")
    timeout = _positive_number(runtime.get("wall_clock_timeout_s"), "wall_clock_timeout_s")
    if timeout > 7200:
        raise ValueError("wall_clock_timeout_s must be at most 7200 seconds")
    if runtime.get("verbose") not in (0, 1):
        raise ValueError("verbose must be 0 or 1")

    ppo = _mapping(config, "ppo")
    _positive_int(ppo.get("n_steps"), "n_steps")
    _positive_int(ppo.get("batch_size"), "batch_size")
    _positive_int(ppo.get("n_epochs"), "n_epochs")
    _positive_int(ppo.get("checkpoint_n_epochs"), "checkpoint_n_epochs")
    if int(ppo["n_envs"]) != 1:
        raise ValueError("n_envs must be 1. One Jackal controller is allowed.")
    if int(ppo["n_steps"]) % int(ppo["batch_size"]) != 0:
        raise ValueError("batch_size must divide n_steps")
    if int(runtime["total_timesteps"]) % int(ppo["n_steps"]) != 0:
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
    if not isinstance(environment.get("scenario_id"), str) or not environment["scenario_id"]:
        raise ValueError("scenario_id is required")
    start = environment.get("start")
    goal = environment.get("goal")
    if not isinstance(start, (list, tuple)) or len(start) != 3:
        raise ValueError("start must be (x, y, yaw)")
    if not isinstance(goal, (list, tuple)) or len(goal) != 2:
        raise ValueError("goal must be (x, y)")
    if not np.all(np.isfinite(np.asarray(list(start) + list(goal), dtype=np.float64))):
        raise ValueError("start and goal must be finite")
    if int(environment.get("expected_max_episode_steps")) != 300:
        raise ValueError("expected_max_episode_steps must remain 300")
    _positive_number(environment.get("sensor_timeout_s"), "sensor_timeout_s")
    if not isinstance(environment.get("collision_monitoring"), bool):
        raise ValueError("collision_monitoring must be a boolean")

    checkpointing = _mapping(config, "checkpointing")
    intermediate = checkpointing.get("intermediate_timesteps")
    if intermediate is not None:
        _positive_int(intermediate, "intermediate_timesteps")
        if int(intermediate) % int(ppo["n_steps"]) != 0:
            raise ValueError("intermediate_timesteps must be a multiple of n_steps")
        if int(intermediate) >= int(runtime["total_timesteps"]):
            raise ValueError("intermediate_timesteps must be before the final checkpoint")
    return config


def parse_topic_info(text):
    """Return ``(publisher_count, subscription_count)`` from ``ros2 topic info -v``."""

    publishers = re.search(r"Publisher count:\s*(\d+)", text or "")
    subscriptions = re.search(r"Subscription count:\s*(\d+)", text or "")
    if publishers is None or subscriptions is None:
        raise ValueError("ROS topic info did not report publisher and subscription counts")
    return int(publishers.group(1)), int(subscriptions.group(1))


def assess_topic_readiness(scan_counts, odom_counts, command_counts):
    """Refuse to start when sensors are absent or another commander is publishing."""

    errors = []
    if int(scan_counts[0]) < 1:
        errors.append(f"{LIDAR_TOPIC} has no publisher")
    if int(odom_counts[0]) < 1:
        errors.append(f"{ODOM_TOPIC} has no publisher")
    if int(command_counts[0]) != 0:
        errors.append(
            f"{COMMAND_TOPIC} already has {int(command_counts[0])} publisher(s)"
        )
    if int(command_counts[1]) < 1:
        errors.append(f"{COMMAND_TOPIC} has no velocity-controller subscription")
    if errors:
        raise SmokeFailure("Refusing to start PPO: " + "; ".join(errors))


def preflight_simulator():
    """Check Gazebo and the command topic before this process publishes."""

    from jackal_rl_navigation.gazebo_controller import gazebo_process_env

    try:
        listing = subprocess.run(
            ["ign", "service", "-l"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=gazebo_process_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmokeFailure(f"Gazebo service listing failed: {exc}") from exc
    if listing.returncode != 0 or GAZEBO_CONTROL_SERVICE not in listing.stdout:
        detail = (listing.stderr or listing.stdout or "").strip()
        raise SmokeFailure(
            "Gazebo world control is unavailable. Launch "
            "jackal_rl_bringup jackal_sim.launch.py before training. "
            f"{detail}"
        )
    assess_topic_readiness(
        _topic_counts(LIDAR_TOPIC),
        _topic_counts(ODOM_TOPIC),
        _topic_counts(COMMAND_TOPIC),
    )


def audit_loaded_policy(model):
    """Check the loaded actor, critic, and Stable-Baselines3 action path."""

    policy = model.policy
    if not isinstance(policy, JackalPretrainedPPOPolicy):
        raise SmokeFailure(
            "Loaded policy is not JackalPretrainedPPOPolicy. "
            "Refusing to fall back to another policy."
        )
    validate_jackal_policy_spaces(policy.observation_space, policy.action_space)
    if policy.squash_output or policy.use_sde or policy.normalize_images:
        raise SmokeFailure("PPO squashing, gSDE, or image normalization is enabled")
    if policy.share_features_extractor:
        raise SmokeFailure("Actor and critic feature extractors must stay separate")
    if not isinstance(policy.action_dist, DiagGaussianDistribution):
        raise SmokeFailure("Actor distribution is not a diagonal Gaussian")
    if list(policy.mlp_extractor.policy_net.children()) or list(
        policy.mlp_extractor.value_net.children()
    ):
        raise SmokeFailure("Extra PPO MLP layers are present")
    log_std = policy.log_std.detach().cpu()
    if tuple(log_std.shape) != (2,) or not torch.equal(log_std, torch.zeros(2)):
        raise SmokeFailure("log_std is not the Phase 6G.1 initialization of zero")
    if model.num_timesteps != 0 or model._n_updates != 0:
        raise SmokeFailure("Checkpoint is not the untrained Phase 6G.1 initialization")

    observations = torch.zeros((2, 40), dtype=torch.float32)
    preprocessed = preprocess_obs(
        observations,
        policy.observation_space,
        normalize_images=policy.normalize_images,
    )
    if preprocessed.dtype != observations.dtype or not torch.equal(preprocessed, observations):
        raise SmokeFailure("Stable-Baselines3 preprocessing changed the observation")
    raw, mean = policy.raw_and_mean(observations)
    if not torch.allclose(mean, torch.tanh(raw), atol=0.0, rtol=0.0):
        raise SmokeFailure("Gaussian mean is not tanh of the actor output")
    distribution = policy.get_distribution(observations)
    gaussian = distribution.distribution
    if not torch.allclose(gaussian.mean, mean, atol=0.0, rtol=0.0):
        raise SmokeFailure("Policy distribution mean does not match tanh(actor)")
    if not torch.allclose(gaussian.stddev, torch.ones_like(gaussian.stddev)):
        raise SmokeFailure("Initial Gaussian standard deviation is not 1")
    sample = gaussian.mean + 2.0 * gaussian.stddev
    manual = Normal(gaussian.mean, gaussian.stddev).log_prob(sample).sum(dim=-1)
    recorded = distribution.log_prob(sample)
    if not torch.allclose(recorded, manual, atol=1e-6, rtol=0.0):
        raise SmokeFailure("PPO log-probability is not the diagonal-Gaussian log-probability")
    critic_trainable = all(
        parameter.requires_grad
        for name, parameter in policy.named_parameters()
        if name.startswith("vf_features_extractor") or name.startswith("value_net")
    )
    if not critic_trainable:
        raise SmokeFailure("Critic parameters are not trainable")
    return {
        "policy_class": "agents.jackal_pretrained_ppo_policy.JackalPretrainedPPOPolicy",
        "squash_output": False,
        "use_sde": False,
        "normalize_images": False,
        "observation_normalized_once_inside_trunk": True,
        "gaussian_mean": "tanh(action_net)",
        "log_std": [0.0, 0.0],
        "initial_std": [1.0, 1.0],
        "sampled_actions_clipped_by_sb3": True,
        "second_tanh_applied": False,
        "critic_trainable": True,
    }


def apply_smoke_hyperparameters(model, config):
    """Require checkpoint hyperparameters, then apply the smoke-test epoch count."""

    ppo = config["ppo"]
    if int(model.n_epochs) != int(ppo["checkpoint_n_epochs"]):
        raise SmokeFailure(
            f"Checkpoint n_epochs is {model.n_epochs}, expected {ppo['checkpoint_n_epochs']}"
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
            raise SmokeFailure(f"Checkpoint {name}={actual} does not match the smoke configuration")
    if model.use_sde or not model.normalize_advantage or int(model.n_envs) != 1:
        raise SmokeFailure("Checkpoint exploration or environment count does not match")
    model.n_epochs = int(ppo["n_epochs"])


def run_smoke(config, output_directory, environment_factory=None, handle_signals=True):
    """Run the bounded smoke test and return its summary."""

    config = validate_config(config)
    if config["initialization"] != "bc":
        raise SmokeFailure(
            "Phase 6G.2 refuses scratch initialization. "
            "It is reserved for a later phase and was not started."
        )
    output = Path(output_directory)
    if not output.is_absolute():
        output = _ROOT / output
    output = output.resolve()
    _reject_output(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    check_sb3_version()
    checkpoint = Path(config["checkpoint"])
    bc_checkpoint = Path(config["bc_checkpoint"])
    checkpoint_hash = file_sha256(checkpoint)
    bc_hash = file_sha256(bc_checkpoint)
    if checkpoint_hash != OFFICIAL_CHECKPOINT_SHA256:
        raise SmokeFailure(
            "Phase 6G.1 checkpoint hash mismatch. "
            "BC-initialized training was not started."
        )
    if bc_hash != OFFICIAL_BC_SHA256:
        raise SmokeFailure("Frozen BC checkpoint hash mismatch. Training was not started.")

    protected_before = _protected_hashes()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    state = _new_state(config)
    state["checkpoint_sha256"] = checkpoint_hash
    state["bc_sha256"] = bc_hash
    state["protected_before"] = protected_before
    _write_json(output / "smoke_config.json", config)
    _write_json(output / "protected_hashes_before.json", protected_before)
    _write_json(output / "framework_versions.json", _versions())
    _write_text(output / "commands.txt", _commands())

    model = None
    env = None
    previous_handler = None
    try:
        if handle_signals and threading_is_main():
            previous_handler = signal.signal(signal.SIGINT, _sigint_handler(state))
        model = PPO.load(str(checkpoint), device="cpu")
        state["policy_audit"] = audit_loaded_policy(model)
        apply_smoke_hyperparameters(model, config)
        if environment_factory is None:
            preflight_simulator()
            environment_factory = lambda: _make_jackal_env(config)  # noqa: E731
        env = _build_training_env(environment_factory(), config)
        _check_environment(env, config)
        model.set_env(env)
        if model.get_env() is None or int(model.get_env().num_envs) != 1:
            raise SmokeFailure("PPO did not attach exactly one environment")
        model.set_random_seed(int(config["runtime"]["seed"]))
        model.set_logger(configure(
            str(output),
            ["stdout", "csv", "json"] if int(config["runtime"]["verbose"]) else ["csv", "json"],
        ))
        state["initial_parameters"] = _parameter_rows(model.policy)
        state["initial_buffers"] = _buffer_rows(model.policy)
        _install_diagnostics(model, state, output)
        state["step_log"] = _CsvLog(output / "steps.csv", STEP_COLUMNS)
        state["episode_log"] = _CsvLog(output / "episodes.csv", EPISODE_COLUMNS)
        callback = _SmokeCallback(state, config)
        state["training_started_at"] = time.monotonic()
        model.learn(
            total_timesteps=int(config["runtime"]["total_timesteps"]),
            callback=callback,
            log_interval=1,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        _flush_logger(model)
        state["final_parameters"] = _parameter_rows(model.policy)
        state["final_buffers"] = _buffer_rows(model.policy)
        state["optimization"] = _optimization_report(model, state, config)
        if state["stop_reason"]:
            state["status"] = _status_for_stop(state["stop_reason"])
            state["failure_kind"] = state["stop_reason"]
            _capture_partial_model(model, state, output)
        elif not state["optimization"]["passed"]:
            state["status"] = "failed"
            state["failure_kind"] = "optimization"
            _capture_partial_model(model, state, output)
        else:
            final_path = output / "ppo_smoke.zip"
            _save_checkpoint(model, final_path)
            state["final_checkpoint"] = str(final_path)
            state["final_checkpoint_sha256"] = file_sha256(final_path)
            state["reload"] = _reload_checkpoint(
                final_path,
                model,
                int(config["runtime"]["total_timesteps"]),
            )
            state["status"] = "passed" if state["reload"]["passed"] else "failed"
            if not state["reload"]["passed"]:
                state["failure_kind"] = "reload"
    except KeyboardInterrupt as exc:
        state["status"] = "interrupted"
        state["failure_kind"] = "interrupted"
        state["error"] = _error_payload(exc)
        _capture_partial_model(model, state, output)
    except Exception as exc:
        state["status"] = "failed"
        state["failure_kind"] = _failure_kind(exc)
        state["error"] = _error_payload(exc)
        _capture_partial_model(model, state, output)
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)
        callback_obj = state.get("callback")
        if callback_obj is not None:
            callback_obj.close_open_episode()
        state["shutdown"] = _shutdown(model, env)
        if state.get("final_parameters") is None and model is not None:
            try:
                state["final_parameters"] = _parameter_rows(model.policy)
                state["final_buffers"] = _buffer_rows(model.policy)
                state["optimization"] = _optimization_report(model, state, config)
            except Exception as exc:
                state["parameter_error"] = f"{type(exc).__name__}: {exc}"
        for log_name in ("step_log", "episode_log"):
            log = state.get(log_name)
            if log is not None:
                log.close()
        protected_after = _protected_hashes()
        state["protected_after"] = protected_after
        state["protected_unchanged"] = protected_after == state["protected_before"]
        if not state["protected_unchanged"]:
            state["status"] = "failed"
            state["failure_kind"] = "protected_artifact_changed"
        if state["status"] == "passed" and not state.get("optimization", {}).get("passed", False):
            state["status"] = "failed"
        _write_reports(output, config, state)
    return _public_summary(output, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.output_directory:
        output = Path(args.output_directory)
    else:
        output = Path(config["output"]["directory"]) / f"run_{_timestamp()}"
    try:
        summary = run_smoke(config, output)
    except (SmokeFailure, ValueError, FileExistsError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": summary["status"],
        "failure_kind": summary.get("failure_kind"),
        "timesteps": summary.get("environment_timesteps"),
        "rollouts": summary.get("rollouts_completed"),
        "n_updates": summary.get("n_updates"),
        "output_directory": summary.get("output_directory"),
        "checkpoint_sha256": summary.get("final_checkpoint_sha256"),
    }, indent=2))
    return _exit_code(summary["status"])


class FixedScenarioWrapper(gym.Wrapper):
    """Reset the Jackal at the configured start and goal."""

    def __init__(self, env, start, goal, scenario_id, seed):
        super().__init__(env)
        self._start = tuple(start)
        self._goal = tuple(goal)
        self.scenario_id = str(scenario_id)
        self.training_seed = int(seed)

    def reset(self, *, seed=None, options=None):
        merged = dict(options or {})
        merged.setdefault("start", self._start)
        merged.setdefault("goal", self._goal)
        observation, info = self.env.reset(seed=seed, options=merged)
        info = dict(info)
        info["scenario_id"] = self.scenario_id
        info["training_seed"] = self.training_seed
        return observation, info


class TimeLimitInfoWrapper(gym.Wrapper):
    """Expose the environment's own timeout to Stable-Baselines3."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        timed_out = bool(truncated) and not bool(terminated) and info.get("time_limit_reached") is True
        info["TimeLimit.truncated"] = timed_out
        return observation, reward, terminated, truncated, info


class StopOnFailureWrapper(gym.Wrapper):
    """Publish a zero command if reset or step fails."""

    def reset(self, *, seed=None, options=None):
        try:
            return self.env.reset(seed=seed, options=options)
        except BaseException:
            _stop_robot(self.env)
            raise

    def step(self, action):
        try:
            return self.env.step(action)
        except BaseException:
            _stop_robot(self.env)
            raise

    def close(self):
        _stop_robot(self.env)
        return super().close()


class _SmokeCallback(BaseCallback):
    """Record transitions and stop at the wall-clock bound."""

    def __init__(self, state, config):
        super().__init__(verbose=0)
        self.state = state
        self.config = config
        self.episode_index = -1
        self.episode_open = False
        self.episode_return = 0.0
        self.episode_length = 0
        state["callback"] = self

    def _on_step(self):
        if self.state["stop_requested"]:
            self.state["stop_reason"] = "interrupted"
            return False
        started = self.state["training_started_at"]
        limit = float(self.config["runtime"]["wall_clock_timeout_s"])
        if time.monotonic() - started > limit:
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
            self._finish_episode("completed", info)
        return True

    def close_open_episode(self):
        if self.episode_open and self.episode_length > 0:
            self._finish_episode("partial", {})

    def _start_episode(self):
        self.episode_index += 1
        self.episode_open = True
        self.episode_return = 0.0
        self.episode_length = 0

    def _finish_episode(self, status, info):
        reason = info.get("termination_reason")
        record = {
            "episode_index": self.episode_index,
            "scenario_id": self.config["environment"]["scenario_id"],
            "seed": int(self.config["runtime"]["seed"]),
            "status": status,
            "episode_return": self.episode_return,
            "length": self.episode_length,
            "success": bool(info.get("goal_reached")) if status == "completed" else False,
            "collision": bool(info.get("collision_detected")) if status == "completed" else False,
            "timeout": bool(info.get("time_limit_reached")) if status == "completed" else False,
            "termination_reason": "" if reason is None or status != "completed" else str(reason),
            "environment_steps_completed": int(self.num_timesteps),
            "start": list(self.config["environment"]["start"]),
            "goal": list(self.config["environment"]["goal"]),
        }
        self.state["episodes"].append(record)
        self.state["episode_log"].write([
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
            record["environment_steps_completed"],
            json.dumps(record["start"]),
            json.dumps(record["goal"]),
        ])
        self.episode_open = False

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


class _CsvLog:
    def __init__(self, path, columns):
        self.path = Path(path)
        self.handle = self.path.open("w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.handle)
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


def _install_diagnostics(model, state, output):
    original_train = model.train
    original_step = model.policy.optimizer.step

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
        original_train()
        metrics = _train_metrics(model)
        state["updates"].append({
            "timesteps": int(model.num_timesteps),
            "n_updates": int(model._n_updates),
            "metrics": metrics,
        })
        required = (
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/loss",
            "train/std",
        )
        if any(key not in metrics or not math.isfinite(metrics[key]) for key in required):
            state["nonfinite_loss"] = True
            raise OptimizationFailure("Non-finite PPO loss")
        intermediate = state["intermediate_timesteps"]
        if (
            intermediate is not None
            and not state["intermediate_saved"]
            and int(model.num_timesteps) >= int(intermediate)
        ):
            path = output / "intermediate_ppo.zip"
            _save_checkpoint(model, path)
            state["intermediate_checkpoint"] = str(path)
            state["intermediate_checkpoint_sha256"] = file_sha256(path)
            state["intermediate_saved"] = True

    model.train = traced_train
    model.policy.optimizer.step = traced_step
    state["hooked_model"] = model


def _save_checkpoint(model, path):
    """Save without pickling the temporary diagnostic hooks."""

    removed_train = model.__dict__.pop("train", None)
    optimizer = model.policy.optimizer
    removed_step = optimizer.__dict__.pop("step", None)
    try:
        model.save(str(path))
        _fsync(path)
    finally:
        if removed_train is not None:
            model.train = removed_train
        if removed_step is not None:
            optimizer.step = removed_step


def _reload_checkpoint(path, model, timesteps):
    report = {"passed": False, "in_process": False, "fresh_process": False}
    loaded = None
    try:
        loaded = PPO.load(str(path), device="cpu")
        if not isinstance(loaded.policy, JackalPretrainedPPOPolicy):
            raise SmokeFailure("Reloaded policy class changed")
        validate_jackal_policy_spaces(loaded.observation_space, loaded.action_space)
        if int(loaded.num_timesteps) != int(timesteps) or int(loaded._n_updates) != int(model._n_updates):
            raise SmokeFailure("Reloaded timestep or update count changed")
        for name, parameter in model.policy.state_dict().items():
            restored = loaded.policy.state_dict()[name]
            if not torch.equal(parameter.detach().cpu(), restored.detach().cpu()):
                raise SmokeFailure(f"Reloaded tensor changed: {name}")
        if not all(torch.isfinite(parameter).all() for parameter in loaded.policy.parameters()):
            raise SmokeFailure("Reloaded parameters are not finite")
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
""" % {"path": str(path), "timesteps": int(timesteps)}
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0 or "RELOAD_OK" not in result.stdout:
        detail = (result.stderr or result.stdout or "").strip()
        return {"loaded": False, "error": detail[-2000:]}
    return {"loaded": True, "timesteps": int(timesteps)}


def _optimization_report(model, state, config):
    comparison = _compare_parameters(
        state.get("initial_parameters") or [],
        state.get("final_parameters") or [],
    )
    buffers = _compare_parameters(
        state.get("initial_buffers") or [],
        state.get("final_buffers") or [],
    )
    requested = int(config["runtime"]["total_timesteps"])
    rollouts = len(state["updates"])
    expected_rollouts = requested // int(config["ppo"]["n_steps"])
    required_metrics = (
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/loss",
        "train/std",
    )
    losses_finite = not state["nonfinite_loss"] and all(
        key in update["metrics"] and math.isfinite(update["metrics"][key])
        for update in state["updates"]
        for key in required_metrics
    )
    explained_finite = all(
        "train/explained_variance" not in update["metrics"]
        or math.isfinite(update["metrics"]["train/explained_variance"])
        for update in state["updates"]
    )
    changed = int(comparison["changed"]) > 0
    critic_trainable = bool(comparison["critic_requires_grad"])
    normalizers_unchanged = bool(buffers["changed"] == 0 and buffers["count"] > 0)
    expected_updates = expected_rollouts * int(config["ppo"]["n_epochs"])
    expected_gradients = expected_updates * (
        int(config["ppo"]["n_steps"]) // int(config["ppo"]["batch_size"])
    )
    completed = (
        int(model.num_timesteps) == requested
        and rollouts == expected_rollouts
        and int(model._n_updates) == expected_updates
        and int(state["gradient_steps"]) == expected_gradients
    )
    passed = bool(
        completed
        and changed
        and critic_trainable
        and losses_finite
        and state["grad_finite"]
        and state["action_contract_ok"]
        and normalizers_unchanged
        and state["clipped_outside_box"] == 0
    )
    return {
        "passed": passed,
        "environment_timesteps": int(model.num_timesteps),
        "requested_timesteps": requested,
        "rollouts_completed": rollouts,
        "expected_rollouts": expected_rollouts,
        "n_updates": int(model._n_updates),
        "gradient_steps": int(state["gradient_steps"]),
        "losses_finite": losses_finite,
        "explained_variance_finite": explained_finite,
        "expected_n_updates": expected_updates,
        "expected_gradient_steps": expected_gradients,
        "gradients_finite": bool(state["grad_finite"]),
        "trainable_changed": changed,
        "actor_changed": bool(comparison["actor_changed"]),
        "critic_changed": bool(comparison["critic_changed"]),
        "log_std_changed": bool(comparison["log_std_changed"]),
        "critic_requires_grad": critic_trainable,
        "normalizer_buffers_unchanged": normalizers_unchanged,
        "action_contract_ok": bool(state["action_contract_ok"]),
        "raw_actions_outside_box": int(state["raw_outside_box"]),
        "clipped_actions_outside_box": int(state["clipped_outside_box"]),
        "clip_examples": state["clip_examples"],
        "gradient_norm_max": max(state["grad_norms"]) if state["grad_norms"] else None,
        "updates": state["updates"],
        "parameters": comparison,
        "buffers": buffers,
    }


def _compare_parameters(initial, final):
    final_by_name = {row["name"]: row for row in final}
    changed = 0
    actor_changed = False
    critic_changed = False
    log_std_changed = False
    critic_requires_grad = True
    critic_count = 0
    rows = []
    for row in initial:
        current = final_by_name.get(row["name"])
        if current is None:
            did_change = True
        else:
            did_change = current["sha256"] != row["sha256"]
        name = row["name"]
        is_critic = name.startswith("vf_features_extractor") or name.startswith("value_net")
        is_actor = name.startswith("pi_features_extractor") or name.startswith("action_net")
        if is_critic:
            critic_count += 1
            critic_requires_grad = critic_requires_grad and bool(current and current["requires_grad"])
            actor_flag = False
        else:
            actor_flag = is_actor or name == "log_std"
        if did_change:
            changed += 1
            actor_changed = actor_changed or actor_flag
            critic_changed = critic_changed or is_critic
            log_std_changed = log_std_changed or name == "log_std"
        rows.append({
            "name": name,
            "requires_grad": None if current is None else current["requires_grad"],
            "changed": did_change,
            "finite": None if current is None else current["finite"],
            "initial_mean": row["mean"],
            "final_mean": None if current is None else current["mean"],
            "initial_sha256": row["sha256"],
            "final_sha256": None if current is None else current["sha256"],
        })
    return {
        "count": len(initial),
        "changed": changed,
        "actor_changed": actor_changed,
        "critic_changed": critic_changed,
        "log_std_changed": log_std_changed,
        "critic_requires_grad": bool(critic_count and critic_requires_grad),
        "rows": rows,
    }


def _parameter_rows(policy):
    rows = []
    for name, parameter in policy.named_parameters():
        rows.append(_tensor_row(name, parameter.detach(), bool(parameter.requires_grad)))
    return rows


def _buffer_rows(policy):
    rows = []
    for name, buffer in policy.named_buffers():
        if "obs_mean" in name or "obs_std" in name or "normalizer_count" in name:
            rows.append(_tensor_row(name, buffer.detach(), False))
    return rows


def _tensor_row(name, tensor, requires_grad):
    array = tensor.detach().cpu()
    finite = bool(torch.isfinite(array).all())
    flat = array.float()
    return {
        "name": name,
        "shape": list(array.shape),
        "requires_grad": requires_grad,
        "finite": finite,
        "mean": float(flat.mean().item()) if finite and flat.numel() else None,
        "sha256": tensor_sha256(array),
    }


def _capture_partial_model(model, state, output):
    if model is None:
        return
    try:
        if not all(torch.isfinite(parameter.detach()).all() for parameter in model.policy.parameters()):
            state["partial_checkpoint"] = None
            state["partial_checkpoint_reason"] = "parameters are not finite"
            return
        path = output / "ppo_smoke_partial.zip"
        if path.exists():
            return
        _save_checkpoint(model, path)
        state["partial_checkpoint"] = str(path)
        state["partial_checkpoint_sha256"] = file_sha256(path)
    except Exception as exc:
        state["partial_checkpoint_reason"] = f"{type(exc).__name__}: {exc}"


def _build_training_env(env, config):
    environment = config["environment"]
    env = FixedScenarioWrapper(
        env,
        environment["start"],
        environment["goal"],
        environment["scenario_id"],
        config["runtime"]["seed"],
    )
    env = TimeLimitInfoWrapper(env)
    return StopOnFailureWrapper(env)


def _check_environment(env, config):
    base = env.unwrapped
    validate_jackal_policy_spaces(base.observation_space, base.action_space)
    expected = int(config["environment"]["expected_max_episode_steps"])
    if int(getattr(base, "max_episode_steps")) != expected:
        raise SmokeFailure(f"Environment timeout is not {expected} steps")


def _make_jackal_env(config):
    from jackal_rl_navigation.collision_config import CollisionConfig
    from jackal_rl_navigation.gazebo_env import JackalGazeboEnv

    environment = config["environment"]
    return JackalGazeboEnv(
        sensor_timeout=float(environment["sensor_timeout_s"]),
        collision_config=CollisionConfig(enabled=bool(environment["collision_monitoring"])),
    )


def _shutdown(model, env):
    report = {"stopped": False, "closed": False, "errors": []}
    robot = _robot_from_model(model) or _robot(env)
    if robot is not None:
        try:
            robot.stop()
            report["stopped"] = True
        except Exception as exc:
            report["errors"].append(f"{type(exc).__name__}: {exc}")
    vec = None if model is None else model.get_env()
    if vec is not None:
        try:
            vec.close()
            report["closed"] = True
        except Exception as exc:
            report["errors"].append(f"{type(exc).__name__}: {exc}")
    if env is not None and not report["closed"]:
        try:
            env.close()
            report["closed"] = True
        except Exception as exc:
            report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report


def _robot_from_model(model):
    if model is None or model.get_env() is None:
        return None
    try:
        return _robot(model.get_env().envs[0])
    except Exception:
        return None


def _stop_robot(env):
    robot = _robot(env)
    if robot is None:
        return
    try:
        robot.stop()
    except Exception:
        return


def _robot(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        robot = getattr(current, "robot", None)
        if robot is not None and hasattr(robot, "stop"):
            return robot
        current = getattr(current, "env", None)
    return None


def _write_reports(output, config, state):
    episodes = state["episodes"]
    completed = [episode for episode in episodes if episode["status"] == "completed"]
    partial = [episode for episode in episodes if episode["status"] == "partial"]
    started = state.get("training_started_at")
    wall = None if started is None else time.monotonic() - started
    optimization = state.get("optimization") or {}
    summary = {
        "phase": PHASE,
        "status": state["status"],
        "failure_kind": state.get("failure_kind"),
        "stop_reason": state.get("stop_reason") or None,
        "initialization": "bc",
        "seed": int(config["runtime"]["seed"]),
        "scenario_id": config["environment"]["scenario_id"],
        "start": list(config["environment"]["start"]),
        "goal": list(config["environment"]["goal"]),
        "environment_timesteps": optimization.get("environment_timesteps", state.get("action_rows", 0)),
        "requested_timesteps": int(config["runtime"]["total_timesteps"]),
        "rollouts_completed": optimization.get("rollouts_completed", len(state["updates"])),
        "n_updates": optimization.get("n_updates"),
        "gradient_steps": optimization.get("gradient_steps", state["gradient_steps"]),
        "n_epochs": int(config["ppo"]["n_epochs"]),
        "n_steps": int(config["ppo"]["n_steps"]),
        "batch_size": int(config["ppo"]["batch_size"]),
        "n_envs": 1,
        "training_wall_time_s": wall,
        "completed_episodes": len(completed),
        "partial_episodes": len(partial),
        "successes": sum(bool(episode["success"]) for episode in completed),
        "collisions": sum(bool(episode["collision"]) for episode in completed),
        "timeouts": sum(bool(episode["timeout"]) for episode in completed),
        "navigation_success_is_not_required": True,
        "initial_checkpoint": config["checkpoint"],
        "initial_checkpoint_sha256": state.get("checkpoint_sha256"),
        "bc_checkpoint": config["bc_checkpoint"],
        "bc_checkpoint_sha256": state.get("bc_sha256"),
        "final_checkpoint": state.get("final_checkpoint"),
        "final_checkpoint_sha256": state.get("final_checkpoint_sha256"),
        "intermediate_checkpoint": state.get("intermediate_checkpoint"),
        "intermediate_checkpoint_sha256": state.get("intermediate_checkpoint_sha256"),
        "partial_checkpoint": state.get("partial_checkpoint"),
        "partial_checkpoint_sha256": state.get("partial_checkpoint_sha256"),
        "protected_artifacts_unchanged": bool(state.get("protected_unchanged")),
        "policy_audit": state.get("policy_audit"),
        "optimization_passed": bool(optimization.get("passed")),
        "actor_changed": optimization.get("actor_changed"),
        "critic_changed": optimization.get("critic_changed"),
        "log_std_changed": optimization.get("log_std_changed"),
        "critic_requires_grad": optimization.get("critic_requires_grad"),
        "losses_finite": optimization.get("losses_finite"),
        "gradients_finite": optimization.get("gradients_finite"),
        "normalizer_buffers_unchanged": optimization.get("normalizer_buffers_unchanged"),
        "raw_actions_outside_box": state["raw_outside_box"],
        "clipped_actions_outside_box": state["clipped_outside_box"],
        "clip_examples": state["clip_examples"],
        "reload": state.get("reload"),
        "shutdown": state.get("shutdown"),
        "error": state.get("error"),
        "output_directory": str(output),
        "vectorized_environment": (
            "Stable-Baselines3 DummyVecEnv around one JackalGazeboEnv. "
            "No second command publisher is created."
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "episodes.json", episodes)
    if optimization:
        _write_json(output / "optimization.json", optimization)
    if state.get("initial_parameters") is not None:
        _write_json(output / "initial_parameters.json", state["initial_parameters"])
    if state.get("final_parameters") is not None:
        _write_json(output / "final_parameters.json", state["final_parameters"])
    if state.get("reload") is not None:
        _write_json(output / "checkpoint_reload.json", state["reload"])
    _write_json(output / "protected_hashes_after.json", state.get("protected_after") or {})
    if state.get("error") is not None:
        _write_json(output / "failure.json", state["error"])


def _public_summary(output, state):
    path = output / "summary.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": state["status"], "output_directory": str(output)}


def _new_state(config):
    return {
        "status": "failed",
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
        "clip_examples": [],
        "episodes": [],
        "intermediate_timesteps": config["checkpointing"]["intermediate_timesteps"],
        "intermediate_saved": False,
        "callback": None,
    }


def _train_metrics(model):
    keys = (
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/explained_variance",
        "train/clip_fraction",
        "train/loss",
        "train/std",
    )
    metrics = {}
    for key in keys:
        if key in model.logger.name_to_value:
            metrics[key] = float(model.logger.name_to_value[key])
    return metrics


def _flush_logger(model):
    """Write the last PPO update. Stable-Baselines3 dumps that update on the next rollout."""
    try:
        if not model.logger.name_to_value:
            return
        rollout = max(int(model._n_updates) // max(int(model.n_epochs), 1), 1)
        model.dump_logs(iteration=rollout)
    except Exception:
        return


def _sigint_handler(state):
    def handler(signum, frame):
        state["sigint_count"] += 1
        state["stop_requested"] = True
        if state["sigint_count"] >= 2:
            raise KeyboardInterrupt
    return handler


def threading_is_main():
    import threading
    return threading.current_thread() is threading.main_thread()


def _failure_kind(exc):
    if isinstance(exc, OptimizationFailure):
        return "optimization"
    if isinstance(exc, TimeoutError):
        return "simulator_failure"
    try:
        from jackal_rl_navigation.gazebo_controller import GazeboControlError
    except Exception:
        GazeboControlError = ()
    if GazeboControlError and isinstance(exc, GazeboControlError):
        return "simulator_failure"
    text = str(exc).lower()
    if "gazebo" in text or "timed out" in text or "lidar" in text or "odometry" in text:
        return "simulator_failure"
    return "failed"


def _status_for_stop(reason):
    if reason == "interrupted":
        return "interrupted"
    if reason in ("nonfinite_transition", "action_contract"):
        return "failed"
    if reason == "wall_clock_timeout":
        return "incomplete"
    return "failed"


def _error_payload(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _protected_hashes():
    files = [
        OFFICIAL_BC_CHECKPOINT,
        _ROOT / "artifacts" / "behavior_cloning" / "jackal_6f2" / "final.pt",
        _ROOT / "artifacts" / "behavior_cloning" / "jackal_6f2" / "preprocessing.json",
    ]
    run = OFFICIAL_CHECKPOINT.parent
    files.extend(sorted(path for path in run.rglob("*") if path.is_file()))
    return {str(path.resolve()): file_sha256(path) for path in files}


def _reject_output(output):
    for root in FORBIDDEN_OUTPUT_ROOTS:
        resolved = root.resolve()
        if output == resolved or resolved in output.parents:
            raise SmokeFailure(f"Refusing to write inside {resolved}")
        if output in resolved.parents:
            raise SmokeFailure(f"Output directory would contain protected artifacts: {resolved}")


def _versions():
    versions = {
        "stable_baselines3": check_sb3_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": gym.__version__,
        "python": platform.python_version(),
        "ros_distro": os.environ.get("ROS_DISTRO"),
    }
    return versions


def _commands():
    return "\n".join((
        "source /opt/ros/humble/setup.bash",
        "source jackal_rl_ws/install/setup.bash",
        "ros2 launch jackal_rl_bringup jackal_sim.launch.py",
        "",
        "source /opt/ros/humble/setup.bash",
        "source jackal_rl_ws/install/setup.bash",
        "source .venv/bin/activate",
        "python -m navigation.jackal_ppo_train --config configs/jackal_ppo_smoke_6g2.yaml",
        "",
        "python -m pytest tests/test_jackal_ppo_smoke.py tests/test_jackal_gazebo_env_contract.py "
        "tests/test_jackal_bc_ppo_transfer.py tests/test_jackal_bc_architecture.py "
        "tests/test_jackal_bc_training.py tests/test_jackal_bc_evaluation.py",
        "",
    ))


def _exit_code(status):
    if status == "passed":
        return 0
    if status == "interrupted":
        return 130
    return 1


def _resolve(path):
    if not path:
        raise ValueError("Path is required")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _ROOT / candidate
    return candidate.resolve()


def _topic_counts(topic):
    try:
        result = subprocess.run(
            ["ros2", "topic", "info", topic, "-v"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmokeFailure(f"ROS topic query failed for {topic}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SmokeFailure(f"ROS topic {topic} is not available: {detail}")
    try:
        return parse_topic_info(result.stdout)
    except ValueError as exc:
        raise SmokeFailure(f"ROS topic {topic} query was not readable: {exc}") from exc


def _schedule_value(value):
    if isinstance(value, bool):
        raise SmokeFailure("Hyperparameter schedule is not numeric")
    if isinstance(value, (int, float)):
        return float(value)
    if callable(value):
        start = float(value(1.0))
        end = float(value(0.0))
        if not _same_number(start, end):
            raise SmokeFailure("Hyperparameter schedule is not constant")
        return start
    raise SmokeFailure(f"Unsupported hyperparameter value {value!r}")


def _same_number(left, right):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _write_json(path, payload):
    path = Path(path)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _fsync(path)


def _write_text(path, text):
    path = Path(path)
    path.write_text(text, encoding="utf-8")
    _fsync(path)


def _fsync(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


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


def _timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
