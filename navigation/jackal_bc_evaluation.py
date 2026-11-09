"""Closed-loop evaluation of the frozen Jackal behavior-cloning policy.

The evaluator loads ``artifacts/behavior_cloning/jackal_6f2/best.pt`` and the
train-split normalizer saved with it. Live control goes through
``JackalGazeboEnv``: one observation in, one Gymnasium action out, and the
environment applies its own velocity limits. A* comparison reuses
``LiveExpertEpisode`` and does not publish on a second command topic.

Preview loads the checkpoint and prints the scenario manifest. It does not
reset Gazebo or publish a velocity. ``--mode smoke``, ``bc``, ``astar``, and
``all`` are the only modes that move the robot.

    python -m navigation.jackal_bc_evaluation --config configs/jackal_behavior_cloning_6f3.yaml
"""

import argparse
import csv
import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from navigation.expert_evaluation_plan import (
    AROUND_GOAL,
    AROUND_START,
    OBSTACLE_BOX,
    SOUTH_GOAL,
    SOUTH_START,
    WORLD_FILENAME,
)
from navigation.expert_evaluation_report import path_length, plot_planned_versus_executed
from navigation.expert_evaluation_runner import verify_stop
from navigation.frame_alignment import alignment_from_poses, points_to_world
from navigation.jackal_bc_data import ObservationNormalizer
from navigation.jackal_bc_policy import load_checkpoint, predict_numpy
from navigation.jackal_bc_schema import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    OBSERVATION_DIM,
    create_new_file,
    feature_names,
)
from navigation.jackal_bc_train import file_sha256
from planners.gazebo_astar import GazeboAStarPlanner


PHASE = "6f.3"
POLICY_NAME = "behavior_cloning"
ASTAR_NAME = "astar_pure_pursuit"
PROTECTED_TRAINING_FILES = (
    "best.pt",
    "final.pt",
    "history.json",
    "metrics.csv",
    "summary.json",
    "architecture.json",
    "preprocessing.json",
    "training_curves.png",
    "resume_state.pt",
    "config.yaml",
)
METRIC_DEFINITIONS = {
    "success": (
        "Behavior cloning: JackalGazeboEnv info goal_reached is true and "
        "collision_detected is false. The environment treats a collision as "
        "terminal before the 0.30 m goal test. "
        "A*: LiveExpertEpisode status is GOAL_REACHED. Pure pursuit uses a "
        "0.15 m goal tolerance, which is stricter than the Gymnasium "
        "environment. A final distance is recorded for both so the two "
        "definitions can be compared, but neither success flag is recomputed "
        "from that distance."
    ),
    "collision": (
        "Behavior cloning: info collision_detected from CollisionMonitor on "
        "/jackal_rl/obstacle_contacts. "
        "A*: status SAFETY_STOP or COLLISION. "
        "These flags are mutually exclusive with success and timeout only "
        "when the corresponding controller reports a single termination "
        "reason. Overlaps are stored rather than cleared."
    ),
    "timeout": (
        "Behavior cloning: info time_limit_reached, which is episode_steps "
        ">= max_episode_steps (300 by default, 5 on the timeout scenario) "
        "without success or collision. "
        "A*: status TIME_LIMIT or TIMEOUT from the wall-clock duration. "
        "A sensor timeout while waiting for LiDAR is an interface failure, "
        "not this timeout."
    ),
    "path_length_m": (
        "Sum of horizontal distances between consecutive recorded poses. "
        "Poses are in the Gazebo world frame when reset provides both a "
        "world pose and odometry. The value is null when no pose was recorded."
    ),
    "planning_only": (
        "A* path length on the static occupancy map. It is not a closed-loop "
        "navigation result and is not a success."
    ),
    "rates": "Each rate uses the number of attempted episodes as its denominator.",
}


@dataclass(frozen=True)
class EvalScenario:
    """One fixed start and goal used by both controllers."""

    scenario_id: str
    category: str
    start: tuple
    goal: tuple
    repeats: int
    astar_duration_s: float
    description: str
    smoke: bool = False
    max_episode_steps: int = None

    def trial_id(self, trial):
        return f"{self.scenario_id}_t{int(trial)}"


def evaluation_scenarios(timeout_steps):
    """Return the fixed scenario list. Definitions do not depend on the dataset."""

    timeout_steps = _positive_int(timeout_steps, "timeout_episode_steps")
    scenarios = (
        EvalScenario(
            "open_south",
            "open",
            tuple(SOUTH_START),
            tuple(SOUTH_GOAL),
            2,
            45.0,
            "Open route south of the box. Same poses as the Phase 6D low-speed case.",
            smoke=True,
        ),
        EvalScenario(
            "obstacle_around",
            "obstacle",
            tuple(AROUND_START),
            tuple(AROUND_GOAL),
            1,
            260.0,
            "South-west to north-east around the box. Same poses as Phase 6D reachable_around.",
        ),
        EvalScenario(
            "obstacle_west_east",
            "obstacle",
            (-1.0, 0.0, 0.0),
            (3.5, 0.0),
            1,
            90.0,
            "West to east. The straight line crosses the inflated box, so A* must go around it.",
        ),
        EvalScenario(
            "obstacle_south_north",
            "obstacle",
            (1.5, -2.2, 0.0),
            (1.5, 2.2),
            1,
            90.0,
            "South to north along the box centerline. The straight line is blocked.",
        ),
        EvalScenario(
            "timeout_open_south",
            "timeout",
            tuple(SOUTH_START),
            tuple(SOUTH_GOAL),
            1,
            0.8,
            "Same open route with a short horizon. This does not command a collision.",
            max_episode_steps=timeout_steps,
        ),
    )
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Duplicate evaluation scenario ids")
    categories = {scenario.category for scenario in scenarios}
    if "open" not in categories or "timeout" not in categories:
        raise RuntimeError("Evaluation scenarios are missing open or timeout coverage")
    if sum(scenario.category == "obstacle" for scenario in scenarios) < 3:
        raise RuntimeError("Evaluation scenarios need at least three obstacle routes")
    return scenarios


def smoke_trials(scenarios):
    """First trial of the designated open-area scenario."""

    selected = [scenario for scenario in scenarios if scenario.smoke]
    if len(selected) != 1:
        raise RuntimeError("Exactly one smoke scenario is required")
    return ((selected[0], 0),)


def iter_trials(scenarios):
    for scenario in scenarios:
        for trial in range(int(scenario.repeats)):
            yield scenario, trial


def load_config(path):
    """Load and check a Phase 6F.3 evaluation config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config)


def validate_config(config):
    """Raise unless this mapping evaluates the frozen Phase 6F.2 checkpoint."""

    if not isinstance(config, dict) or config.get("phase") != PHASE:
        raise ValueError("Configuration phase must be 6f.3")
    if config.get("read_test") is not False:
        raise ValueError("Phase 6F.3 cannot read the test split")
    if config.get("device") not in ("cpu", "auto", "cuda"):
        raise ValueError("device must be cpu, auto, or cuda")
    _nonnegative_int(config.get("seed"), "seed")
    _positive_int(config.get("timeout_episode_steps"), "timeout_episode_steps")
    if config.get("collision_monitoring") is not True:
        raise ValueError("Collision monitoring must stay enabled")
    radius = config.get("inflation_radius")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise ValueError("inflation_radius must be numeric")
    if not math.isfinite(float(radius)) or float(radius) < 0:
        raise ValueError("inflation_radius must be nonnegative")
    for name in (
        "checkpoint",
        "preprocessing",
        "architecture",
        "statistics",
        "training_summary",
        "world",
        "output_directory",
        "expected_checkpoint_sha256",
        "expected_observation_sha256",
    ):
        if not config.get(name):
            raise ValueError(f"{name} is required")
    hashes = config.get("expected_statistics_sha256")
    if not isinstance(hashes, dict) or not hashes.get("json") or not hashes.get("npz"):
        raise ValueError("expected_statistics_sha256.json and .npz are required")
    _positive_int(config.get("expected_normalizer_count"), "expected_normalizer_count")
    return config


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def verify_artifacts(config):
    """Load the frozen checkpoint and reject any normalization mismatch.

    Statistics are read from the Phase 6F.1 file and compared with the
    checkpoint buffers. They are not refit on evaluation data.
    """

    config = validate_config(config)
    checkpoint_path = resolve_repo_path(config["checkpoint"])
    preprocessing_path = resolve_repo_path(config["preprocessing"])
    architecture_path = resolve_repo_path(config["architecture"])
    statistics_path = resolve_repo_path(config["statistics"])
    summary_path = resolve_repo_path(config["training_summary"])
    npz_path = statistics_path.with_suffix(".npz")
    for path in (
        checkpoint_path,
        preprocessing_path,
        architecture_path,
        statistics_path,
        npz_path,
        summary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation artifact is missing: {path}")

    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != config["expected_checkpoint_sha256"]:
        raise ValueError("Checkpoint hash does not match the Phase 6F.2 best.pt pin")
    statistics_hashes = {
        "json": file_sha256(statistics_path),
        "npz": file_sha256(npz_path),
    }
    if statistics_hashes != {
        "json": config["expected_statistics_sha256"]["json"],
        "npz": config["expected_statistics_sha256"]["npz"],
    }:
        raise ValueError("Normalization file hash does not match the Phase 6F.1 pin")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_split_read") is not False:
        raise ValueError("Training summary reports that the test split was read")
    if int(summary.get("normalizer_count", -1)) != int(config["expected_normalizer_count"]):
        raise ValueError("Training summary normalizer count does not match the config pin")
    if summary.get("observation_normalization_sha256") != config["expected_observation_sha256"]:
        raise ValueError("Training summary normalization hash does not match the config pin")
    protected = summary.get("protected_input_sha256") or {}
    if protected.get(str(statistics_path)) != statistics_hashes["json"]:
        raise ValueError("Training summary does not record the statistics JSON hash")
    if protected.get(str(npz_path)) != statistics_hashes["npz"]:
        raise ValueError("Training summary does not record the statistics NPZ hash")

    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    observation = preprocessing.get("observation_normalization") or {}
    action = preprocessing.get("action_scaling") or {}
    if observation.get("observation_sha256") != config["expected_observation_sha256"]:
        raise ValueError("Preprocessing metadata hash does not match the frozen statistics")
    if observation.get("method") != "per_feature_population_zscore":
        raise ValueError("Preprocessing method is not the frozen train z-score")
    if observation.get("fit_during_training") is not False:
        raise ValueError("Preprocessing metadata says statistics were refit during training")
    if int(observation.get("count", -1)) != int(config["expected_normalizer_count"]):
        raise ValueError("Preprocessing normalizer count does not match the config pin")
    if action.get("method") != "identity" or action.get("regression_target") != "gymnasium_action":
        raise ValueError("Action targets must stay identity-scaled Gymnasium actions")
    if list(action.get("order") or []) != ["linear", "angular"]:
        raise ValueError("Action order must be linear, angular")

    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    if int(architecture.get("observation_dim", -1)) != OBSERVATION_DIM:
        raise ValueError("Architecture metadata is not 40-dimensional")
    if int(architecture.get("action_dim", -1)) != ACTION_DIM:
        raise ValueError("Architecture metadata is not 2-dimensional")
    if architecture.get("output_activation") != "tanh":
        raise ValueError("Policy output activation must stay tanh")

    policy, payload = load_checkpoint(checkpoint_path, device=config["device"])
    if payload.get("trained") is not True or payload.get("training_phase") != "6f.2":
        raise ValueError("Checkpoint is not the trained Phase 6F.2 policy")
    if payload.get("action_normalization") != "identity":
        raise ValueError("Checkpoint action normalization is not identity")
    if list(payload.get("action_names") or []) != ["linear", "angular"]:
        raise ValueError("Checkpoint action order is not linear, angular")
    if payload.get("normalization", {}).get("observation_sha256") != config["expected_observation_sha256"]:
        raise ValueError("Checkpoint normalization hash does not match preprocessing")
    if int(payload["normalization"]["count"]) != int(config["expected_normalizer_count"]):
        raise ValueError("Checkpoint normalizer count does not match preprocessing")
    if list(payload["normalization"]["feature_names"]) != list(feature_names()):
        raise ValueError("Checkpoint feature names do not match the 40D layout")
    if architecture.get("network_config") != payload.get("network_config"):
        raise ValueError("architecture.json does not match the checkpoint network")

    normalizer = ObservationNormalizer.load(statistics_path)
    if normalizer.observation_sha256 != config["expected_observation_sha256"]:
        raise ValueError("Loaded statistics hash does not match the checkpoint")
    if int(normalizer.count) != int(config["expected_normalizer_count"]):
        raise ValueError("Loaded statistics count does not match the checkpoint")
    if normalizer.source_split != "train":
        raise ValueError("Evaluation statistics must come from the train split")
    _assert_close(policy.obs_mean.detach().cpu().numpy(), normalizer.mean, "mean")
    _assert_close(policy.obs_std.detach().cpu().numpy(), normalizer.std, "std")
    _assert_close(
        np.asarray(observation["mean"], dtype=np.float64),
        normalizer.mean,
        "preprocessing mean",
    )
    _assert_close(
        np.asarray(observation["std"], dtype=np.float64),
        normalizer.std,
        "preprocessing std",
    )
    prepare_policy(policy)
    return {
        "policy": policy,
        "payload": payload,
        "normalizer": normalizer,
        "checkpoint_sha256": checkpoint_sha,
        "statistics_sha256": statistics_hashes,
        "preprocessing": preprocessing,
        "architecture": architecture,
        "summary": summary,
    }


def prepare_policy(policy):
    """Freeze the module in evaluation mode."""

    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    if policy.training:
        raise RuntimeError("Policy remained in training mode")
    if not policy.normalizer_is_ready:
        raise RuntimeError("Train-split normalizer is not installed")
    return policy


def validate_observation(observation, observation_space=None):
    """Return a contiguous float32 observation or raise."""

    array = np.asarray(observation, dtype=np.float32)
    if array.shape != (OBSERVATION_DIM,):
        raise ValueError(
            f"Observation shape {tuple(array.shape)} is not {(OBSERVATION_DIM,)}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Observation contains NaN or infinity")
    array = np.ascontiguousarray(array, dtype=np.float32)
    if observation_space is not None and not observation_space.contains(array):
        raise ValueError("Observation is outside the Gymnasium observation space")
    return array


def bound_action(action, action_space=None):
    """Accept a 2D action and clip only float noise onto the Gymnasium box.

    Values more than 1e-5 outside [-1, 1] are rejected. This does not convert
    the action into a velocity.
    """

    array = np.asarray(action, dtype=np.float32).reshape(-1)
    if array.shape != (ACTION_DIM,):
        raise ValueError(f"Action shape {tuple(array.shape)} is not {(ACTION_DIM,)}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Action contains NaN or infinity")
    low, high = _action_bounds(action_space)
    if np.any(array < low - 1e-5) or np.any(array > high + 1e-5):
        raise ValueError("Action is outside the Gymnasium action space")
    clipped = np.clip(array, low, high).astype(np.float32, copy=False)
    clipped = np.ascontiguousarray(clipped, dtype=np.float32)
    if action_space is not None and not action_space.contains(clipped):
        raise ValueError("Action is outside the Gymnasium action space")
    return clipped


def infer_action(policy, observation, action_space=None):
    """Score one raw environment observation with gradients disabled."""

    prepare_policy(policy)
    checked = validate_observation(observation, None)
    action = predict_numpy(policy, checked)
    return bound_action(action, action_space)


def run_closed_loop_episode(
    env,
    policy,
    scenario,
    trial,
    *,
    seed,
    predict=None,
    clock=None,
):
    """Run one BC episode and always attempt to stop the robot."""

    predict = infer_action if predict is None else predict
    clock = time.perf_counter if clock is None else clock
    started = clock()
    record = _blank_record(POLICY_NAME, scenario, trial, seed)
    steps = []
    previous_limit = getattr(env, "max_episode_steps", None)
    failure = ""
    info = {}
    try:
        if scenario.max_episode_steps is not None:
            env.max_episode_steps = int(scenario.max_episode_steps)
        observation, info = env.reset(
            seed=int(seed),
            options={"start": scenario.start, "goal": scenario.goal},
        )
        observation = validate_observation(
            observation,
            getattr(env, "observation_space", None),
        )
        alignment = _alignment(env, info)
        record["initial_goal_distance_m"] = _goal_distance(env, info)
        record["frame_alignment"] = alignment
        poses = [_world_pose(env, info, alignment, prefer_world=True)]
        record["poses"] = poses
        limit = int(getattr(env, "max_episode_steps"))
        terminated = False
        truncated = False
        for _ in range(limit):
            policy_observation = observation
            infer_started = clock()
            action = predict(
                policy,
                policy_observation,
                getattr(env, "action_space", None),
            )
            latency = float(clock() - infer_started)
            action = bound_action(action, getattr(env, "action_space", None))
            observation, reward, terminated, truncated, info = env.step(action)
            pose = _world_pose(env, info, alignment, prefer_world=False)
            if pose is not None:
                poses.append(pose)
            steps.append(_step_row(
                scenario,
                trial,
                len(steps),
                policy_observation,
                action,
                reward,
                info,
                pose,
                latency,
                env,
            ))
            if not (terminated or truncated):
                observation = validate_observation(
                    observation,
                    getattr(env, "observation_space", None),
                )
            if terminated or truncated:
                break
        else:
            failure = "evaluator_horizon"
        record["termination_reason"] = _termination_reason(
            info,
            terminated,
            truncated,
            failure,
        )
        record["final_goal_distance_m"] = _goal_distance(env, info)
        record["poses"] = poses
        _apply_outcome_flags(record, info, failure)
        record["return"] = float(sum(step["reward"] for step in steps))
    except Exception as exc:
        failure = _failure_name(exc)
        record["termination_reason"] = failure
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["success"] = False
        record["collision"] = False
        record["timeout"] = False
        if failure in ("invalid_observation", "invalid_action", "interface_failure"):
            record["safety_stop"] = True
    finally:
        if previous_limit is not None and scenario.max_episode_steps is not None:
            env.max_episode_steps = previous_limit
        verified, stop_error = confirm_stop(env)
        record["stop_verified"] = verified
        if stop_error:
            record["error"] = (
                f"{record['error']}; {stop_error}" if record["error"] else stop_error
            )
            record["termination_reason"] = "cleanup_failed"
            record["safety_stop"] = True
    record["wall_time_s"] = float(clock() - started)
    record["simulation_time_s"] = _simulation_time(steps)
    record["control_steps"] = len(steps)
    record["invalid_observations"] = int(failure == "invalid_observation")
    record["invalid_actions"] = int(failure == "invalid_action")
    record["safety_stop_events"] = int(bool(record["safety_stop"]))
    record["completed"] = bool(record["termination_reason"]) and verified
    _fill_distances_and_diagnostics(record, steps, scenario)
    return record, steps


def run_astar_episode(runner, scenario, trial, *, seed, clock=None):
    """Run one existing A* episode. The runner owns the command publisher."""

    clock = time.perf_counter if clock is None else clock
    started = clock()
    record = _blank_record(ASTAR_NAME, scenario, trial, seed)
    record["astar_duration_s"] = float(scenario.astar_duration_s)
    result = None
    try:
        runner.prepare(start=scenario.start, goal=scenario.goal)
        result = runner.run(duration=float(scenario.astar_duration_s))
        status = str(result.status)
        record["termination_reason"] = status
        record["success"] = status == "GOAL_REACHED"
        record["collision"] = status in ("SAFETY_STOP", "COLLISION")
        record["timeout"] = status in ("TIME_LIMIT", "TIMEOUT")
        record["flags_overlap"] = (
            int(record["success"]) + int(record["collision"]) + int(record["timeout"])
        ) > 1
        poses = _astar_world_poses(result)
        record["poses"] = poses
        record["planned_path"] = _planned_world_path(result)
        record["control_steps"] = int(getattr(result, "accepted_commands", 0))
        record["rejected_commands"] = int(getattr(result, "rejected_commands", 0))
        if poses:
            record["initial_goal_distance_m"] = _hypot(poses[0], scenario.goal)
            record["final_goal_distance_m"] = _hypot(poses[-1], scenario.goal)
        record["return"] = None
    except Exception as exc:
        record["termination_reason"] = "interface_failure"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["safety_stop"] = True
    finally:
        robot = getattr(runner, "robot", None)
        env = getattr(runner, "env", None)
        target = env if env is not None and getattr(env, "robot", None) is not None else runner
        if robot is not None or env is not None:
            verified, stop_error = confirm_stop(target if env is not None else _RobotEnv(robot))
        else:
            verified, stop_error = False, "A* runner has no robot stop interface"
        record["stop_verified"] = verified
        if stop_error:
            record["error"] = (
                f"{record['error']}; {stop_error}" if record["error"] else stop_error
            )
            record["termination_reason"] = "cleanup_failed"
            record["safety_stop"] = True
    record["wall_time_s"] = float(clock() - started)
    record["completed"] = bool(record["termination_reason"]) and record["stop_verified"]
    record["path_length_m"] = path_length(record["poses"])
    record["safety_stop_events"] = int(bool(record["safety_stop"]))
    record["diagnosis"] = diagnose_episode(record, [])
    return record


def plan_scenarios(scenarios, world_path, inflation_radius):
    """Plan every scenario on the static map. This does not move the robot."""

    planner = GazeboAStarPlanner(
        world_path,
        inflation_radius=float(inflation_radius),
    )
    plans = {}
    for scenario in scenarios:
        path = planner.plan(scenario.start[:2], scenario.goal)
        valid = bool(path) and bool(planner.validate_path(path))
        plans[scenario.scenario_id] = {
            "planning_only": True,
            "reachable": valid,
            "planned_path": [list(map(float, point[:2])) for point in path] if valid else [],
            "planned_length_m": path_length(path) if valid else None,
            "waypoint_count": 0 if not valid else len(path),
        }
    return plans


def aggregate_episodes(episodes):
    """Rate each flag against the number of attempted episodes."""

    attempted = len(episodes)
    counts = {
        name: sum(bool(episode.get(name)) for episode in episodes)
        for name in ("success", "collision", "timeout", "safety_stop", "completed")
    }
    rates = {
        name: {
            "count": counts[name],
            "attempted": attempted,
            "rate": (counts[name] / attempted) if attempted else None,
        }
        for name in counts
    }
    return {
        "attempted_episodes": attempted,
        "completed_episodes": counts["completed"],
        **rates,
    }


def diagnose_episode(record, steps):
    """Separate measured trajectory facts from later-phase hypotheses."""

    facts = []
    hypotheses = []
    angular = [float(step["angular_action"]) for step in steps if step.get("angular_action") is not None]
    linear = [float(step["linear_action"]) for step in steps if step.get("linear_action") is not None]
    sign_changes = _sign_changes(angular)
    progress_values = [
        float(step["progress_m"])
        for step in steps
        if step.get("progress_m") is not None
    ]
    stall_fraction = None
    if progress_values:
        stalled = sum(value <= 0.0 for value in progress_values)
        stall_fraction = stalled / len(progress_values)
    headings = [
        abs(float(step["goal_heading_norm"]))
        for step in steps
        if step.get("goal_heading_norm") is not None
    ]
    initial = record.get("initial_goal_distance_m")
    final = record.get("final_goal_distance_m")
    reduction = None
    if initial is not None and final is not None:
        reduction = float(initial) - float(final)
        facts.append(
            f"Measured goal distance changed from {initial:.3f} m to {final:.3f} m."
        )
    recorded_steps = len(steps) if steps else int(record.get("control_steps") or 0)
    facts.append(f"Control steps: {recorded_steps}.")
    if angular:
        facts.append(
            f"Angular action changed sign {sign_changes} times; "
            f"mean absolute angular action was {float(np.mean(np.abs(angular))):.3f}."
        )
    if linear:
        facts.append(
            f"Mean absolute linear action was {float(np.mean(np.abs(linear))):.3f}."
        )
    if stall_fraction is not None:
        facts.append(
            f"Fraction of steps with no positive progress was {stall_fraction:.3f}."
        )
    if (
        record.get("path_length_m") is not None
        and record.get("simulation_time_s") not in (None, 0)
    ):
        speed = float(record["path_length_m"]) / float(record["simulation_time_s"])
        facts.append(
            f"Average speed along the recorded path was {speed:.3f} m/s "
            "over the recorded simulation time."
        )
    if headings:
        facts.append(
            f"Mean absolute normalized goal heading was {float(np.mean(headings)):.3f}."
        )
    rejected = sum(step.get("action_accepted") is False for step in steps)
    if rejected:
        facts.append(f"The environment safety interface rejected {rejected} actions.")
    if record.get("collision"):
        facts.append("The episode termination was a collision or contact safety stop.")
    if record.get("timeout"):
        facts.append("The episode termination was the controller time horizon.")
    if record.get("invalid_observations") or record.get("invalid_actions"):
        facts.append("Inference input or output failed the finite-value check.")

    if record.get("success"):
        hypotheses.append(
            "The recorded termination met that controller's goal condition, "
            "so this episode is not an uncorrected compounding failure."
        )
    elif (
        sign_changes >= 4
        and angular
        and float(np.mean(np.abs(angular))) >= 0.25
        and not record.get("collision")
    ):
        hypotheses.append(
            "Repeated angular sign changes at a large magnitude are consistent "
            "with oscillation. That pattern can come from compounding imitation "
            "error, from a live observation the training set rarely contained, "
            "or from the velocity smoother. The trace does not isolate one cause."
        )
    elif record.get("timeout") and reduction is not None and reduction > 0:
        hypotheses.append(
            "Distance decreased, and the episode then hit the controller horizon. "
            "One explanation is that the commanded speed cannot cover the route "
            "inside that horizon. That is separate from a claim that heading "
            "error compounded."
        )
    elif (
        reduction is not None
        and reduction < 0.25
        and not record.get("collision")
        and not record.get("invalid_actions")
        and record.get("termination_reason") not in ("invalid_observation", "interface_failure")
    ):
        hypotheses.append(
            "Goal distance did not decrease by a quarter of a metre. "
            "Possible causes include a poor initial action, a safety rejection, "
            "or accumulated heading error. This is a hypothesis, not a finding "
            "that the network weights drifted."
        )
    if record.get("policy") == POLICY_NAME and record.get("category") == "obstacle" and not record.get("success"):
        hypotheses.append(
            "A later phase could compare these live observations with the "
            "frozen training distribution, or add a recovery controller. "
            "Phase 6F.3 does not retrain the policy."
        )
    return {
        "facts": facts,
        "hypotheses": hypotheses,
        "angular_sign_changes": sign_changes,
        "stall_step_fraction": stall_fraction,
        "distance_reduction_m": reduction,
    }


def comparison_table(bc_episodes, astar_episodes):
    """Join BC and A* on scenario id and trial."""

    astar_index = {
        (episode["scenario_id"], episode["trial"]): episode
        for episode in astar_episodes
    }
    rows = []
    for episode in bc_episodes:
        other = astar_index.get((episode["scenario_id"], episode["trial"]))
        rows.append({
            "scenario_id": episode["scenario_id"],
            "trial": episode["trial"],
            "category": episode["category"],
            "bc_success": episode["success"],
            "bc_collision": episode["collision"],
            "bc_timeout": episode["timeout"],
            "bc_termination_reason": episode["termination_reason"],
            "bc_path_length_m": episode["path_length_m"],
            "bc_final_goal_distance_m": episode["final_goal_distance_m"],
            "astar_success": None if other is None else other["success"],
            "astar_collision": None if other is None else other["collision"],
            "astar_timeout": None if other is None else other["timeout"],
            "astar_termination_reason": None if other is None else other["termination_reason"],
            "astar_path_length_m": None if other is None else other["path_length_m"],
            "astar_final_goal_distance_m": None if other is None else other["final_goal_distance_m"],
            "same_start_goal": True,
        })
    return rows


def allocate_output(root):
    """Create a new output directory and refuse to reuse a non-empty one."""

    root = Path(root)
    if not root.exists():
        root.mkdir(parents=True)
        return root
    if not any(root.iterdir()):
        return root
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = root / f"run_{stamp}"
    if candidate.exists():
        candidate = root / f"run_{stamp}_{os.getpid()}"
    if candidate.exists():
        raise FileExistsError(f"Refusing to overwrite {candidate}")
    candidate.mkdir()
    return candidate


def save_evaluation(output_dir, summary, episodes, step_rows, plans):
    """Write results. Existing files in the directory are left untouched."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "scenario_manifest.json", summary["scenario_manifest"])
    _write_json(output_dir / "metric_definitions.json", METRIC_DEFINITIONS)
    _write_json(output_dir / "planning_only.json", plans)
    for episode, steps in zip(episodes, step_rows):
        stem = f"{episode['policy']}_{episode['episode_id']}"
        _write_json(output_dir / f"{stem}.json", _public_episode(episode))
        _write_step_csv(output_dir / f"{stem}_steps.csv", steps)
        _write_step_npz(output_dir / f"{stem}_steps.npz", steps)
        _plot_episode(output_dir / f"{stem}.png", episode, plans)
        _plot_diagnostics(output_dir / f"{stem}_diagnostics.png", episode, steps)
    _write_episode_csv(output_dir / "episodes.csv", episodes)
    _plot_overview(output_dir / "trajectories.png", episodes, plans)
    _plot_comparison(output_dir / "comparison.png", summary)
    _write_json(output_dir / "comparison.json", summary.get("comparison", []))
    _write_json(output_dir / "summary.json", summary)
    report = render_report(summary)
    _write_text(output_dir / "report.md", report)
    return output_dir / "report.md"


def render_report(summary):
    """Describe measured results. Missing live trials stay missing."""

    lines = [
        "# Phase 6F.3 closed-loop behavior-cloning evaluation",
        "",
        "This report records the frozen Phase 6F.2 policy. The checkpoint was not trained or modified.",
        "",
        f"- Checkpoint: `{summary.get('checkpoint')}`",
        f"- Checkpoint sha256: `{summary.get('checkpoint_sha256')}`",
        f"- Observation normalization sha256: `{summary.get('observation_sha256')}`",
        f"- Normalizer count: {summary.get('normalizer_count')}",
        f"- World: `{summary.get('world')}`",
        f"- Live robot execution: {summary.get('live_executed')}",
        f"- Output: `{summary.get('output_directory')}`",
        "",
        "## Metric definitions",
        "",
    ]
    for name, text in METRIC_DEFINITIONS.items():
        lines.append(f"- **{name}.** {text}")
    lines.extend([
        "",
        "## Controller differences",
        "",
        "Behavior cloning steps `JackalGazeboEnv`. Each action is the tanh network output in `[-1, 1]` and is passed to `env.step` without a second velocity scaling. The environment's `ActionMapper` then limits speed to 0.30 m/s forward, 0.15 m/s reverse, and 0.60 rad/s. Goal success uses the environment tolerance of 0.30 m. The default horizon is 300 physics steps of 100 iterations each.",
        "",
        "A* uses `LiveExpertEpisode`, `GazeboAStarPlanner`, and `PurePursuitController` (`goal_tolerance` 0.15 m, `max_linear` 0.25 m/s, `max_angular` 0.60 rad/s). Desired velocities are converted once with `velocity_to_action` and published by the same robot interface. The episode horizon is a wall-clock duration while Gazebo is resumed, not the 300-step Gymnasium horizon. Only one of these controllers is constructed at a time.",
        "",
        "Planning-only path lengths come from the static occupancy map. They are not navigation successes.",
        "",
        "## Episodes",
        "",
    ])
    episodes = summary.get("episodes") or []
    if not episodes:
        lines.append("No closed-loop episode was run. Unit tests and planning-only checks do not establish live navigation success.")
    else:
        lines.append("| Episode | Policy | Reason | Success | Collision | Timeout | Path m | Final goal m | Steps |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for episode in episodes:
            lines.append(
                "| {episode_id} | {policy} | {termination_reason} | {success} | {collision} | {timeout} | {path} | {final} | {steps} |".format(
                    episode_id=episode["episode_id"],
                    policy=episode["policy"],
                    termination_reason=episode["termination_reason"],
                    success=episode["success"],
                    collision=episode["collision"],
                    timeout=episode["timeout"],
                    path=_format_number(episode.get("path_length_m")),
                    final=_format_number(episode.get("final_goal_distance_m")),
                    steps=episode.get("control_steps"),
                )
            )
        lines.append("")
        errors = [episode for episode in episodes if episode.get("error")]
        if errors:
            lines.append("Interface and cleanup errors:")
            lines.append("")
            for episode in errors:
                lines.append(
                    f"- `{episode['policy']}` `{episode['episode_id']}`: {episode['error']}"
                )
    lines.extend(["", "## Aggregates", ""])
    for name in ("behavior_cloning", "astar_pure_pursuit"):
        block = (summary.get("aggregates") or {}).get(name)
        if not block:
            lines.append(f"{name}: not run.")
            continue
        lines.append(
            f"{name}: attempted {block['attempted_episodes']}, "
            f"completed {block['completed_episodes']}, "
            f"success {block['success']['count']}/{block['success']['attempted']}, "
            f"collision {block['collision']['count']}/{block['collision']['attempted']}, "
            f"timeout {block['timeout']['count']}/{block['timeout']['attempted']}."
        )
    lines.extend(["", "## Failure notes", ""])
    notes = summary.get("diagnoses") or []
    if not notes:
        lines.append("No trajectory diagnosis was computed.")
    for note in notes:
        lines.append(f"### {note['episode_id']}")
        lines.append("")
        lines.append("Observed:")
        for fact in note["facts"]:
            lines.append(f"- {fact}")
        lines.append("")
        lines.append("Hypotheses for a later phase:")
        if note["hypotheses"]:
            for item in note["hypotheses"]:
                lines.append(f"- {item}")
        else:
            lines.append("- None from the recorded trace.")
        lines.append("")
    blocked = summary.get("blocked_reason")
    if blocked:
        lines.extend(["## Blocked", "", blocked, ""])
    lines.extend([
        "## Limitations",
        "",
        "- The Gymnasium goal tolerance is 0.30 m and the A* pure-pursuit tolerance is 0.15 m. Success flags are therefore not the same predicate.",
        "- Behavior-cloning episodes advance a fixed number of physics iterations. A* episodes resume the simulator against a wall clock. Durations are not interchangeable.",
        "- The timeout scenario shortens the behavior-cloning horizon to the configured step limit and the A* horizon to 0.8 s. It does not drive the robot into the obstacle.",
        "- Evaluation does not open the dataset test split and does not update normalization statistics.",
        "",
    ])
    if summary.get("gazebo_backend") is False:
        lines.extend([
            "This output used injected test interfaces. It is not a Gazebo navigation result.",
            "",
        ])
    lines.extend([
        "## Commands",
        "",
        "```text",
        "python -m navigation.jackal_bc_evaluation --config configs/jackal_behavior_cloning_6f3.yaml",
        "python -m navigation.jackal_bc_evaluation --config configs/jackal_behavior_cloning_6f3.yaml --mode smoke",
        "python -m navigation.jackal_bc_evaluation --config configs/jackal_behavior_cloning_6f3.yaml --mode all",
        "```",
        "",
    ])
    return "\n".join(lines)


def execute(
    config,
    mode,
    *,
    output_dir=None,
    preflight=None,
    publisher_count=None,
    environment_factory=None,
    astar_runner_factory=None,
    clock=None,
):
    """Verify artifacts, then evaluate only the requested mode.

    ``environment_factory`` and ``astar_runner_factory`` exist so tests can
    supply fakes. The default factories construct the real Gymnasium
    environment and the existing live A* runner.
    """

    if mode not in ("smoke", "bc", "astar", "all"):
        raise ValueError("Execution mode must be smoke, bc, astar, or all")
    gazebo_backend = environment_factory is None and astar_runner_factory is None
    verified = verify_artifacts(config)
    scenarios = evaluation_scenarios(config["timeout_episode_steps"])
    world = resolve_repo_path(config["world"])
    if world.name != WORLD_FILENAME:
        raise ValueError(f"Scenarios are defined for {WORLD_FILENAME}")
    plans = plan_scenarios(scenarios, world, config["inflation_radius"])
    if any(not plan["reachable"] for plan in plans.values()):
        raise RuntimeError("A fixed evaluation scenario is not reachable on the static map")

    root = resolve_repo_path(config["output_directory"])
    output = Path(output_dir) if output_dir is not None else allocate_output(root)
    training_dir = resolve_repo_path(config["checkpoint"]).parent
    before = training_snapshot(training_dir)
    manifest = scenario_manifest(config, scenarios, plans, verified)
    summary = _base_summary(config, verified, manifest, output, mode)
    summary["gazebo_backend"] = gazebo_backend
    episodes = []
    step_rows = []
    smoke_safe = False
    env = None
    runner = None
    try:
        if preflight is None:
            from navigation.expert_collection_preflight import run_preflight
            preflight = run_preflight
        preflight_result = preflight()
        summary["preflight"] = {
            "passed": bool(preflight_result.passed),
            "checks": dict(preflight_result.checks),
        }
        if not preflight_result.passed:
            summary["blocked_reason"] = (
                "Preflight failed. No behavior-cloning or A* episode was started."
            )
            summary["live_executed"] = False
            return _finish(output, summary, episodes, step_rows, plans, training_dir, before)

        if publisher_count is None:
            from navigation.evaluate_expert_navigation import _publisher_count
            publisher_count = _publisher_count
        publishers = int(publisher_count())
        summary["publisher_count_before"] = publishers
        if publishers != 0:
            summary["blocked_reason"] = (
                "The velocity topic already has "
                f"{publishers} publisher(s). No second controller was started."
            )
            summary["live_executed"] = False
            return _finish(output, summary, episodes, step_rows, plans, training_dir, before)

        if mode in ("smoke", "bc", "all"):
            env = _make_environment(config, environment_factory)
            summary["goal_tolerance_m"] = float(env.goal_tolerance)
            summary["max_episode_steps"] = int(env.max_episode_steps)
            summary["live_executed"] = True
            bc_trials = list(smoke_trials(scenarios) if mode == "smoke" else iter_trials(scenarios))
            for index, (scenario, trial) in enumerate(bc_trials):
                if index > 0 and not smoke_safe:
                    summary["blocked_reason"] = (
                        "The open-area smoke trial did not finish safely. "
                        "Obstacle, timeout, and A* trials were not started."
                    )
                    break
                print(
                    f"behavior_cloning {scenario.scenario_id} trial {trial}",
                    flush=True,
                )
                episode, steps = run_closed_loop_episode(
                    env,
                    verified["policy"],
                    scenario,
                    trial,
                    seed=int(config["seed"]),
                    clock=clock,
                )
                _attach_plan(episode, plans)
                episodes.append(episode)
                step_rows.append(steps)
                if scenario.smoke and trial == 0:
                    smoke_safe = _smoke_is_safe(episode)
                    summary["bc_smoke_safe"] = smoke_safe
            confirm_stop(env)
        else:
            summary["bc_smoke_safe"] = None

        if mode in ("astar", "all") and (mode == "astar" or smoke_safe):
            if env is not None:
                env.close()
                env = None
            publishers_after_bc = int(publisher_count())
            summary["publisher_count_after_bc"] = publishers_after_bc
            if publishers_after_bc != 0:
                summary["blocked_reason"] = (
                    "A velocity publisher was still active after the "
                    "behavior-cloning environment closed. A* was not started."
                )
            else:
                runner = _make_astar_runner(config, world, astar_runner_factory)
                summary["live_executed"] = True
                for scenario, trial in iter_trials(scenarios):
                    print(
                        f"astar_pure_pursuit {scenario.scenario_id} trial {trial}",
                        flush=True,
                    )
                    episode = run_astar_episode(
                        runner,
                        scenario,
                        trial,
                        seed=int(config["seed"]),
                        clock=clock,
                    )
                    _attach_plan(episode, plans)
                    episodes.append(episode)
                    step_rows.append([])
    finally:
        if env is not None:
            try:
                confirm_stop(env)
            finally:
                close = getattr(env, "close", None)
                if close is not None:
                    close()
        if runner is not None:
            close = getattr(runner, "close", None)
            if close is not None:
                close()
    return _finish(output, summary, episodes, step_rows, plans, training_dir, before)


def training_snapshot(training_dir):
    """Hash the Phase 6F.2 artifacts that evaluation must not modify."""

    training_dir = Path(training_dir)
    snapshot = {}
    for name in PROTECTED_TRAINING_FILES:
        path = training_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Protected training artifact is missing: {path}")
        snapshot[name] = file_sha256(path)
    statistics = resolve_repo_path(
        "artifacts/behavior_cloning/jackal_6f1/observation_stats.json"
    )
    snapshot["observation_stats.json"] = file_sha256(statistics)
    snapshot["observation_stats.npz"] = file_sha256(statistics.with_suffix(".npz"))
    return snapshot


def assert_training_unchanged(training_dir, before):
    after = training_snapshot(training_dir)
    if after != before:
        changed = sorted(name for name in before if before[name] != after.get(name))
        raise RuntimeError(f"Evaluation modified training artifacts: {changed}")
    return after


def preview(config):
    """Load the checkpoint and describe the scenarios without robot motion."""

    verified = verify_artifacts(config)
    scenarios = evaluation_scenarios(config["timeout_episode_steps"])
    world = resolve_repo_path(config["world"])
    plans = plan_scenarios(scenarios, world, config["inflation_radius"])
    payload = scenario_manifest(config, scenarios, plans, verified)
    payload["motion"] = False
    payload["checkpoint_sha256"] = verified["checkpoint_sha256"]
    payload["observation_sha256"] = verified["normalizer"].observation_sha256
    payload["normalizer_count"] = int(verified["normalizer"].count)
    payload["device"] = str(next(verified["policy"].parameters()).device)
    payload["policy_training_mode"] = bool(verified["policy"].training)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 6F.3 closed-loop Jackal behavior-cloning evaluation",
    )
    parser.add_argument(
        "--config",
        default="configs/jackal_behavior_cloning_6f3.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=("preview", "smoke", "bc", "astar", "all"),
        default="preview",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional empty directory. An existing non-empty directory is rejected.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.mode == "preview":
        print(json.dumps(preview(config), indent=2, sort_keys=True))
        return 0
    if args.output_dir is not None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    summary = execute(config, args.mode, output_dir=args.output_dir)
    print(json.dumps({
        "output_directory": summary["output_directory"],
        "live_executed": summary["live_executed"],
        "blocked_reason": summary.get("blocked_reason", ""),
        "bc_smoke_safe": summary.get("bc_smoke_safe"),
        "aggregates": summary.get("aggregates"),
    }, indent=2, sort_keys=True))
    return 0 if not summary.get("blocked_reason") else 2


class _RobotEnv:
    """Expose a runner robot to ``confirm_stop``."""

    def __init__(self, robot):
        self.robot = robot


def confirm_stop(env):
    """Stop through the robot interface and record whether zero was published."""

    robot = getattr(env, "robot", None)
    if robot is None or not hasattr(robot, "stop"):
        return False, "Environment has no robot.stop interface"
    try:
        info = verify_stop(robot)
    except Exception as exc:
        return False, f"Cleanup failure: {type(exc).__name__}: {exc}"
    if info.get("stop_verified"):
        return True, ""
    return False, "Stop command was not zero linear and angular velocity"


def scenario_manifest(config, scenarios, plans, verified):
    return {
        "phase": PHASE,
        "world": str(resolve_repo_path(config["world"])),
        "world_filename": WORLD_FILENAME,
        "obstacle_box": list(OBSTACLE_BOX),
        "inflation_radius": float(config["inflation_radius"]),
        "seed": int(config["seed"]),
        "checkpoint": str(resolve_repo_path(config["checkpoint"])),
        "checkpoint_sha256": verified["checkpoint_sha256"],
        "observation_sha256": config["expected_observation_sha256"],
        "normalizer_count": int(config["expected_normalizer_count"]),
        "action_scaling": "identity",
        "bc_goal_tolerance_m": 0.30,
        "astar_goal_tolerance_m": 0.15,
        "read_test": False,
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "category": scenario.category,
                "start": list(scenario.start),
                "goal": list(scenario.goal),
                "repeats": scenario.repeats,
                "smoke": scenario.smoke,
                "max_episode_steps": scenario.max_episode_steps,
                "astar_duration_s": scenario.astar_duration_s,
                "description": scenario.description,
                "planning_only": plans[scenario.scenario_id],
            }
            for scenario in scenarios
        ],
    }


def _make_environment(config, environment_factory):
    if environment_factory is not None:
        return environment_factory()
    from jackal_rl_navigation.collision_config import CollisionConfig
    from jackal_rl_navigation.gazebo_env import JackalGazeboEnv

    return JackalGazeboEnv(collision_config=CollisionConfig(enabled=True))


def _make_astar_runner(config, world, astar_runner_factory):
    if astar_runner_factory is not None:
        return astar_runner_factory()
    from navigation.expert_live_runner_factory import create_live_runner

    return create_live_runner(
        None,
        world_path=world,
        inflation_radius=float(config["inflation_radius"]),
        allow_live=True,
    )


def _finish(output, summary, episodes, step_rows, plans, training_dir, before):
    summary["episodes"] = [_public_episode(episode) for episode in episodes]
    summary["aggregates"] = {
        POLICY_NAME: aggregate_episodes(
            [episode for episode in episodes if episode["policy"] == POLICY_NAME]
        ),
        ASTAR_NAME: aggregate_episodes(
            [episode for episode in episodes if episode["policy"] == ASTAR_NAME]
        ),
    }
    summary["comparison"] = comparison_table(
        [episode for episode in episodes if episode["policy"] == POLICY_NAME],
        [episode for episode in episodes if episode["policy"] == ASTAR_NAME],
    )
    summary["diagnoses"] = [
        {"episode_id": episode["episode_id"], **episode["diagnosis"]}
        for episode in episodes
        if episode.get("diagnosis")
    ]
    summary["training_artifacts_unchanged"] = True
    assert_training_unchanged(training_dir, before)
    save_evaluation(output, summary, episodes, step_rows, plans)
    summary["report"] = str(Path(output) / "report.md")
    return summary


def _base_summary(config, verified, manifest, output, mode):
    import torch

    return {
        "phase": PHASE,
        "mode": mode,
        "live_executed": False,
        "blocked_reason": "",
        "bc_smoke_safe": False,
        "checkpoint": str(resolve_repo_path(config["checkpoint"])),
        "checkpoint_sha256": verified["checkpoint_sha256"],
        "observation_sha256": verified["normalizer"].observation_sha256,
        "normalizer_count": int(verified["normalizer"].count),
        "action_scaling": "identity",
        "world": manifest["world"],
        "seed": int(config["seed"]),
        "output_directory": str(output),
        "scenario_manifest": manifest,
        "read_test": False,
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "metric_definitions": METRIC_DEFINITIONS,
    }


def _blank_record(policy, scenario, trial, seed):
    return {
        "policy": policy,
        "episode_id": scenario.trial_id(trial),
        "scenario_id": scenario.scenario_id,
        "trial": int(trial),
        "category": scenario.category,
        "seed": int(seed),
        "start": list(scenario.start),
        "goal": list(scenario.goal),
        "description": scenario.description,
        "success": False,
        "collision": False,
        "timeout": False,
        "safety_stop": False,
        "flags_overlap": False,
        "termination_reason": "",
        "completed": False,
        "stop_verified": False,
        "error": "",
        "wall_time_s": None,
        "simulation_time_s": None,
        "path_length_m": None,
        "initial_goal_distance_m": _hypot(scenario.start, scenario.goal),
        "final_goal_distance_m": None,
        "control_steps": 0,
        "invalid_observations": 0,
        "invalid_actions": 0,
        "safety_stop_events": 0,
        "return": None,
        "mean_inference_latency_s": None,
        "max_inference_latency_s": None,
        "planned_path": [],
        "planned_length_m": None,
        "planning_only": False,
        "poses": [],
        "rejected_commands": 0,
        "diagnosis": None,
    }


def _step_row(scenario, trial, index, observation, action, reward, info, pose, latency, env):
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    heading = None
    if observation.shape == (OBSERVATION_DIM,) and np.all(np.isfinite(observation)):
        heading = float(observation[37])
        linear_velocity = float(observation[38])
        angular_velocity = float(observation[39])
    else:
        linear_velocity = None
        angular_velocity = None
    robot = getattr(env, "robot", None)
    return {
        "episode_id": scenario.trial_id(trial),
        "step": int(index),
        "linear_action": float(action[0]),
        "angular_action": float(action[1]),
        "observation": np.asarray(observation, dtype=np.float32).copy(),
        "goal_distance_m": _optional_float(info.get("current_distance")),
        "goal_heading_norm": heading,
        "linear_velocity_norm": linear_velocity,
        "angular_velocity_norm": angular_velocity,
        "progress_m": _optional_float(info.get("progress")),
        "pose_x": None if pose is None else float(pose[0]),
        "pose_y": None if pose is None else float(pose[1]),
        "pose_yaw": None if pose is None else float(pose[2]),
        "reward": float(reward),
        "scan_stamp": _optional_float(info.get("scan_stamp")),
        "odom_stamp": _optional_float(info.get("odom_stamp")),
        "inference_latency_s": float(latency),
        "action_accepted": info.get("action_accepted"),
        "safety_reason": str(getattr(robot, "last_safety_reason", "") or ""),
        "termination_reason": info.get("termination_reason"),
    }


def _apply_outcome_flags(record, info, failure):
    if failure:
        record["success"] = False
        record["collision"] = bool(info.get("collision_detected"))
        record["timeout"] = bool(info.get("time_limit_reached"))
        record["safety_stop"] = True
    else:
        record["collision"] = bool(info.get("collision_detected"))
        record["timeout"] = bool(info.get("time_limit_reached"))
        record["success"] = bool(info.get("goal_reached")) and not record["collision"]
        record["safety_stop"] = record["collision"]
    record["flags_overlap"] = (
        int(record["success"]) + int(record["collision"]) + int(record["timeout"])
    ) > 1


def _termination_reason(info, terminated, truncated, failure):
    if failure:
        return failure
    reason = info.get("termination_reason")
    if reason:
        return str(reason)
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return "running"


def _failure_name(exc):
    message = str(exc).lower()
    if isinstance(exc, TimeoutError) or "sensor" in message or "unavailable" in message:
        return "interface_failure"
    if "observation" in message:
        return "invalid_observation"
    if "action" in message:
        return "invalid_action"
    return "interface_failure"


def _fill_distances_and_diagnostics(record, steps, scenario):
    if record.get("final_goal_distance_m") is None and steps:
        record["final_goal_distance_m"] = steps[-1].get("goal_distance_m")
    poses = [pose for pose in record.get("poses") or [] if pose is not None]
    record["path_length_m"] = path_length(poses)
    latencies = [step["inference_latency_s"] for step in steps]
    if latencies:
        record["mean_inference_latency_s"] = float(np.mean(latencies))
        record["max_inference_latency_s"] = float(np.max(latencies))
    if record.get("initial_goal_distance_m") is None:
        record["initial_goal_distance_m"] = _hypot(scenario.start, scenario.goal)
    record["diagnosis"] = diagnose_episode(record, steps)


def _public_episode(episode):
    hidden = {"poses"}
    return {
        key: _json_ready(value)
        for key, value in episode.items()
        if key not in hidden
    }


def _alignment(env, info):
    world = info.get("robot_world_pose") if info else None
    robot = getattr(env, "robot", None)
    odom = getattr(robot, "odom", None) if robot is not None else None
    if world is None or odom is None or len(world) < 4:
        return None
    from navigation.live_expert_episode import LiveExpertEpisode

    return alignment_from_poses(
        (float(world[0]), float(world[1]), float(world[3])),
        LiveExpertEpisode.pose_from_odom(odom),
    )


def _world_pose(env, info, alignment, prefer_world):
    if prefer_world and info and info.get("robot_world_pose") is not None:
        world = info["robot_world_pose"]
        if len(world) >= 4:
            return (float(world[0]), float(world[1]), float(world[3]))
    if info and info.get("robot_pose") is not None:
        pose = tuple(float(value) for value in info["robot_pose"][:3])
        return pose
    robot = getattr(env, "robot", None)
    odom = getattr(robot, "odom", None) if robot is not None else None
    if odom is None:
        return None
    from navigation.live_expert_episode import LiveExpertEpisode

    pose = LiveExpertEpisode.pose_from_odom(odom)
    if alignment is None:
        return pose
    world = points_to_world([pose], alignment)[0]
    return (float(world[0]), float(world[1]), float(pose[2]))


def _goal_distance(env, info):
    if info and info.get("current_distance") is not None:
        return float(info["current_distance"])
    robot = getattr(env, "robot", None)
    if robot is not None and hasattr(robot, "_goal_relative"):
        relative = robot._goal_relative()
        if relative is not None and math.isfinite(float(relative[0])):
            return float(relative[0])
    distance = getattr(env, "previous_goal_distance", None)
    if distance is not None and math.isfinite(float(distance)):
        return float(distance)
    return None


def _astar_world_poses(result):
    poses = [tuple(map(float, pose[:3])) for pose in getattr(result, "executed_poses", [])]
    alignment = getattr(result, "frame_alignment", None)
    if not poses or not alignment:
        return poses
    converted = points_to_world(poses, alignment)
    return [
        (float(point[0]), float(point[1]), float(pose[2]))
        for point, pose in zip(converted, poses)
    ]


def _planned_world_path(result):
    path = [tuple(map(float, point[:2])) for point in getattr(result, "planned_path", [])]
    alignment = getattr(result, "frame_alignment", None)
    if not path or not alignment:
        return [list(point) for point in path]
    return [list(map(float, point[:2])) for point in points_to_world(path, alignment)]


def _attach_plan(episode, plans):
    plan = plans.get(episode["scenario_id"]) or {}
    episode["planned_path"] = plan.get("planned_path") or episode.get("planned_path") or []
    episode["planned_length_m"] = plan.get("planned_length_m")
    episode["planning_only"] = False


def _simulation_time(steps):
    stamps = [
        step["odom_stamp"]
        for step in steps
        if step.get("odom_stamp") is not None
    ]
    if len(stamps) < 2:
        return None
    return float(stamps[-1] - stamps[0])


def _smoke_is_safe(episode):
    """A smoke trial is safe when control was valid and the robot stopped.

    Reaching the goal is not required. A collision, an invalid action, or a
    failed stop blocks every later scenario.
    """

    if not episode.get("stop_verified"):
        return False
    if episode.get("invalid_actions") or episode.get("invalid_observations"):
        return False
    if episode.get("termination_reason") in (
        "cleanup_failed",
        "invalid_action",
        "invalid_observation",
        "interface_failure",
        "evaluator_horizon",
    ):
        return False
    if episode.get("collision"):
        return False
    initial = episode.get("initial_goal_distance_m")
    final = episode.get("final_goal_distance_m")
    if episode.get("success"):
        return True
    if initial is None or final is None:
        return False
    return float(final) < float(initial)


def _action_bounds(action_space):
    if action_space is None:
        return (
            np.asarray([ACTION_LOW, ACTION_LOW], dtype=np.float32),
            np.asarray([ACTION_HIGH, ACTION_HIGH], dtype=np.float32),
        )
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
        raise ValueError("Gymnasium action space is not 2-dimensional")
    return low, high


def _plot_episode(path, episode, plans):
    plot_record = {
        "scenario_id": episode["episode_id"],
        "status": f"{episode['policy']}:{episode['termination_reason'] or 'not_run'}",
        "start": episode["start"],
        "goal": episode["goal"],
        "planned_path": (plans.get(episode["scenario_id"]) or {}).get("planned_path") or [],
        "executed_poses": episode.get("poses") or [],
    }
    plot_planned_versus_executed(plot_record, path, OBSTACLE_BOX)


def _plot_diagnostics(path, episode, steps):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Plot already exists: {path}")
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    indices = [step["step"] for step in steps]
    axes[0].plot(indices, [step["linear_action"] for step in steps], label="linear")
    axes[0].plot(indices, [step["angular_action"] for step in steps], label="angular")
    axes[0].set_ylabel("Gymnasium action")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    distances = [step["goal_distance_m"] for step in steps]
    axes[1].plot(indices, distances, color="#2463a8")
    axes[1].set_ylabel("Goal distance (m)")
    axes[1].set_xlabel("Control step")
    axes[1].grid(alpha=0.3)
    title = f"{episode['episode_id']} ({episode['termination_reason'] or 'incomplete'})"
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_overview(path, episodes, plans):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Plot already exists: {path}")
    figure, axis = plt.subplots(figsize=(8, 7))
    center_x, center_y, size_x, size_y = OBSTACLE_BOX
    axis.add_patch(Rectangle(
        (center_x - size_x / 2, center_y - size_y / 2),
        size_x,
        size_y,
        facecolor="#b33",
        alpha=0.35,
        label="obstacle",
    ))
    for episode in episodes:
        poses = episode.get("poses") or []
        if len(poses) < 2:
            continue
        style = "-" if episode["policy"] == POLICY_NAME else "--"
        axis.plot(
            [pose[0] for pose in poses],
            [pose[1] for pose in poses],
            style,
            linewidth=1.5,
            label=episode["episode_id"],
        )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.3)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("Recorded trajectories")
    axis.legend(loc="best", fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_comparison(path, summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Plot already exists: {path}")
    labels = ["success", "collision", "timeout"]
    policies = [POLICY_NAME, ASTAR_NAME]
    figure, axis = plt.subplots(figsize=(7, 4))
    width = 0.35
    positions = np.arange(len(labels))
    for offset, policy in enumerate(policies):
        block = (summary.get("aggregates") or {}).get(policy) or {}
        attempted = block.get("attempted_episodes") or 0
        heights = []
        for label in labels:
            count = (block.get(label) or {}).get("count") or 0
            heights.append(0 if attempted == 0 else count / attempted)
        axis.bar(positions + offset * width, heights, width=width, label=f"{policy} (n={attempted})")
    axis.set_xticks(positions + width / 2)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rate of attempted episodes")
    axis.set_title("Closed-loop rates")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _write_step_csv(path, steps):
    fields = [
        "episode_id",
        "step",
        "linear_action",
        "angular_action",
        "goal_distance_m",
        "goal_heading_norm",
        "linear_velocity_norm",
        "angular_velocity_norm",
        "progress_m",
        "pose_x",
        "pose_y",
        "pose_yaw",
        "reward",
        "scan_stamp",
        "odom_stamp",
        "inference_latency_s",
        "action_accepted",
        "safety_reason",
        "termination_reason",
    ]

    def _write(temporary):
        with Path(temporary).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for step in steps:
                writer.writerow({key: step.get(key) for key in fields})

    create_new_file(path, _write)


def _write_step_npz(path, steps):
    observations = [
        np.asarray(step["observation"], dtype=np.float32)
        for step in steps
        if isinstance(step.get("observation"), np.ndarray)
    ]
    if observations:
        stacked = np.stack(observations)
    else:
        stacked = np.zeros((0, OBSERVATION_DIM), dtype=np.float32)
    actions = np.asarray(
        [[step["linear_action"], step["angular_action"]] for step in steps],
        dtype=np.float32,
    ).reshape(-1, ACTION_DIM)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, observations=stacked, actions=actions)

    def _write(temporary):
        Path(temporary).write_bytes(buffer.getvalue())

    create_new_file(path, _write)


def _write_episode_csv(path, episodes):
    fields = [
        "episode_id",
        "policy",
        "scenario_id",
        "trial",
        "seed",
        "success",
        "collision",
        "timeout",
        "termination_reason",
        "wall_time_s",
        "simulation_time_s",
        "path_length_m",
        "initial_goal_distance_m",
        "final_goal_distance_m",
        "control_steps",
        "mean_inference_latency_s",
        "max_inference_latency_s",
        "invalid_observations",
        "invalid_actions",
        "safety_stop_events",
        "return",
        "stop_verified",
    ]

    def _write(temporary):
        with Path(temporary).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for episode in episodes:
                writer.writerow({key: episode.get(key) for key in fields})

    create_new_file(path, _write)


def _write_json(path, payload):
    text = json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"

    def _write(temporary):
        Path(temporary).write_text(text, encoding="utf-8")

    create_new_file(path, _write)


def _write_text(path, text):
    def _write(temporary):
        Path(temporary).write_text(text, encoding="utf-8")

    create_new_file(path, _write)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _format_number(value):
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _optional_float(value):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _hypot(origin, goal):
    return float(math.hypot(float(goal[0]) - float(origin[0]), float(goal[1]) - float(origin[1])))


def _sign_changes(values):
    signs = [1 if value > 1e-3 else -1 if value < -1e-3 else 0 for value in values]
    previous = 0
    changes = 0
    for sign in signs:
        if sign == 0:
            continue
        if previous != 0 and sign != previous:
            changes += 1
        previous = sign
    return changes


def _assert_close(actual, expected, label):
    if not np.allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(f"Normalization {label} does not match the frozen statistics")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
