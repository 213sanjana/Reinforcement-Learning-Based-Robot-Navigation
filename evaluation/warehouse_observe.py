"""Record one stationary observation from the external warehouse.

This process does not launch Gazebo, publish a velocity, step the simulator,
train a policy, or load a checkpoint. Attaching to a live ROS graph requires
``--authorize-observation`` and
``WAREHOUSE_RL_OBSERVE_AUTHORIZATION=I_AUTHORIZE_STATIONARY_OBSERVATION``.
The motion-authorization variable does not satisfy that gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from jackal_rl_navigation.warehouse_config import load_warehouse_config, repo_root


OBSERVE_ENV = "WAREHOUSE_RL_OBSERVE_AUTHORIZATION"
OBSERVE_VALUE = "I_AUTHORIZE_STATIONARY_OBSERVATION"
ALLOWED_PREFIX = "artifacts/warehouse_ppo/observations"
PROTECTED_PREFIXES = (
    "artifacts/ppo",
    "artifacts/behavior_cloning",
    "artifacts/experiments",
    "artifacts/project_audit",
    "artifacts/benchmarks",
    "artifacts/imitation",
    "artifacts/curriculum",
    "artifacts/qlearning",
    "artifacts/dqn",
    "artifacts/warehouse_ppo/runs",
)


def observation_authorized(cli_flag: bool) -> bool:
    return bool(cli_flag) and os.environ.get(OBSERVE_ENV) == OBSERVE_VALUE


def assert_observation_output(path: Path, root: Path | None = None) -> Path:
    """Accept a new observation directory and refuse historical trees."""

    root = repo_root() if root is None else Path(root)
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite {resolved}")
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved
    for prefix in PROTECTED_PREFIXES:
        if relative == prefix or relative.startswith(prefix + "/"):
            raise RuntimeError(f"Refusing to write inside protected evidence tree {prefix}")
    if not (relative == ALLOWED_PREFIX or relative.startswith(ALLOWED_PREFIX + "/")):
        raise RuntimeError(
            "In-repository observation output must be a new directory under "
            "artifacts/warehouse_ppo/observations"
        )
    return resolved


def _train_scenario(config: dict) -> dict:
    for item in config.get("scenarios") or []:
        if str(item.get("split", "train")) == str(config.get("split", "train")):
            return {
                "id": item.get("id"),
                "start": item.get("start"),
                "goal": item.get("goal"),
            }
    raise RuntimeError("No scenario is configured for this split")


def _publishers(session) -> list[str]:
    transport = getattr(session, "transport", None)
    query = getattr(transport, "command_publishers", None)
    if not callable(query):
        return []
    return [str(name) for name in query()]


def _record(observation, info, *, status, error, evidence_class, scenario) -> dict:
    vector = None if observation is None else np.asarray(observation, dtype=np.float32)
    valid = (
        vector is not None
        and vector.shape == (40,)
        and vector.dtype == np.float32
        and bool(np.all(np.isfinite(vector)))
    )
    if status == "observation_collected" and not valid:
        status = "failed"
        error = error or "observation contract failed"
    return {
        "status": status,
        "error": error,
        "evidence_class": evidence_class,
        "world_name": None if not info else info.get("world_name", "industrial-warehouse"),
        "gazebo_partition": None if not info else info.get("gazebo_partition", "jackal_rl_external"),
        "scenario_id": None if scenario is None else scenario.get("id"),
        "configured_goal_odom": None if scenario is None else list(scenario.get("goal") or []),
        "pose_xy": None if not info or info.get("pose_xy") is None else [float(v) for v in info["pose_xy"]],
        "goal_distance": None if not info or info.get("goal_distance") is None else float(info["goal_distance"]),
        "observation_shape": None if not valid else [40],
        "observation_dtype": None if not valid else "float32",
        "observation_finite": valid,
        "observation": None if not valid else [float(value) for value in vector],
        "gazebo_launched_by_this_process": False,
        "robot_commanded_by_this_process": False,
        "training": False,
        "checkpoint_loaded": False,
        "navigation_result": False,
        "real_world": False,
    }


def observe(
    config: dict,
    output: Path,
    *,
    authorize: bool,
    session=None,
    connect_session=None,
) -> dict:
    """Read one observation and write a new record.

    A passed ``session`` is an injected transport. The default constructor
    attaches to ROS and is the live path.
    """

    if not observation_authorized(authorize):
        raise SystemExit(
            "Refusing the stationary observation. It attaches to a live ROS graph. "
            f"Re-run only with --authorize-observation and {OBSERVE_ENV}={OBSERVE_VALUE}. "
            "Motion authorization does not enable this command."
        )
    output = assert_observation_output(output)
    evidence = "injected_transport" if session is not None else "ros_attachment"
    scenario = _train_scenario(config)
    observation = None
    info = {}
    error = None
    try:
        if session is None:
            if connect_session is None:
                from jackal_rl_navigation.warehouse_observation import StationaryObservationSession

                session = StationaryObservationSession(config, connect=True)
            else:
                session = connect_session(config)
        observation, info = session.reset(seed=0, scenario=scenario)
        publishers = _publishers(session)
        if publishers:
            raise RuntimeError(
                "A velocity command publisher appeared during observation: "
                + ", ".join(publishers)
            )
    except Exception as exc:
        error = str(exc)
        observation = None
    finally:
        closer = None if session is None else getattr(session, "close", None)
        if closer is not None:
            closer()
    status = "failed" if error else "observation_collected"
    record = _record(
        observation,
        info,
        status=status,
        error=error,
        evidence_class=evidence,
        scenario=scenario,
    )
    output.mkdir(parents=True)
    (output / "observation.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    record["output_dir"] = str(output)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Read one stationary warehouse observation. Does not launch Gazebo or command the robot.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-observation", action="store_true")
    args = parser.parse_args(argv)
    config = load_warehouse_config(args.config)
    config["execution"] = "stationary_observation"
    record = observe(
        config,
        Path(args.output),
        authorize=observation_authorized(args.authorize_observation),
    )
    print(record["output_dir"])
    return 0 if record["status"] == "observation_collected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
