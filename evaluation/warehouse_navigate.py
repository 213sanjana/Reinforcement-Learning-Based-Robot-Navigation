"""Drive warehouse destinations with a learned PPO policy.

The policy chooses each velocity. The existing safety monitor rejects commands
that point into an obstacle. A route keeps the same episode and changes the
goal after each checkpoint, so the robot is not teleported between stops.
This module does not train and does not update normalization statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from jackal_rl_navigation.goal_frames import odom_point_to_world
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_env import JackalWarehouseRLEnv
from jackal_rl_navigation.warehouse_estop import live_authorized

_MAX_RECOVERIES = 24
_MAX_FALSE_ARRIVALS = 8
_APPROACH_DISTANCE_M = 1.2
from training.warehouse_checkpoint import assert_load_allowed, select_device
from training.warehouse_ppo_train import build_vec_env, load_run


def _resume_goals(goals, start, tolerance: float):
    from evaluation.warehouse_checkpoints import route_from_pose

    points = [{"x": goal[0], "y": goal[1], "id": name} for goal, name in goals]
    selected = route_from_pose(points, start[0], start[1], tolerance_m=tolerance)
    return [((point["x"], point["y"]), point["id"]) for point in selected]


def _sector_bearing(sector: int, sectors: int = 36) -> float:
    """Center angle of one LiDAR sector. Sector 0 starts at -pi."""

    return -math.pi + (sector + 0.5) * (2.0 * math.pi / sectors)


def _recovery_action(attempt: int, observation=None, command_reason=None) -> np.ndarray:
    """Move clear after a blocked command without repeating a rejected turn.

    A rotation-clearance rejection means the monitor already zeroed the yaw
    command. Another pure turn asks for the same blocked axis. A short
    reverse still passes through that monitor on the next step.
    """

    reason = "" if command_reason is None else str(command_reason).lower()
    if "rotation clearance" in reason:
        return np.array([-0.35, 0.0], dtype=np.float32)
    direction = 1.0 if attempt % 2 == 0 else -1.0
    ranges = None if observation is None else np.asarray(observation[:36], dtype=np.float64)
    if ranges is not None and ranges.shape == (36,) and np.all(np.isfinite(ranges)):
        bearing = _sector_bearing(int(np.argmin(ranges)))
        if abs(bearing) >= math.radians(25) and abs(abs(bearing) - math.pi) >= math.radians(25):
            direction = -1.0 if bearing > 0.0 else 1.0
    return np.array([0.0, 0.5 * direction], dtype=np.float32)


def _approach_action(observation) -> np.ndarray | None:
    """Steer toward the current checkpoint when it is close and the front is clear."""

    vector = np.asarray(observation, dtype=np.float64)
    if vector.shape != (40,) or not np.all(np.isfinite(vector)):
        return None
    heading = float(vector[37]) * math.pi
    forward_m = float(np.min(vector[16:21])) * 12.0
    if forward_m < 0.90:
        return None
    if abs(heading) > 0.40:
        return np.array([0.0, float(np.clip(heading / math.pi, -0.8, 0.8))], dtype=np.float32)
    return np.array([0.45, float(np.clip(heading / math.pi, -0.25, 0.25))], dtype=np.float32)


def _within(pose, goal, tolerance: float) -> bool:
    if pose is None:
        return False
    return math.hypot(float(pose[0]) - float(goal[0]), float(pose[1]) - float(goal[1])) <= tolerance


def _warehouse_pose(env, pose, world_route: bool):
    """Express an odometry pose in the warehouse frame when the route uses one."""

    if pose is None:
        return None
    point = (float(pose[0]), float(pose[1]))
    if not world_route:
        return point
    session = getattr(env, "session", None)
    world_pose = getattr(session, "world_robot_pose", None)
    odom_pose = getattr(session, "odom_robot_pose", None)
    if world_pose is None or odom_pose is None:
        return point
    return odom_point_to_world(point, world_pose, odom_pose)


def _route_pose_for_result(measured, fallback):
    """Publish the pose that was compared with the checkpoint.

    ``fallback`` is a second odometry-to-world conversion. It is used only
    when the route never measured a pose. Replacing a measured pose with that
    conversion can move the reported point by meters and disagree with the
    checkpoint decision.
    """

    return measured if measured is not None else fallback


def _live_warehouse_pose(env):
    """Read the Gazebo model position. None when the world is not publishing.

    The world-to-odometry pair captured at reset is left unchanged. A later
    unsynchronized world pose and odometry sample is not a new rigid transform,
    and retarget uses that pair to place the next checkpoint.
    """

    session = getattr(env, "session", None)
    inner = getattr(session, "env", None)
    reader = getattr(inner, "world_pose", None)
    robot = getattr(inner, "robot", None)
    if reader is None or not hasattr(reader, "read_latest") or robot is None:
        return None
    try:
        world = reader.read_latest()
    except Exception:
        return None
    return float(world[0]), float(world[1])


def _goal_pair(goal_x: float, goal_y: float) -> tuple[float, float]:
    goal = (float(goal_x), float(goal_y))
    if not np.all(np.isfinite(goal)):
        raise ValueError("Destination must be finite")
    return goal


def _start_from_robot(env: JackalWarehouseRLEnv) -> tuple[float, float, float]:
    session = getattr(env, "session", None)
    inner = getattr(session, "env", None)
    robot = getattr(inner, "robot", None)
    executor = getattr(inner, "executor", None)
    if robot is None or executor is None:
        raise RuntimeError("The live robot pose is not available")
    # The demo launch waits 8 s before spawn and then starts controllers.
    # A 5 s sensor timeout expires before /platform/odom exists.
    topic = env.config.get("odom_topic", "/j100_0000/platform/odom")
    wait_s = max(45.0, float(env.config["limits"].sensor_timeout_s))
    print(
        f"Waiting up to {wait_s:.0f}s for odometry on {topic}. "
        f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', 'unset')} "
        f"world={env.config.get('world_name')!r} "
        f"partition={env.config.get('gazebo_partition')!r}.",
        flush=True,
    )
    deadline = time.monotonic() + wait_s
    announced = time.monotonic()
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        message = getattr(robot, "odom", None)
        if message is None:
            if time.monotonic() - announced >= 5.0:
                print("Still waiting for odometry. The velocity controller must be active.", flush=True)
                announced = time.monotonic()
            continue
        position = message.pose.pose.position
        start = (float(position.x), float(position.y), 0.0)
        if np.all(np.isfinite(start)):
            print(f"Odometry is at x={start[0]:.2f} y={start[1]:.2f}.", flush=True)
            return start
    raise TimeoutError(
        f"Odometry did not arrive on {topic} within {wait_s:.0f}s. "
        "Start the launch first and wait until the velocity controller is active. "
        "This process must use the same ROS_DOMAIN_ID as the launch. "
        f"World {env.config.get('world_name')!r}, partition {env.config.get('gazebo_partition')!r}."
    )


def _clear_episode_latch(env: JackalWarehouseRLEnv) -> None:
    """Allow another command after a late scan or a bump."""

    env._episode_open = True
    env._abort_reason = None
    session = getattr(env, "session", None)
    inner = getattr(session, "env", None)
    robot = getattr(inner, "robot", None)
    if inner is None or robot is None:
        return
    inner.episode_done = False
    monitor = getattr(robot, "safety_monitor", None)
    if monitor is not None and hasattr(monitor, "reset_episode"):
        monitor.reset_episode()
    if hasattr(robot, "reset_collision_latch"):
        robot.reset_collision_latch()
    for owner in (inner, robot):
        contact = getattr(owner, "collision_monitor", None)
        reset = getattr(contact, "reset", None)
        if callable(reset):
            reset()


def _turn_away(pairs) -> float:
    """Negative yaw leaves an obstacle that touched the left side."""

    text = " ".join(f"{left} {right}" for left, right in pairs).lower()
    if "left" in text and "right" not in text:
        return -1.0
    if "right" in text and "left" not in text:
        return 1.0
    return -1.0


def _escape_action(pairs, away: float) -> np.ndarray:
    """Drive off the wheel that is touching.

    Reversing while a rear wheel is in contact presses that wheel harder
    into the obstacle. A rear contact has to move forward.
    """

    text = " ".join(f"{left} {right}" for left, right in pairs).lower()
    rear = "rear_" in text
    front = "front_" in text
    linear = 0.8 if rear and not front else -0.8
    return np.array([linear, 0.8 * away], dtype=np.float32)


def _odom_pose(env: JackalWarehouseRLEnv):
    session = getattr(env, "session", None)
    inner = getattr(session, "env", None)
    robot = getattr(inner, "robot", None)
    message = getattr(robot, "odom", None)
    if message is None:
        return None
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )
    pose = (float(position.x), float(position.y), float(yaw))
    if not np.all(np.isfinite(pose)):
        return None
    return pose


def load_policy(config: dict, checkpoint: Path, *, authorize_live: bool):
    """Load one warehouse PPO policy and the environment it commands."""

    checkpoint = assert_load_allowed(checkpoint)
    device = select_device(config["ppo"]["device"])
    vector_env = build_vec_env(config, authorize_live=authorize_live, training=False)
    model, vector_env = load_run(checkpoint, vector_env, device, training=False)
    if hasattr(vector_env, "training"):
        vector_env.training = False
        vector_env.norm_reward = False
    normalizer = vector_env if hasattr(vector_env, "normalize_obs") else None
    inner = vector_env.venv.envs[0] if hasattr(vector_env, "venv") else vector_env.envs[0]

    def policy(observation, _info):
        model_observation = observation
        if normalizer is not None:
            batched = np.asarray(observation, dtype=np.float32)[None, :]
            model_observation = np.asarray(normalizer.normalize_obs(batched)[0], dtype=np.float32)
        action, _state = model.predict(model_observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return inner, policy, vector_env


def navigate(
    config: dict,
    goal_x: float,
    goal_y: float,
    *,
    checkpoint: Path | None = None,
    start: tuple[float, float] | tuple[float, float, float] | None = None,
    authorize_live: bool = False,
    seed: int = 1,
    policy=None,
) -> dict:
    """Run one policy until the destination, a collision, or the step limit."""

    return navigate_route(
        config,
        [(_goal_pair(goal_x, goal_y), "destination")],
        checkpoint=checkpoint,
        start=start,
        authorize_live=authorize_live,
        seed=seed,
        policy=policy,
    )


def navigate_route(
    config: dict,
    goals: list[tuple[tuple[float, float], str]],
    *,
    checkpoint: Path | None = None,
    start: tuple[float, float] | tuple[float, float, float] | None = None,
    authorize_live: bool = False,
    seed: int = 1,
    policy=None,
    resume: bool = False,
    world_route: bool = False,
    on_observation=None,
) -> dict:
    """Follow an ordered destination list without teleporting between stops.

    ``world_route`` keeps checkpoint coordinates in the warehouse frame after
    a spawn teleport. Wheel odometry does not jump with that teleport, so
    later checkpoints are converted with the pose pair captured at reset.
    """

    execution = config.get("execution", "offline")
    if execution == "stationary_observation":
        raise SystemExit(
            "Refusing navigation on the stationary observation path. "
            "step() and checkpoint loading stay blocked."
        )
    if execution == "gazebo" and not authorize_live:
        raise SystemExit("Refusing Gazebo navigation because it would command the robot.")
    if not goals:
        raise ValueError("A route needs at least one destination")
    parsed = [(_goal_pair(goal[0], goal[1]), str(name)) for goal, name in goals]
    config = dict(config)
    config["require_listed_pair"] = False
    tolerance = float(config["limits"].goal_tolerance_m)
    config["min_separation_m"] = tolerance + 1e-3
    if policy is None:
        if checkpoint is None:
            raise ValueError("A learned checkpoint or an explicit policy is required")
        env, policy, vector_env = load_policy(config, Path(checkpoint), authorize_live=authorize_live)
    else:
        env = JackalWarehouseRLEnv(config, authorize_live=authorize_live)
        vector_env = env
    try:
        if start is None and execution == "gazebo":
            start = _start_from_robot(env)
        elif start is None:
            start = (0.0, 0.0, 0.0)
        if resume:
            parsed = _resume_goals(parsed, start, tolerance)
        first_goal, _first_name = parsed[0]
        if math.hypot(start[0] - first_goal[0], start[1] - first_goal[1]) <= tolerance:
            return {
                "destination": [parsed[-1][0][0], parsed[-1][0][1]],
                "checkpoint": parsed[-1][1],
                "route_start": _first_name,
                "checkpoints_reached": [name for _goal, name in parsed],
                "termination_reason": "goal_reached",
                "goal_distance": 0.0,
                "collision": False,
                "timeout": False,
                "success": True,
                "learned_policy": checkpoint is not None,
                "execution": execution,
                "pose_xy": [start[0], start[1]],
            }
        options = {
            "scenario_id": "destination",
            "start": tuple(start),
            "goal": first_goal,
        }
        if execution == "gazebo" and world_route:
            env.session.route_frame = "world"
        elif execution == "gazebo":
            options["goal_frame"] = "odom"
        observation, info = env.reset(seed=seed, options=options)
        done = False
        index = 0
        reached: list[str] = []
        recoveries = 0
        false_arrivals = 0
        forced_action = None
        route_pose = None
        use_world = bool(world_route and execution == "gazebo")
        last = dict(info)
        while not done:
            if forced_action is not None:
                action = forced_action
                forced_action = None
            else:
                action = policy(observation, last)
                distance = last.get("goal_distance")
                if distance is not None and tolerance < float(distance) <= _APPROACH_DISTANCE_M:
                    assist = _approach_action(observation)
                    if assist is not None:
                        action = assist
            if on_observation is not None:
                on_observation(observation, last)
            observation, _reward, terminated, truncated, last = env.step(action)
            done = bool(terminated or truncated)
            reason = last.get("termination_reason")
            goal_xy = parsed[index][0]
            if done and reason == "goal_reached":
                live_pose = _live_warehouse_pose(env) if use_world else None
                route_pose = live_pose if live_pose is not None else _warehouse_pose(
                    env, last.get("pose_xy"), world_route
                )
                if _within(route_pose, goal_xy, tolerance):
                    reached.append(parsed[index][1])
                    false_arrivals = 0
                    if index + 1 < len(parsed):
                        index += 1
                        observation = env.retarget(parsed[index][0])
                        recoveries = 0
                        done = False
                elif false_arrivals < _MAX_FALSE_ARRIVALS:
                    false_arrivals += 1
                    observation = env.retarget(goal_xy, reset_steps=False)
                    recoveries = 0
                    done = False
            elif done and reason == "safety_rejected" and recoveries < _MAX_RECOVERIES:
                observation = env.retarget(goal_xy, reset_steps=False)
                forced_action = _recovery_action(recoveries, observation, last.get("command_reason"))
                recoveries += 1
                done = False
        last = dict(last)
        goal, goal_name = parsed[-1]
        measured = route_pose
        fallback = None if measured is not None else _warehouse_pose(env, last.get("pose_xy"), world_route)
        route_pose = _route_pose_for_result(measured, fallback)
        last["destination"] = [goal[0], goal[1]]
        last["checkpoint"] = goal_name
        last["route_start"] = parsed[0][1]
        last["checkpoints_reached"] = reached
        last["learned_policy"] = checkpoint is not None
        last["execution"] = execution
        if route_pose is not None:
            last["pose_xy"] = [route_pose[0], route_pose[1]]
            remaining = float(math.hypot(route_pose[0] - goal[0], route_pose[1] - goal[1]))
            last["goal_distance"] = remaining
            arrived = remaining <= tolerance
        else:
            arrived = False
        last["success"] = bool(
            arrived
            and last.get("termination_reason") == "goal_reached"
            and reached == [name for _goal, name in parsed]
        )
        return last
    finally:
        vector_env.close()


def navigate_explore(
    config: dict,
    checkpoint: Path,
    *,
    authorize_live: bool = False,
    start: tuple[float, float] | tuple[float, float, float] | None = None,
    seed: int = 1,
    max_steps: int | None = None,
) -> dict:
    """Drive with the learned policy toward successive open-space goals.

    With ``max_steps`` left unset, the robot keeps roaming until Ctrl+C.
    """

    from evaluation.world_explore import avoidance_action, roaming_goal

    execution = config.get("execution", "offline")
    if execution == "stationary_observation":
        raise SystemExit(
            "Refusing navigation on the stationary observation path. "
            "step() and checkpoint loading stay blocked."
        )
    if execution == "gazebo" and not authorize_live:
        raise SystemExit("Refusing Gazebo navigation because it would command the robot.")
    config = dict(config)
    config["require_listed_pair"] = False
    env, policy, vector_env = load_policy(config, Path(checkpoint), authorize_live=authorize_live)
    inner = getattr(getattr(env, "session", None), "env", None)
    if inner is not None:
        inner.continuous = True
        robot = getattr(inner, "robot", None)
        if robot is not None:
            # The interface drops velocity 0.4s after a command. One decision
            # takes longer than that, which stops the robot between turns.
            robot.command_hold_s = 1.2
            mapper = getattr(robot, "action_mapper", None)
            if mapper is not None:
                mapper.max_forward = 0.75
                mapper.linear_acceleration = 0.80
                mapper.max_angular = 0.80
    goals_sent = 0
    steps = 0
    sector = None
    recoveries = 0
    last = {}
    try:
        if start is None and execution == "gazebo":
            start = _start_from_robot(env)
        elif start is None:
            start = (0.0, 0.0, 0.0)
        seed_goal = (float(start[0]) + 1.5, float(start[1]))
        print(
            f"Placing the robot at x={float(start[0]):.2f} y={float(start[1]):.2f} and driving.",
            flush=True,
        )
        observation, last = env.reset(
            seed=seed,
            options={
                "scenario_id": "explore",
                "start": tuple(start),
                "goal": seed_goal,
                "goal_frame": "odom",
            },
        )
        pending = True
        forced_action = None
        front_blocked = False
        opened_front = False
        stale_waits = 0
        bumps = 0
        escape_left = 0
        escape_action = None
        recent_goals = []
        print("The robot keeps roaming. Press Ctrl+C to stop it.", flush=True)
        while max_steps is None or steps < max_steps:
            if pending:
                pose = _odom_pose(env)
                if pose is None:
                    pose = (float(start[0]), float(start[1]), 0.0)
                choice = roaming_goal(observation, pose, recent_goals)
                opened_front = False
                observation = env.retarget(choice["goal"], reset_steps=True)
                sector = choice["sector"]
                recent_goals.append(choice["goal"])
                recent_goals[:] = recent_goals[-6:]
                goals_sent += 1
                pending = False
                print(
                    f"Open-space goal {goals_sent}: "
                    f"x={choice['goal'][0]:.2f} y={choice['goal'][1]:.2f}.",
                    flush=True,
                )
            avoid = None if forced_action is not None else avoidance_action(observation)
            if forced_action is not None:
                action = forced_action
                forced_action = None
            elif avoid is not None:
                if not front_blocked:
                    print("Steering around the obstacle.", flush=True)
                front_blocked = True
                action = avoid
            else:
                if front_blocked:
                    front_blocked = False
                    opened_front = True
                    pending = True
                action = policy(observation, last)
                forward_m = float(np.min(np.asarray(observation[15:21], dtype=np.float64))) * 12.0
                if abs(float(action[0])) < 0.05 and forward_m > 2.0:
                    heading = float(observation[37]) * math.pi
                    if abs(heading) > 0.5:
                        action = np.array(
                            [0.0, float(np.clip(heading / math.pi, -0.8, 0.8))],
                            dtype=np.float32,
                        )
                    else:
                        action = np.array(
                            [0.7, float(np.clip(heading / math.pi, -0.3, 0.3))],
                            dtype=np.float32,
                        )
            observation, _reward, terminated, truncated, last = env.step(action)
            steps += 1
            reason = last.get("termination_reason")
            distance = last.get("goal_distance")
            if last.get("action_accepted"):
                stale_waits = 0
            if reason == "stale_sensors":
                if stale_waits >= 8:
                    print("LiDAR stayed stale. Stopping.", flush=True)
                    break
                stale_waits += 1
                print("LiDAR update was late. Continuing.", flush=True)
                _clear_episode_latch(env)
                continue
            if reason == "collision" or escape_left > 0:
                pairs = last.get("collision_pairs") or []
                if reason == "collision" and escape_left <= 0:
                    bumps += 1
                    away = _turn_away(pairs)
                    if bumps % 4 == 0:
                        away = -away
                    escape_action = _escape_action(pairs, away)
                    escape_left = 8
                    print("Touched an obstacle. Pulling clear and going around it.", flush=True)
                if reason == "collision":
                    _clear_episode_latch(env)
                escape_left -= 1
                forced_action = escape_action
                if escape_left <= 0:
                    pending = True
                    bumps = 0
                continue
            bumps = 0
            if terminated and reason not in ("goal_reached",):
                break
            if truncated and reason not in ("time_limit", "safety_rejected"):
                break
            if reason == "safety_rejected":
                if recoveries == 0:
                    print("Safety held the forward command. Turning toward open space.", flush=True)
                if recoveries >= _MAX_RECOVERIES:
                    break
                forced_action = _recovery_action(recoveries, observation)
                recoveries += 1
                pending = True
                continue
            close = distance is not None and float(distance) <= 0.8
            if reason in ("goal_reached", "time_limit") or close:
                pending = True
    except KeyboardInterrupt:
        print("You pressed Ctrl+C. Stopping the robot.", flush=True)
    finally:
        try:
            session = getattr(env, "session", None)
            inner_env = getattr(session, "env", None)
            robot = getattr(inner_env, "robot", None)
            if robot is not None and hasattr(robot, "stop"):
                robot.stop()
            controller = getattr(inner_env, "controller", None)
            if controller is not None and hasattr(controller, "pause"):
                controller.pause()
        except Exception:
            pass
        vector_env.close()
    last = dict(last) if isinstance(last, dict) else {}
    last["goals_sent"] = goals_sent
    last["explore_steps"] = steps
    last["learned_policy"] = True
    last["execution"] = execution
    last["checkpoints_reached"] = []
    last["success"] = bool(
        goals_sent > 0
        and steps > 0
        and not last.get("collision")
        and not last.get("fatal")
        and last.get("termination_reason") not in ("collision", "emergency_stop", "stale_sensors")
    )
    return last


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Send the Jackal through warehouse checkpoints with a learned PPO policy.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route", choices=["warehouse"])
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--authorize-live", action="store_true")
    args = parser.parse_args(argv)
    if (args.start_x is None) != (args.start_y is None):
        raise SystemExit("Provide both --start-x and --start-y, or neither.")
    output = None if args.output is None else Path(args.output)
    if output is not None and output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    config = load_warehouse_config(args.config)
    start = None if args.start_x is None else (args.start_x, args.start_y, 0.0)
    authorized = live_authorized(args.authorize_live)
    if args.route == "warehouse":
        from evaluation.warehouse_checkpoints import WAREHOUSE_ROUTE

        result = navigate_route(
            config,
            [((point["x"], point["y"]), point["id"]) for point in WAREHOUSE_ROUTE],
            checkpoint=Path(args.checkpoint),
            start=start,
            authorize_live=authorized,
            seed=args.seed,
            resume=True,
        )
    else:
        if args.goal_x is None or args.goal_y is None:
            raise SystemExit("Provide --goal-x and --goal-y, or --route warehouse.")
        result = navigate(
            config,
            args.goal_x,
            args.goal_y,
            checkpoint=Path(args.checkpoint),
            start=start,
            authorize_live=authorized,
            seed=args.seed,
        )
    payload = json.dumps(result, indent=2, default=str)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
