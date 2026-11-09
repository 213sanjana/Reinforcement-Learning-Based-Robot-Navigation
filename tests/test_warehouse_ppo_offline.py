"""Offline PPO checkpoint tests. They do not open historical zips or Gazebo."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from evaluation.warehouse_compare import compare
from evaluation.warehouse_evaluate import evaluate
from evaluation.warehouse_autonomous import decide, read_scene
from evaluation.warehouse_checkpoints import (
    SHELF_BOX_BAY,
    WAREHOUSE_ROUTE,
    clearance_m,
    route_from_pose,
)
from evaluation.warehouse_navigate import (
    _live_warehouse_pose,
    _recovery_action,
    _route_pose_for_result,
    _within,
    navigate,
    navigate_route,
)
from jackal_rl_navigation.goal_frames import odom_point_to_world, world_goal_to_odom
from evaluation.warehouse_inspect import inspect_run
from evaluation.warehouse_metrics import aggregate_episodes
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from training.warehouse_callbacks import WarehouseTrainingCallback
from training.warehouse_checkpoint import assert_load_allowed, assert_output_allowed
from training.warehouse_ppo_train import train
from visualization.warehouse_dashboard import plot_run


ROOT = Path(__file__).resolve().parents[1]


def _tiny(tmp_path, *, normalize=False, timesteps=32):
    config = load_warehouse_config(None)
    config["execution"] = "offline"
    config["ppo"] = dict(config["ppo"])
    config["ppo"].update(
        {
            "n_steps": 16,
            "batch_size": 16,
            "n_epochs": 1,
            "total_timesteps": timesteps,
            "checkpoint_interval": 16,
            "normalize_observations": normalize,
            "device": "cpu",
            "seed": 7,
            "min_episodes_for_best": 1,
        }
    )
    limits = config["limits"]
    config["limits"] = type(limits)(
        goal_tolerance_m=limits.goal_tolerance_m,
        max_episode_steps=4,
        sensor_timeout_s=limits.sensor_timeout_s,
        control_frequency_hz=limits.control_frequency_hz,
    )
    return config


def test_protected_checkpoints_and_existing_directories_are_refused(tmp_path):
    with pytest.raises(RuntimeError):
        assert_load_allowed(ROOT / "artifacts" / "ppo" / "jackal_6g1")
    existing = tmp_path / "already"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        assert_output_allowed(existing)
    with pytest.raises(RuntimeError):
        assert_output_allowed(ROOT / "artifacts" / "ppo" / "new_run")


def test_callback_records_failure_and_stops(tmp_path):
    callback = WarehouseTrainingCallback(tmp_path, checkpoint_interval=100, min_episodes_for_best=1)

    class Model:
        num_timesteps = 4

        @staticmethod
        def save(path):
            Path(str(path) + ".zip").write_bytes(b"ok")

    callback.model = Model()
    callback.n_calls = 1
    callback._on_training_start()
    callback.locals = {
        "infos": [
            {
                "termination_reason": "safety_fault",
                "fatal": True,
                "success": False,
                "collision": False,
                "timeout": False,
                "episode_return": -1.0,
                "episode_length": 2,
                "goal_distance": 1.0,
                "total_reward": -1.0,
            }
        ]
    }
    assert callback._on_step() is False
    assert "safety fault" in callback.stop_reason
    assert (tmp_path / "episodes.csv").is_file()


def test_short_offline_training_can_resume(tmp_path):
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"
    result = train(config=_tiny(tmp_path, timesteps=16), output_dir=first)
    assert result["trained"] is True
    assert result["device"] == "cpu"
    metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["observation_interface"] == "jackal_40d_v1"
    assert metadata["normalize_actions"] is False
    assert (first / "ppo_model.zip").is_file()
    inspected = inspect_run(first)
    assert inspected["loaded_weights"] is False
    assert inspected["model_present"] is True
    from stable_baselines3 import PPO
    loaded_model = PPO.load(str(first / "ppo_model.zip"), device="cpu")
    action, _state = loaded_model.predict(np.zeros(40, dtype=np.float32), deterministic=True)
    assert np.asarray(action).shape == (2,)
    assert np.all(np.isfinite(action))
    resumed = train(config=_tiny(tmp_path, timesteps=32), output_dir=second, resume_from=first)
    assert resumed["resumed_timesteps"] == 16
    assert resumed["timesteps"] == 32
    summary = evaluate(
        _tiny(tmp_path, timesteps=16),
        checkpoint=None,
        episodes=1,
        seed=11,
        authorize_live=False,
        policy=lambda _observation, _info: np.array([1.0, 0.0], dtype=np.float32),
    )
    assert summary["summary"]["episodes"] == 1
    assert summary["summary"]["claimed_success"] is False
    aggregated = aggregate_episodes(summary["episodes"])
    assert aggregated["success_rate"] + aggregated["collision_rate"] + aggregated["timeout_rate"] <= 1.0


def test_destination_navigation_uses_the_policy_until_the_goal():
    config = load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml")
    config["execution"] = "offline"
    result = navigate(
        config,
        1.0,
        0.0,
        start=(0.0, 0.0, 0.0),
        policy=lambda _observation, _info: np.array([1.0, 0.0], dtype=np.float32),
    )
    assert result["success"] is True
    assert result["termination_reason"] == "goal_reached"
    assert result["destination"] == [1.0, 0.0]
    assert result["goal_distance"] <= 0.30

    config["execution"] = "gazebo"
    with pytest.raises(SystemExit, match="Gazebo navigation"):
        navigate(
            config,
            1.5,
            0.0,
            policy=lambda _observation, _info: np.zeros(2, dtype=np.float32),
            authorize_live=False,
        )


def test_route_continues_after_the_first_checkpoint():
    config = load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml")
    config["execution"] = "offline"
    result = navigate_route(
        config,
        [((0.8, 0.0), "first"), ((1.6, 0.0), "second")],
        start=(0.0, 0.0, 0.0),
        world_route=True,
        policy=lambda _observation, _info: np.array([1.0, 0.0], dtype=np.float32),
    )
    assert result["success"] is True
    assert result["checkpoints_reached"] == ["first", "second"]
    assert result["destination"] == [1.6, 0.0]
    assert result["goal_distance"] <= 0.30


def test_multi_routes_use_different_starts_aisles_and_destinations(tmp_path):
    from evaluation.warehouse_routes import WAREHOUSE_ROUTES, assert_routes_clear, run_routes

    assert_routes_clear()
    assert len({route["start"] for route in WAREHOUSE_ROUTES}) == 4
    assert len({route["goals"][-1][:2] for route in WAREHOUSE_ROUTES}) == 4
    assert len({route["aisle"] for route in WAREHOUSE_ROUTES}) == 4
    calls = []

    def fake_navigate(config, goals, **kwargs):
        calls.append((goals, kwargs))
        reached = [name for _goal, name in goals]
        success = goals[-1][1] == "east_goal"
        return {
            "success": success,
            "collision": not success,
            "timeout": False,
            "goal_distance": 0.2 if success else 1.5,
            "termination_reason": "goal_reached" if success else "safety_rejected",
            "checkpoints_reached": reached if success else reached[:1],
            "path_length_m": 2.0,
            "duration_s": 8.0,
            "pose_xy": [goals[-1][0][0], goals[-1][0][1]],
        }

    summary = run_routes(
        {"execution": "offline"},
        tmp_path / "checkpoint",
        tmp_path / "routes",
        navigate=fake_navigate,
    )
    assert [kwargs["start"] for _goals, kwargs in calls] == [route["start"] for route in WAREHOUSE_ROUTES]
    assert all(kwargs["resume"] is False and kwargs["world_route"] is True for _goals, kwargs in calls)
    assert summary["episodes"] == 4
    assert summary["successes"] == 1
    assert summary["collisions"] == 3
    assert summary["routes"][0]["route_id"] == "familiar_east"
    assert summary["routes"][2]["destination"] == [5.7, 1.97]
    assert (tmp_path / "routes" / "summary.json").is_file()


def test_a_checkpoint_counts_only_inside_the_arrival_circle():
    assert _within((1.32, 6.35), (1.2, 6.6), 0.30) is True
    assert _within((0.72, 6.67), (1.2, 6.6), 0.30) is False
    world = (0.0, 0.0, 0.3)
    odom = (4.0, -2.0, -0.4)
    dustbin = (1.2, 6.6)
    restored = odom_point_to_world(world_goal_to_odom(dustbin, world, odom), world, odom)
    assert restored == pytest.approx(dustbin)
    # Trial 5's published pose is outside the circle. A second conversion must
    # not replace that measurement with a point meters away.
    assert _route_pose_for_result((0.719, 6.671), (-5.77, 0.57)) == (0.719, 6.671)
    assert _route_pose_for_result(None, (-5.77, 0.57)) == (-5.77, 0.57)
    assert math.hypot(0.719 - 1.2, 6.671 - 6.6) > 0.30


def test_live_pose_does_not_replace_the_reset_transform():
    class Reader:
        def read_latest(self):
            return (1.2, 6.6, 0.0, 1.7)

    class Robot:
        odom = object()

    class Inner:
        world_pose = Reader()
        robot = Robot()

    class Session:
        env = Inner()
        world_robot_pose = (0.0, 0.0, 0.0)
        odom_robot_pose = (4.0, -2.0, -0.4)

    class Env:
        session = Session()

    pose = _live_warehouse_pose(Env())
    assert pose == (1.2, 6.6)
    assert Env.session.world_robot_pose == (0.0, 0.0, 0.0)
    assert Env.session.odom_robot_pose == (4.0, -2.0, -0.4)


def test_rotation_rejection_backs_away_instead_of_turning_again():
    observation = np.ones(40, dtype=np.float32)
    observation[22] = 0.05
    blocked = _recovery_action(0, observation, "Rotation clearance insufficient")
    assert float(blocked[0]) < 0.0
    assert float(blocked[1]) == 0.0
    turn = _recovery_action(0, observation, "Forward obstacle too close")
    assert float(turn[0]) == 0.0
    assert float(turn[1]) != 0.0


def test_recovery_turns_away_from_the_nearest_obstacle():
    observation = np.ones(40, dtype=np.float32)
    observation[22] = 0.05
    action = _recovery_action(0, observation)
    assert float(action[0]) == 0.0
    assert float(action[1]) < 0.0


def test_warehouse_checkpoints_stay_in_the_aisle():
    names = [point["id"] for point in WAREHOUSE_ROUTE]
    assert names[0] == "toward_green_circle"
    assert "green_circle" in names
    assert "shelf_row" in names
    assert names[-1] == "green_dustbin"
    by_id = {point["id"]: (point["x"], point["y"]) for point in WAREHOUSE_ROUTE}
    assert by_id["green_circle"] == (0.0, -3.0)
    assert by_id["shelf_row"] == (1.5, -1.2)
    assert by_id["green_dustbin"] == (1.2, 6.6)
    bay = SHELF_BOX_BAY
    for name in ("shelf_box_west", "shelf_box_middle", "shelf_box_east"):
        x, y = by_id[name]
        assert bay["x_min"] < x < bay["x_max"]
        assert bay["shelf_face_y"] < y < bay["box_face_y"]
    for point in WAREHOUSE_ROUTE:
        assert clearance_m(point["x"], point["y"]) >= 0.90
    continued = [point["id"] for point in route_from_pose(WAREHOUSE_ROUTE, 1.90, 2.06)]
    assert continued[0] == "bay_exit"
    assert continued[-1] == "green_dustbin"
    assert "shelf_box_east" not in continued
    restarted = [point["id"] for point in route_from_pose(WAREHOUSE_ROUTE, 0.0, 0.0)]
    assert restarted[0] == "toward_green_circle"


def test_spawn_repeat_records_success_collision_timeout_and_goal_distance(tmp_path):
    from evaluation.warehouse_repeat import run_spawn_trials

    calls = []

    def fake_navigate(config, goals, **kwargs):
        calls.append(kwargs)
        reached = [name for _goal, name in goals]
        if len(calls) == 1:
            return {
                "success": True,
                "collision": False,
                "timeout": False,
                "goal_distance": 0.22,
                "termination_reason": "goal_reached",
                "checkpoints_reached": reached,
                "path_length_m": 18.0,
                "duration_s": 40.0,
                "pose_xy": [1.25, 6.45],
            }
        return {
            "success": False,
            "collision": True,
            "timeout": False,
            "goal_distance": 2.4,
            "termination_reason": "collision",
            "checkpoints_reached": reached[:3],
            "path_length_m": 4.0,
            "duration_s": 12.0,
            "pose_xy": [0.4, -1.1],
        }

    summary = run_spawn_trials(
        {"execution": "offline"},
        tmp_path / "checkpoint",
        tmp_path / "repeat",
        trials=2,
        navigate=fake_navigate,
    )
    assert calls[0]["start"] == (0.0, 0.0, 0.0)
    assert calls[0]["resume"] is False
    assert calls[0]["world_route"] is True
    assert calls[0]["seed"] == 1
    assert calls[1]["seed"] == 2
    assert summary["episodes"] == 2
    assert summary["successes"] == 1
    assert summary["collisions"] == 1
    assert summary["timeouts"] == 0
    assert summary["success_rate"] == 0.5
    assert summary["mean_terminal_goal_distance_m"] == pytest.approx(1.31)
    assert summary["trials"][0]["checkpoints_reached"][-1] == "green_dustbin"
    assert (tmp_path / "repeat" / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        run_spawn_trials(
            {"execution": "offline"},
            tmp_path / "checkpoint",
            tmp_path / "repeat",
            trials=1,
            navigate=fake_navigate,
        )


def test_lidar_scene_names_the_obstacle_and_the_destination():
    observation = np.ones(40, dtype=np.float32)
    observation[4] = 0.25
    observation[36] = 0.5
    observation[37] = 0.5
    scene = read_scene(observation)
    assert scene["nearest_sector"] == 4
    assert scene["nearest_obstacle_m"] == pytest.approx(3.0)
    assert scene["destination_distance_m"] == pytest.approx(10.0)
    assert scene["destination_heading_rad"] == pytest.approx(math.pi / 2)
    action = decide(
        lambda _observation, _info: np.array([0.2, -0.4], dtype=np.float32),
        observation,
        {},
    )
    assert np.allclose(action, [0.2, -0.4])


def test_normalization_statistics_freeze_during_evaluation(tmp_path):
    directory = tmp_path / "norm"
    result = train(config=_tiny(tmp_path, normalize=True, timesteps=16), output_dir=directory)
    assert result["trained"] is True
    assert (directory / "vecnormalize.pkl").is_file()
    from training.warehouse_ppo_train import build_vec_env, load_run

    config = _tiny(tmp_path, normalize=True, timesteps=16)
    vector = build_vec_env(config, authorize_live=False, training=False)
    _model, loaded = load_run(directory, vector, "cpu", training=False)
    assert loaded.training is False
    assert loaded.norm_reward is False
    loaded.reset()
    count = int(loaded.obs_rms.count)
    loaded.step(np.zeros((1, 2), dtype=np.float32))
    assert int(loaded.obs_rms.count) == count
    loaded.close()
    evaluated = evaluate(
        config,
        directory,
        episodes=1,
        seed=3,
        authorize_live=False,
    )
    assert evaluated["summary"]["episodes"] == 1
    assert evaluated["summary"]["claimed_success"] is False


def test_comparison_and_dashboard_read_records_only(tmp_path):
    report = compare({"ppo": tmp_path / "missing.json", "astar": None, "classical": None, "behavior_cloning": None})
    assert report["ppo"]["status"] == "NOT_RUN"
    assert report["ranking"] is None
    recorded = tmp_path / "ppo.json"
    recorded.write_text(
        json.dumps({"success_rate": 0.0, "collision_rate": 1.0, "timeout_rate": 0.0, "episodes": 2}),
        encoding="utf-8",
    )
    loaded = compare({"ppo": recorded, "astar": None, "classical": None, "behavior_cloning": None})
    assert loaded["ppo"]["status"] == "RECORDED"
    assert loaded["astar"]["status"] == "NOT_RUN"
    assert loaded["ranking"] is None
    run = tmp_path / "plotsrc"
    run.mkdir()
    (run / "progress.csv").write_text(
        "timestep,episodes,mean_return,success_rate,collision_rate,timeout_rate,mean_episode_length,mean_goal_distance,mean_duration_s,mean_min_clearance_m\n"
        "16,1,-0.2,0.0,0.0,1.0,4,0.8,0.8,2.0\n",
        encoding="utf-8",
    )
    (run / "episodes.csv").write_text(
        "episode_index,goal_distance,mean_linear_speed_mps,mean_angular_speed_radps\n"
        "0,0.8,0.1,0.0\n",
        encoding="utf-8",
    )
    (run / "trajectory_0.json").write_text(
        json.dumps({"points": [[0, 0], [0.2, 0]], "start": [0, 0], "goal": [1, 0], "obstacles": []}),
        encoding="utf-8",
    )
    written = plot_run(run, tmp_path / "plots")
    assert written
    assert all(path.is_file() for path in written)
