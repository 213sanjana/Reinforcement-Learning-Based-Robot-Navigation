"""Autonomous warehouse navigation for the Jackal.

The robot reads its LiDAR as 36 range sectors. The same 40-D observation
carries the current checkpoint as a destination distance and heading. A PPO
policy turns those readings into a linear and angular command. The existing
safety monitor rejects a command that points into an obstacle, and the route
advances to the next aisle checkpoint after each arrival.

Repeated training is `training.warehouse_ppo_train`. Reward feedback comes
from `configs/warehouse_reward.yaml`: progress toward the checkpoint, a goal
bonus, a collision penalty, and a penalty for driving close to an obstacle.
This module loads a finished policy and drives the checkpoint route. It does
not train during a live run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluation.warehouse_checkpoints import WAREHOUSE_ROUTE
from evaluation.warehouse_navigate import navigate_explore, navigate_route
from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_estop import live_authorized
from jackal_rl_navigation.world_catalog import apply_world, world_names


def read_scene(observation, builder: ObservationBuilder | None = None) -> dict:
    """Identify the nearest obstacle and the current destination from LiDAR."""

    builder = ObservationBuilder() if builder is None else builder
    vector = np.asarray(observation, dtype=np.float32)
    if vector.shape != (builder.observation_dim,) or not np.all(np.isfinite(vector)):
        raise ValueError("LiDAR observation must be a finite 40-vector")
    ranges = np.asarray(vector[: builder.lidar_bins], dtype=np.float32) * builder.lidar_max_range
    nearest = int(np.argmin(ranges))
    return {
        "nearest_obstacle_m": float(ranges[nearest]),
        "nearest_sector": nearest,
        "destination_distance_m": float(vector[builder.lidar_bins] * builder.goal_max_distance),
        "destination_heading_rad": float(vector[builder.lidar_bins + 1] * math.pi),
    }


def decide(policy, observation, info) -> np.ndarray:
    """Ask the PPO policy for one forward and one turn command."""

    action = np.asarray(policy(observation, info), dtype=np.float32).reshape(2)
    if not np.all(np.isfinite(action)):
        raise ValueError("PPO action must be finite")
    return action


def run_checkpoint_route(
    config: dict,
    checkpoint: Path,
    *,
    authorize_live: bool = False,
    start=None,
    seed: int = 1,
    world: str | None = None,
) -> dict:
    """Drive the learned policy through the selected world.

    Omitting ``world`` or passing ``warehouse`` keeps the warehouse aisle
    route. ``house``, ``ground``, and ``open`` follow open space from LiDAR.
    """

    use_warehouse_route = world is None or world == "warehouse"
    if world is not None:
        config = apply_world(config, world)
    scene = {}

    def _observe(observation, info):
        scene.clear()
        scene.update(read_scene(observation))
        info["lidar_scene"] = dict(scene)

    if use_warehouse_route:
        result = navigate_route(
            config,
            [((point["x"], point["y"]), point["id"]) for point in WAREHOUSE_ROUTE],
            checkpoint=Path(checkpoint),
            start=start,
            authorize_live=authorize_live,
            seed=seed,
            resume=True,
            on_observation=_observe,
        )
    else:
        result = navigate_explore(
            config,
            Path(checkpoint),
            start=start,
            authorize_live=authorize_live,
            seed=seed,
        )
    result["world"] = None if world is None else config.get("selected_world", world)
    result["world_name"] = config.get("world_name")
    if scene:
        result["lidar_scene"] = dict(scene)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drive the Jackal with a learned PPO policy. "
            "House, ground, and open follow open space from LiDAR."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--world", choices=world_names())
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
    result = run_checkpoint_route(
        config,
        Path(args.checkpoint),
        authorize_live=live_authorized(args.authorize_live),
        start=start,
        seed=args.seed,
        world=args.world,
    )
    payload = json.dumps(result, indent=2, default=str)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
