"""Roam the large aisle warehouse from LiDAR.

The world is the Apache-2.0 F3-05 hall from
https://github.com/mertgulerx/world-collection
adapted for Gazebo Fortress. It is not the checkpoint warehouse and this
file does not follow that route.

Launch jackal_rl_ws/worlds/aisle_warehouse/jackal_aisle_warehouse.sdf with
jackal_demo_sim.launch.py, then run this module with --authorize-live.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.warehouse_navigate import navigate_explore
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_estop import live_authorized
import jackal_rl_navigation.world_catalog as world_catalog


WORLD_NAME = "jackal_aisle_warehouse"
PARTITION = "jackal_rl"
_ORIGINAL_REQUIRE = world_catalog.require_navigation_world


def _allow_aisle_warehouse(config: dict) -> None:
    """Accept only this world. Every other world keeps the existing check."""

    if (
        config.get("world_name") == WORLD_NAME
        and config.get("gazebo_partition") == PARTITION
    ):
        if config.get("namespace", "j100_0000") != "j100_0000":
            raise RuntimeError("The aisle warehouse uses namespace 'j100_0000'.")
        return
    _ORIGINAL_REQUIRE(config)


def aisle_config(config: dict) -> dict:
    """Point a loaded policy config at the aisle warehouse."""

    updated = dict(config)
    updated["world_name"] = WORLD_NAME
    updated["gazebo_partition"] = PARTITION
    updated["world_file"] = "jackal_rl_ws/worlds/aisle_warehouse/jackal_aisle_warehouse.sdf"
    updated["selected_world"] = "aisle_warehouse"
    updated.pop("route", None)
    return updated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Roam the large aisle warehouse with the saved PPO policy.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--authorize-live", action="store_true")
    args = parser.parse_args(argv)
    output = None if args.output is None else Path(args.output)
    if output is not None and output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    world_catalog.require_navigation_world = _allow_aisle_warehouse
    config = aisle_config(load_warehouse_config(args.config))
    result = navigate_explore(
        config,
        Path(args.checkpoint),
        authorize_live=live_authorized(args.authorize_live),
        seed=args.seed,
    )
    result["world"] = "aisle_warehouse"
    result["world_name"] = WORLD_NAME
    payload = json.dumps(result, indent=2, default=str)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
