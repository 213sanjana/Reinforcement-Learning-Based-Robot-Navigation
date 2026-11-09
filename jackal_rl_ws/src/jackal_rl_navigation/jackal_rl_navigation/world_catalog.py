"""Named worlds for autonomous Jackal navigation.

A name such as ``house`` or ``ground`` selects a Gazebo Fortress world file
and that world's checkpoint route. Resolving a name does not launch Gazebo,
train a policy, or publish a command.
"""

from __future__ import annotations

from pathlib import Path

from jackal_rl_navigation.warehouse_scenarios import assert_disjoint_splits


WORLD_NAMES = ("house", "ground", "open", "warehouse")

_STAGE_BY_WORLD = {
    "house": "house",
    "ground": "ground",
    "open": "open",
    "warehouse": "external_warehouse",
}


def world_names() -> tuple[str, ...]:
    return WORLD_NAMES


_NAVIGATION_TARGETS = {
    "house": ("jackal_house", "jackal_rl"),
    "ground": ("jackal_ground", "jackal_rl"),
    "open": ("jackal_demo_open", "jackal_rl"),
}


def require_navigation_world(config: dict) -> None:
    """Accept the external warehouse, or one explicitly selected Fortress world.

    Stationary observation keeps its own warehouse-only check. This function
    does not launch Gazebo and does not change the process partition.
    """

    selected = config.get("selected_world")
    if selected in _NAVIGATION_TARGETS:
        world, partition = _NAVIGATION_TARGETS[selected]
        if config.get("world_name") != world or config.get("gazebo_partition") != partition:
            raise RuntimeError(
                f"World {selected!r} requires world {world!r} and partition {partition!r}. "
                f"Got world {config.get('world_name')!r} and partition {config.get('gazebo_partition')!r}."
            )
        if config.get("namespace", "j100_0000") != "j100_0000":
            raise RuntimeError(
                "Navigation worlds use namespace 'j100_0000'. "
                f"Got {config.get('namespace')!r}."
            )
        return
    from jackal_rl_navigation.warehouse_observation import require_external_warehouse_target

    require_external_warehouse_target(config)


def apply_world(config: dict, name: str) -> dict:
    """Return a config whose active stage is the named world."""

    key = str(name).strip().lower()
    if key not in _STAGE_BY_WORLD:
        known = ", ".join(WORLD_NAMES)
        raise ValueError(f"Unknown world {name!r}. Known worlds: {known}")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("Config has no environment catalog to select a world from")
    stages = environment.get("stages") or {}
    stage_id = _STAGE_BY_WORLD[key]
    if stage_id not in stages:
        raise ValueError(f"Environment catalog has no stage {stage_id!r} for world {key!r}")
    stage = dict(stages[stage_id])
    world_file = stage.get("world_file")
    if not world_file:
        raise ValueError(f"World {key!r} has no world file")
    root = Path(config.get("repo_root") or _repo_root())
    candidate = Path(world_file)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.is_file():
        raise ValueError(f"World file for {key!r} is missing: {candidate}")
    scenarios = stage.get("scenarios") or []
    assert_disjoint_splits(scenarios)
    route = stage.get("route")
    if route is not None:
        _validate_route(route, key)
    updated = dict(config)
    updated["stage_id"] = stage_id
    updated["scenarios"] = scenarios
    updated["obstacles"] = stage.get("obstacles", [])
    updated["world_file"] = stage.get("world_file")
    updated["world_name"] = stage.get("world_name")
    updated["bounds_xy"] = stage.get("bounds_xy")
    updated["require_listed_pair"] = stage.get(
        "require_listed_pair",
        environment.get("require_listed_pair", False),
    )
    partition = stage.get("gazebo_partition", environment.get("gazebo_partition"))
    if partition is None:
        updated.pop("gazebo_partition", None)
    else:
        updated["gazebo_partition"] = partition
    if route is None:
        updated.pop("route", None)
    else:
        updated["route"] = route
    updated["selected_world"] = key
    return updated


def _validate_route(route, name: str) -> None:
    if not isinstance(route, list) or not route:
        raise ValueError(f"World {name!r} needs a non-empty checkpoint route")
    seen = set()
    for index, point in enumerate(route):
        if not isinstance(point, dict):
            raise ValueError(f"World {name!r} route item {index} must be a mapping")
        point_id = str(point.get("id", "")).strip()
        if not point_id or point_id in seen:
            raise ValueError(f"World {name!r} route ids must be unique and non-empty")
        seen.add(point_id)
        for axis in ("x", "y"):
            value = point.get(axis)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"World {name!r} route point {point_id!r} needs a numeric {axis}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
