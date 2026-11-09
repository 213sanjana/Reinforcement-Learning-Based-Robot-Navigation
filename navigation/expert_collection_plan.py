"""Side-effect-free planning for expert demonstration collection."""

from dataclasses import dataclass
from pathlib import Path
import math

from jackal_rl_navigation.collision_config import CollisionConfig


@dataclass(frozen=True)
class CollectionEpisodePlan:
    episode_id: int
    start: tuple
    goal: tuple
    duration: float


@dataclass(frozen=True)
class CollectionPlan:
    world: Path
    output_dir: Path
    episodes: tuple
    inflation_radius: float
    collision_config: CollisionConfig
    movement_enabled: bool = False
    simulation_reset_enabled: bool = False


def build_collection_plan(args, episode_ids):
    """Validate parameters and describe episodes without executing."""

    if not isinstance(args.world, (str, Path)):
        raise TypeError("Invalid world path")

    world = Path(args.world).expanduser()
    if not world.is_file():
        raise FileNotFoundError(f"World not found: {world}")

    output_dir = Path(args.output_dir).expanduser()

    ids = tuple(episode_ids)

    if len(ids) != args.episodes:
        raise ValueError("Episode ID count mismatch")

    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in ids
    ):
        raise ValueError("Episode IDs must be nonnegative integers")

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode IDs")

    start = tuple(args.start)
    goal = tuple(args.goal)

    if len(start) != 3 or len(goal) != 2:
        raise ValueError("Invalid start or goal dimensions")

    numeric = (
        *start,
        *goal,
        args.duration,
        args.inflation_radius,
    )

    if not all(math.isfinite(float(v)) for v in numeric):
        raise ValueError("Nonfinite collection parameter")

    if args.duration <= 0:
        raise ValueError("Duration must be positive")

    if args.inflation_radius <= 0:
        raise ValueError("Inflation radius must be positive")

    collision_config = CollisionConfig(enabled=True)

    episodes = tuple(
        CollectionEpisodePlan(
            episode_id=episode_id,
            start=start,
            goal=goal,
            duration=float(args.duration),
        )
        for episode_id in ids
    )

    return CollectionPlan(
        world=world,
        output_dir=output_dir,
        episodes=episodes,
        inflation_radius=float(args.inflation_radius),
        collision_config=collision_config,
    )
