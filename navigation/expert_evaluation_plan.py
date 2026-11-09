"""Side-effect-free scenarios for live A* expert evaluation.

Coordinates assume jackal_obstacle_test_collision.sdf:
a 0.5 m box centered at (1.5, 0) and planner bounds [-2, 5] x [-3, 3].
Building a plan does not start ROS, Gazebo, or a velocity publisher.
"""

from dataclasses import dataclass
import math


WORLD_FILENAME = "jackal_obstacle_test_collision.sdf"
OBSTACLE_BOX = (1.5, 0.0, 0.5, 0.5)
INFLATION_RADIUS = 0.45

# Clear of the inflated box, so a straight route stays in free space.
SOUTH_START = (0.0, -1.5, 0.0)
SOUTH_GOAL = (2.5, -1.5)
NORTH_START = (0.0, 1.5, 0.0)
NORTH_GOAL = (2.5, 1.5)
AROUND_START = (-1.0, -1.5, 0.0)
AROUND_GOAL = (3.0, 1.5)


@dataclass(frozen=True)
class EvaluationScenario:
    scenario_id: str
    category: str
    start: tuple
    goal: tuple
    duration: float
    expected_statuses: tuple
    mode: str
    reachable: bool
    description: str


@dataclass(frozen=True)
class EvaluationPlan:
    suite: str
    world_filename: str
    inflation_radius: float
    scenarios: tuple
    movement_enabled: bool = False
    simulation_reset_enabled: bool = False


def _scenario(
    scenario_id,
    category,
    start,
    goal,
    duration,
    expected_statuses,
    mode,
    reachable,
    description,
):
    start = tuple(float(v) for v in start)
    goal = tuple(float(v) for v in goal)
    expected = tuple(expected_statuses)

    if len(start) != 3 or len(goal) != 2:
        raise ValueError("Invalid start or goal dimensions")

    if not all(math.isfinite(v) for v in (*start, *goal, float(duration))):
        raise ValueError("Nonfinite evaluation parameter")

    if duration <= 0:
        raise ValueError("Duration must be positive")

    if mode not in ("navigation", "collision_probe"):
        raise ValueError("Unknown evaluation mode")

    if not expected:
        raise ValueError("Expected statuses cannot be empty")

    return EvaluationScenario(
        scenario_id=scenario_id,
        category=category,
        start=start,
        goal=goal,
        duration=float(duration),
        expected_statuses=expected,
        mode=mode,
        reachable=bool(reachable),
        description=description,
    )


def diagnostic_scenarios():
    """One pass over every required Phase 6D.4 case."""

    return (
        _scenario(
            "low_speed_single",
            "low_speed",
            SOUTH_START,
            SOUTH_GOAL,
            45.0,
            ("GOAL_REACHED",),
            "navigation",
            True,
            "Low-speed straight goal south of the obstacle.",
        ),
        _scenario(
            "reachable_north",
            "reachable",
            NORTH_START,
            NORTH_GOAL,
            70.0,
            ("GOAL_REACHED",),
            "navigation",
            True,
            "Straight goal north of the obstacle.",
        ),
        _scenario(
            "reachable_around",
            "obstacle",
            AROUND_START,
            AROUND_GOAL,
            260.0,
            ("GOAL_REACHED",),
            "navigation",
            True,
            "Route from the south-west to the north-east around the box.",
        ),
        _scenario(
            "unreachable_goal",
            "unreachable",
            SOUTH_START,
            (1.5, 0.0),
            20.0,
            ("PLANNING_FAILED", "INVALID_PATH"),
            "navigation",
            False,
            "Goal lies inside the static box, so A* must not return a route.",
        ),
        _scenario(
            "invalid_start",
            "invalid_start",
            (8.0, 0.0, 0.0),
            SOUTH_GOAL,
            20.0,
            ("PLANNING_FAILED", "INVALID_PATH"),
            "navigation",
            False,
            "Start is outside the planner map bounds.",
        ),
        _scenario(
            "timeout_short",
            "timeout",
            SOUTH_START,
            SOUTH_GOAL,
            0.8,
            ("TIME_LIMIT", "TIMEOUT"),
            "navigation",
            True,
            "Same reachable goal with a wall-clock limit shorter than the route.",
        ),
        _scenario(
            "collision_probe",
            "collision",
            (0.35, 0.0, 0.0),
            (1.5, 0.0),
            20.0,
            ("SAFETY_STOP",),
            "collision_probe",
            False,
            "Slow forward command into the box through the existing velocity publisher.",
        ),
    )


def benchmark_scenarios(count=20):
    """Repeat the three reachable routes in a fixed cycle."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Benchmark count must be a positive integer")

    bases = (
        diagnostic_scenarios()[0],
        diagnostic_scenarios()[1],
        diagnostic_scenarios()[2],
    )
    selected = []

    for index in range(count):
        base = bases[index % len(bases)]
        selected.append(
            _scenario(
                f"bench_{index:02d}_{base.scenario_id}",
                base.category,
                base.start,
                base.goal,
                base.duration,
                base.expected_statuses,
                base.mode,
                base.reachable,
                base.description,
            )
        )

    return tuple(selected)


def build_evaluation_plan(suite):
    """Return an immutable plan. This function does not execute it."""

    if suite == "diagnostics":
        scenarios = diagnostic_scenarios()
    elif suite == "benchmark":
        scenarios = benchmark_scenarios(20)
    else:
        raise ValueError("Unknown evaluation suite")

    ids = [scenario.scenario_id for scenario in scenarios]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scenario ids")

    return EvaluationPlan(
        suite=suite,
        world_filename=WORLD_FILENAME,
        inflation_radius=INFLATION_RADIUS,
        scenarios=scenarios,
    )
