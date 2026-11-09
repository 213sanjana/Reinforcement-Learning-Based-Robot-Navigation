import pytest

from navigation.expert_evaluation_plan import (
    WORLD_FILENAME,
    benchmark_scenarios,
    build_evaluation_plan,
    diagnostic_scenarios,
)


def test_diagnostics_cover_required_cases():
    scenarios = diagnostic_scenarios()
    categories = {scenario.category for scenario in scenarios}
    reachable = [scenario for scenario in scenarios if scenario.reachable and scenario.mode == "navigation"]

    assert categories >= {
        "low_speed",
        "reachable",
        "obstacle",
        "unreachable",
        "invalid_start",
        "timeout",
        "collision",
    }
    assert len(reachable) >= 3
    assert any(scenario.mode == "collision_probe" for scenario in scenarios)
    assert all(scenario.duration > 0 for scenario in scenarios)


def test_plan_does_not_enable_motion():
    plan = build_evaluation_plan("diagnostics")

    assert plan.world_filename == WORLD_FILENAME
    assert plan.movement_enabled is False
    assert plan.simulation_reset_enabled is False
    assert plan.inflation_radius == 0.45


def test_benchmark_is_a_fixed_20_episode_cycle():
    first = benchmark_scenarios()
    second = benchmark_scenarios()

    assert len(first) == 20
    assert [scenario.scenario_id for scenario in first] == [
        scenario.scenario_id for scenario in second
    ]
    assert len({scenario.scenario_id for scenario in first}) == 20
    assert {scenario.category for scenario in first} >= {
        "low_speed",
        "reachable",
        "obstacle",
    }
    assert all(scenario.mode == "navigation" for scenario in first)
    assert all(scenario.reachable for scenario in first)
    assert first[0].start == first[3].start
    assert first[1].goal == first[4].goal


def test_unknown_suite_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        build_evaluation_plan("all")
