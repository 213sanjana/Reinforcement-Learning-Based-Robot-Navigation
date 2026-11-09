import pytest
from navigation import jackal_generalization_diagnostic as d


def test_existing_default_plans_are_preserved():
    pilot = d.episode_plan("pilot")
    matrix = d.episode_plan("matrix")
    assert len(pilot) == 4
    assert {r["scenario"] for r in pilot} == {"anchor"}
    assert len(matrix) == len(d.matrix()) * len(d.PILOT)


@pytest.mark.parametrize("name", ["distance_1p5", "obstacle_west_east"])
def test_selector_uses_existing_scenario_and_matched_order(name):
    rows = d.episode_plan("pilot", name)
    scenario = next(s for s in d.matrix() if s.scenario_id == name)
    assert len(rows) == 4
    assert [(r["policy"], r["seed"]) for r in rows] == list(d.PILOT)
    assert all(r["scenario"] == name for r in rows)
    assert all(r["start"] == list(scenario.start) for r in rows)
    assert all(r["goal"] == list(scenario.goal) for r in rows)


def test_unknown_selector_is_rejected():
    with pytest.raises(ValueError):
        d.episode_plan("pilot", "does_not_exist")


def test_selected_preview_does_not_execute(monkeypatch, capsys):
    import json

    def forbidden(*args, **kwargs):
        raise AssertionError("Live execution must not occur")

    monkeypatch.setattr(d, "verify_checkpoints", lambda: {})
    monkeypatch.setattr(d, "validate_matrix", lambda: {})
    monkeypatch.setattr(d, "run_diagnostic", forbidden)
    monkeypatch.setattr(d, "pilot", forbidden)

    assert d.main(["preview", "--scenario", "distance_1p5"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["max_episodes"] == 4
    assert {r["scenario"] for r in result["episode_order"]} == {"distance_1p5"}
