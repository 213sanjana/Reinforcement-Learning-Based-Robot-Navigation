from types import SimpleNamespace

import pytest

from navigation.expert_live_runner_factory import (
    create_live_runner,
)


def test_disabled_by_default():
    with pytest.raises(RuntimeError, match="disabled"):
        create_live_runner(
            SimpleNamespace(),
            world_path="test.sdf",
            inflation_radius=0.95,
        )


def test_disabled_gate_does_not_call_factories():
    def forbidden(*args, **kwargs):
        pytest.fail("Factory must not be called")

    with pytest.raises(RuntimeError, match="disabled"):
        create_live_runner(
            SimpleNamespace(),
            world_path="test.sdf",
            inflation_radius=0.95,
            environment_factory=forbidden,
            planner_factory=forbidden,
            runner_factory=forbidden,
        )


def test_authorized_factory_wiring():
    calls = {}
    env = SimpleNamespace(close=lambda: None)
    planner = object()
    runner = object()

    def make_env(**kwargs):
        calls["collision"] = kwargs["collision_config"]
        return env

    def make_planner(**kwargs):
        calls["planner"] = kwargs
        return planner

    def make_runner(**kwargs):
        calls["runner"] = kwargs
        return runner

    result = create_live_runner(
        SimpleNamespace(),
        world_path="test.sdf",
        inflation_radius=0.95,
        allow_live=True,
        environment_factory=make_env,
        planner_factory=make_planner,
        runner_factory=make_runner,
    )

    assert result is runner
    assert calls["collision"].enabled
    assert calls["planner"]["world_path"] == "test.sdf"
    assert calls["planner"]["inflation_radius"] == 0.95
    assert calls["runner"]["env"] is env
    assert calls["runner"]["planner"] is planner
    assert calls["runner"]["enable_motion"] is True
    assert calls["runner"]["controller_type"] == "pure-pursuit"


def test_planner_failure_closes_environment():
    state = {"closed": False}

    class FakeEnv:
        def close(self):
            state["closed"] = True

    def fail_planner(**kwargs):
        raise RuntimeError("Planner failed")

    with pytest.raises(RuntimeError, match="Planner failed"):
        create_live_runner(
            SimpleNamespace(),
            world_path="test.sdf",
            inflation_radius=0.95,
            allow_live=True,
            environment_factory=lambda **kwargs: FakeEnv(),
            planner_factory=fail_planner,
            runner_factory=lambda **kwargs: object(),
        )

    assert state["closed"]


def test_runner_failure_closes_environment():
    state = {"closed": False}

    class FakeEnv:
        def close(self):
            state["closed"] = True

    def fail_runner(**kwargs):
        raise RuntimeError("Runner failed")

    with pytest.raises(RuntimeError, match="Runner failed"):
        create_live_runner(
            SimpleNamespace(),
            world_path="test.sdf",
            inflation_radius=0.95,
            allow_live=True,
            environment_factory=lambda **kwargs: FakeEnv(),
            planner_factory=lambda **kwargs: object(),
            runner_factory=fail_runner,
        )

    assert state["closed"]


def test_import_does_not_initialize_ros():
    import navigation.expert_live_runner_factory as factory

    assert callable(factory.create_live_runner)
