from types import SimpleNamespace

import pytest

from navigation.expert_collection_orchestrator import (
    collect_episode,
    collect_plan,
)


def episode(episode_id=0):
    return SimpleNamespace(
        episode_id=episode_id,
        start=(0.0, 0.0, 0.0),
        goal=(3.0, 0.0),
        duration=45.0,
    )


def result(status="GOAL_REACHED", rejected=0):
    return SimpleNamespace(
        status=status,
        rejected_commands=rejected,
    )


class FakeRunner:
    def __init__(self, result_value=None, failure=None):
        self.result_value = result_value or result()
        self.failure = failure
        self.prepared = False
        self.ran = False
        self.closed = False

    def prepare(self, start, goal):
        self.prepared = True

        if self.failure == "prepare":
            raise RuntimeError("Preparation failed")

    def run(self, duration):
        self.ran = True

        if self.failure == "run":
            raise RuntimeError("Execution failed")

        return self.result_value

    def close(self):
        self.closed = True


def test_execution_disabled_by_default(tmp_path):
    plan = SimpleNamespace(
        episodes=(episode(),),
        output_dir=tmp_path,
    )

    def forbidden_factory(_):
        raise AssertionError("Runner must not initialize")

    with pytest.raises(RuntimeError, match="disabled"):
        collect_plan(
            plan,
            forbidden_factory,
            lambda *args: (),
        )


def test_successful_episode_exports(tmp_path):
    runner = FakeRunner()
    calls = []

    def exporter(*args):
        calls.append(args)
        return ("episode.npz", "episode.json")

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        exporter,
        tmp_path,
    )

    assert outcome.exported
    assert outcome.status == "GOAL_REACHED"
    assert runner.prepared
    assert runner.ran
    assert runner.closed
    assert len(calls) == 1


def test_collision_does_not_export(tmp_path):
    runner = FakeRunner(result("SAFETY_STOP"))

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        lambda *args: pytest.fail("Exporter called"),
        tmp_path,
    )

    assert not outcome.exported
    assert outcome.status == "SAFETY_STOP"
    assert runner.closed


def test_rejected_commands_do_not_export(tmp_path):
    runner = FakeRunner(result(rejected=1))

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        lambda *args: pytest.fail("Exporter called"),
        tmp_path,
    )

    assert not outcome.exported
    assert outcome.status == "REJECTED_COMMANDS"
    assert runner.closed


def test_prepare_failure_closes_runner(tmp_path):
    runner = FakeRunner(failure="prepare")

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        lambda *args: (),
        tmp_path,
    )

    assert outcome.status == "ERROR"
    assert not outcome.exported
    assert runner.closed


def test_run_failure_closes_runner(tmp_path):
    runner = FakeRunner(failure="run")

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        lambda *args: (),
        tmp_path,
    )

    assert outcome.status == "ERROR"
    assert not outcome.exported
    assert runner.closed


def test_export_failure_closes_runner(tmp_path):
    runner = FakeRunner()

    def failing_exporter(*args):
        raise OSError("Disk unavailable")

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        failing_exporter,
        tmp_path,
    )

    assert outcome.status == "ERROR"
    assert not outcome.exported
    assert runner.closed


def test_factory_failure_returns_error(tmp_path):
    def failing_factory(_):
        raise RuntimeError("Factory unavailable")

    outcome = collect_episode(
        episode(),
        failing_factory,
        lambda *args: (),
        tmp_path,
    )

    assert outcome.status == "ERROR"
    assert not outcome.exported


def test_plan_stops_after_failure(tmp_path):
    runners = []

    def factory(_):
        runner = FakeRunner(result("SAFETY_STOP"))
        runners.append(runner)
        return runner

    plan = SimpleNamespace(
        episodes=(episode(0), episode(1)),
        output_dir=tmp_path,
    )

    outcomes = collect_plan(
        plan,
        factory,
        lambda *args: (),
        allow_execution=True,
    )

    assert len(outcomes) == 1
    assert not outcomes[0].exported
    assert len(runners) == 1


def test_plan_processes_successful_episodes(tmp_path):
    runners = []

    def factory(_):
        runner = FakeRunner()
        runners.append(runner)
        return runner

    plan = SimpleNamespace(
        episodes=(episode(0), episode(1)),
        output_dir=tmp_path,
    )

    outcomes = collect_plan(
        plan,
        factory,
        lambda *args: ("file.npz", "file.json"),
        allow_execution=True,
    )

    assert len(outcomes) == 2
    assert all(outcome.exported for outcome in outcomes)
    assert all(runner.closed for runner in runners)



def test_close_failure_returns_unsuccessful_outcome(tmp_path):
    class CloseFailureRunner(FakeRunner):
        def close(self):
            raise RuntimeError("Close failed")

    runner = CloseFailureRunner()

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        lambda *args: ("episode.npz", "episode.json"),
        tmp_path,
    )

    assert outcome.status == "CLEANUP_FAILED"
    assert not outcome.exported
    assert "Close failed" in outcome.error


def test_cleanup_failure_after_run_failure(tmp_path):
    class DoubleFailureRunner(FakeRunner):
        def close(self):
            raise RuntimeError("Close failed")

    runner = DoubleFailureRunner(failure="run")

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        lambda *args: pytest.fail("Exporter called"),
        tmp_path,
    )

    assert outcome.status == "CLEANUP_FAILED"
    assert not outcome.exported
    assert "Execution failed" in outcome.error
    assert "Close failed" in outcome.error


def test_plan_stops_after_cleanup_failure(tmp_path):
    created = []

    class CloseFailureRunner(FakeRunner):
        def close(self):
            raise RuntimeError("Close failed")

    def factory(_):
        runner = CloseFailureRunner()
        created.append(runner)
        return runner

    plan = SimpleNamespace(
        episodes=(episode(0), episode(1)),
        output_dir=tmp_path,
    )

    outcomes = collect_plan(
        plan,
        factory,
        lambda *args: ("episode.npz", "episode.json"),
        allow_execution=True,
    )

    assert len(outcomes) == 1
    assert outcomes[0].status == "CLEANUP_FAILED"
    assert len(created) == 1


def test_cleanup_failure_does_not_change_execution_gate(tmp_path):
    plan = SimpleNamespace(
        episodes=(episode(),),
        output_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="disabled"):
        collect_plan(
            plan,
            lambda _: pytest.fail("Runner created"),
            lambda *args: pytest.fail("Exporter called"),
        )



def test_export_happens_after_close(tmp_path):
    events = []

    class OrderedRunner(FakeRunner):
        def close(self):
            events.append("close")
            super().close()

    runner = OrderedRunner()

    def exporter(*args):
        events.append("export")
        return ("episode.npz", "episode.json")

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        exporter,
        tmp_path,
    )

    assert outcome.exported
    assert events == ["close", "export"]


def test_failed_close_never_invokes_exporter(tmp_path):
    class CloseFailureRunner(FakeRunner):
        def close(self):
            raise RuntimeError("Cleanup failed")

    runner = CloseFailureRunner()
    calls = []

    def exporter(*args):
        calls.append(args)
        return ("episode.npz", "episode.json")

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        exporter,
        tmp_path,
    )

    assert outcome.status == "CLEANUP_FAILED"
    assert not outcome.exported
    assert calls == []


def test_failed_close_does_not_create_dataset(tmp_path):
    class CloseFailureRunner(FakeRunner):
        def close(self):
            raise RuntimeError("Cleanup failed")

    output_dir = tmp_path / "dataset"

    outcome = collect_episode(
        episode(),
        lambda _: CloseFailureRunner(),
        lambda *args: pytest.fail("Exporter must not run"),
        output_dir,
    )

    assert outcome.status == "CLEANUP_FAILED"
    assert not output_dir.exists()


def test_exporter_failure_after_close(tmp_path):
    runner = FakeRunner()

    def exporter(*args):
        assert runner.closed
        raise OSError("Storage failure")

    outcome = collect_episode(
        episode(),
        lambda _: runner,
        exporter,
        tmp_path,
    )

    assert outcome.status == "ERROR"
    assert not outcome.exported
    assert runner.closed
