from types import SimpleNamespace

import pytest

from navigation.expert_evaluation_plan import diagnostic_scenarios
from navigation.expert_evaluation_runner import run_suite, verify_stop


class FakeRobot:
    def __init__(self):
        self.expert_publish_callback = None
        self.stopped = 0
        self.odom = SimpleNamespace(
            twist=SimpleNamespace(
                twist=SimpleNamespace(
                    linear=SimpleNamespace(x=0.2),
                    angular=SimpleNamespace(z=0.05),
                )
            )
        )

    def stop(self):
        self.stopped += 1
        callback = self.expert_publish_callback
        if callback is not None:
            callback(0.0, 0.0, 100)


def along(start, goal):
    x0, y0 = start[0], start[1]
    x1, y1 = goal
    return [
        (x0, y0, 0.0),
        ((x0 + x1) / 2, (y0 + y1) / 2, 0.0),
        (x1, y1, 0.0),
    ]


class FakeRunner:
    def __init__(self):
        self.robot = FakeRobot()
        self.calls = []
        self.closed = False

    def close(self):
        self.closed = True
        self.calls.append("close")

    def prepare(self, start, goal):
        self.calls.append(("prepare", tuple(start), tuple(goal)))
        if abs(start[0] - 8.0) < 1e-6:
            raise RuntimeError("Expert planning failed: PLANNING_FAILED")
        if tuple(goal) == (1.5, 0.0):
            raise RuntimeError("Expert planning failed: PLANNING_FAILED")

    def run(self, duration):
        self.calls.append(("run", duration))
        start, goal = self.calls[-2][1], self.calls[-2][2]
        if duration < 1.0:
            status = "TIME_LIMIT"
            poses = [(start[0], start[1], start[2])]
        else:
            status = "GOAL_REACHED"
            poses = along(start, goal)
        return SimpleNamespace(
            status=status,
            planned_path=[(start[0], start[1]), goal],
            executed_poses=poses,
            accepted_commands=len(poses),
            rejected_commands=0,
        )


def probe(runner, scenario):
    runner.calls.append(("probe", scenario.scenario_id))
    return SimpleNamespace(
        status="SAFETY_STOP",
        planned_path=[],
        executed_poses=[(0.35, 0.0, 0.0), (0.7, 0.0, 0.0)],
        accepted_commands=4,
        rejected_commands=0,
    )


def test_verify_stop_observes_the_published_zero_command():
    robot = FakeRobot()
    info = verify_stop(robot)

    assert robot.stopped == 1
    assert info["stop_verified"] is True
    assert info["stop_command_linear"] == 0.0
    assert info["stop_command_angular"] == 0.0
    assert info["odom_linear_after_stop"] == pytest.approx(0.2)
    assert robot.expert_publish_callback is None


def test_suite_records_every_contract_and_stops_between_episodes(tmp_path):
    runner = FakeRunner()
    records = run_suite(
        diagnostic_scenarios(),
        runner,
        tmp_path,
        probe=probe,
    )

    assert [record["scenario_id"] for record in records] == [
        scenario.scenario_id for scenario in diagnostic_scenarios()
    ]
    assert all(record["passed"] for record in records)
    assert all(record["stop_verified"] for record in records)
    assert runner.robot.stopped == len(records)
    assert ("probe", "collision_probe") in runner.calls
    assert sum(call[0] == "prepare" for call in runner.calls) == 6
    assert (tmp_path / "episode_unreachable_goal.json").is_file()
    assert (tmp_path / "episode_low_speed_single.png").is_file()
    unreachable = next(record for record in records if record["scenario_id"] == "unreachable_goal")
    assert unreachable["status"] == "PLANNING_FAILED"
    assert unreachable["planned_path"] == []
    timeout = next(record for record in records if record["scenario_id"] == "timeout_short")
    assert timeout["status"] == "TIME_LIMIT"
    assert timeout["passed"] is True


def test_cleanup_failure_stops_the_remaining_suite(tmp_path):
    runner = FakeRunner()
    calls = {"n": 0}

    def stop_check(_robot):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Stop failed")
        return verify_stop(_robot)

    records = run_suite(
        diagnostic_scenarios(),
        runner,
        tmp_path,
        probe=probe,
        stop_check=stop_check,
    )

    assert len(records) == 1
    assert records[0]["status"] == "CLEANUP_FAILED"
    assert records[0]["passed"] is False
    assert "Stop failed" in records[0]["error"]
    assert sum(call[0] == "prepare" for call in runner.calls) == 1
