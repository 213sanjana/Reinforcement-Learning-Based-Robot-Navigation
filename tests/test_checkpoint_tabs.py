"""Offline checks for the checkpoint tab page. They do not start Gazebo."""

import json
import threading
from pathlib import Path

import pytest

from checkpoint_tabs.catalog import checkpoint_by_id, checkpoint_for_number, checkpoint_tabs, numbered_checkpoints
from checkpoint_tabs.panel import CheckpointPopup
from checkpoint_tabs.selector import CheckpointSelector
from evaluation.warehouse_checkpoints import WAREHOUSE_ROUTE
from jackal_rl_navigation.warehouse_config import load_warehouse_config

ROOT = Path(__file__).resolve().parents[1]


def _config(execution: str) -> dict:
    config = load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml")
    config["execution"] = execution
    return config


def test_tabs_list_every_warehouse_checkpoint_once():
    seen = [point["id"] for tab in checkpoint_tabs() for point in tab["checkpoints"]]
    assert seen == [point["id"] for point in WAREHOUSE_ROUTE]
    assert len(seen) == len(set(seen))
    assert checkpoint_by_id("green_dustbin")["y"] == 6.6


def test_popup_buttons_are_only_the_existing_green_circles():
    numbered = numbered_checkpoints()
    record = (ROOT / "checkpoint_tabs" / "four_route_record.html").read_text(encoding="utf-8")
    assert [point["id"] for point in numbered] == [
        "green_circle",
        "shelf_row",
        "shelf_face",
        "shelf_box_west",
        "shelf_box_middle",
        "shelf_box_east",
        "green_dustbin",
    ]
    assert [point["number"] for point in numbered] == [1, 2, 3, 4, 5, 6, 7]
    assert "toward_green_circle" not in {point["id"] for point in numbered}
    assert "bay_entry" not in {point["id"] for point in numbered}
    assert checkpoint_for_number(1) == {
        "number": 1,
        "id": "green_circle",
        "x": 0.0,
        "y": -3.0,
        "label": "green circle",
    }
    assert checkpoint_for_number(7)["id"] == "green_dustbin"
    with pytest.raises(KeyError):
        checkpoint_for_number(8)
    assert "4 of 4 routes reached" in record
    assert "north_box_lane" in record


def test_a_click_drives_to_that_checkpoint_from_the_current_world_pose(tmp_path):
    captured = {}
    pose = (1.2, 3.2, 0.3, 0.1)

    def navigator(_config, goals, **kwargs):
        captured["goals"] = goals
        captured["kwargs"] = kwargs
        return {
            "success": True,
            "termination_reason": "goal_reached",
            "goal_distance": 0.2,
            "checkpoints_reached": ["green_dustbin"],
        }

    selector = CheckpointSelector(
        _config("gazebo"),
        ROOT / "artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311",
        authorize_live=True,
        navigator=navigator,
        pose_reader=lambda: pose,
        output_dir=tmp_path,
    )
    status = selector.start("green_dustbin")
    assert status["state"] == "driving"
    selector._worker.join(timeout=2)
    finished = selector.status()
    assert finished["state"] == "finished"
    assert captured["goals"] == [((1.2, 6.6), "green_dustbin")]
    assert captured["kwargs"]["start"] == (1.2, 3.2, 0.1)
    assert captured["kwargs"]["world_route"] is True
    assert captured["kwargs"]["resume"] is False
    saved = list(tmp_path.glob("green_dustbin_*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text(encoding="utf-8"))["selected_checkpoint"] == "green_dustbin"


def test_gazebo_click_without_authorization_does_not_navigate():
    called = []
    selector = CheckpointSelector(
        _config("gazebo"),
        Path("checkpoint"),
        authorize_live=False,
        navigator=lambda *_args, **_kwargs: called.append(True),
        pose_reader=lambda: (_ for _ in ()).throw(AssertionError("pose was read")),
    )
    with pytest.raises(SystemExit, match="Gazebo navigation"):
        selector.start("shelf_row")
    assert called == []
    assert selector.status()["state"] == "idle"


def test_a_second_click_is_refused_while_the_first_is_driving():
    started = threading.Event()
    release = threading.Event()

    def navigator(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=2)
        return {"success": True, "termination_reason": "goal_reached"}

    selector = CheckpointSelector(
        _config("offline"),
        Path("checkpoint"),
        navigator=navigator,
    )
    selector.start("bay_entry")
    assert started.wait(timeout=2)
    with pytest.raises(RuntimeError, match="already running"):
        selector.start("bay_exit")
    release.set()
    selector._worker.join(timeout=2)
    assert selector.status()["state"] == "finished"


def test_pressing_a_numbered_button_drives_to_that_checkpoint():
    selector = CheckpointSelector(
        _config("offline"),
        Path("checkpoint"),
        navigator=lambda *_args, **_kwargs: {"success": True, "termination_reason": "goal_reached"},
    )
    popup = CheckpointPopup(selector)
    status = popup.press(3)
    assert status["state"] == "driving"
    assert status["checkpoint_id"] == "shelf_face"
    selector._worker.join(timeout=2)
    assert selector.status()["state"] == "finished"
