"""Send the Jackal to one clicked checkpoint.

A click reads the current Gazebo pose and drives to that warehouse point.
It does not teleport between stops, and it does not run the full aisle tour.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from checkpoint_tabs.catalog import checkpoint_by_id


def read_industrial_warehouse_pose():
    """Read the Jackal world pose on the external warehouse partition."""

    from jackal_rl_navigation.world_pose import GazeboWorldPose

    return GazeboWorldPose(
        robot="j100_0000",
        timeout=8.0,
        world="industrial-warehouse",
        partition="jackal_rl_external",
    ).read_latest()


def _navigate_selected(config, goals, **kwargs):
    from evaluation.warehouse_navigate import navigate_route

    return navigate_route(config, goals, **kwargs)


class CheckpointSelector:
    """One live navigation at a time, started by a checkpoint id."""

    def __init__(
        self,
        config: dict,
        checkpoint: Path,
        *,
        authorize_live: bool = False,
        seed: int = 1,
        navigator=None,
        pose_reader=None,
        output_dir: Path | None = None,
    ):
        self.config = config
        self.checkpoint = Path(checkpoint)
        self.authorize_live = bool(authorize_live)
        self.seed = int(seed)
        self.navigator = _navigate_selected if navigator is None else navigator
        self.pose_reader = read_industrial_warehouse_pose if pose_reader is None else pose_reader
        self.output_dir = None if output_dir is None else Path(output_dir)
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._status = {
            "state": "idle",
            "checkpoint_id": None,
            "result": None,
            "error": None,
        }

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def start(self, checkpoint_id: str) -> dict:
        """Begin driving to one checkpoint. A second click is refused."""

        point = checkpoint_by_id(checkpoint_id)
        execution = self.config.get("execution", "offline")
        if execution == "stationary_observation":
            raise SystemExit(
                "Refusing navigation on the stationary observation path. "
                "step() and checkpoint loading stay blocked."
            )
        if execution == "gazebo" and not self.authorize_live:
            raise SystemExit("Refusing Gazebo navigation because it would command the robot.")
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("A checkpoint navigation is already running")
            self._status = {
                "state": "driving",
                "checkpoint_id": point["id"],
                "result": None,
                "error": None,
            }
            self._worker = threading.Thread(
                target=self._drive,
                args=(point,),
                name=f"checkpoint-{point['id']}",
                daemon=True,
            )
            self._worker.start()
            return dict(self._status)

    def _drive(self, point: dict) -> None:
        try:
            start = None
            if self.config.get("execution") == "gazebo":
                pose = self.pose_reader()
                start = (float(pose[0]), float(pose[1]), float(pose[3]))
            result = self.navigator(
                self.config,
                [((point["x"], point["y"]), point["id"])],
                checkpoint=self.checkpoint,
                start=start,
                authorize_live=self.authorize_live,
                seed=self.seed,
                resume=False,
                world_route=True,
            )
            payload = dict(result)
            payload["selected_checkpoint"] = point["id"]
            saved = self._write(payload)
            if saved is not None:
                payload["saved_result"] = str(saved)
            status = {
                "state": "finished",
                "checkpoint_id": point["id"],
                "result": payload,
                "error": None,
            }
        except Exception as exc:
            status = {
                "state": "error",
                "checkpoint_id": point["id"],
                "result": None,
                "error": str(exc),
            }
        with self._lock:
            self._status = status

    def _write(self, payload: dict) -> Path | None:
        if self.output_dir is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"{payload['selected_checkpoint']}_{stamp}.json"
        suffix = 1
        while path.exists():
            path = self.output_dir / f"{payload['selected_checkpoint']}_{stamp}_{suffix}.json"
            suffix += 1
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        return path
