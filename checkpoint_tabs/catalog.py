"""Checkpoint tabs for the external warehouse.

The coordinates stay in evaluation.warehouse_checkpoints. The popup numbers
only the green floor circles that already exist in the warehouse world.
"""

from __future__ import annotations

import re
from pathlib import Path

from evaluation.warehouse_checkpoints import WAREHOUSE_ROUTE

_WORLD_SDF = (
    Path(__file__).resolve().parents[1]
    / "jackal_rl_ws/worlds/external_warehouse/jackal_industrial_warehouse.sdf"
)
_MARKER_POSE = re.compile(
    r'<model name="(goal_marker|checkpoint_[A-Za-z0-9_]+)">\s*'
    r"<static>true</static>\s*"
    r"<pose>\s*([-+0-9.]+)\s+([-+0-9.]+)\s+",
)

_TAB_LABELS = {
    "green_circle": "Green circle",
    "shelf_row": "Shelf row",
    "shelf_face": "Shelf face",
    "shelf_box_bay": "Shelf and box bay",
    "box_row": "Box row",
    "green_dustbin": "Green dustbin",
}


def checkpoint_tabs() -> list[dict]:
    """Group the aisle checkpoints into one tab per landmark."""

    groups: list[dict] = []
    index: dict[str, int] = {}
    for point in WAREHOUSE_ROUTE:
        key = str(point["landmark"])
        if key not in index:
            index[key] = len(groups)
            groups.append(
                {
                    "id": key,
                    "label": _TAB_LABELS.get(key, key.replace("_", " ")),
                    "checkpoints": [],
                }
            )
        groups[index[key]]["checkpoints"].append(
            {
                "id": str(point["id"]),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "label": str(point["id"]).replace("_", " "),
            }
        )
    return groups


def _marker_checkpoint_id(model_name: str) -> str:
    if model_name == "goal_marker":
        return "green_circle"
    return model_name.removeprefix("checkpoint_")


def green_circle_checkpoints() -> list[dict]:
    """Return the green floor circles already placed in the warehouse, in world order."""

    by_id = {str(point["id"]): point for point in WAREHOUSE_ROUTE}
    text = _WORLD_SDF.read_text(encoding="utf-8")
    circles = []
    for model_name, x_text, y_text in _MARKER_POSE.findall(text):
        checkpoint_id = _marker_checkpoint_id(model_name)
        point = by_id.get(checkpoint_id)
        if point is None:
            raise KeyError(f"{model_name} has no warehouse checkpoint")
        x = float(x_text)
        y = float(y_text)
        if abs(x - float(point["x"])) > 1e-6 or abs(y - float(point["y"])) > 1e-6:
            raise ValueError(f"{model_name} does not match checkpoint {checkpoint_id}")
        circles.append(
            {
                "id": checkpoint_id,
                "x": float(point["x"]),
                "y": float(point["y"]),
                "label": checkpoint_id.replace("_", " "),
                "model": model_name,
            }
        )
    if not circles:
        raise ValueError("No green circle markers were found in the warehouse world")
    return circles


def numbered_checkpoints() -> list[dict]:
    """Number the existing green circles from 1."""

    numbered = []
    for number, point in enumerate(green_circle_checkpoints(), start=1):
        numbered.append(
            {
                "number": number,
                "id": point["id"],
                "x": point["x"],
                "y": point["y"],
                "label": point["label"],
            }
        )
    return numbered


def checkpoint_for_number(number: int) -> dict:
    """Return the checkpoint behind button ``number``. Unknown numbers raise KeyError."""

    for point in numbered_checkpoints():
        if point["number"] == int(number):
            return point
    raise KeyError(number)


def checkpoint_by_id(checkpoint_id: str) -> dict:
    """Return one warehouse checkpoint. Unknown ids raise KeyError."""

    for point in WAREHOUSE_ROUTE:
        if str(point["id"]) == str(checkpoint_id):
            return {
                "id": str(point["id"]),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "landmark": str(point["landmark"]),
            }
    raise KeyError(checkpoint_id)
