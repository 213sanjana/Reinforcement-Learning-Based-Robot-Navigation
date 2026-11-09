"""Aisle checkpoints for the external industrial warehouse.

Positions are in the warehouse world frame. The bay checkpoints sit in the
open gap between the north face of the east shelf row and the south face of
the box stacks, which is the green-outlined floor space in the warehouse.
Footprints below are the collision-mesh bounds after each COLLADA node
transform. The perimeter wall is not included because its bounding box covers
the whole floor.
"""

from __future__ import annotations

import math

# xmin, xmax, ymin, ymax in meters.
OBSTACLE_BOXES = (
    ("shelf_e_1", 2.77, 6.69, 0.14, 1.02),
    ("shelf_d_1", 2.77, 6.69, -1.68, -0.80),
    ("shelf_d_2", 2.77, 6.69, -3.48, -2.60),
    ("shelf_e_2", 2.77, 6.69, -5.27, -4.39),
    ("shelf_d_3", 2.77, 6.69, -7.19, -6.31),
    ("shelf_e_3", 2.77, 6.69, -9.11, -8.23),
    ("shelf_f", -6.84, -4.74, -9.98, 8.07),
    ("boxes_west", 2.29, 4.36, 2.93, 4.72),
    ("boxes_east", 4.50, 6.58, 2.92, 4.72),
    ("boxes_north_east", 4.49, 6.28, 5.10, 7.17),
    ("boxes_north_west", 2.34, 4.13, 5.10, 7.17),
    ("boxes_far_west", -2.47, -0.68, 1.26, 3.34),
    ("boxes_aisle_west", -2.50, -0.48, 4.13, 6.32),
    ("boxes_far_north_east", 4.62, 6.78, 7.62, 9.62),
    ("boxes_far_north_west", 2.32, 4.49, 7.62, 9.62),
    ("boxes_top_west", -2.26, -0.18, 8.51, 10.30),
    ("clutter_south", -2.14, -1.13, -8.57, -7.05),
    ("green_dustbin", -2.33, -0.85, 7.26, 8.17),
    ("bucket_north", -0.18, 1.05, 9.16, 10.11),
    ("bucket_south", -2.45, -1.22, -6.85, -5.90),
    ("bucket_mid", -0.18, 1.05, 8.11, 9.07),
)

# North face of the first east shelf and south face of the facing box row.
SHELF_BOX_BAY = {
    "shelf_face_y": 1.02,
    "box_face_y": 2.92,
    "x_min": 2.77,
    "x_max": 6.58,
}

# Ordered stops. The shelf-box points are inside the bay between the racks and the boxes.
WAREHOUSE_ROUTE = (
    {"id": "toward_green_circle", "x": 0.0, "y": -1.4, "landmark": "green_circle"},
    {"id": "green_circle", "x": 0.0, "y": -3.0, "landmark": "green_circle"},
    {"id": "toward_shelf_row", "x": 0.8, "y": -2.2, "landmark": "shelf_row"},
    {"id": "shelf_row", "x": 1.5, "y": -1.2, "landmark": "shelf_row"},
    {"id": "shelf_face", "x": 1.5, "y": 0.6, "landmark": "shelf_face"},
    {"id": "bay_entry", "x": 1.5, "y": 1.97, "landmark": "shelf_box_bay"},
    {"id": "shelf_box_west", "x": 3.3, "y": 1.97, "landmark": "shelf_box_bay"},
    {"id": "shelf_box_middle", "x": 4.5, "y": 1.97, "landmark": "shelf_box_bay"},
    {"id": "shelf_box_east", "x": 5.7, "y": 1.97, "landmark": "shelf_box_bay"},
    {"id": "bay_exit", "x": 1.5, "y": 1.97, "landmark": "shelf_box_bay"},
    {"id": "box_row_pass", "x": 1.2, "y": 3.2, "landmark": "box_row"},
    {"id": "toward_dustbin", "x": 1.2, "y": 4.6, "landmark": "green_dustbin"},
    {"id": "dustbin_approach", "x": 1.2, "y": 5.6, "landmark": "green_dustbin"},
    {"id": "green_dustbin", "x": 1.2, "y": 6.6, "landmark": "green_dustbin"},
)


def route_from_pose(route, x: float, y: float, *, tolerance_m: float = 0.30):
    """Continue a route from the checkpoint nearest the robot.

    A later checkpoint wins when two are equally close, so a repeated point
    does not send the robot back to the start of the bay. Points already
    inside the arrival tolerance are skipped.
    """

    nearest = 0
    nearest_distance = math.inf
    for index, point in enumerate(route):
        distance = math.hypot(float(point["x"]) - float(x), float(point["y"]) - float(y))
        if distance <= nearest_distance + 1e-9:
            nearest = index
            nearest_distance = distance
    while nearest < len(route) - 1 and nearest_distance <= float(tolerance_m):
        nearest += 1
        point = route[nearest]
        nearest_distance = math.hypot(float(point["x"]) - float(x), float(point["y"]) - float(y))
    return tuple(route[nearest:])


def clearance_m(x: float, y: float) -> float:
    """Distance from a floor point to the nearest recorded obstacle box."""

    nearest = math.inf
    for _name, xmin, xmax, ymin, ymax in OBSTACLE_BOXES:
        dx = max(xmin - x, 0.0, x - xmax)
        dy = max(ymin - y, 0.0, y - ymax)
        nearest = min(nearest, math.hypot(dx, dy))
    return float(nearest)
