"""Draw a VLM answer on the camera image, next to the ground-truth label image (pure, no file I/O).

Left: the camera image with each hinted cell tinted by its traversability and labelled with
its terrain, the waypoint route, and the goal. Right: the label image in the ontology colors
with the same grid, route, and goal, so the hints can be checked against the truth by eye.
"""

import textwrap

import cv2
import numpy as np

from offroad.viz.grid_overlay import draw_grid
from offroad.vlm.grid import ImageGrid
from offroad.vlm.runner import downscale

# Tint per traversability level, best to worst (RGB).
LEVEL_RGB = {
    "preferred": (0, 200, 0),
    "ok": (150, 220, 0),
    "caution": (255, 210, 0),
    "avoid": (255, 120, 0),
    "impassable": (230, 0, 0),
}
TERRAIN_SHORT = {
    "dirt_or_gravel": "dirt", "short_grass": "grass", "tall_grass_or_bush": "bush", "mud": "mud",
    "water": "water", "rocks": "rocks", "tree_or_log": "tree", "other_obstacle": "obst",
    "person_or_vehicle": "person",
}
ROUTE_RGB = (255, 255, 255)
GOAL_RGB = (255, 0, 255)
CONFLICT_RGB = (255, 0, 0)


def colorize_labels(label_image: np.ndarray, id_to_color: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """(H, W) class IDs -> (H, W, 3) uint8 RGB in the ontology colors (unknown IDs black)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for class_id, rgb in id_to_color.items():
        if 0 <= class_id < 256:
            lut[class_id] = rgb
    return lut[label_image]


def draw_hints(image: np.ndarray, grid: ImageGrid, hints: list[dict], alpha: float = 0.4) -> np.ndarray:
    """Copy of the image with each hinted cell tinted by traversability and labelled with its terrain."""
    tinted = image.copy()
    for hint in hints:
        u0, v0, u1, v1 = (round(x) for x in grid.bounds(*grid.parse(hint["cell"])))
        cv2.rectangle(tinted, (u0, v0), (u1 - 1, v1 - 1), LEVEL_RGB[hint["traversability"]], thickness=-1)
    out = cv2.addWeighted(tinted, alpha, image, 1 - alpha, 0)

    scale = min(grid.cell_width, grid.cell_height) / 150
    for hint in hints:
        u0, v0, u1, v1 = (round(x) for x in grid.bounds(*grid.parse(hint["cell"])))
        cv2.rectangle(out, (u0, v0), (u1 - 1, v1 - 1), LEVEL_RGB[hint["traversability"]], max(1, round(3 * scale)))
        _put_text(out, TERRAIN_SHORT[hint["terrain"]], (u0 + round(8 * scale), v1 - round(12 * scale)), 0.9 * scale)
    return out


def draw_route(
    image: np.ndarray, grid: ImageGrid, waypoints: list[str], goal_cell: str | None, conflicts: list[str] = ()
) -> np.ndarray:
    """Copy of the image with the route through the waypoint cell centers and a star on the goal cell.

    Waypoints in `conflicts` (cells the model itself marked avoid/impassable) get a red dot.
    """
    out = image.copy()
    scale = min(grid.cell_width, grid.cell_height) / 150
    thick = max(2, round(6 * scale))
    points = [tuple(round(x) for x in grid.center(*grid.parse(cell))) for cell in waypoints]
    if len(points) > 1:
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 0, 0), thick + 4, cv2.LINE_AA)
        cv2.polylines(out, [pts], False, ROUTE_RGB, thick, cv2.LINE_AA)
    for cell, point in zip(waypoints, points):
        color = CONFLICT_RGB if cell in conflicts else ROUTE_RGB
        cv2.circle(out, point, thick * 2 + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, point, thick * 2, color, -1, cv2.LINE_AA)
    if goal_cell is not None:
        center = tuple(round(x) for x in grid.center(*grid.parse(goal_cell)))
        size = round(60 * scale)
        cv2.drawMarker(out, center, (0, 0, 0), cv2.MARKER_STAR, size + 6, thick + 4, cv2.LINE_AA)
        cv2.drawMarker(out, center, GOAL_RGB, cv2.MARKER_STAR, size, thick, cv2.LINE_AA)
    return out


def draw_answer_figure(
    image: np.ndarray,
    label_image: np.ndarray | None,
    grid: ImageGrid,
    answer: dict,
    goal_cell: str | None,
    id_to_color: dict[int, tuple[int, int, int]],
    title: str,
    max_side: int = 1280,
) -> np.ndarray:
    """Side-by-side picture (camera + hints | labels + route) with a title and the model's reasoning.

    image: (H, W, 3) uint8 RGB without the grid; label_image: (H, W) class IDs or None.
    answer: a checked answer as saved by `runner.answer_to_dict`.
    Returns an RGB image; each panel is shrunk to `max_side` px on its long side.
    """
    conflicts = answer.get("route_conflicts", [])
    left = draw_route(draw_hints(draw_grid(image, grid), grid, answer["hints"]),
                      grid, answer["waypoints"], goal_cell, conflicts)
    panels = [downscale(left, max_side)]
    if label_image is not None:
        right = draw_route(draw_grid(colorize_labels(label_image, id_to_color), grid),
                           grid, answer["waypoints"], goal_cell, conflicts)
        panels.append(downscale(right, max_side))
    body = np.concatenate(panels, axis=1)

    width = body.shape[1]
    reasoning = answer["reasoning"] or "(no reasoning)"
    status = "VALID" if answer["valid"] else "NOT USABLE"
    legend = ("Tint: green = preferred, lime = ok, yellow = caution, orange = avoid, red = impassable.  "
              "White line = route (red dot = in a cell it marked avoid), magenta star = goal.  Right: label image.")
    lines = [f"{title}   [{status}, {len(answer['issues'])} issues]", *textwrap.wrap(reasoning, width=width // 14), legend]
    line_h = 34
    header = np.full((line_h * len(lines) + 14, width, 3), 255, dtype=np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(header, line, (10, 28 + k * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),
                    2 if k == 0 else 1, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


def _put_text(image: np.ndarray, text: str, origin: tuple[int, int], font_scale: float):
    """White text with a black outline, readable on any background."""
    thickness = max(1, round(2 * font_scale))
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
