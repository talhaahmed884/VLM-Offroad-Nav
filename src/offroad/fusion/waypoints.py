"""The VLM's waypoints as soft guidance on the cost map (pure functions, no file I/O).

Each waypoint cell is traced onto the ground (`GridLookup.ground_point`), and the route from
the vehicle through those points becomes a corridor whose cost is slightly reduced. The
planner is then nudged toward the VLM's route but still free to leave it: the discount
never unblocks anything, and phase 4 showed the routes are much weaker than the hints.
"""

import numpy as np

from offroad.fusion.grid_lookup import GridLookup
from offroad.mapping.elevation import GridSpec


def waypoint_ground_points(lookup: GridLookup, waypoints: list[str]) -> list[tuple[float, float]]:
    """FLU (x, y) of each waypoint on the ground, in order; waypoints that show no ground are skipped."""
    points = [lookup.ground_point(cell) for cell in waypoints]
    return [p for p in points if p is not None]


def distance_to_polyline(spec: GridSpec, points: list[tuple[float, float]]) -> np.ndarray:
    """(nx, ny) distance in meters from each map cell center to the polyline through `points`.

    A single point gives the distance to that point; no points gives inf everywhere.
    """
    nx, ny = spec.shape
    xs = spec.x_min + (np.arange(nx) + 0.5) * spec.resolution
    ys = spec.y_min + (np.arange(ny) + 0.5) * spec.resolution
    x, y = np.meshgrid(xs, ys, indexing="ij")
    if not points:
        return np.full((nx, ny), np.inf)
    if len(points) == 1:
        return np.hypot(x - points[0][0], y - points[0][1])

    best = np.full((nx, ny), np.inf)
    for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        # Position of the closest point along the segment, 0 = at a, 1 = at b.
        t = np.zeros_like(x) if length2 == 0 else np.clip(((x - ax) * dx + (y - ay) * dy) / length2, 0.0, 1.0)
        best = np.minimum(best, np.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best


def route_corridor(
    spec: GridSpec, points: list[tuple[float, float]], half_width: float, max_distance: float
) -> np.ndarray:
    """(nx, ny) bool: map cells within `half_width` m of the route from the vehicle (0, 0) through `points`.

    Cells farther than `max_distance` m from the vehicle are left out, like the cost hints.
    No points gives an empty corridor.
    """
    if not points:
        return np.zeros(spec.shape, dtype=bool)
    nx, ny = spec.shape
    xs = spec.x_min + (np.arange(nx) + 0.5) * spec.resolution
    ys = spec.y_min + (np.arange(ny) + 0.5) * spec.resolution
    near = np.hypot(xs[:, None], ys[None, :]) <= max_distance
    return (distance_to_polyline(spec, [(0.0, 0.0), *points]) <= half_width) & near
