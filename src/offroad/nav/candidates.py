"""Candidate paths: the drivable options the VLM chooses between (pure, no file I/O).

Following DriveVLN, a planner proposes a few feasible, direction-agnostic paths; the VLM picks
one. Ours fans out from the vehicle: for each bearing in FAN_BEARINGS_DEG it takes the cheapest
path toward a point CANDIDATE_LENGTH_M ahead, from one shortest-path search on a cost grid.
The paths are smoothed, cut to CANDIDATE_LENGTH_M, and near-duplicates are dropped; up to
MAX_CANDIDATES are kept. The VLM only sees the camera image, so a path must stay in the
camera's view (about +-19 deg) from the first tick on; the fan covers a bit more than the view
because a path aimed just outside it often bends back in.

Two cost grids are compared (user decision 2026-11-22):
- **strict**: the geometric cost map from approach 1. Anything taller than 0.3 m blocks, so tall
  grass and bush at a trail's edge act like walls.
- **relaxed** (chosen for phase 3 on, CHOSEN_VARIANT): only *hard* obstacles block (taller than HARD_OBSTACLE_HEIGHT_M, e.g. trees and
  walls). Lower vegetation-height obstacles are passable but expensive (VEGETATION_COST), so
  candidates can go through grass and bush, and it's the VLM's job to say whether that's safe.
"""

from dataclasses import dataclass

import numpy as np

from offroad.mapping.costmap import LETHAL, CostMap, CostParams, drivable_cost, inflate
from offroad.mapping.elevation import GridSpec
from offroad.planning.astar import clear_footprint, goal_from_costs, path_from_tree, shortest_path_tree

CANDIDATE_LENGTH_M = 15.0
FAN_BEARINGS_DEG = tuple(float(b) for b in range(-30, 31, 5))
MAX_CANDIDATES = 4
MIN_SEPARATION_M = 2.0  # a path that never gets this far from a kept one is a duplicate
MIN_CANDIDATE_LENGTH_M = 3.0  # shorter paths are dead ends right in front of the vehicle
FULL_LENGTH_TOLERANCE_M = 1.0  # a path this close to CANDIDATE_LENGTH_M counts as full length
GOAL_COST_WEIGHT = 0.1  # as for the baseline's goal choice
FOOTPRINT_RADIUS_M = 2.0
SMOOTH_WINDOW = 5  # cells (1.25 m) in the moving average that removes grid zigzags
RESAMPLE_SPACING_M = 0.25
IN_VIEW_FROM_M = 5.0  # the camera can't see the first ~4.4 m, so check from the first tick on
IN_VIEW_MARGIN_DEG = 2.0  # stay this far inside the image edges
MIN_IN_VIEW_SHARE = 0.8  # share of the checked points that must be in view

HARD_OBSTACLE_HEIGHT_M = 1.5
VEGETATION_COST = 10.0  # twice the roughest drivable cost: passable, but only if worth it

# Display colors for the candidates, chosen to stand out from grass, trails and sky (RGB).
CANDIDATE_COLORS = (("red", (230, 40, 40)), ("blue", (40, 110, 255)), ("yellow", (255, 215, 0)),
                    ("magenta", (220, 60, 220)))
VARIANTS = ("strict", "relaxed")
CHOSEN_VARIANT = "relaxed"  # used from phase 3 on (user decision 2026-11-22, after the phase 2 comparison)


@dataclass
class Candidate:
    """One candidate path, in the FLU frame of the decision frame.

    xy: (K, 2) points every RESAMPLE_SPACING_M meters from the vehicle along the path.
    target_bearing_deg: the fan direction it was planned toward (+ = left).
    cost_per_m: mean cost of the cells it crosses (from the cost grid it was planned on).
    """

    xy: np.ndarray
    target_bearing_deg: float
    cost_per_m: float

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum()) if len(self.xy) > 1 else 0.0

    @property
    def end_bearing_deg(self) -> float:
        x, y = self.xy[-1]
        return float(np.degrees(np.arctan2(y, x)))

    @property
    def full_length(self) -> bool:
        return self.length_m >= CANDIDATE_LENGTH_M - FULL_LENGTH_TOLERANCE_M


def strict_cost(costmap: CostMap) -> np.ndarray:
    """The approach-1 geometric cost grid: everything taller than max_step (0.3 m) blocks."""
    return costmap.cost


def relaxed_cost(
    costmap: CostMap,
    params: CostParams = CostParams(),
    hard_height: float = HARD_OBSTACLE_HEIGHT_M,
    vegetation_cost: float = VEGETATION_COST,
) -> np.ndarray:
    """Cost grid where only hard obstacles (taller than hard_height) block; lower ones cost vegetation_cost.

    Hard obstacles are grown by the usual inflation radius; vegetation is not. Heights are the
    cost map's obstacle heights (tallest point below vehicle height above the local ground).
    """
    height = costmap.obstacle_height
    cost = drivable_cost(height, params)
    hard = costmap.obstacle & (np.nan_to_num(height) > hard_height)
    cost[costmap.obstacle & ~hard] = vegetation_cost
    cost[inflate(hard, round(params.inflation_radius / costmap.spec.resolution))] = LETHAL
    return cost


def smooth_path(xy: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Centered moving average over `window` points; the window shrinks near the ends, so they stay fixed.

    A straight, evenly spaced path is left unchanged.
    """
    xy = np.asarray(xy, dtype=float)
    n, half = len(xy), window // 2
    smoothed = xy.copy()
    for k in range(n):
        h = min(half, k, n - 1 - k)
        smoothed[k] = xy[k - h:k + h + 1].mean(axis=0)
    return smoothed


def resample_and_cut(xy: np.ndarray, spacing: float, max_length: float) -> np.ndarray:
    """(K, 2) points every `spacing` meters along the polyline, up to max_length (or its end)."""
    xy = np.asarray(xy, dtype=float)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    end = min(along[-1], max_length)
    distances = np.arange(0.0, end + 1e-9, spacing)
    if end - distances[-1] > 1e-6:
        distances = np.append(distances, end)
    return np.column_stack([np.interp(distances, along, xy[:, 0]), np.interp(distances, along, xy[:, 1])])


def mean_distance_to(path_xy: np.ndarray, other_xy: np.ndarray) -> float:
    """Mean over path_xy's points of the distance to the nearest point of other_xy, meters."""
    d = np.hypot(path_xy[:, None, 0] - other_xy[None, :, 0], path_xy[:, None, 1] - other_xy[None, :, 1])
    return float(d.min(axis=1).mean())


def camera_half_fov_deg(K: np.ndarray, image_width: int) -> tuple[float, float]:
    """(left, right) horizontal half-angles of the camera view, degrees (camera assumed at the vehicle origin)."""
    fx, cx = K[0, 0], K[0, 2]
    return float(np.degrees(np.arctan(cx / fx))), float(np.degrees(np.arctan((image_width - cx) / fx)))


def in_view_share(xy: np.ndarray, half_fov_deg: tuple[float, float], from_m: float = IN_VIEW_FROM_M,
                  margin_deg: float = IN_VIEW_MARGIN_DEG) -> float:
    """Share of the path's points (from `from_m` along it on) whose bearing is inside the camera view.

    half_fov_deg: (left, right) half-angles, e.g. from `camera_half_fov_deg`. 1.0 if no point is that far.
    """
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    far = xy[along >= from_m]
    if len(far) == 0:
        return 1.0
    bearing = np.degrees(np.arctan2(far[:, 1], far[:, 0]))
    inside = (bearing <= half_fov_deg[0] - margin_deg) & (bearing >= -(half_fov_deg[1] - margin_deg))
    return float(inside.mean())


def max_separation(a: np.ndarray, b: np.ndarray) -> float:
    """Largest distance between two paths at equal distances along them, meters.

    a, b: (K, 2) paths resampled at the same spacing from the same start (as candidates are);
    compared over the shorter one's length. Two paths that share the trail for a while and then
    split count as different once their ends are far enough apart.
    """
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n], axis=1).max()) if n else 0.0


def mean_cost_along(cost: np.ndarray, spec: GridSpec, xy: np.ndarray) -> float:
    """Mean cost of the grid cells under the points (inf if any is blocked)."""
    nx, ny = spec.shape
    values = []
    for x, y in xy:
        i, j = spec.cell_index(x, y)
        if 0 <= i < nx and 0 <= j < ny:
            values.append(cost[i, j])
    return float(np.mean(values)) if values else float("inf")


def generate_candidates(
    cost: np.ndarray,
    spec: GridSpec,
    bearings_deg: tuple[float, ...] = FAN_BEARINGS_DEG,
    length: float = CANDIDATE_LENGTH_M,
    max_candidates: int = MAX_CANDIDATES,
    min_separation: float = MIN_SEPARATION_M,
    half_fov_deg: tuple[float, float] | None = None,
) -> list[Candidate]:
    """Up to max_candidates distinct paths fanning out from the vehicle, sorted left to right.

    cost: (nx, ny) cost grid (strict_cost or relaxed_cost). The vehicle's own footprint is
    cleared first. For each bearing, the goal is the reachable cell that best trades getting
    close to the point `length` m away against driving cost (as the baseline's goal choice);
    the path to it comes from one shortest-path tree. Full-length paths are preferred, then
    cheaper ones; a path that never gets min_separation away from an already kept path (see
    `max_separation`) is a duplicate. Paths shorter than MIN_CANDIDATE_LENGTH_M are dropped, and so are paths that
    leave the camera view (less than MIN_IN_VIEW_SHARE in view) if half_fov_deg is given.
    """
    cleared = clear_footprint(cost, spec, FOOTPRINT_RADIUS_M)
    start = spec.cell_index(0.0, 0.0)
    come, parent = shortest_path_tree(cleared, start)
    proposals = []
    for bearing in bearings_deg:
        tx, ty = length * np.cos(np.radians(bearing)), length * np.sin(np.radians(bearing))
        goal = goal_from_costs(come, spec, tx, ty, GOAL_COST_WEIGHT)
        if goal is None:
            continue
        cells = path_from_tree(parent, goal)
        xy = np.array([spec.cell_center(i, j) for i, j in cells])
        xy[0] = (0.0, 0.0)  # start exactly at the vehicle
        xy = resample_and_cut(smooth_path(xy), RESAMPLE_SPACING_M, length)
        candidate = Candidate(xy=xy, target_bearing_deg=bearing, cost_per_m=mean_cost_along(cleared, spec, xy))
        if candidate.length_m < MIN_CANDIDATE_LENGTH_M:
            continue
        if half_fov_deg is not None and in_view_share(xy, half_fov_deg) < MIN_IN_VIEW_SHARE:
            continue
        proposals.append(candidate)

    proposals.sort(key=lambda c: (not c.full_length, c.cost_per_m))
    kept: list[Candidate] = []
    for candidate in proposals:
        if all(max_separation(candidate.xy, k.xy) >= min_separation for k in kept):
            kept.append(candidate)
        if len(kept) == max_candidates:
            break
    return sorted(kept, key=lambda c: -c.end_bearing_deg)  # left to right
