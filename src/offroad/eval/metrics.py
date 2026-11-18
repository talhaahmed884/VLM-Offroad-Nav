"""Evaluation metrics for one planned path (pure, no file I/O).

The metrics from the proposal, with the definitions the user approved on 2026-11-18:
- share of the path through water (water + puddle) and through mud, from the two-source
  ground truth (`path_truth.py`). Mud is reported separately: RELLIS-3D often labels plain
  trail ruts as mud, so it is not a failure by itself.
- share through compliant vegetation (grass, bush: drivable) and non-compliant vegetation
  (tree, log).
- distance to hard obstacles: for each path cell, the distance to the nearest LiDAR point
  labelled as a hard obstacle (HARD_OBSTACLE_CLASSES); mean and minimum along the path. Points
  above vehicle height (canopy the vehicle passes under) and points on the vehicle itself are ignored.
- roughness: the mean ground slope between consecutive path cells (cm of height change per m)
  and the mean bump height (the cost map's obstacle height) along the path.
- reached / shortfall / detour / length, plus "goal blocked": a hard obstacle (e.g. a person)
  within GOAL_BLOCKED_RADIUS_M of the target, where stopping at the obstacle is the right
  behavior. Success = reached, or goal blocked and stopped within BLOCKED_STOP_MAX_M of the target
  (tightened on 2026-11-18: without the distance limit, a path stopping 4.4 m short of a
  tree-blocked target counted as a success).
"""

from dataclasses import dataclass

import numpy as np

from offroad.eval.path_truth import PathTruth, UNSEEN
from offroad.fusion.grid_lookup import fill_ground
from offroad.mapping.costmap import CostParams
from offroad.mapping.elevation import GridSpec, lidar_to_flu
from offroad.planning.baseline import REACHED_TOLERANCE_M, BaselineResult

WATER_CLASSES = ("water", "puddle")
MUD_CLASSES = ("mud",)
COMPLIANT_VEGETATION = ("grass", "bush")
NON_COMPLIANT_VEGETATION = ("tree", "log")
HARD_OBSTACLE_CLASSES = ("tree", "pole", "fence", "barrier", "building", "vehicle", "person", "log", "object", "rubble")
GOAL_BLOCKED_RADIUS_M = 1.5
# A path to a blocked target still counts as a success if it stops this close to the target
# (the blocked radius plus the usual 1 m "reached" tolerance).
BLOCKED_STOP_MAX_M = GOAL_BLOCKED_RADIUS_M + REACHED_TOLERANCE_M
VEHICLE_IGNORE_RADIUS_M = 2.0  # LiDAR points this close to the sensor are the vehicle itself


@dataclass
class PathMetrics:
    """Metrics of one path. Shares are of all path cells (unseen cells included); None = not defined.

    success: reached the target, or the target is blocked and the path stops close to it
        (within BLOCKED_STOP_MAX_M), since stopping at the obstacle is correct but stopping far
        before it is not.
    """

    found: bool
    reached: bool
    goal_blocked: bool
    shortfall_m: float | None
    length_m: float | None
    detour_ratio: float | None
    share_water: float | None
    share_mud: float | None
    share_compliant: float | None
    share_noncompliant: float | None
    share_unseen: float | None
    hard_dist_mean_m: float | None
    hard_dist_min_m: float | None
    slope_cm_per_m: float | None
    bump_height_m: float | None

    @property
    def success(self) -> bool:
        stopped_at_obstacle = self.shortfall_m is not None and self.shortfall_m <= BLOCKED_STOP_MAX_M
        return self.reached or (self.found and self.goal_blocked and stopped_at_obstacle)


def ids_of(id_to_name: dict[int, str], names: tuple[str, ...]) -> list[int]:
    """Ontology IDs of the classes with these names."""
    return [i for i, name in id_to_name.items() if name in names]


def hard_obstacle_xy(
    points_lidar: np.ndarray,
    point_labels: np.ndarray,
    hard_ids: list[int],
    ground_z: np.ndarray,
    spec: GridSpec,
    vehicle_height: float = CostParams.vehicle_height,
    ignore_radius: float = VEHICLE_IGNORE_RADIUS_M,
) -> np.ndarray:
    """FLU (x, y) of the LiDAR points labelled as hard obstacles that the vehicle could hit: (P, 2).

    Points higher than vehicle_height above the local ground (branches overhead) and points
    within ignore_radius of the sensor (the vehicle) are left out.
    ground_z: (nx, ny) ground height per map cell (gaps are filled); points outside the map are dropped.
    """
    points_flu = lidar_to_flu(points_lidar)
    keep = np.isin(point_labels, hard_ids) & (np.hypot(points_flu[:, 0], points_flu[:, 1]) > ignore_radius)
    points = points_flu[keep]
    if len(points) == 0:
        return np.zeros((0, 2))
    i = np.floor((points[:, 0] - spec.x_min) / spec.resolution).astype(int)
    j = np.floor((points[:, 1] - spec.y_min) / spec.resolution).astype(int)
    nx, ny = spec.shape
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    points, i, j = points[inside], i[inside], j[inside]
    ground = fill_ground(ground_z)[i, j]
    low = ~np.isfinite(ground) | (points[:, 2] - ground <= vehicle_height)
    return points[low, :2]


def distance_to_obstacles(path_xy: np.ndarray, obstacle_xy: np.ndarray) -> tuple[float, float] | None:
    """(mean, min) over the path of the distance to the nearest obstacle point, meters; None if there are none."""
    if len(path_xy) == 0 or len(obstacle_xy) == 0:
        return None
    nearest = np.full(len(path_xy), np.inf)
    for start in range(0, len(obstacle_xy), 4096):  # in chunks, to keep memory small
        chunk = obstacle_xy[start:start + 4096]
        d = np.hypot(path_xy[:, None, 0] - chunk[None, :, 0], path_xy[:, None, 1] - chunk[None, :, 1])
        nearest = np.minimum(nearest, d.min(axis=1))
    return float(nearest.mean()), float(nearest.min())


def goal_is_blocked(target_xy: tuple[float, float], obstacle_xy: np.ndarray, radius: float = GOAL_BLOCKED_RADIUS_M) -> bool:
    """True if a hard obstacle point lies within `radius` m of the target."""
    if len(obstacle_xy) == 0:
        return False
    return bool((np.hypot(obstacle_xy[:, 0] - target_xy[0], obstacle_xy[:, 1] - target_xy[1]) <= radius).any())


def path_roughness(
    path: list[tuple[int, int]], ground_z: np.ndarray, obstacle_height: np.ndarray, spec: GridSpec
) -> tuple[float | None, float | None]:
    """(mean slope in cm per m between consecutive path cells, mean bump height in m) along a path.

    Ground gaps are filled first; steps where either end has no ground are skipped, and so are
    cells with no bump height. None where nothing can be measured.
    """
    if len(path) < 2:
        return None, None
    ground = fill_ground(ground_z)
    z = np.array([ground[c] for c in path])
    xy = np.array([spec.cell_center(*c) for c in path])
    step = np.hypot(*(xy[1:] - xy[:-1]).T)
    rise = np.abs(z[1:] - z[:-1])
    ok = np.isfinite(rise) & (step > 0)
    slope = float(100.0 * rise[ok].sum() / step[ok].sum()) if ok.any() else None
    bumps = np.array([obstacle_height[c] for c in path])
    bumps = bumps[np.isfinite(bumps)]
    return slope, (float(bumps.mean()) if bumps.size else None)


def class_share(truth: PathTruth, ids: list[int]) -> float | None:
    return float(np.isin(truth.labels, ids).mean()) if len(truth.labels) else None


def compute_path_metrics(
    plan: BaselineResult,
    truth: PathTruth | None,
    obstacle_xy: np.ndarray,
    id_to_name: dict[int, str],
) -> PathMetrics:
    """All metrics for one plan.

    truth: two-source ground truth of plan.path (from `path_truth`), or None if there is no path.
    obstacle_xy: (P, 2) hard obstacle points from `hard_obstacle_xy` for the same scan.
    """
    blocked = goal_is_blocked(plan.target_xy, obstacle_xy)
    if not plan.found or truth is None:
        return PathMetrics(False, False, blocked, plan.shortfall_m, None, None,
                           None, None, None, None, None, None, None, None, None)
    clearance = distance_to_obstacles(plan.path_xy(), obstacle_xy)
    slope, bumps = path_roughness(plan.path, plan.costmap.ground_z, plan.costmap.obstacle_height, plan.spec)
    return PathMetrics(
        found=True,
        reached=plan.reached,
        goal_blocked=blocked,
        shortfall_m=plan.shortfall_m,
        length_m=plan.length_m,
        detour_ratio=plan.detour_ratio,
        share_water=class_share(truth, ids_of(id_to_name, WATER_CLASSES)),
        share_mud=class_share(truth, ids_of(id_to_name, MUD_CLASSES)),
        share_compliant=class_share(truth, ids_of(id_to_name, COMPLIANT_VEGETATION)),
        share_noncompliant=class_share(truth, ids_of(id_to_name, NON_COMPLIANT_VEGETATION)),
        share_unseen=class_share(truth, [UNSEEN]),
        hard_dist_mean_m=clearance[0] if clearance else None,
        hard_dist_min_m=clearance[1] if clearance else None,
        slope_cm_per_m=slope,
        bump_height_m=bumps,
    )
