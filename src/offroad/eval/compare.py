"""Evaluate the baseline and every hybrid condition on one frame, and compare conditions (pure, no file I/O).

All conditions share the same elevation map, cost map, labels, grid lookup, and hard-obstacle
points, so they differ only in the VLM answer that re-weights the costs. Each condition's path
is scored with `metrics.compute_path_metrics` on its own two-source ground truth.

Timing per condition: `compute_seconds` is what the vehicle's computer would spend on the
geometry side (map + plan for the baseline; map + grid lookup + re-weighting + plan for a
hybrid). `vlm_seconds` is the VLM's answer time, saved by run_vlm.py. In the asynchronous
design the VLM runs alongside the planner, so it delays the hints, not each plan.
"""

import time
from dataclasses import dataclass

import numpy as np

from offroad.eval.metrics import HARD_OBSTACLE_CLASSES, PathMetrics, compute_path_metrics, hard_obstacle_xy, ids_of
from offroad.eval.path_truth import PathTruth, path_truth
from offroad.fusion.grid_lookup import build_grid_lookup
from offroad.fusion.reweight import FusionParams, reweight_cost_map
from offroad.mapping.costmap import CostParams, geometric_cost_map
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu, majority_label_grid
from offroad.planning.baseline import UNSEEN, BaselineResult, plan_on_cost
from offroad.vlm.grid import ImageGrid

BASELINE = "baseline"


@dataclass
class ConditionResult:
    """One condition's plan on one frame, and its scores.

    separation: (mean, max) distance of this path from the baseline path, meters (None if either has no path).
    """

    name: str
    plan: BaselineResult
    truth: PathTruth | None
    metrics: PathMetrics
    compute_seconds: float
    vlm_seconds: float | None
    separation: tuple[float, float] | None


def path_separation(plan: BaselineResult, base: BaselineResult) -> tuple[float, float] | None:
    """(mean, max) over plan's cells of the distance to the nearest base path cell, meters."""
    if not (plan.found and base.found):
        return None
    a, b = plan.path_xy(), base.path_xy()
    nearest = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]).min(axis=1)
    return float(nearest.mean()), float(nearest.max())


def evaluate_frame(
    points_lidar: np.ndarray,
    point_labels: np.ndarray,
    T_cam_from_lidar: np.ndarray,
    K: np.ndarray,
    grid: ImageGrid,
    label_image: np.ndarray | None,
    answers: dict[str, tuple[dict, float | None]],
    id_to_name: dict[int, str],
    spec: GridSpec = GridSpec(),
    params: CostParams = CostParams(),
    fusion: FusionParams = FusionParams(),
    goal_distance: float = 20.0,
    footprint_radius: float = 2.0,
    goal_cost_weight: float = 0.1,
) -> dict[str, ConditionResult]:
    """Plan and score the baseline plus one hybrid per answer. Returns {condition name: result}, baseline first.

    answers: condition name -> (checked answer dict with "hints" and "waypoints", VLM seconds or None).
    label_image: (H, W) camera labels for the ground truth, or None (LiDAR labels only).
    """
    plan_args = (params, goal_distance, footprint_radius, goal_cost_weight)
    ignore_ids = tuple(c for c, name in id_to_name.items() if name in ("void", "sky")) + (UNSEEN,)

    started = time.perf_counter()
    points_flu = lidar_to_flu(points_lidar)
    costmap = geometric_cost_map(points_flu, build_elevation_map(points_flu, spec), params)
    labels = majority_label_grid(points_flu, point_labels, spec, empty_label=UNSEEN)
    map_seconds = time.perf_counter() - started

    obstacles = hard_obstacle_xy(points_lidar, point_labels, ids_of(id_to_name, HARD_OBSTACLE_CLASSES),
                                 costmap.ground_z, spec, vehicle_height=params.vehicle_height)

    def score(name: str, plan: BaselineResult, seconds: float, vlm_seconds: float | None, base=None) -> ConditionResult:
        truth = (path_truth(plan.path, labels, costmap.ground_z, spec, T_cam_from_lidar, K, label_image,
                            points_lidar, ignore_ids) if plan.found else None)
        return ConditionResult(name, plan, truth, compute_path_metrics(plan, truth, obstacles, id_to_name),
                               seconds, vlm_seconds, path_separation(plan, base or plan))

    started = time.perf_counter()
    baseline = plan_on_cost(costmap.cost, costmap, labels, *plan_args)
    results = {BASELINE: score(BASELINE, baseline, map_seconds + time.perf_counter() - started, None)}

    started = time.perf_counter()
    lookup = build_grid_lookup(costmap.ground_z, spec, T_cam_from_lidar, K, grid)
    lookup_seconds = time.perf_counter() - started
    for name, (answer, vlm_seconds) in answers.items():
        started = time.perf_counter()
        fused = reweight_cost_map(costmap, lookup, answer["hints"], fusion, params, answer["waypoints"])
        plan = plan_on_cost(fused.cost, costmap, labels, *plan_args)
        seconds = map_seconds + lookup_seconds + time.perf_counter() - started
        results[name] = score(name, plan, seconds, vlm_seconds, base=baseline)
    return results


def paired_comparison(
    base: list[float | None], other: list[float | None], lower_is_better: bool, tolerance: float
) -> dict[str, float | int]:
    """Frame-by-frame comparison of one metric against the baseline.

    Frames where either value is None are skipped. A frame is a win if `other` is better than
    `base` by more than `tolerance`, a loss if worse by more than it, else a tie.
    Returns {"wins", "losses", "ties", "frames", "mean_diff"} with mean_diff = mean(other - base).
    """
    pairs = [(b, o) for b, o in zip(base, other) if b is not None and o is not None]
    sign = -1.0 if lower_is_better else 1.0
    gains = [sign * (o - b) for b, o in pairs]
    return {
        "wins": sum(g > tolerance for g in gains),
        "losses": sum(g < -tolerance for g in gains),
        "ties": sum(abs(g) <= tolerance for g in gains),
        "frames": len(pairs),
        "mean_diff": float(np.mean([o - b for b, o in pairs])) if pairs else float("nan"),
    }
