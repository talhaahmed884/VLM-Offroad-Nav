"""The hybrid planner for one scan: the baseline cost map changed by a VLM answer, then the same A* (pure).

Both planners share everything except the cost grid: the same elevation map, the same goal
choice, the same footprint clearing, and the same A*. So any difference between the two
paths comes from the VLM's hints and route alone.
"""

from dataclasses import dataclass

import numpy as np

from offroad.fusion.grid_lookup import GridLookup, build_grid_lookup
from offroad.fusion.reweight import FusedCostMap, FusionParams, reweight_cost_map
from offroad.mapping.costmap import CostParams, geometric_cost_map
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu, majority_label_grid
from offroad.planning.baseline import UNSEEN, BaselineResult, plan_on_cost
from offroad.vlm.grid import ImageGrid


@dataclass
class HybridResult:
    """The baseline and hybrid plans for one scan, plus what the VLM changed.

    baseline / hybrid: plans on the geometric and on the VLM-adjusted cost grid.
    fused: the VLM-adjusted cost map (hints, unblocked cells, route corridor).
    lookup: which image grid cell each map cell appears in.
    """

    baseline: BaselineResult
    hybrid: BaselineResult
    fused: FusedCostMap
    lookup: GridLookup

    @property
    def path_changed(self) -> bool:
        """True if the hybrid path differs from the baseline path (or one of them has none)."""
        return self.baseline.path != self.hybrid.path

    def path_separation(self) -> tuple[float, float] | None:
        """How far the hybrid path strays from the baseline path: (mean, max) meters, or None if either has no path.

        For each hybrid path cell, the distance to the nearest baseline path cell. A small mean
        means the VLM only nudged the path; a large max means it took a different way somewhere.
        """
        if not (self.baseline.found and self.hybrid.found):
            return None
        base, hybrid = self.baseline.path_xy(), self.hybrid.path_xy()
        nearest = np.hypot(hybrid[:, None, 0] - base[None, :, 0], hybrid[:, None, 1] - base[None, :, 1]).min(axis=1)
        return float(nearest.mean()), float(nearest.max())


def run_hybrid_planner(
    points_lidar: np.ndarray,
    point_labels: np.ndarray,
    T_cam_from_lidar: np.ndarray,
    K: np.ndarray,
    grid: ImageGrid,
    answer: dict,
    spec: GridSpec = GridSpec(),
    params: CostParams = CostParams(),
    fusion: FusionParams = FusionParams(),
    goal_distance: float = 20.0,
    footprint_radius: float = 2.0,
    goal_cost_weight: float = 0.1,
) -> HybridResult:
    """Plan the baseline and the hybrid path for one scan.

    points_lidar: (N, 3+) Ouster points in the LiDAR frame, meters; point_labels: (N,) for reporting only.
    grid: the image grid the VLM answered in, at the camera's full resolution.
    answer: a checked VLM answer as saved by the runner ({"waypoints", "hints", ...}); an answer
        with no hints and no waypoints gives the same path as the baseline.
    The other arguments are as in `run_geometric_baseline`.
    """
    points_flu = lidar_to_flu(points_lidar)
    costmap = geometric_cost_map(points_flu, build_elevation_map(points_flu, spec), params)
    labels = majority_label_grid(points_flu, point_labels, spec, empty_label=UNSEEN)
    lookup = build_grid_lookup(costmap.ground_z, spec, T_cam_from_lidar, K, grid)
    fused = reweight_cost_map(costmap, lookup, answer["hints"], fusion, params, answer["waypoints"])

    plan_args = (params, goal_distance, footprint_radius, goal_cost_weight)
    return HybridResult(
        baseline=plan_on_cost(costmap.cost, costmap, labels, *plan_args),
        hybrid=plan_on_cost(fused.cost, costmap, labels, *plan_args),
        fused=fused,
        lookup=lookup,
    )
