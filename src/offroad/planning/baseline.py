"""The geometry-only baseline for one scan: elevation map -> cost map -> A* path (pure, no file I/O).

The vehicle is at the FLU origin. The target is a point straight ahead. The planning goal
is a reachable cell chosen by `choose_goal`: as close to the target as it can get, unless
getting a little closer would need a disproportionately long drive. How far short of the
target it stops is reported as the shortfall.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from offroad.mapping.costmap import CostMap, CostParams, geometric_cost_map
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu, majority_label_grid
from offroad.planning.astar import Cell, astar, choose_goal, clear_footprint, path_length

UNSEEN = 0  # label used for path cells with no LiDAR points (same ID as "void")
REACHED_TOLERANCE_M = 1.0  # a path ending this close to the target counts as reaching it


@dataclass
class BaselineResult:
    """Everything the baseline computed for one scan.

    costmap: the geometric cost map (before the vehicle footprint was cleared).
    cost: (nx, ny) the cost grid the planner used (footprint cleared).
    labels: (nx, ny) most common ground-truth class per cell; 0 where no points.
    target_xy: FLU (x, y) of the point the vehicle is trying to reach, meters.
    start, goal: grid cells; goal comes from `choose_goal`, or is None if the vehicle
        cannot move at all (start blocked).
    path: cells from start to goal, or None if there is no goal.
    total_cost: A* cost of the path, or None.
    """

    spec: GridSpec
    params: CostParams
    costmap: CostMap
    cost: np.ndarray
    labels: np.ndarray
    target_xy: tuple[float, float]
    start: Cell
    goal: Cell | None
    path: list[Cell] | None
    total_cost: float | None

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def shortfall_m(self) -> float | None:
        """Distance from the goal cell's center to the target, meters (about 0 = target reached).

        None if there is no goal.
        """
        if self.goal is None:
            return None
        gx, gy = self.spec.cell_center(*self.goal)
        return float(np.hypot(gx - self.target_xy[0], gy - self.target_xy[1]))

    @property
    def reached(self) -> bool:
        """True if the path ends within REACHED_TOLERANCE_M of the target."""
        return self.shortfall_m is not None and self.shortfall_m <= REACHED_TOLERANCE_M

    @property
    def length_m(self) -> float | None:
        """Path length in meters, or None if no path."""
        return path_length(self.path, self.spec) if self.found else None

    @property
    def straight_m(self) -> float | None:
        """Straight-line distance from start to goal cell, meters, or None if no goal."""
        if self.goal is None:
            return None
        return float(np.hypot(self.goal[0] - self.start[0], self.goal[1] - self.start[1]) * self.spec.resolution)

    @property
    def detour_ratio(self) -> float | None:
        """Path length / straight-line distance (1.0 = straight), or None if no path."""
        if not self.found or not self.straight_m:
            return None
        return self.length_m / self.straight_m

    def path_xy(self) -> np.ndarray:
        """(M, 2) FLU (x, y) of the path cell centers, meters."""
        return np.array([self.spec.cell_center(*c) for c in self.path])

    def class_shares(self) -> dict[int, float]:
        """Share of path cells per ground-truth class ID (UNSEEN = no LiDAR points). Empty if no path."""
        if not self.found:
            return {}
        counts = Counter(int(self.labels[c]) for c in self.path)
        return {class_id: n / len(self.path) for class_id, n in counts.most_common()}


def run_geometric_baseline(
    points_lidar: np.ndarray,
    point_labels: np.ndarray,
    spec: GridSpec = GridSpec(),
    params: CostParams = CostParams(),
    goal_distance: float = 20.0,
    footprint_radius: float = 2.0,
    goal_cost_weight: float = 0.1,
) -> BaselineResult:
    """Plan the geometry-only path for one scan.

    points_lidar: (N, 3+) Ouster points in the LiDAR frame, meters.
    point_labels: (N,) ground-truth class per point (only used for reporting, not planning).
    goal_distance: the target is this far straight ahead, meters.
    footprint_radius: cells this close to the vehicle are treated as drivable, meters.
    goal_cost_weight: trade-off used by `choose_goal`; 0.1 means getting 1 m closer to the
        target is worth up to 10 m of extra smooth driving.
    """
    points_flu = lidar_to_flu(points_lidar)
    elevation = build_elevation_map(points_flu, spec)
    costmap = geometric_cost_map(points_flu, elevation, params)
    labels = majority_label_grid(points_flu, point_labels, spec, empty_label=UNSEEN)

    cost = clear_footprint(costmap.cost, spec, footprint_radius)
    start = spec.cell_index(0.0, 0.0)
    target_xy = (goal_distance, 0.0)
    goal = choose_goal(cost, spec, start, *target_xy, cost_weight=goal_cost_weight)
    planned = astar(cost, start, goal) if goal is not None else None
    path, total_cost = planned if planned is not None else (None, None)

    return BaselineResult(
        spec=spec,
        params=params,
        costmap=costmap,
        cost=cost,
        labels=labels,
        target_xy=target_xy,
        start=start,
        goal=goal,
        path=path,
        total_cost=total_cost,
    )
