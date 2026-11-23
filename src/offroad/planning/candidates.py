"""Geometric candidate trajectory generator for the DriveVLN-style selector (pure, no file I/O).

Generates up to K distinct, collision-free candidate paths on an existing geometric CostMap:
1. Nominal geometric path (center).
2. Optional soft-obstacle path (allows traversing low obstacles <= 1.5 m at a higher cost,
   giving the VLM a bush/tall-grass option while keeping tall obstacles LETHAL).
3. Laterally biased geometric paths (left/right corridors) in the FLU frame (x forward, y left).
"""

from dataclasses import dataclass

import numpy as np

from offroad.mapping.costmap import LETHAL, CostMap, CostParams
from offroad.mapping.elevation import GridSpec
from offroad.planning.astar import Cell, astar, choose_goal, clear_footprint, path_length
from offroad.planning.baseline import REACHED_TOLERANCE_M, BaselineResult

CANDIDATE_COLORS = ("red", "blue", "yellow", "green")


@dataclass(frozen=True)
class CandidateParams:
    """Settings for generating laterally diverse and soft-obstacle candidate paths."""

    # Lateral offsets in FLU y (meters): 0 = center, >0 = left, <0 = right.
    lateral_offsets_m: tuple[float, ...] = (0.0, 2.5, -2.5, 5.0, -5.0)
    lateral_bias_weight: float = 0.25  # cost multiplier slope per meter from offset
    goal_y_scale: float = 0.5  # fraction of lateral_offset applied to the target y
    min_separation_m: float = 0.75  # drop candidates closer than this to a kept path
    max_candidates: int = 4
    # Soft-obstacle candidate settings (tall grass / penetrable bush option).
    max_soft_height_m: float = 1.5
    soft_obstacle_cost: float = 6.0
    hard_inflation_radius_m: float = 0.75


@dataclass
class CandidatePath:
    """One geometrically feasible candidate trajectory."""

    color: str
    kind: str  # "geometric" or "soft_obstacle"
    lateral_offset_m: float
    path: list[Cell]
    total_cost: float
    start: Cell
    goal: Cell
    target_xy: tuple[float, float]
    spec: GridSpec

    @property
    def length_m(self) -> float:
        return path_length(self.path, self.spec)

    @property
    def shortfall_m(self) -> float:
        gx, gy = self.spec.cell_center(*self.goal)
        return float(np.hypot(gx - self.target_xy[0], gy - self.target_xy[1]))

    @property
    def reached(self) -> bool:
        return self.shortfall_m <= REACHED_TOLERANCE_M

    def path_xy(self) -> np.ndarray:
        """FLU (x, y) coordinates along the path: (P, 2)."""
        if not self.path:
            return np.zeros((0, 2), dtype=float)
        return np.array([self.spec.cell_center(i, j) for i, j in self.path], dtype=float)


def _dilate_mask(mask: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    """Inflate a boolean (nx, ny) mask by `radius_m` using a circular footprint."""
    if radius_m <= 0.0 or not mask.any():
        return mask.copy()
    r = int(np.ceil(radius_m / resolution))
    di, dj = np.mgrid[-r : r + 1, -r : r + 1]
    offsets = np.argwhere((di**2 + dj**2) * (resolution**2) <= radius_m**2) - r
    nx, ny = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for oi, oj in offsets:
        src_i = slice(max(0, -oi), min(nx, nx - oi))
        src_j = slice(max(0, -oj), min(ny, ny - oj))
        dst_i = slice(max(0, oi), min(nx, nx + oi))
        dst_j = slice(max(0, oj), min(ny, ny + oj))
        out[dst_i, dst_j] |= mask[src_i, src_j]
    return out


def soft_obstacle_cost(
    cost: np.ndarray,
    obstacle_height: np.ndarray,
    spec: GridSpec,
    max_soft_height_m: float = 1.5,
    soft_cost: float = 6.0,
    hard_inflation_radius_m: float = 0.75,
) -> np.ndarray:
    """Return a copy of `cost` where low obstacles (<= max_soft_height_m) become traversable at `soft_cost`.

    Tall obstacles (> max_soft_height_m) and their inflation rings remain LETHAL.
    cost, obstacle_height: (nx, ny) float arrays.
    """
    hard = np.isfinite(obstacle_height) & (obstacle_height > max_soft_height_m)
    hard_inflated = _dilate_mask(hard, hard_inflation_radius_m, spec.resolution)
    soft = cost.copy()
    low_blocked = (~np.isfinite(cost)) & (~hard_inflated)
    soft[low_blocked] = soft_cost
    soft[hard_inflated] = LETHAL
    return soft


def lateral_biased_cost(
    cost: np.ndarray,
    spec: GridSpec,
    y_offset_m: float,
    bias_weight: float,
) -> np.ndarray:
    """Return a copy of `cost` biased toward FLU lateral coordinate `y_offset_m`.

    LETHAL cells stay LETHAL; drivable cells are scaled by `1 + bias_weight * |y - y_offset_m|`.
    cost: (nx, ny) float array.
    """
    if bias_weight <= 0.0:
        return cost.copy()
    _, ny = spec.shape
    y_centers = spec.y_min + (np.arange(ny) + 0.5) * spec.resolution
    scale = 1.0 + bias_weight * np.abs(y_centers - y_offset_m)  # (ny,)
    biased = cost * scale[None, :]
    biased[~np.isfinite(cost)] = LETHAL
    return biased


def path_mean_distance(path_a_xy: np.ndarray, path_b_xy: np.ndarray) -> float:
    """Mean nearest-point Euclidean distance (meters) from path_a to path_b."""
    if len(path_a_xy) == 0 or len(path_b_xy) == 0:
        return float("inf")
    d = np.hypot(
        path_a_xy[:, None, 0] - path_b_xy[None, :, 0],
        path_a_xy[:, None, 1] - path_b_xy[None, :, 1],
    )
    return float(d.min(axis=1).mean())


def _plan_single(
    cost_grid: np.ndarray,
    spec: GridSpec,
    start: Cell,
    target_xy: tuple[float, float],
    goal_cost_weight: float,
) -> tuple[Cell, list[Cell], float, np.ndarray] | None:
    goal = choose_goal(cost_grid, spec, start, target_xy[0], target_xy[1], cost_weight=goal_cost_weight)
    if goal is None:
        return None
    planned = astar(cost_grid, start, goal)
    if planned is None:
        return None
    path, total_cost = planned
    xy = np.array([spec.cell_center(i, j) for i, j in path], dtype=float)
    return goal, path, total_cost, xy


def generate_candidates(
    cost: np.ndarray,
    spec: GridSpec,
    obstacle_height: np.ndarray | None = None,
    goal_distance: float = 20.0,
    footprint_radius: float = 2.0,
    goal_cost_weight: float = 0.1,
    params: CandidateParams = CandidateParams(),
    colors: tuple[str, ...] = CANDIDATE_COLORS,
) -> list[CandidatePath]:
    """Generate up to `params.max_candidates` distinct candidate paths.

    cost: (nx, ny) geometric cost grid (LETHAL = inf).
    obstacle_height: optional (nx, ny) obstacle height above local ground (m). When provided,
        an additional straight-ahead candidate is planned on a soft-obstacle costmap so the
        VLM can choose to drive through penetrable tall grass/bushes if appropriate.
    """
    base_cost = clear_footprint(cost, spec, radius=footprint_radius)
    start = spec.cell_index(0.0, 0.0)
    if not np.isfinite(base_cost[start]):
        return []

    # Build ordered list of (kind, lateral_offset_m, cost_grid) proposals.
    proposals: list[tuple[str, float, np.ndarray]] = []
    for offset in params.lateral_offsets_m:
        grid = (
            base_cost
            if offset == 0.0
            else lateral_biased_cost(base_cost, spec, offset, params.lateral_bias_weight)
        )
        proposals.append(("geometric", offset, grid))
        # Try the soft-obstacle path right after the nominal center path so it isn't
        # crowded out by lateral detours on bush-blocked frames.
        if offset == 0.0 and obstacle_height is not None:
            soft = soft_obstacle_cost(
                cost,
                obstacle_height,
                spec,
                max_soft_height_m=params.max_soft_height_m,
                soft_cost=params.soft_obstacle_cost,
                hard_inflation_radius_m=params.hard_inflation_radius_m,
            )
            proposals.append(("soft_obstacle", 0.0, clear_footprint(soft, spec, radius=footprint_radius)))

    kept: list[CandidatePath] = []
    kept_xy: list[np.ndarray] = []
    limit = min(params.max_candidates, len(colors))

    for kind, offset, grid in proposals:
        if len(kept) >= limit:
            break
        target_xy = (goal_distance, offset * params.goal_y_scale)
        planned = _plan_single(grid, spec, start, target_xy, goal_cost_weight)
        if planned is None:
            continue
        goal, path, total_cost, xy = planned

        if any(path_mean_distance(xy, prev_xy) < params.min_separation_m for prev_xy in kept_xy):
            continue

        kept.append(
            CandidatePath(
                color=colors[len(kept)],
                kind=kind,
                lateral_offset_m=offset,
                path=path,
                total_cost=total_cost,
                start=start,
                goal=goal,
                target_xy=target_xy,
                spec=spec,
            )
        )
        kept_xy.append(xy)

    return kept


def candidate_to_baseline_result(
    candidate: CandidatePath,
    costmap: CostMap,
    labels: np.ndarray,
    params: CostParams = CostParams(),
    goal_distance: float = 20.0,
) -> BaselineResult:
    """Wrap a chosen CandidatePath as a BaselineResult so `compute_path_metrics` can score it."""
    return BaselineResult(
        spec=candidate.spec,
        params=params,
        costmap=costmap,
        cost=costmap.cost,
        labels=labels,
        target_xy=(goal_distance, 0.0),
        start=candidate.start,
        goal=candidate.goal,
        path=candidate.path,
        total_cost=candidate.total_cost,
    )