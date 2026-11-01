"""Geometry-only traversability cost map (pure functions, no file I/O).

This is the baseline the VLM approach is compared against. It only knows shapes:
anything sticking up more than `max_step` above the local ground is an obstacle,
whether it is a rock or a patch of tall grass the vehicle could drive through.

All grids are (nx, ny) in the FLU frame, same layout as `ElevationMap`.
"""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from offroad.mapping.elevation import ElevationMap, GridSpec, points_to_cells

LETHAL = np.inf


@dataclass(frozen=True)
class CostParams:
    """Settings for the geometric cost map. Lengths in meters.

    max_step: anything taller than this above the local ground blocks the vehicle.
    vehicle_height: points higher than this above the ground (e.g. branches) are ignored,
        since the vehicle passes under them.
    ground_window: size of the neighborhood used to estimate the ground under each cell.
        Wide enough that cells under a tree canopy still see real ground at the edge.
    roughness_weight: drivable cells cost 1 + roughness_weight * (obstacle height / max_step),
        so bumpier ground costs up to 1 + roughness_weight.
    unknown_cost: cost of cells with no usable points (not seen, or only overhead branches).
    inflation_radius: obstacles are grown by this much so the path keeps the vehicle body
        clear, roughly half the vehicle's width.
    """

    max_step: float = 0.3
    vehicle_height: float = 1.8
    ground_window: float = 6.0
    roughness_weight: float = 4.0
    unknown_cost: float = 3.0
    inflation_radius: float = 0.75


@dataclass
class CostMap:
    """Result of `geometric_cost_map`. All arrays are (nx, ny).

    cost: cost of driving through each cell; LETHAL (inf) where blocked.
    ground_z: estimated ground height, meters (NaN if nothing was seen nearby).
    obstacle_height: tallest point below vehicle_height, measured from ground_z, meters
        (NaN where unknown).
    obstacle: True where obstacle_height > max_step (before inflation).
    unknown: True where there are no usable points.
    """

    spec: GridSpec
    cost: np.ndarray
    ground_z: np.ndarray
    obstacle_height: np.ndarray
    obstacle: np.ndarray
    unknown: np.ndarray


def local_ground(min_z: np.ndarray, window_cells: int) -> np.ndarray:
    """Estimate the ground under each cell as the lowest point in a square neighborhood.

    min_z: (nx, ny) lowest point per cell, NaN where empty.
    window_cells: odd neighborhood size in cells, e.g. 25 for 6 m at 0.25 m.
    Returns (nx, ny); NaN where the whole neighborhood is empty.
    """
    radius = window_cells // 2
    filled = np.where(np.isnan(min_z), np.inf, min_z)
    padded = np.pad(filled, radius, constant_values=np.inf)
    ground = sliding_window_view(padded, (window_cells, window_cells)).min(axis=(-2, -1))
    ground[np.isinf(ground)] = np.nan
    return ground


def obstacle_height_map(
    points_flu: np.ndarray, ground_z: np.ndarray, spec: GridSpec, vehicle_height: float
) -> np.ndarray:
    """Tallest point per cell that the vehicle could hit, measured from the local ground.

    points_flu: (N, 3+) FLU frame, meters. ground_z: (nx, ny) from `local_ground`.
    Points more than vehicle_height above the ground are ignored (the vehicle passes under).
    Returns (nx, ny) meters; NaN where the cell has no such points.
    """
    i, j, inside = points_to_cells(points_flu, spec)
    height = points_flu[inside, 2] - ground_z[i, j]
    below = height < vehicle_height  # False where ground_z is NaN
    flat = np.ravel_multi_index((i[below], j[below]), spec.shape)

    tallest = np.full(spec.shape[0] * spec.shape[1], -np.inf)
    np.maximum.at(tallest, flat, height[below])
    tallest[np.isinf(tallest)] = np.nan
    return tallest.reshape(spec.shape)


def inflate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Grow True cells by a disk of the given radius (in cells). Returns a new (nx, ny) bool array."""
    nx, ny = mask.shape
    out = mask.copy()
    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            if di * di + dj * dj > radius_cells * radius_cells:
                continue
            # out[i, j] |= mask[i - di, j - dj], clipped to the grid.
            dst_i = slice(max(0, di), nx + min(0, di))
            src_i = slice(max(0, -di), nx - max(0, di))
            dst_j = slice(max(0, dj), ny + min(0, dj))
            src_j = slice(max(0, -dj), ny - max(0, dj))
            out[dst_i, dst_j] |= mask[src_i, src_j]
    return out


def geometric_cost_map(
    points_flu: np.ndarray, elevation: ElevationMap, params: CostParams = CostParams()
) -> CostMap:
    """Build the geometry-only cost map for one scan.

    points_flu: (N, 3+) FLU frame, meters (the same points `elevation` was built from).
    """
    spec = elevation.spec
    window_cells = 2 * round(params.ground_window / (2 * spec.resolution)) + 1
    ground_z = local_ground(elevation.min_z, window_cells)
    height = obstacle_height_map(points_flu, ground_z, spec, params.vehicle_height)

    unknown = np.isnan(height)
    obstacle = ~unknown & (height > params.max_step)

    roughness = np.clip(np.nan_to_num(height) / params.max_step, 0.0, 1.0)
    cost = 1.0 + params.roughness_weight * roughness
    cost[unknown] = params.unknown_cost
    blocked = inflate(obstacle, round(params.inflation_radius / spec.resolution))
    cost[blocked] = LETHAL

    return CostMap(
        spec=spec,
        cost=cost,
        ground_z=ground_z,
        obstacle_height=height,
        obstacle=obstacle,
        unknown=unknown,
    )
