"""Tests for the geometry-only cost map. No dataset needed."""

import numpy as np

from offroad.mapping.costmap import (
    LETHAL,
    CostParams,
    geometric_cost_map,
    inflate,
    local_ground,
    obstacle_height_map,
)
from offroad.mapping.elevation import GridSpec, build_elevation_map

# 4 m x 4 m grid with 0.5 m cells: 8 x 8 cells, x in [0, 4), y in [-2, 2).
SPEC = GridSpec(x_min=0.0, x_max=4.0, y_min=-2.0, y_max=2.0, resolution=0.5)
PARAMS = CostParams(
    max_step=0.3, vehicle_height=1.8, ground_window=1.0, roughness_weight=4.0, unknown_cost=3.0, inflation_radius=0.5
)


def cell_point(i: int, j: int, z: float) -> list[float]:
    """A point at the center of cell [i, j] of SPEC, at height z."""
    x, y = SPEC.cell_center(i, j)
    return [x, y, z]


def test_local_ground_takes_neighborhood_minimum():
    min_z = np.full((5, 5), np.nan)
    min_z[2, 2] = 3.0  # e.g. a tree canopy cell with no ground seen
    min_z[2, 3] = -1.0  # real ground next to it
    ground = local_ground(min_z, window_cells=3)
    assert ground[2, 2] == -1.0
    assert ground[1, 4] == -1.0  # empty cell, but its neighborhood has ground
    assert np.isnan(ground[0, 0])  # nothing within one cell


def test_obstacle_height_ignores_points_above_vehicle():
    ground = np.zeros(SPEC.shape)
    points = np.array([cell_point(1, 1, 0.1), cell_point(1, 1, 0.5), cell_point(1, 1, 2.5)])
    height = obstacle_height_map(points, ground, SPEC, vehicle_height=1.8)
    assert np.isclose(height[1, 1], 0.5)


def test_overhead_only_cell_is_unknown():
    ground = np.zeros(SPEC.shape)
    points = np.array([cell_point(2, 2, 3.0)])  # only a branch above the vehicle
    height = obstacle_height_map(points, ground, SPEC, vehicle_height=1.8)
    assert np.isnan(height[2, 2])


def test_inflate_grows_by_disk():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    grown = inflate(mask, radius_cells=1)
    assert grown.sum() == 5  # center plus its 4 direct neighbors
    assert grown[1, 2] and grown[3, 2] and grown[2, 1] and grown[2, 3]
    assert not grown[1, 1]  # diagonal is outside a radius-1 disk
    assert np.array_equal(inflate(mask, radius_cells=0), mask)


def test_inflate_at_grid_edge():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    grown = inflate(mask, radius_cells=1)
    assert grown[0, 0] and grown[1, 0] and grown[0, 1]
    assert grown.sum() == 3


def test_flat_ground_with_a_post():
    # Flat ground at z = 0 in every cell except (7, 7), plus a 1 m post in cell (4, 4).
    points = [cell_point(i, j, 0.0) for i in range(8) for j in range(8) if (i, j) != (7, 7)]
    points.append(cell_point(4, 4, 1.0))
    points = np.array(points)
    elevation = build_elevation_map(points, SPEC)
    costmap = geometric_cost_map(points, elevation, PARAMS)

    assert costmap.cost[0, 0] == 1.0  # flat, drivable
    assert costmap.obstacle[4, 4]
    assert costmap.cost[4, 4] == LETHAL
    assert costmap.cost[4, 5] == LETHAL  # inflated neighbor
    assert costmap.cost[5, 5] == 1.0  # diagonal, outside the inflation radius
    assert costmap.unknown[7, 7]
    assert costmap.cost[7, 7] == PARAMS.unknown_cost


def test_small_bumps_cost_more_but_do_not_block():
    points = np.array([cell_point(i, j, 0.0) for i in range(8) for j in range(8)] + [cell_point(3, 3, 0.15)])
    costmap = geometric_cost_map(points, build_elevation_map(points, SPEC), PARAMS)
    # 0.15 m is half of max_step, so cost = 1 + 4 * 0.5.
    assert np.isclose(costmap.cost[3, 3], 3.0)
    assert not costmap.obstacle[3, 3]
