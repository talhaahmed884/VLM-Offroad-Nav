"""Tests for the top-down elevation map. No dataset needed."""

import numpy as np

from offroad.mapping.elevation import (
    GridSpec,
    build_elevation_map,
    lidar_to_flu,
    majority_label_grid,
    points_to_cells,
)

# Small 2 m x 2 m grid with 0.5 m cells: 4 x 4 cells, covering x in [0, 2), y in [-1, 1).
SPEC = GridSpec(x_min=0.0, x_max=2.0, y_min=-1.0, y_max=1.0, resolution=0.5)


def test_lidar_forward_and_left_map_to_flu_axes():
    # On RELLIS-3D the Ouster's -x is forward and -y is left.
    points = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(lidar_to_flu(points), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_lidar_to_flu_ignores_intensity_column():
    assert lidar_to_flu(np.array([[1.0, 2.0, 3.0, 0.5]])).shape == (1, 3)


def test_default_grid_shape():
    assert GridSpec().shape == (160, 120)  # 40 m forward x 30 m wide at 0.25 m


def test_cell_center():
    assert SPEC.cell_center(0, 0) == (0.25, -0.75)


def test_points_to_cells_and_bounds():
    points = np.array(
        [
            [0.0, -1.0, 0.0],  # exactly at the grid corner -> cell (0, 0)
            [1.9, 0.9, 0.0],  # last cell (3, 3)
            [2.0, 0.0, 0.0],  # x == x_max is outside
            [-0.1, 0.0, 0.0],  # behind the grid
        ]
    )
    i, j, inside = points_to_cells(points, SPEC)
    assert inside.tolist() == [True, True, False, False]
    assert i.tolist() == [0, 3]
    assert j.tolist() == [0, 3]


def test_build_elevation_map_statistics():
    points = np.array(
        [
            [0.1, -0.9, 0.1],  # two points in cell (0, 0)
            [0.2, -0.8, 0.5],
            [1.2, 0.2, -0.3],  # one point in cell (2, 2)
        ]
    )
    elevation = build_elevation_map(points, SPEC)
    assert elevation.count.shape == (4, 4)
    assert elevation.count[0, 0] == 2
    assert np.isclose(elevation.min_z[0, 0], 0.1)
    assert np.isclose(elevation.max_z[0, 0], 0.5)
    assert np.isclose(elevation.mean_z[0, 0], 0.3)
    assert np.isclose(elevation.step_height[0, 0], 0.4)
    assert elevation.step_height[2, 2] == 0.0
    assert elevation.count.sum() == 3


def test_empty_cells_are_nan():
    elevation = build_elevation_map(np.array([[0.1, -0.9, 0.0]]), SPEC)
    assert elevation.count[1, 1] == 0
    assert np.isnan(elevation.min_z[1, 1])
    assert np.isnan(elevation.mean_z[1, 1])
    assert np.isnan(elevation.step_height[1, 1])


def test_majority_label_grid():
    points = np.array(
        [
            [0.1, -0.9, 0.0],  # cell (0, 0): grass, grass, bush -> grass
            [0.2, -0.9, 0.0],
            [0.3, -0.9, 0.0],
            [1.2, 0.2, 0.0],  # cell (2, 2): mud
            [5.0, 5.0, 0.0],  # outside the grid, ignored
        ]
    )
    labels = np.array([3, 3, 19, 33, 4], dtype=np.uint16)
    grid = majority_label_grid(points, labels, SPEC)
    assert grid[0, 0] == 3
    assert grid[2, 2] == 33
    assert grid[1, 1] == 0  # empty cell
