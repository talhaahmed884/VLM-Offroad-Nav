"""Tests for matching top-down map cells to image grid cells. No dataset needed."""

import numpy as np
import pytest

from offroad.fusion.grid_lookup import NO_CELL, build_grid_lookup, fill_ground, grid_cell_ids, project_map_cells
from offroad.mapping.elevation import GridSpec
from offroad.vlm.grid import ImageGrid

SPEC = GridSpec()  # x -5..35, y -15..15, 0.25 m cells
GRID = ImageGrid(height=1200, width=1920)  # 150 x 160 px cells
K = np.array([[2800.0, 0.0, 960.0], [0.0, 2800.0, 600.0], [0.0, 0.0, 1.0]])

# A level camera at the LiDAR origin looking forward (same as in test_vlm_runner.py).
T_CAM_FROM_LIDAR = np.eye(4)
T_CAM_FROM_LIDAR[:3, :3] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]


def flat_ground(height: float = -1.5) -> np.ndarray:
    return np.full(SPEC.shape, height)


def lookup_on_flat_ground():
    return build_grid_lookup(flat_ground(), SPEC, T_CAM_FROM_LIDAR, K, GRID)


# --- filling ground gaps ---


def test_fill_ground_fills_a_hole_from_its_neighbors():
    ground = flat_ground(-1.2)
    ground[50:60, 50:60] = np.nan
    filled = fill_ground(ground)
    assert np.allclose(filled, -1.2)
    assert np.isnan(ground[55, 55])  # the input is not changed


def test_fill_ground_stops_after_max_steps():
    ground = np.full((20, 20), np.nan)
    ground[0, 0] = 1.0
    filled = fill_ground(ground, max_steps=3)
    assert filled[3, 3] == 1.0
    assert np.isnan(filled[4, 4])


def test_fill_ground_all_unknown_stays_unknown():
    assert np.isnan(fill_ground(np.full((5, 5), np.nan))).all()


# --- projecting map cells ---


def test_project_map_cells_matches_the_pinhole_formula():
    uv = project_map_cells(flat_ground(), SPEC, T_CAM_FROM_LIDAR, K, (1200, 1920))
    i, j = SPEC.cell_index(20.1, -0.1)
    x, y = SPEC.cell_center(i, j)
    assert uv[i, j, 0] == pytest.approx(960.0 - 2800.0 * y / x)
    assert uv[i, j, 1] == pytest.approx(600.0 + 2800.0 * 1.5 / x)


def test_cells_behind_or_too_close_are_not_in_view():
    uv = project_map_cells(flat_ground(), SPEC, T_CAM_FROM_LIDAR, K, (1200, 1920))
    assert np.isnan(uv[SPEC.cell_index(-2.0, 0.0)]).all()  # behind the vehicle
    assert np.isnan(uv[SPEC.cell_index(3.0, 0.0)]).all()  # below the bottom of the image


def test_unknown_ground_is_not_in_view():
    ground = flat_ground()
    i, j = SPEC.cell_index(20.1, -0.1)
    ground[i, j] = np.nan
    uv = project_map_cells(ground, SPEC, T_CAM_FROM_LIDAR, K, (1200, 1920))
    assert np.isnan(uv[i, j]).all()


def test_grid_cell_ids_marks_nan_as_no_cell():
    uv = np.array([[[np.nan, np.nan], [970.0, 820.0]]])
    assert grid_cell_ids(uv, GRID).tolist() == [[NO_CELL, 5 * 12 + 6]]  # F7 = row 5, column 6


# --- the lookup ---


def test_lookup_cell_straight_ahead():
    lookup = lookup_on_flat_ground()
    i, j = SPEC.cell_index(20.1, -0.1)  # 1.5 m below the camera at 20 m -> v = 810 (row F), u = 977 (column 7)
    assert lookup.label_of(int(lookup.cell_id[i, j])) == "F7"
    assert lookup.mask("F7")[i, j]


def test_lookup_near_is_lower_and_left_is_left():
    lookup = lookup_on_flat_ground()
    near = lookup.label_of(int(lookup.cell_id[SPEC.cell_index(8.0, -0.1)]))
    far = lookup.label_of(int(lookup.cell_id[SPEC.cell_index(20.1, -0.1)]))
    assert GRID.parse(near)[0] > GRID.parse(far)[0]  # nearer ground is in a lower row
    left = lookup.label_of(int(lookup.cell_id[SPEC.cell_index(20.1, 3.0)]))
    assert GRID.parse(left)[1] < GRID.parse(far)[1]  # ground to the left is in a column further left


def test_lookup_centroid():
    lookup = lookup_on_flat_ground()
    x, y = lookup.centroid("F7")
    assert 15.0 < x < 30.0 and -1.5 < y < 0.0
    assert lookup.centroid("A1") is None  # the sky has no ground


def test_ground_point_is_the_ground_at_the_cell_center():
    lookup = lookup_on_flat_ground()
    # F7's center pixel is (1040, 825): 2800 * 1.5 / (825 - 600) = 18.7 m ahead, 80 px right of center.
    x, y = lookup.ground_point("F7")
    expected_x = 2800.0 * 1.5 / (825.0 - 600.0)
    # At ~19 m one 0.25 m map cell moves the image ~37 px sideways but only ~3 px up/down, so the
    # nearest map cell in pixels can be up to ~0.5 m off in distance.
    assert x == pytest.approx(expected_x, abs=0.6)
    assert y == pytest.approx(-80.0 * expected_x / 2800.0, abs=0.3)
    assert lookup.ground_point("A1") is None


def test_lookup_marks_filled_ground():
    ground = flat_ground()
    i, j = SPEC.cell_index(20.1, -0.1)
    ground[i, j] = np.nan
    lookup = build_grid_lookup(ground, SPEC, T_CAM_FROM_LIDAR, K, GRID)
    assert lookup.ground_filled[i, j] and lookup.visible[i, j]  # filled in, so it gets a grid cell
    assert lookup.ground_filled.sum() == 1
