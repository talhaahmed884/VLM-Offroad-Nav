"""Tests for the end-to-end geometry-only baseline on synthetic scans. No dataset needed."""

import numpy as np
import pytest

from offroad.planning.baseline import UNSEEN, run_geometric_baseline

GROUND_Z = -1.3  # ground about 1.3 m below the LiDAR
GRASS = 3


def flu_to_lidar(points_flu: np.ndarray) -> np.ndarray:
    """RELLIS-3D Ouster: forward is -x and left is -y, so flip x and y."""
    return points_flu * np.array([-1.0, -1.0, 1.0])


def flat_ground(x_max: float = 25.0, half_width: float = 6.0, step: float = 0.2) -> np.ndarray:
    xs = np.arange(-4.0, x_max, step)
    ys = np.arange(-half_width, half_width, step)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, GROUND_Z)])


def wall(x: float, y_from: float, y_to: float, height: float = 1.0) -> np.ndarray:
    """Vertical wall of points at forward distance x, spanning y_from..y_to."""
    ys = np.arange(y_from, y_to, 0.1)
    zs = np.arange(GROUND_Z, GROUND_Z + height, 0.1)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    return np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])


def run(points_flu: np.ndarray):
    labels = np.full(len(points_flu), GRASS, dtype=np.uint16)
    return run_geometric_baseline(flu_to_lidar(points_flu), labels, goal_distance=20.0)


def test_flat_ground_gives_straight_path():
    result = run(flat_ground())
    assert result.found
    assert result.reached
    assert result.shortfall_m < 0.5
    assert result.detour_ratio == pytest.approx(1.0, abs=0.02)
    assert result.length_m == pytest.approx(20.0, abs=0.5)
    shares = result.class_shares()
    assert shares[GRASS] > 0.9
    assert sum(shares.values()) == pytest.approx(1.0)


def test_wall_with_gap_causes_detour():
    # Wall at 10 m ahead from y = -6 to 3 m, leaving a gap on the left (y from 3 to 6 m).
    result = run(np.vstack([flat_ground(), wall(10.0, -6.0, 3.0)]))
    assert result.found
    assert result.reached
    assert result.detour_ratio > 1.05
    gap_side = [result.spec.cell_center(*c)[1] for c in result.path if abs(result.spec.cell_center(*c)[0] - 10.0) < 0.3]
    assert min(gap_side) > 3.0  # crosses x = 10 m through the gap


def test_wall_across_the_whole_map_stops_short():
    # The wall spans the full 30 m grid width, so the target behind it is cut off. The path
    # goes as close as it can: to just before the wall's 0.75 m safety margin (x ~ 9.1 m).
    result = run(np.vstack([flat_ground(), wall(10.0, -15.0, 15.0)]))
    assert result.found
    assert not result.reached
    goal_x, _ = result.spec.cell_center(*result.goal)
    assert 8.5 < goal_x < 10.0
    assert 10.0 < result.shortfall_m < 11.5


def test_obstacle_at_vehicle_does_not_block_start():
    # Points sticking up right at the vehicle (e.g. its own body) are inside the cleared footprint.
    result = run(np.vstack([flat_ground(), wall(0.5, -0.5, 0.5)]))
    assert result.found


def test_unseen_cells_are_labelled_unseen():
    # Ground only up to 10 m: the rest of the path to 20 m has no points.
    result = run(flat_ground(x_max=10.0))
    assert result.found
    assert result.class_shares()[UNSEEN] > 0.4
