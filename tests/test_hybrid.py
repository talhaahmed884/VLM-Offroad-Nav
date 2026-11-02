"""Tests for the hybrid planner on synthetic scans. No dataset or model needed."""

import numpy as np

from offroad.planning.hybrid import run_hybrid_planner
from offroad.vlm.grid import ImageGrid

GROUND_Z = -1.3  # ground 1.3 m below the LiDAR (and the camera)
GRASS = 3
GRID = ImageGrid(height=1200, width=1920)
K = np.array([[2800.0, 0.0, 960.0], [0.0, 2800.0, 600.0], [0.0, 0.0, 1.0]])
T_CAM_FROM_LIDAR = np.eye(4)
T_CAM_FROM_LIDAR[:3, :3] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]  # level camera at the LiDAR, looking forward
# With this camera, image row G (v = 900-1050) shows the ground about 8-12 m ahead,
# and row F (v = 750-900) about 12-24 m ahead.


def flu_to_lidar(points_flu: np.ndarray) -> np.ndarray:
    return points_flu * np.array([-1.0, -1.0, 1.0])


def flat_ground(x_max: float = 25.0, half_width: float = 6.0, step: float = 0.2) -> np.ndarray:
    xs = np.arange(-4.0, x_max, step)
    ys = np.arange(-half_width, half_width, step)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, GROUND_Z)])


def wall(x: float, y_from: float, y_to: float, height: float = 1.0) -> np.ndarray:
    ys = np.arange(y_from, y_to, 0.1)
    zs = np.arange(GROUND_Z, GROUND_Z + height, 0.1)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    return np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])


def plan(points_flu: np.ndarray, hints=(), waypoints=()):
    labels = np.full(len(points_flu), GRASS, dtype=np.uint16)
    answer = {"waypoints": list(waypoints),
              "hints": [{"cell": c, "terrain": t, "traversability": lv} for c, t, lv in hints]}
    return run_hybrid_planner(flu_to_lidar(points_flu), labels, T_CAM_FROM_LIDAR, K, GRID, answer)


def row_hints(row: str, terrain: str, level: str):
    return [(f"{row}{col}", terrain, level) for col in range(1, 13)]


GRASS_WALL = np.vstack([flat_ground(), wall(10.0, -15.0, 15.0)])  # 1 m tall, across the whole map


def test_no_answer_gives_the_baseline_path():
    result = plan(np.vstack([flat_ground(), wall(10.0, -6.0, 3.0)]))
    assert result.hybrid.path == result.baseline.path
    assert not result.path_changed
    assert result.path_separation() == (0.0, 0.0)


def test_vlm_grass_hint_opens_a_grass_wall():
    result = plan(GRASS_WALL, hints=row_hints("G", "tall_grass_or_bush", "ok"))
    assert not result.baseline.reached  # geometry sees a wall and stops short
    assert result.hybrid.reached  # the VLM says it is grass, so the hybrid drives through
    assert result.path_changed
    assert result.fused.unblocked.any()
    mean_apart, max_apart = result.path_separation()
    assert max_apart > 1.0 and 0.0 < mean_apart <= max_apart


def test_tree_hint_does_not_open_the_wall():
    result = plan(GRASS_WALL, hints=row_hints("G", "tree_or_log", "ok"))
    assert not result.fused.unblocked.any()
    assert result.hybrid.path == result.baseline.path


def test_water_hint_steers_the_path_around():
    hints = [("F6", "water", "avoid"), ("F7", "water", "avoid")]
    result = plan(flat_ground(), hints=hints)
    avoided = result.fused.hints.has_level(("avoid",))

    def cells_in_water(path):
        return sum(avoided[c] for c in path)

    assert cells_in_water(result.baseline.path) > 0  # the baseline drives straight through
    assert cells_in_water(result.hybrid.path) < cells_in_water(result.baseline.path)
    assert result.hybrid.reached
