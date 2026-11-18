"""Tests for evaluating conditions on one frame and comparing them. No dataset or model needed."""

import numpy as np
import pytest

from offroad.eval.compare import BASELINE, evaluate_frame, paired_comparison
from offroad.planning.hybrid import run_hybrid_planner
from offroad.vlm.grid import ImageGrid

GROUND_Z = -1.3
GRASS = 3
ID_TO_NAME = {0: "void", 3: "grass", 4: "tree", 7: "sky", 19: "bush"}
GRID = ImageGrid(height=1200, width=1920)
K = np.array([[2800.0, 0.0, 960.0], [0.0, 2800.0, 600.0], [0.0, 0.0, 1.0]])
T_CAM_FROM_LIDAR = np.eye(4)
T_CAM_FROM_LIDAR[:3, :3] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]  # level camera at the LiDAR, looking forward


def flu_to_lidar(points_flu) -> np.ndarray:
    return np.asarray(points_flu, dtype=float) * np.array([-1.0, -1.0, 1.0])


def flat_ground(x_max=25.0, half_width=6.0, step=0.2) -> np.ndarray:
    xs, ys = np.arange(-4.0, x_max, step), np.arange(-half_width, half_width, step)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, GROUND_Z)])


def grass_wall(x=10.0) -> np.ndarray:
    ys, zs = np.arange(-15.0, 15.0, 0.1), np.arange(GROUND_Z, GROUND_Z + 1.0, 0.1)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    return np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])


def hints(row, terrain, level):
    return {"waypoints": [], "hints": [{"cell": f"{row}{c}", "terrain": terrain, "traversability": level}
                                       for c in range(1, 13)]}


def evaluate(points_flu, answers):
    labels = np.full(len(points_flu), GRASS, dtype=np.uint16)
    return evaluate_frame(flu_to_lidar(points_flu), labels, T_CAM_FROM_LIDAR, K, GRID, None, answers, ID_TO_NAME)


# --- paired comparison ---


def test_paired_comparison_lower_is_better():
    c = paired_comparison([0.5, 0.5, 0.5, None], [0.2, 0.9, 0.505, 0.1], lower_is_better=True, tolerance=0.01)
    assert (c["wins"], c["losses"], c["ties"], c["frames"]) == (1, 1, 1, 3)
    assert c["mean_diff"] == pytest.approx((-0.3 + 0.4 + 0.005) / 3)


def test_paired_comparison_higher_is_better():
    c = paired_comparison([1.0, 1.0], [2.0, 0.5], lower_is_better=False, tolerance=0.1)
    assert (c["wins"], c["losses"], c["ties"]) == (1, 1, 0)


# --- evaluating one frame ---


def test_empty_answer_matches_the_baseline():
    results = evaluate(flat_ground(), {"empty": ({"waypoints": [], "hints": []}, None)})
    assert list(results) == [BASELINE, "empty"]
    assert results["empty"].plan.path == results[BASELINE].plan.path
    assert results["empty"].separation == (0.0, 0.0)
    assert results["empty"].metrics == results[BASELINE].metrics


def test_grass_hint_turns_failure_into_success():
    scan = np.vstack([flat_ground(), grass_wall()])
    results = evaluate(scan, {"vlm": (hints("G", "tall_grass_or_bush", "ok"), 12.5),
                              "tree": (hints("G", "tree_or_log", "ok"), 9.0)})
    assert not results[BASELINE].metrics.success
    assert results["vlm"].metrics.success and results["vlm"].metrics.reached
    assert not results["tree"].metrics.success  # a tree hint never unblocks
    assert results["vlm"].vlm_seconds == 12.5 and results[BASELINE].vlm_seconds is None
    assert results["vlm"].separation[1] > 1.0


def test_compute_time_is_measured():
    results = evaluate(flat_ground(), {"empty": ({"waypoints": [], "hints": []}, None)})
    assert results[BASELINE].compute_seconds > 0
    assert results["empty"].compute_seconds > 0


def test_hybrid_result_timing_fields_are_filled():
    points = flat_ground()
    result = run_hybrid_planner(flu_to_lidar(points), np.full(len(points), GRASS, dtype=np.uint16),
                                T_CAM_FROM_LIDAR, K, GRID, {"waypoints": [], "hints": []})
    assert result.baseline_seconds > 0 and result.hybrid_plan_seconds > 0
