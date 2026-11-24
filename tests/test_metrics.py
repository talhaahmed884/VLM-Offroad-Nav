"""Tests for the evaluation metrics of one path. No dataset needed."""

import numpy as np
import pytest

from offroad.eval.metrics import (
    PathMetrics,
    compute_path_metrics,
    distance_to_obstacles,
    goal_is_blocked,
    hard_obstacle_xy,
    ids_of,
    path_roughness,
)
from offroad.eval.path_truth import SOURCE_LIDAR, PathTruth
from offroad.mapping.elevation import GridSpec
from offroad.planning.baseline import run_geometric_baseline

ID_TO_NAME = {0: "void", 3: "grass", 4: "tree", 17: "person", 19: "bush", 31: "puddle", 33: "mud"}
GRASS, TREE, PERSON, BUSH, PUDDLE, MUD = 3, 4, 17, 19, 31, 33
SPEC = GridSpec()
GROUND_Z = -1.3


def flu_to_lidar(points_flu) -> np.ndarray:
    return np.asarray(points_flu, dtype=float) * np.array([-1.0, -1.0, 1.0])


def flat_scan(x_max: float = 25.0, half_width: float = 6.0, step: float = 0.2) -> np.ndarray:
    xs, ys = np.arange(-4.0, x_max, step), np.arange(-half_width, half_width, step)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, GROUND_Z)])


def test_ids_of():
    assert sorted(ids_of(ID_TO_NAME, ("tree", "person", "lava"))) == [TREE, PERSON]


# --- hard obstacles ---


def test_hard_obstacle_points_are_filtered():
    points_flu = np.array([
        [10.0, 2.0, GROUND_Z + 0.5],  # tree trunk: kept
        [10.0, -2.0, GROUND_Z + 4.0],  # canopy above vehicle height: dropped
        [0.5, 0.0, GROUND_Z + 0.5],  # on the vehicle: dropped
        [10.0, 0.0, GROUND_Z],  # grass: not a hard class
        [50.0, 0.0, GROUND_Z],  # outside the map: dropped
    ])
    labels = np.array([TREE, TREE, PERSON, GRASS, TREE])
    ground = np.full(SPEC.shape, GROUND_Z)
    xy = hard_obstacle_xy(flu_to_lidar(points_flu), labels, [TREE, PERSON], ground, SPEC)
    assert xy.tolist() == [[10.0, 2.0]]


def test_hard_obstacle_points_lying_flat_can_be_dropped():
    points_flu = np.array([[10.0, 2.0, GROUND_Z + 0.5], [8.0, 1.0, GROUND_Z + 0.05]])  # a person, and "person" on the ground
    labels = np.array([PERSON, PERSON])
    ground = np.full(SPEC.shape, GROUND_Z)
    assert len(hard_obstacle_xy(flu_to_lidar(points_flu), labels, [PERSON], ground, SPEC)) == 2  # approach 1: kept
    xy = hard_obstacle_xy(flu_to_lidar(points_flu), labels, [PERSON], ground, SPEC, min_height=0.3)
    assert xy.tolist() == [[10.0, 2.0]]


def test_distance_to_obstacles():
    path = np.array([[0.0, 0.0], [4.0, 0.0]])
    assert distance_to_obstacles(path, np.array([[4.0, 3.0]])) == pytest.approx((4.0, 3.0))  # (mean of 5 and 3, min 3)
    assert distance_to_obstacles(path, np.zeros((0, 2))) is None


def test_distance_to_many_obstacles_uses_the_nearest():
    far = np.column_stack([np.full(10000, 100.0), np.arange(10000.0)])
    near = np.vstack([far, [[0.0, 1.0]]])
    assert distance_to_obstacles(np.array([[0.0, 0.0]]), near) == pytest.approx((1.0, 1.0))


def test_goal_is_blocked():
    assert goal_is_blocked((20.0, 0.0), np.array([[20.5, 1.0]]))
    assert not goal_is_blocked((20.0, 0.0), np.array([[23.0, 0.0]]))
    assert not goal_is_blocked((20.0, 0.0), np.zeros((0, 2)))


# --- roughness ---


def test_flat_path_has_no_slope():
    path = [SPEC.cell_index(x, 0.0) for x in np.arange(1.0, 10.0, 0.25)]
    slope, bumps = path_roughness(path, np.full(SPEC.shape, GROUND_Z), np.zeros(SPEC.shape), SPEC)
    assert slope == pytest.approx(0.0) and bumps == pytest.approx(0.0)


def test_ten_percent_slope():
    xs = SPEC.x_min + (np.arange(SPEC.shape[0]) + 0.5) * SPEC.resolution
    ground = np.repeat((0.1 * xs)[:, None], SPEC.shape[1], axis=1)  # rises 10 cm per meter forward
    path = [SPEC.cell_index(x, 0.0) for x in np.arange(1.0, 10.0, 0.25)]
    bumps = np.full(SPEC.shape, np.nan)
    bumps[path[0]] = 0.2
    slope, mean_bump = path_roughness(path, ground, bumps, SPEC)
    assert slope == pytest.approx(10.0)
    assert mean_bump == pytest.approx(0.2)  # NaN cells are skipped


# --- all metrics for a real plan ---


def plan_on_flat_ground():
    points = flat_scan()
    return run_geometric_baseline(flu_to_lidar(points), np.full(len(points), GRASS, dtype=np.uint16))


def truth_for(plan, labels_along_path):
    labels = np.array(labels_along_path)
    return PathTruth(labels=labels, source=np.full(len(labels), SOURCE_LIDAR))


def test_compute_path_metrics_shares_and_clearance():
    plan = plan_on_flat_ground()
    n = len(plan.path)
    labels = [PUDDLE] * (n // 4) + [MUD] * (n // 4) + [BUSH] * (n // 4)
    labels += [GRASS] * (n - len(labels) - 2) + [0, 0]
    metrics = compute_path_metrics(plan, truth_for(plan, labels), np.array([[10.0, 3.0]]), ID_TO_NAME)
    assert metrics.found and metrics.reached and metrics.success and not metrics.goal_blocked
    assert metrics.share_water == pytest.approx((n // 4) / n)
    assert metrics.share_mud == pytest.approx((n // 4) / n)
    assert metrics.share_compliant == pytest.approx((n - 2 * (n // 4) - 2) / n)  # bush + grass
    assert metrics.share_noncompliant == 0.0
    assert metrics.share_unseen == pytest.approx(2 / n)
    assert metrics.hard_dist_min_m == pytest.approx(3.0, abs=0.2)
    assert metrics.slope_cm_per_m == pytest.approx(0.0, abs=0.5)


def test_goal_blocked_counts_as_success():
    plan = plan_on_flat_ground()
    person_at_target = np.array([[20.0, 0.5]])
    metrics = compute_path_metrics(plan, truth_for(plan, [GRASS] * len(plan.path)), person_at_target, ID_TO_NAME)
    assert metrics.goal_blocked and metrics.success
    assert PathMetrics(True, False, True, 2.0, *[None] * 11).success  # stops at the obstacle: fine
    assert not PathMetrics(True, False, True, 4.4, *[None] * 11).success  # stops far before it: not fine
    assert not PathMetrics(True, False, False, 2.0, *[None] * 11).success  # nothing in the way


# --- the BaselineResult helpers (roughness, hard-obstacle distance) ---


def test_baseline_result_roughness_and_hard_distance():
    plan = plan_on_flat_ground()
    assert plan.mean_roughness_m == pytest.approx(0.0, abs=0.05)
    assert plan.hard_obstacle_distance([TREE]) is None  # no tree cells on this map
