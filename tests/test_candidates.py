"""Tests for candidate generation (strict and relaxed cost grids). No dataset needed."""

import numpy as np
import pytest

from offroad.mapping.costmap import LETHAL, CostMap, CostParams, drivable_cost, inflate
from offroad.mapping.elevation import GridSpec
from offroad.nav.candidates import (
    CANDIDATE_LENGTH_M,
    MAX_CANDIDATES,
    MIN_SEPARATION_M,
    VEGETATION_COST,
    camera_half_fov_deg,
    generate_candidates,
    in_view_share,
    max_separation,
    relaxed_cost,
    resample_and_cut,
    smooth_path,
    strict_cost,
)

SPEC = GridSpec(x_min=-5.0, x_max=25.0, y_min=-12.0, y_max=12.0, resolution=0.25)
PARAMS = CostParams()


def make_costmap(height: np.ndarray, spec: GridSpec = SPEC, params: CostParams = PARAMS) -> CostMap:
    """A CostMap built from obstacle heights the same way as `geometric_cost_map` does."""
    obstacle = ~np.isnan(height) & (np.nan_to_num(height) > params.max_step)
    cost = drivable_cost(height, params)
    cost[inflate(obstacle, round(params.inflation_radius / spec.resolution))] = LETHAL
    return CostMap(spec=spec, cost=cost, ground_z=np.zeros(spec.shape), obstacle_height=height,
                   obstacle=obstacle, unknown=np.isnan(height))


def rows_at(x_from: float, x_to: float) -> slice:
    return slice(SPEC.cell_index(x_from, 0.0)[0], SPEC.cell_index(x_to, 0.0)[0])


def cols_at(y_from: float, y_to: float) -> slice:
    return slice(SPEC.cell_index(0.0, y_from)[1], SPEC.cell_index(0.0, y_to)[1])


# --- cost grids ---


def test_relaxed_cost_blocks_only_hard_obstacles():
    height = np.zeros(SPEC.shape)
    height[rows_at(8, 9), cols_at(2, 4)] = 0.8  # a bush
    height[rows_at(8, 9), cols_at(-4, -3)] = 3.0  # a tree
    costmap = make_costmap(height)
    relaxed = relaxed_cost(costmap)

    bush = SPEC.cell_index(8.5, 3.0)
    tree = SPEC.cell_index(8.5, -3.5)
    next_to_bush = SPEC.cell_index(8.5, 1.5)
    next_to_tree = SPEC.cell_index(8.5, -2.5)
    assert np.isinf(strict_cost(costmap)[bush]) and np.isinf(strict_cost(costmap)[next_to_bush])
    assert relaxed[bush] == VEGETATION_COST
    assert np.isfinite(relaxed[next_to_bush])  # vegetation isn't inflated
    assert np.isinf(relaxed[tree]) and np.isinf(relaxed[next_to_tree])  # trees are, as usual
    assert relaxed[SPEC.cell_index(15.0, 0.0)] == 1.0  # flat ground is untouched


# --- path helpers ---


def test_smooth_path_keeps_ends_and_straight_lines():
    line = np.column_stack([np.arange(10.0), np.zeros(10)])
    assert np.allclose(smooth_path(line), line)
    zigzag = np.column_stack([np.arange(10.0), np.tile([0.0, 0.5], 5)])
    smoothed = smooth_path(zigzag)
    assert np.allclose(smoothed[[0, -1]], zigzag[[0, -1]])
    assert np.ptp(smoothed[2:-2, 1]) < np.ptp(zigzag[:, 1])


def test_resample_and_cut():
    path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    cut = resample_and_cut(path, 0.25, 15.0)
    assert np.allclose(cut[-1], [10.0, 5.0])
    assert np.allclose(np.linalg.norm(np.diff(cut, axis=0), axis=1), 0.25)
    short = resample_and_cut(np.array([[0.0, 0.0], [1.1, 0.0]]), 0.25, 15.0)
    assert np.allclose(short[-1], [1.1, 0.0])  # ends exactly at the end of a shorter path


def test_max_separation_sees_paths_that_split_late():
    straight = resample_and_cut(np.array([[0.0, 0.0], [15.0, 0.0]]), 0.25, 15.0)
    split = resample_and_cut(np.array([[0.0, 0.0], [10.0, 0.0], [14.0, 3.0]]), 0.25, 15.0)
    assert max_separation(straight, straight) == 0.0
    assert max_separation(straight, split) > 2.0  # 3 m apart at the end, though the first 10 m are shared


def test_camera_view():
    K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])
    left, right = camera_half_fov_deg(K, 1000)
    assert left == pytest.approx(right) == pytest.approx(np.degrees(np.arctan(0.5)))
    ahead = resample_and_cut(np.array([[0.0, 0.0], [15.0, 0.0]]), 0.25, 15.0)
    sideways = resample_and_cut(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 12.0]]), 0.25, 15.0)
    assert in_view_share(ahead, (left, right)) == 1.0
    assert in_view_share(sideways, (left, right)) == 0.0  # after 5 m it runs off to the left
    assert in_view_share(ahead[:10], (left, right)) == 1.0  # nothing beyond 5 m to check


# --- generate_candidates ---


def test_open_field_gives_distinct_full_candidates_left_to_right():
    candidates = generate_candidates(np.ones(SPEC.shape), SPEC)
    assert len(candidates) == MAX_CANDIDATES
    assert all(c.full_length and c.length_m <= CANDIDATE_LENGTH_M + 1e-6 for c in candidates)
    bearings = [c.end_bearing_deg for c in candidates]
    assert bearings == sorted(bearings, reverse=True)  # left (+) to right (-)
    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            assert max_separation(candidates[a].xy, candidates[b].xy) >= MIN_SEPARATION_M
    assert all(np.allclose(c.xy[0], 0.0) for c in candidates)


def test_candidates_stay_in_the_camera_view():
    fov = (12.0, 12.0)
    candidates = generate_candidates(np.ones(SPEC.shape), SPEC, half_fov_deg=fov)
    assert candidates
    assert all(in_view_share(c.xy, fov) >= 0.8 for c in candidates)
    assert all(abs(c.end_bearing_deg) < 15.0 for c in candidates)


def test_narrow_corridor_gives_one_candidate():
    cost = np.full(SPEC.shape, LETHAL)
    cost[:, cols_at(-0.75, 0.75)] = 1.0  # a 1.5 m wide lane straight ahead
    candidates = generate_candidates(cost, SPEC)
    assert len(candidates) == 1
    assert candidates[0].full_length and abs(candidates[0].end_bearing_deg) < 3.0


def test_relaxed_grid_lets_candidates_cross_a_vegetation_wall():
    height = np.zeros(SPEC.shape)
    height[rows_at(8, 9), :] = 0.8  # a line of bushes across the whole map, 8 m ahead
    costmap = make_costmap(height)
    strict = generate_candidates(strict_cost(costmap), SPEC)
    relaxed = generate_candidates(relaxed_cost(costmap), SPEC)
    assert strict and all(not c.full_length and c.xy[:, 0].max() < 8.0 for c in strict)  # they stop in front of it
    assert relaxed and all(c.full_length for c in relaxed)


def test_no_candidates_when_everything_is_blocked():
    # Only the vehicle's own footprint (2 m around it) is cleared: shorter than the minimum length.
    assert generate_candidates(np.full(SPEC.shape, LETHAL), SPEC) == []
