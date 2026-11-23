"""Unit tests for the geometric + soft-obstacle candidate trajectory generator."""

import numpy as np

from offroad.mapping.costmap import LETHAL
from offroad.mapping.elevation import GridSpec
from offroad.planning.candidates import (
    CANDIDATE_COLORS,
    CandidateParams,
    generate_candidates,
    lateral_biased_cost,
    path_mean_distance,
    soft_obstacle_cost,
)


def _flat_spec() -> GridSpec:
    return GridSpec(x_min=-2.0, x_max=22.0, y_min=-10.0, y_max=10.0, resolution=0.5)


def test_lateral_biased_cost_preserves_lethal_and_favors_offset() -> None:
    spec = _flat_spec()
    cost = np.ones(spec.shape, dtype=float)
    cost[10, 10] = LETHAL

    biased = lateral_biased_cost(cost, spec, y_offset_m=3.0, bias_weight=0.5)
    assert not np.isfinite(biased[10, 10])

    _, j_target = spec.cell_index(5.0, 3.0)
    _, j_far = spec.cell_index(5.0, -3.0)
    assert biased[10, j_target] < biased[10, j_far]


def test_generate_candidates_flat_ground_returns_distinct_paths() -> None:
    spec = _flat_spec()
    cost = np.ones(spec.shape, dtype=float)

    candidates = generate_candidates(cost, spec, goal_distance=15.0)
    assert len(candidates) == 4
    assert [c.color for c in candidates] == list(CANDIDATE_COLORS)
    assert candidates[0].lateral_offset_m == 0.0

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            dist = path_mean_distance(candidates[i].path_xy(), candidates[j].path_xy())
            assert dist >= CandidateParams().min_separation_m


def test_generate_candidates_deduplicates_narrow_corridor() -> None:
    spec = _flat_spec()
    cost = np.full(spec.shape, LETHAL, dtype=float)

    _, j_lo = spec.cell_index(0.0, -0.25)
    _, j_hi = spec.cell_index(0.0, 0.25)
    cost[:, j_lo : j_hi + 1] = 1.0

    candidates = generate_candidates(cost, spec, goal_distance=15.0)
    assert len(candidates) == 1
    assert candidates[0].color == "red"
    assert candidates[0].reached


def test_soft_obstacle_cost_unblocks_low_bush_but_keeps_tall_tree_lethal() -> None:
    spec = _flat_spec()
    cost = np.ones(spec.shape, dtype=float)
    height = np.zeros(spec.shape, dtype=float)

    # Low bush (1.0 m) at x=8, y=0; tall tree (3.0 m) at x=8, y=5.
    bi, bj = spec.cell_index(8.0, 0.0)
    ti, tj = spec.cell_index(8.0, 5.0)
    cost[bi, bj] = LETHAL
    height[bi, bj] = 1.0
    cost[ti, tj] = LETHAL
    height[ti, tj] = 3.0

    soft = soft_obstacle_cost(cost, height, spec, max_soft_height_m=1.5, soft_cost=6.0)
    assert soft[bi, bj] == 6.0
    assert not np.isfinite(soft[ti, tj])


def test_generate_candidates_adds_soft_obstacle_path_through_bush_wall() -> None:
    spec = _flat_spec()
    cost = np.ones(spec.shape, dtype=float)
    height = np.zeros(spec.shape, dtype=float)

    # Full-width 1.0 m bush wall at x in [7, 9]: geometric planner stops short,
    # soft-obstacle candidate drives straight through to the 15 m goal.
    i0, _ = spec.cell_index(7.0, 0.0)
    i1, _ = spec.cell_index(9.0, 0.0)
    cost[i0 : i1 + 1, :] = LETHAL
    height[i0 : i1 + 1, :] = 1.0

    candidates = generate_candidates(cost, spec, obstacle_height=height, goal_distance=15.0)
    soft_cands = [c for c in candidates if c.kind == "soft_obstacle"]
    assert len(soft_cands) == 1
    assert soft_cands[0].reached
    assert not candidates[0].reached