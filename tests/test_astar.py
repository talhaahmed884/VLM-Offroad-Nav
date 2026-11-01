"""Tests for the A* planner and its grid helpers. No dataset needed."""

import math

import numpy as np

from offroad.mapping.elevation import GridSpec
from offroad.planning.astar import astar, choose_goal, clear_footprint, cost_to_come, path_length

INF = math.inf


def test_straight_line_on_uniform_grid():
    cost = np.ones((5, 5))
    path, total = astar(cost, (0, 2), (4, 2))
    assert path == [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    assert math.isclose(total, 4.0)


def test_diagonal_move_costs_sqrt2():
    cost = np.ones((3, 3))
    path, total = astar(cost, (0, 0), (2, 2))
    assert path == [(0, 0), (1, 1), (2, 2)]
    assert math.isclose(total, 2 * math.sqrt(2))


def test_goes_through_gap_in_wall():
    cost = np.ones((5, 5))
    cost[2, :] = INF
    cost[2, 4] = 1.0  # the only gap
    path, _ = astar(cost, (0, 0), (4, 0))
    assert (2, 4) in path
    assert all(np.isfinite(cost[c]) for c in path)


def test_returns_none_when_goal_is_cut_off():
    cost = np.ones((5, 5))
    cost[2, :] = INF
    assert astar(cost, (0, 0), (4, 0)) is None


def test_returns_none_when_start_or_goal_blocked():
    cost = np.ones((3, 3))
    cost[0, 0] = INF
    assert astar(cost, (0, 0), (2, 2)) is None
    assert astar(cost, (2, 2), (0, 0)) is None


def test_prefers_longer_cheap_route_over_short_expensive_one():
    cost = np.ones((3, 7))
    cost[:2, 3] = 10.0  # expensive band across the direct route, cheap only in row 2
    path, _ = astar(cost, (0, 0), (0, 6))
    assert (2, 3) in path


def test_no_corner_cutting_between_blocked_cells():
    cost = np.ones((2, 2))
    cost[0, 1] = INF
    cost[1, 0] = INF
    assert astar(cost, (0, 0), (1, 1)) is None


def test_start_equals_goal():
    path, total = astar(np.ones((2, 2)), (1, 1), (1, 1))
    assert path == [(1, 1)]
    assert total == 0.0


SPEC = GridSpec(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, resolution=0.5)  # 8 x 8, origin at a cell corner


def test_clear_footprint_frees_cells_near_origin_only():
    cost = np.full(SPEC.shape, INF)
    cleared = clear_footprint(cost, SPEC, radius=0.5)
    # The four cells touching the origin have centers 0.35 m away; the next ring is farther than 0.5 m.
    assert np.isfinite(cleared).sum() == 4
    assert cleared[3, 3] == 1.0 and cleared[4, 4] == 1.0
    assert np.isinf(cost).all()  # the input is not modified


def test_cost_to_come_stops_at_a_wall():
    cost = np.ones(SPEC.shape)
    cost[4, :] = INF  # wall across the whole grid
    reachable = np.isfinite(cost_to_come(cost, (0, 0)))
    assert reachable[3, 7]
    assert not reachable[4].any()
    assert not reachable[5:].any()


def test_cost_to_come_respects_corner_rule_and_blocked_start():
    cost = np.ones((2, 2))
    cost[0, 1] = INF
    cost[1, 0] = INF
    assert np.isinf(cost_to_come(cost, (0, 0))[1, 1])  # same corner rule as astar
    assert np.isinf(cost_to_come(cost, (0, 1))).all()  # start itself blocked


def test_cost_to_come_matches_astar_cost():
    cost = np.ones((5, 5))
    cost[2, :] = INF
    cost[2, 4] = 1.0
    cost[0, 2] = 4.0  # an expensive cell to make it less trivial
    _, astar_total = astar(cost, (0, 0), (4, 0))
    assert math.isclose(cost_to_come(cost, (0, 0))[4, 0], astar_total)


def test_choose_goal_stops_in_front_of_wall():
    cost = np.ones(SPEC.shape)
    cost[4, :] = INF
    # The target (1.25, 1.25) is the center of cell (6, 6), behind the wall. The goal is in
    # the last row before the wall, in the same column.
    assert choose_goal(cost, SPEC, (0, 0), 1.25, 1.25, cost_weight=0.0) == (3, 6)
    assert choose_goal(cost, SPEC, (0, 0), 1.25, 1.25, cost_weight=0.1) == (3, 6)
    # With no wall, the target cell itself.
    assert choose_goal(np.ones(SPEC.shape), SPEC, (0, 0), 1.25, 1.25, cost_weight=0.1) == (6, 6)


def test_choose_goal_skips_long_loop_for_small_gain():
    # 10 x 10 grid of 1 m cells. Row 7 is a wall with a gap only at the far end (column 9), so
    # the target cell (8, 0) is reachable only by a ~22-cost loop through the gap, while
    # (6, 0), 2 m short of the target, costs 6 to reach.
    spec = GridSpec(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, resolution=1.0)
    cost = np.ones(spec.shape)
    cost[7, :9] = INF
    assert choose_goal(cost, spec, (0, 0), 8.5, 0.5, cost_weight=0.0) == (8, 0)  # closest, whatever it costs
    assert choose_goal(cost, spec, (0, 0), 8.5, 0.5, cost_weight=0.5) == (6, 0)  # the loop isn't worth 2 m


def test_choose_goal_none_when_start_blocked():
    cost = np.ones(SPEC.shape)
    cost[0, 0] = INF
    assert choose_goal(cost, SPEC, (0, 0), 1.25, 1.25, cost_weight=0.1) is None


def test_path_length_in_meters():
    assert math.isclose(path_length([(0, 0), (1, 0), (2, 1)], SPEC), 0.5 + 0.5 * math.sqrt(2))
