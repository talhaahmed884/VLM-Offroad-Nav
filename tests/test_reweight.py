"""Tests for re-weighting the cost map with VLM hints. No dataset needed."""

import numpy as np
import pytest

from offroad.fusion.grid_lookup import NO_CELL, GridLookup
from offroad.fusion.reweight import FusionParams, paint_hints, reweight_cost_map
from offroad.mapping.costmap import CostMap, CostParams, drivable_cost, inflate
from offroad.mapping.elevation import GridSpec
from offroad.vlm.grid import ImageGrid

# 1 m cells so the numbers are easy: x 0..40 m ahead, y -5..5 m. Inflation 0.75 m -> 1 cell.
SPEC = GridSpec(x_min=0.0, x_max=40.0, y_min=-5.0, y_max=5.0, resolution=1.0)
GRID = ImageGrid(height=80, width=120)
PARAMS = CostParams()

# Which map rows (meters ahead) each image grid cell covers, all across the full width.
FOOTPRINTS = {"H7": range(5, 10), "G7": range(10, 15), "D7": range(30, 35)}


def make_lookup() -> GridLookup:
    cell_id = np.full(SPEC.shape, NO_CELL)
    for label, xs in FOOTPRINTS.items():
        row, col = GRID.parse(label)
        cell_id[list(xs), :] = row * GRID.cols + col
    return GridLookup(SPEC, GRID, cell_id, np.full((*SPEC.shape, 2), np.nan), np.zeros(SPEC.shape, dtype=bool))


def make_costmap(height: np.ndarray) -> CostMap:
    """A cost map built the same way as `geometric_cost_map`, from obstacle heights."""
    unknown = np.isnan(height)
    obstacle = ~unknown & (height > PARAMS.max_step)
    cost = drivable_cost(height, PARAMS)
    cost[inflate(obstacle, 1)] = np.inf
    return CostMap(SPEC, cost, np.zeros(SPEC.shape), height, obstacle, unknown)


def hint(cell, terrain, level):
    return {"cell": cell, "terrain": terrain, "traversability": level}


def smooth() -> np.ndarray:
    return np.zeros(SPEC.shape)


LOOKUP = make_lookup()


def test_drivable_cost():
    height = np.array([[0.0, 0.15, 0.3, 1.0, np.nan]])
    assert drivable_cost(height, PARAMS).tolist() == [[1.0, 3.0, 5.0, 5.0, 3.0]]


def test_no_hints_leaves_the_cost_map_unchanged():
    height = smooth()
    height[12, 4] = 1.0
    height[20, 2] = np.nan
    costmap = make_costmap(height)
    assert np.array_equal(reweight_cost_map(costmap, LOOKUP, []).cost, costmap.cost)


@pytest.mark.parametrize("level, expected", [("preferred", 0.7), ("ok", 1.0), ("caution", 2.0), ("avoid", 5.0)])
def test_level_multipliers(level, expected):
    fused = reweight_cost_map(make_costmap(smooth()), LOOKUP, [hint("H7", "mud", level)])
    assert fused.cost[7, 5] == pytest.approx(expected)
    assert fused.cost[12, 5] == 1.0  # G7 has no hint


def test_avoid_on_unknown_ground():
    height = smooth()
    height[7, :] = np.nan
    fused = reweight_cost_map(make_costmap(height), LOOKUP, [hint("H7", "water", "avoid")])
    assert fused.cost[7, 5] == pytest.approx(PARAMS.unknown_cost * 5.0)


def test_low_grass_obstacle_is_unblocked():
    height = smooth()
    height[10:15, :] = 1.0  # a 1 m "wall" of tall grass across the whole map in G7
    costmap = make_costmap(height)
    assert np.isinf(costmap.cost[12, 5]) and np.isinf(costmap.cost[9, 5])  # blocked, plus margin in H7

    fused = reweight_cost_map(costmap, LOOKUP, [hint("G7", "tall_grass_or_bush", "ok")])
    assert fused.unblocked[10:15, :].all()
    assert fused.cost[12, 5] == pytest.approx(1.0 + PARAMS.roughness_weight)  # rough, not free
    assert fused.cost[9, 5] == 1.0  # the safety margin went away with the obstacle


def test_caution_grass_is_unblocked_but_dearer():
    height = smooth()
    height[10:15, :] = 1.0
    fused = reweight_cost_map(make_costmap(height), LOOKUP, [hint("G7", "short_grass", "caution")])
    assert fused.cost[12, 5] == pytest.approx((1.0 + PARAMS.roughness_weight) * 2.0)


@pytest.mark.parametrize("obstacle_height, terrain, level", [
    (1.6, "tall_grass_or_bush", "ok"),     # too tall to trust
    (1.0, "tree_or_log", "ok"),            # not vegetation the vehicle can push through
    (1.0, "rocks", "preferred"),
    (1.0, "tall_grass_or_bush", "avoid"),  # the VLM itself says avoid
])
def test_cases_that_stay_blocked(obstacle_height, terrain, level):
    height = smooth()
    height[10:15, :] = obstacle_height
    fused = reweight_cost_map(make_costmap(height), LOOKUP, [hint("G7", terrain, level)])
    assert not fused.unblocked.any()
    assert np.isinf(fused.cost[12, 5])


def test_impassable_blocks_with_a_margin():
    fused = reweight_cost_map(make_costmap(smooth()), LOOKUP, [hint("G7", "person_or_vehicle", "impassable")])
    assert np.isinf(fused.cost[10:15, :]).all()
    assert np.isinf(fused.cost[9, 5]) and np.isinf(fused.cost[15, 5])  # inflated by one cell
    assert fused.cost[8, 5] == 1.0


def test_hints_beyond_the_distance_limit_are_ignored():
    costmap = make_costmap(smooth())
    fused = reweight_cost_map(costmap, LOOKUP, [hint("D7", "water", "avoid")])  # 30-35 m ahead
    assert np.array_equal(fused.cost, costmap.cost)
    longer = reweight_cost_map(costmap, LOOKUP, [hint("D7", "water", "avoid")], FusionParams(max_hint_distance=40.0))
    assert longer.cost[32, 5] == pytest.approx(5.0)


def test_paint_hints():
    hints = paint_hints(LOOKUP, [hint("H7", "water", "avoid")], max_distance=25.0)
    assert hints.hinted[5:10, :].all() and not hints.hinted[10:, :].any()
    assert hints.has_level(("avoid",))[7, 0] and hints.has_terrain(("water",))[7, 0]
    assert not hints.has_level(("ok",))[7, 0]
