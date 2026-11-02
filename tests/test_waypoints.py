"""Tests for the VLM route as a cost corridor. No dataset needed."""

import numpy as np
import pytest

from offroad.fusion.grid_lookup import build_grid_lookup
from offroad.fusion.reweight import FusionParams, reweight_cost_map
from offroad.fusion.waypoints import distance_to_polyline, route_corridor, waypoint_ground_points
from offroad.mapping.costmap import CostMap, CostParams, drivable_cost, inflate
from offroad.mapping.elevation import GridSpec
from offroad.vlm.grid import ImageGrid

SPEC = GridSpec()  # x -5..35, y -15..15, 0.25 m cells
GRID = ImageGrid(height=1200, width=1920)
K = np.array([[2800.0, 0.0, 960.0], [0.0, 2800.0, 600.0], [0.0, 0.0, 1.0]])
T_CAM_FROM_LIDAR = np.eye(4)
T_CAM_FROM_LIDAR[:3, :3] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]  # level camera looking forward
LOOKUP = build_grid_lookup(np.full(SPEC.shape, -1.5), SPEC, T_CAM_FROM_LIDAR, K, GRID)


def make_costmap(height: np.ndarray) -> CostMap:
    params = CostParams()
    unknown = np.isnan(height)
    obstacle = ~unknown & (height > params.max_step)
    cost = drivable_cost(height, params)
    cost[inflate(obstacle, 3)] = np.inf
    return CostMap(SPEC, cost, np.full(SPEC.shape, -1.5), height, obstacle, unknown)


def at(grid: np.ndarray, x: float, y: float):
    return grid[SPEC.cell_index(x, y)]


# --- geometry ---


def test_distance_to_a_segment():
    d = distance_to_polyline(SPEC, [(0.0, 0.0), (10.0, 0.0)])
    assert at(d, 5.1, 2.1) == pytest.approx(2.0, abs=0.2)  # beside the middle
    assert at(d, 13.1, 0.1) == pytest.approx(3.0, abs=0.2)  # past the end: distance to the end point


def test_distance_with_one_or_no_points():
    assert at(distance_to_polyline(SPEC, [(10.0, 0.0)]), 13.1, 0.1) == pytest.approx(3.0, abs=0.2)
    assert np.isinf(distance_to_polyline(SPEC, [])).all()


def test_corridor_starts_at_the_vehicle_and_stops_at_max_distance():
    corridor = route_corridor(SPEC, [(10.0, 0.0), (30.0, 0.0)], half_width=1.0, max_distance=25.0)
    assert at(corridor, 2.1, 0.1)  # between the vehicle and the first waypoint
    assert at(corridor, 20.1, 0.6)
    assert not at(corridor, 20.1, 2.1)  # too far to the side
    assert not at(corridor, 28.1, 0.1)  # beyond max_distance
    assert not route_corridor(SPEC, [], 1.0, 25.0).any()


def test_waypoint_ground_points_skip_the_sky():
    points = waypoint_ground_points(LOOKUP, ["H7", "F7", "A1"])
    assert len(points) == 2
    assert points[0][0] < points[1][0]  # H7 is nearer than F7


# --- in the re-weighted cost map ---


def test_route_corridor_is_cheaper():
    fused = reweight_cost_map(make_costmap(np.zeros(SPEC.shape)), LOOKUP, [], waypoints=["H7", "G7", "F7"])
    x, y = fused.route_xy[-1]
    assert fused.cost[SPEC.cell_index(x, y)] == pytest.approx(FusionParams().corridor_factor)
    assert at(fused.cost, 10.1, 8.1) == 1.0  # off the route
    assert len(fused.route_xy) == 3


def test_route_never_unblocks():
    height = np.zeros(SPEC.shape)
    height[SPEC.cell_index(12.0, 0.0)[0], :] = 1.0  # a wall across the map at 12 m
    fused = reweight_cost_map(make_costmap(height), LOOKUP, [], waypoints=["H7", "G7", "F7"])
    assert np.isinf(at(fused.cost, 12.0, 0.0))


def test_no_waypoints_no_corridor():
    costmap = make_costmap(np.zeros(SPEC.shape))
    fused = reweight_cost_map(costmap, LOOKUP, [], waypoints=[])
    assert not fused.corridor.any() and fused.route_xy == []
    assert np.array_equal(fused.cost, costmap.cost)
