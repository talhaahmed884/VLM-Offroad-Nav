"""Tests for the path follower (no MAVS needed)."""

import numpy as np
import pytest

from offroad.sim.control import (
    FollowerParams,
    follow,
    lookahead_point,
    pursuit_curvature,
    speed_command,
    steering_command,
    to_vehicle,
    to_world,
)


def test_world_and_vehicle_frames_round_trip():
    path = np.array([[1.0, 0.0], [5.0, 2.0]])
    world = to_world(path, np.array([10.0, -3.0]), np.pi / 2)  # facing +y
    assert world[0] == pytest.approx([10.0, -2.0])  # 1 m ahead = +y
    assert world[1] == pytest.approx([8.0, 2.0])  # 2 m left of +y heading = -x
    assert to_vehicle(world, np.array([10.0, -3.0]), np.pi / 2) == pytest.approx(path)


def test_lookahead_and_curvature():
    straight = np.column_stack([np.arange(0, 15.1, 0.25), np.zeros(61)])
    assert lookahead_point(straight, 3.0) == pytest.approx([3.0, 0.0])
    assert pursuit_curvature(np.array([3.0, 0.0])) == 0.0
    # A point 3 m ahead and 1 m left: the arc through it bends left (+).
    assert pursuit_curvature(np.array([3.0, 1.0])) == pytest.approx(2 * 1.0 / 10.0)
    assert lookahead_point(straight - [20.0, 0.0], 3.0) is None  # all behind the vehicle
    assert lookahead_point(np.array([[0.5, 0.0], [1.0, 0.0]]), 3.0) == pytest.approx([1.0, 0.0])  # short: its end


def test_steering_and_speed_commands():
    p = FollowerParams(curvature_per_steer=0.5, max_steer=1.0)
    assert steering_command(0.25, p) == pytest.approx(0.5) and steering_command(-5.0, p) == -1.0
    assert steering_command(5.0, FollowerParams()) == FollowerParams().max_steer
    throttle, brake = speed_command(0.0, 1.5, p)
    assert throttle > 0 and brake == 0
    throttle, brake = speed_command(3.0, 1.5, p)
    assert throttle == 0 and brake > 0  # too fast: brake
    assert speed_command(0.1, 0.0, p) == (0.0, 0.0)  # never brake at standstill (it would reverse)


def test_follow_turns_toward_a_left_path_and_finishes():
    p = FollowerParams()
    left = np.column_stack([np.arange(0, 15.1, 0.25), 0.02 * np.arange(0, 15.1, 0.25) ** 2])
    throttle, steering, brake, done = follow(left, np.zeros(2), 0.0, 1.0, p)
    assert steering > 0 and throttle > 0 and not done
    _, _, _, done = follow(left, np.array([30.0, 0.0]), 0.0, 1.0, p)
    assert done
