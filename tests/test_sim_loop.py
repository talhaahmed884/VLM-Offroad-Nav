"""Tests for the trail metrics and the closed loop, with a stand-in world (no MAVS needed)."""

import math

import numpy as np
import pytest

from offroad.sim.choosers import Chooser, Decision, StraightestChooser
from offroad.sim.loop import LoopParams, run_episode, start_pose
from offroad.sim.mavs_world import Capture
from offroad.sim.metrics import cumulative_length, footprint_gaps, is_stuck, project_on_trail
from offroad.sim.rig import RigSpec, camera_calibration, to_lidar_axes

RIG = RigSpec()
CALIB = camera_calibration(RIG)


class FlatWorld:
    """A kinematic stand-in for MavsWorld on flat open ground: speed follows the throttle,
    curvature = 0.105 1/m x steering (as the MAVS MRZR4); the LiDAR sees flat ground."""

    def __init__(self, trail, frozen: bool = False):
        self.trail = np.asarray(trail, dtype=float)
        self.frozen = frozen
        xs, ys = np.meshgrid(np.arange(-4.0, 34.0, 0.25), np.arange(-14.0, 14.0, 0.25), indexing="ij")
        self.scan = to_lidar_axes(np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -1.2)]))

    def place(self, x, y, yaw):
        self.x, self.y, self.yaw, self.v = x, y, yaw, 0.0

    def pose(self):
        return np.array([self.x, self.y, 0.0]), [math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)]

    def heading_and_speed(self):
        return self.yaw, self.v

    def capture(self):
        w, h = RIG.image_size
        return Capture(np.zeros((h, w, 3), np.uint8), self.scan, np.full(len(self.scan), -1),
                       np.array([self.x, self.y, 0.0]), self.yaw)

    def step(self, throttle, steering, braking, dt):
        if self.frozen:
            return
        self.v = max(0.0, self.v + (10.0 * throttle - 2.0 * self.v - 5.0 * braking) * dt)
        self.yaw += 0.105 * steering * self.v * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt


class BlockedSometimes(FlatWorld):
    """FlatWorld whose scan shows a wall 2.5 m ahead (no candidate) at the given capture numbers."""

    def __init__(self, trail, blocked: set[int]):
        super().__init__(trail)
        self.blocked, self.count = blocked, 0
        wy, wz = np.meshgrid(np.arange(-14.0, 14.0, 0.1), np.arange(-1.2, 1.5, 0.1), indexing="ij")
        wall = np.column_stack([np.full(wy.size, 2.5), wy.ravel(), wz.ravel()])
        self.wall_scan = np.vstack([self.scan, to_lidar_axes(wall)])

    def capture(self):
        c = super().capture()
        self.count += 1
        if self.count - 1 in self.blocked:
            c.points_lidar = self.wall_scan
        return c


class AlwaysStop(Chooser):
    name = "stop"

    def decide(self, image, plan, ctx):
        return Decision(None, {"said": "stop"})


def straight_trail(length=30.0):
    x = np.arange(0.0, length + 0.01, 0.5)
    return np.column_stack([x, np.zeros_like(x)])


def test_project_on_trail_and_stuck():
    trail = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert cumulative_length(trail)[-1] == pytest.approx(20.0)
    assert project_on_trail(trail, np.array([4.0, 1.0])) == pytest.approx((4.0, 1.0))  # left = +
    assert project_on_trail(trail, np.array([4.0, -2.0])) == pytest.approx((4.0, -2.0))
    assert project_on_trail(trail, np.array([11.0, 5.0])) == pytest.approx((15.0, -1.0))
    assert not is_stuck([0.0, 5.0], [0.0, 0.0], 10.0, 0.5)  # not long enough to tell
    assert is_stuck([0.0, 5.0, 10.0], [3.0, 3.1, 3.2], 10.0, 0.5)
    assert not is_stuck([0.0, 5.0, 10.0], [3.0, 5.0, 8.0], 10.0, 0.5)


def test_footprint_gaps():
    pts = np.array([[2.0, 0.0], [0.0, 1.5], [3.0, 3.0]])
    gaps = footprint_gaps(pts, np.array([0.3, 0.3, 0.3]), np.zeros(2), 0.0, 3.5, 1.6)
    assert gaps[0] < 0  # 0.25 m beyond the front, radius 0.3: touches
    assert gaps[1] == pytest.approx(1.5 - 0.8 - 0.3)
    assert gaps[2] == pytest.approx(np.hypot(3.0 - 1.75, 3.0 - 0.8) - 0.3)
    turned = footprint_gaps(np.array([[0.0, 2.0]]), np.array([0.3]), np.zeros(2), math.pi / 2, 3.5, 1.6)
    assert turned[0] < 0  # facing +y, the point 2 m ahead is in front of the bumper


def test_the_oracle_follows_a_gentle_bend_that_straightest_leaves():
    from offroad.sim.choosers import OracleChooser

    t = np.radians(np.arange(0, 60, 0.6))
    trail = np.vstack([straight_trail(8.0), np.column_stack([8 + 40 * np.sin(t), 40 * (1 - np.cos(t))])])
    params = LoopParams(start_waypoint=0, max_time_s=40.0)
    ep = run_episode(FlatWorld(trail), StraightestChooser(), CALIB, "flat", params=params)
    assert ep.outcome == "off_trail"
    ep = run_episode(FlatWorld(trail), OracleChooser(trail), CALIB, "flat", params=params)
    assert ep.outcome in ("finished", "time_limit") and ep.summary["max_offset_m"] < 1.5


def test_a_tree_the_lidar_misses_is_a_collision():
    world = FlatWorld(straight_trail(30.0))
    world.obstacles = (np.array([[12.0, 0.0]]), np.array([0.4]))
    ep = run_episode(world, StraightestChooser(), CALIB, "flat", params=LoopParams(start_waypoint=0))
    assert ep.outcome == "collision" and ep.summary["collision_with"] == [12.0, 0.0]
    assert ep.summary["min_gap_m"] < 0


def test_start_pose_faces_along_the_trail():
    x, y, yaw = start_pose(np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]), 0, look_ahead=2)
    assert (x, y) == (0.0, 0.0) and yaw == pytest.approx(math.pi / 2)


def test_straightest_finishes_a_straight_trail():
    world = FlatWorld(straight_trail(30.0))
    ep = run_episode(world, StraightestChooser(), CALIB, "flat", params=LoopParams(start_waypoint=0))
    assert ep.outcome == "finished"
    # On open flat ground nothing marks the trail for the planner, so it only has to stay on it (4 m).
    assert ep.summary["max_offset_m"] < 2.0 and ep.summary["completion"] > 0.8
    assert ep.decisions and all(d["chosen"] is not None for d in ep.decisions)
    assert ep.trajectory.shape[1] == 3 and ep.trajectory[-1, 0] > 20.0


def test_a_short_gap_without_candidates_keeps_the_previous_path():
    params = LoopParams(start_waypoint=0)
    ep = run_episode(BlockedSometimes(straight_trail(30.0), {3, 4}), StraightestChooser(), CALIB, "flat", params=params)
    assert ep.outcome == "finished"
    assert ep.summary["held_decisions"] == 2 and [d["held"] for d in ep.decisions[3:5]] == [True, True]
    ep = run_episode(BlockedSometimes(straight_trail(30.0), {3, 4, 5}), StraightestChooser(), CALIB, "flat",
                     params=params)
    assert ep.outcome == "stopped" and ep.summary["stop_reason"] == "no candidates" and len(ep.decisions) == 6


def test_stop_and_stuck_and_off_trail_end_the_run():
    ep = run_episode(FlatWorld(straight_trail()), AlwaysStop(), CALIB, "flat", params=LoopParams(start_waypoint=0))
    assert ep.outcome == "stopped" and len(ep.decisions) == 1 and ep.decisions[0]["answer"] == {"said": "stop"}
    assert ep.summary["stop_reason"] == "chooser said stop"

    ep = run_episode(FlatWorld(straight_trail(), frozen=True), StraightestChooser(), CALIB, "flat",
                     params=LoopParams(start_waypoint=0, stuck_window_s=3.0))
    assert ep.outcome == "stuck"

    # The trail turns sharply left after 6 m; on open flat ground "straightest" keeps going straight.
    t = np.radians(np.arange(0, 180, 3.0))
    trail = np.vstack([straight_trail(6.0), np.column_stack([6 + 4 * np.sin(t), 4 * (1 - np.cos(t))])])
    ep = run_episode(FlatWorld(trail), StraightestChooser(), CALIB, "flat", params=LoopParams(start_waypoint=0))
    assert ep.outcome == "off_trail"


class NeedsAWord(StraightestChooser):
    """Fails like the VLM's Mode B prompt if a decision comes without a direction word."""

    def decide(self, image, plan, ctx):
        assert ctx.word is not None, f"decision {ctx.number} has no direction word"
        return super().decide(image, plan, ctx)


def test_mode_b_keeps_the_last_word_near_the_trail_end():
    # The trail's last 20 m give no word; the 6.5 batch crashed there (2026-11-27).
    ep = run_episode(FlatWorld(straight_trail(30.0)), NeedsAWord(), CALIB, "flat", mode="B",
                     params=LoopParams(start_waypoint=0))
    assert ep.outcome == "finished"
    assert {d["word"] for d in ep.decisions} == {"straight"}
