"""Tests for the simulator's per-frame planning (no MAVS needed)."""

import numpy as np
import pytest

from offroad.nav.candidates import Candidate
from offroad.sim.planning import cheapest_index, plan_frame, straightest_index, trail_ahead_flu, trail_direction_word
from offroad.sim.rig import RigSpec, camera_calibration, to_lidar_axes


def flat_ground_scan(with_wall_at: float | None = None) -> np.ndarray:
    """A dense synthetic scan of flat ground 1.2 m below the LiDAR (MAVS FLU axes), optionally a wall ahead."""
    xs, ys = np.meshgrid(np.arange(-4.0, 34.0, 0.15), np.arange(-14.0, 14.0, 0.15), indexing="ij")
    ground = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -1.2)])
    if with_wall_at is None:
        return ground
    wy, wz = np.meshgrid(np.arange(-14.0, 14.0, 0.1), np.arange(-1.2, 1.5, 0.1), indexing="ij")
    wall = np.column_stack([np.full(wy.size, with_wall_at), wy.ravel(), wz.ravel()])
    return np.vstack([ground, wall])


def test_flat_ground_gives_straight_full_length_candidates():
    rig = RigSpec()
    plan = plan_frame(to_lidar_axes(flat_ground_scan()), camera_calibration(rig), rig.image_size[0])
    assert 1 <= len(plan.candidates) <= 4
    assert max(c.length_m for c in plan.candidates) > 14.0
    k = straightest_index(plan.candidates)
    assert abs(plan.candidates[k].end_bearing_deg) < 5.0
    # Starts at the vehicle and stays within half a cell of straight ahead (not 12.5 cm to the left).
    straight = plan.candidates[k].xy
    assert straight[0] == pytest.approx([0.0, 0.0])
    assert abs(straight[12, 1]) < 0.05 and abs(straight[len(straight) // 2, 1]) < 0.1
    unshaped = plan_frame(to_lidar_axes(flat_ground_scan()), camera_calibration(rig), rig.image_size[0],
                          straighten=False)
    assert abs(unshaped.candidates[straightest_index(unshaped.candidates)].xy[20, 1]) < 0.05  # shifted instead


def test_straightened_candidates_turn_toward_their_goal_from_the_start():
    rig = RigSpec()
    plan = plan_frame(to_lidar_axes(flat_ground_scan()), camera_calibration(rig), rig.image_size[0])
    for c in plan.candidates:
        if abs(c.end_bearing_deg) > 5:  # a side path is already off to its side at 3 m, not only at the end
            assert np.sign(c.xy[12, 1]) == np.sign(c.end_bearing_deg) and abs(c.xy[12, 1]) > 0.2


def test_a_wall_across_the_trail_shortens_every_candidate():
    rig = RigSpec()
    plan = plan_frame(to_lidar_axes(flat_ground_scan(with_wall_at=8.0)), camera_calibration(rig), rig.image_size[0])
    assert all(c.xy[:, 0].max() < 8.0 for c in plan.candidates)


def test_overhanging_foliage_does_not_block_but_a_standing_post_does():
    rig = RigSpec()
    ground = flat_ground_scan()
    # A canopy hanging 1.4-1.6 m above the ground (z = 0.2..0.4 at the LiDAR) across the trail at 8-10 m.
    cx, cy = np.meshgrid(np.arange(8.0, 10.0, 0.1), np.arange(-6.0, 6.0, 0.1), indexing="ij")
    canopy = np.column_stack([cx.ravel(), cy.ravel(), 0.2 + 0.2 * (np.arange(cx.size) % 2)])
    plan = plan_frame(to_lidar_axes(np.vstack([ground, canopy])), camera_calibration(rig), rig.image_size[0])
    assert max(c.length_m for c in plan.candidates) > 14.0  # drives under it
    blocked = plan_frame(to_lidar_axes(np.vstack([ground, canopy])), camera_calibration(rig), rig.image_size[0],
                         ignore_overhangs=False)
    assert all(c.xy[:, 0].max() < 8.0 for c in blocked.candidates)  # the old rule: the canopy blocks
    # A 1.7 m post straight ahead at 6 m stands on the ground: still blocked, paths go round it.
    pz = np.arange(-1.2, 0.5, 0.05)
    post = np.column_stack([np.full(pz.size, 6.0), np.zeros(pz.size), pz])
    plan = plan_frame(to_lidar_axes(np.vstack([ground, post])), camera_calibration(rig), rig.image_size[0])
    for c in plan.candidates:
        near = np.abs(c.xy[:, 0] - 6.0) < 0.2
        assert np.all(np.abs(c.xy[near, 1]) > 0.5)


def test_baseline_picks():
    cands = [Candidate(np.zeros((2, 2)), 20.0, 2.0), Candidate(np.zeros((2, 2)), -3.0, 1.5),
             Candidate(np.zeros((2, 2)), -25.0, 1.0)]
    for c, bearing in zip(cands, (20.0, -3.0, -25.0)):  # end bearing comes from xy: set it directly
        c.xy = np.array([[0.0, 0.0], [10 * np.cos(np.radians(bearing)), 10 * np.sin(np.radians(bearing))]])
    assert straightest_index(cands) == 1 and cheapest_index(cands) == 2
    assert straightest_index([]) is None and cheapest_index([]) is None


def test_trail_ahead_and_word():
    # A trail along +x that bends left after 20 m; the vehicle stands at x=5 facing +x.
    straight = np.column_stack([np.arange(0, 21.0, 0.5), np.zeros(42)])
    t = np.radians(np.arange(0, 90, 2.0))
    bend = np.column_stack([20 + 30 * np.sin(t), 30 * (1 - np.cos(t))])
    trail = np.vstack([straight, bend])
    ahead = trail_ahead_flu(trail, np.array([5.0, 0.0]), 0.0)
    assert ahead[0] == pytest.approx([0.0, 0.0]) and ahead[1] == pytest.approx([0.0, 0.0])
    assert trail_direction_word(trail, np.array([5.0, 0.0]), 0.0) in ("slight left", "bear left")
    assert trail_direction_word(straight, np.array([0.0, 0.0]), 0.0) == "straight"
    # Facing the other way (+y), the same trail ahead lies to the right.
    assert "right" in trail_direction_word(trail, np.array([5.0, 0.0]), np.pi / 2)
