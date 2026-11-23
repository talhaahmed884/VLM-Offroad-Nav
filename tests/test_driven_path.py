"""Tests for the driven path, direction words, and decision frames. No dataset needed."""

import numpy as np
import pytest

from offroad.nav.driven_path import (
    direction_bearing_deg,
    direction_word,
    driven_path_flu,
    path_length,
    point_at_distance,
    poses_relative_to,
    resample_path,
    sample_by_time,
)
from offroad.nav.frames import build_decision_frames, prefer_labelled, speed_ahead


def lidar_pose(x, y, yaw_deg=0.0):
    """T_world_from_lidar for a LiDAR at world (x, y) whose forward axis (LiDAR -x) points at yaw_deg.

    yaw 0 means the vehicle drives toward world +x, so the LiDAR's +x axis points to world -x.
    """
    heading = np.radians(yaw_deg)
    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
    left = np.array([-np.sin(heading), np.cos(heading), 0.0])
    pose = np.eye(4)
    pose[:3, 0] = -forward  # LiDAR +x points backward
    pose[:3, 1] = -left  # LiDAR +y points right
    pose[:3, 2] = [0.0, 0.0, 1.0]
    pose[:3, 3] = [x, y, 0.0]
    return pose


def straight_drive(n=400, step=0.125):
    """Poses driving straight along world +x, `step` m per scan."""
    return np.array([lidar_pose(k * step, 0.0) for k in range(n)])


# --- the driven path ---


def test_relative_pose_of_itself_is_identity():
    poses = straight_drive(10)
    assert np.allclose(poses_relative_to(poses, 3)[3], np.eye(4))


def test_straight_drive_is_straight_ahead_in_flu():
    path = driven_path_flu(straight_drive(), 100, max_distance=20.0)
    assert np.allclose(path[0], 0.0)
    assert path[-1, 0] == pytest.approx(20.0, abs=0.13)  # stops at the first point at or past 20 m
    assert np.allclose(path[:, 1], 0.0) and np.allclose(path[:, 2], 0.0)
    assert np.all(np.diff(path[:, 0]) > 0)


def test_driven_path_turning_left_goes_to_plus_y():
    # Drive along a circle turning left (counter-clockwise), radius 20 m.
    angles = np.radians(np.arange(0, 120, 0.4))
    poses = np.array([lidar_pose(20 * np.sin(a), 20 * (1 - np.cos(a)), np.degrees(a)) for a in angles])
    path = driven_path_flu(poses, 0, max_distance=30.0)
    assert path[-1, 1] > 5.0  # ends up well to the left
    assert direction_word(direction_bearing_deg(path, distance=30.0)) == "bear left"


def test_driven_path_is_seen_from_the_chosen_frame():
    # From frame 200 the earlier part of the drive is behind; the path still starts at the origin.
    path = driven_path_flu(straight_drive(), 200, max_distance=5.0)
    assert np.allclose(path[0], 0.0) and path[-1, 0] > 0


def test_path_ends_at_the_end_of_the_recording():
    path = driven_path_flu(straight_drive(n=50), 40, max_distance=20.0)
    assert path_length(path) == pytest.approx(9 * 0.125)


# --- path helpers ---


def test_point_at_distance_and_resample():
    path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert np.allclose(point_at_distance(path, 15.0), [10.0, 5.0])
    assert point_at_distance(path, 25.0) is None
    assert path_length(path) == pytest.approx(20.0)
    resampled = resample_path(path, 5.0)
    assert len(resampled) == 5 and np.allclose(resampled[-1], [10.0, 10.0])


@pytest.mark.parametrize("bearing, word", [
    (0.0, "straight"), (9.0, "straight"), (-9.0, "straight"),
    (15.0, "slight left"), (-15.0, "slight right"),
    (40.0, "bear left"), (-90.0, "bear right"), (None, None),
])
def test_direction_words(bearing, word):
    assert direction_word(bearing) == word


def test_point_at_exactly_the_path_length():
    # Many short random steps: the total length and the running sum can differ in the last digit.
    rng = np.random.default_rng(0)
    path = np.cumsum(rng.uniform(0.0, 0.2, size=(500, 2)), axis=0)
    assert point_at_distance(path, path_length(path)) is not None
    assert direction_bearing_deg(path) is not None


def test_bearing_uses_the_end_of_a_shorter_path():
    path = np.array([[0.0, 0.0], [30.0, 30.0]])  # 42 m long, 45 deg left
    assert direction_bearing_deg(path, distance=50.0) == pytest.approx(45.0)
    assert direction_bearing_deg(np.array([[0.0, 0.0], [10.0, 0.0]])) is None  # shorter than 20 m


# --- decision frames ---


def test_sample_by_time():
    times = {i: i * 0.1 for i in range(25)}  # 10 Hz for 2.5 s
    assert sample_by_time(times, rate_hz=2.0) == [0, 5, 10, 15, 20]


def test_speed_ahead():
    poses = straight_drive()
    times = np.arange(len(poses)) * 0.1
    assert speed_ahead(poses[:, :3, 3], times, 10) == pytest.approx(1.25)
    assert speed_ahead(poses[:, :3, 3], times, len(poses) - 2) == 0.0  # less than a second left


def test_prefer_labelled_moves_to_a_labelled_neighbour():
    labelled = np.array([True, False, True, False, False, True, False, False, False])
    # 1 -> 2 (next), 3 -> 2 (previous) then merged, 4 -> 5 (next), 7 has no labelled neighbour: stays.
    assert prefer_labelled([0, 1, 3, 4, 7], labelled) == [0, 2, 5, 7]


def test_build_decision_frames_uses_labelled_frames():
    poses = straight_drive(n=300)
    times = np.arange(len(poses)) * 0.1
    labelled = np.arange(len(poses)) % 2 == 1  # only odd frames have label images
    frames, _ = build_decision_frames("00000", poses, times, labelled=labelled)
    assert frames and all(f.frame % 2 == 1 and f.labelled for f in frames)


def test_build_decision_frames_skips_stops_and_the_end():
    moving = straight_drive(n=300)
    stopped = np.repeat(moving[-1:], 100, axis=0)  # then stands still for 10 s
    poses = np.concatenate([moving, stopped])
    times = np.arange(len(poses)) * 0.1
    frames, counts = build_decision_frames("00004", poses, times)
    assert counts["sampled"] == 80
    kept = [f.frame for f in frames]
    assert kept[0] == 0 and all(f % 5 == 0 for f in kept)
    assert all(f.split == "test" and f.direction == "straight" for f in frames)
    assert max(kept) < 300 - 20 / 0.125 + 5  # nothing within 20 m of where the drive stops
    assert counts["near_end"] > 0
    assert counts["sampled"] == len(frames) + counts["near_end"] + counts["stopped"]
