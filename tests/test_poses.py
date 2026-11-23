"""Tests for reading poses and image timestamps. No dataset needed."""

import numpy as np
import pytest

from offroad.data.rellis import image_timestamp, load_poses


def write_poses(path, matrices, trailing_newline=True):
    text = "\n".join(" ".join(f"{v:.6f}" for v in m[:3, :].ravel()) for m in matrices)
    path.write_text(text + ("\n" if trailing_newline else ""))


def test_load_poses_round_trip(tmp_path):
    a = np.eye(4)
    b = np.eye(4)
    b[:3, 3] = [1.0, -2.0, 0.5]
    b[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    path = tmp_path / "poses.txt"
    write_poses(path, [a, b])
    poses = load_poses(path)
    assert poses.shape == (2, 4, 4)
    assert np.allclose(poses[1], b)
    assert np.allclose(poses[:, 3], [0, 0, 0, 1])


def test_load_poses_without_trailing_newline(tmp_path):
    # Two RELLIS sequences end without a newline; every line must still be read.
    path = tmp_path / "poses.txt"
    write_poses(path, [np.eye(4)] * 3, trailing_newline=False)
    assert len(load_poses(path)) == 3


def test_load_poses_single_line(tmp_path):
    path = tmp_path / "poses.txt"
    write_poses(path, [np.eye(4)])
    assert load_poses(path).shape == (1, 4, 4)


def test_load_poses_rejects_wrong_width(tmp_path):
    path = tmp_path / "poses.txt"
    path.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError):
        load_poses(path)


def test_image_timestamp():
    assert image_timestamp("frame000123-1581624652_750.jpg") == pytest.approx(1581624652.750)
    assert image_timestamp("some/dir/frame002846-1581624937_350.jpg") == pytest.approx(1581624937.350)
    with pytest.raises(ValueError):
        image_timestamp("000123.bin")
