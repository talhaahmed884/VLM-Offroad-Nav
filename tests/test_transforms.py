"""Tests for rigid transform helpers and calibration parsing. No dataset needed."""

import numpy as np

from offroad.data.calibration import load_calibration
from offroad.geometry.transforms import (
    invert_transform,
    make_transform,
    quaternion_to_rotation_matrix,
)


def test_identity_quaternion_gives_identity_rotation():
    np.testing.assert_allclose(quaternion_to_rotation_matrix(1, 0, 0, 0), np.eye(3))


def test_quaternion_90_degrees_about_z():
    half = np.sqrt(0.5)  # cos(45 deg) = sin(45 deg)
    R = quaternion_to_rotation_matrix(half, 0, 0, half)
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_unnormalized_quaternion_still_gives_a_rotation():
    R = quaternion_to_rotation_matrix(2.0, 0.2, -0.4, 1.0)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_invert_transform_undoes_transform():
    R = quaternion_to_rotation_matrix(0.9, 0.1, -0.3, 0.2)
    T_b_from_a = make_transform(R, np.array([1.0, -2.0, 0.5]))
    T_a_from_b = invert_transform(T_b_from_a)
    np.testing.assert_allclose(T_a_from_b @ T_b_from_a, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(T_b_from_a @ T_a_from_b, np.eye(4), atol=1e-12)


def write_calibration_files(folder, q, t):
    (folder / "camera_info.txt").write_text("2813.6 2808.3 969.3 624.0\n")
    (folder / "transforms.yaml").write_text(
        "os1_cloud_node-pylon_camera_node:\n"
        f"  q:\n    w: {q[0]}\n    x: {q[1]}\n    y: {q[2]}\n    z: {q[3]}\n"
        f"  t:\n    x: {t[0]}\n    y: {t[1]}\n    z: {t[2]}\n"
    )


def test_load_calibration_parses_intrinsics(tmp_path):
    write_calibration_files(tmp_path, q=(1, 0, 0, 0), t=(0, 0, 0))
    calib = load_calibration(tmp_path)
    np.testing.assert_allclose(calib.K, [[2813.6, 0, 969.3], [0, 2808.3, 624.0], [0, 0, 1]])
    np.testing.assert_allclose(calib.dist_coeffs, np.zeros(5))


def test_load_calibration_inverts_stored_transform(tmp_path):
    # transforms.yaml stores the camera pose in the LiDAR frame (T_lidar_from_cam).
    # So the stored translation is where the camera sits in LiDAR coordinates,
    # and T_cam_from_lidar must map that point to the camera origin.
    camera_position_in_lidar = np.array([-0.13, 0.04, -0.17])
    write_calibration_files(tmp_path, q=(-0.505, 0.512, 0.490, -0.492), t=camera_position_in_lidar)
    T = load_calibration(tmp_path).T_cam_from_lidar
    np.testing.assert_allclose(T[:3, :3] @ camera_position_in_lidar + T[:3, 3], [0, 0, 0], atol=1e-12)


def test_rellis_rotation_maps_lidar_axes_to_camera_axes(tmp_path):
    # With the real RELLIS-3D quaternion, LiDAR "up" (+z) must become camera "up" (-y)
    # and LiDAR forward (-x for this mounting) must become camera forward (+z).
    write_calibration_files(tmp_path, q=(-0.50507811, 0.51206185, 0.49024953, -0.49228464), t=(0, 0, 0))
    R = load_calibration(tmp_path).T_cam_from_lidar[:3, :3]
    np.testing.assert_allclose(R @ [0, 0, 1], [0, -1, 0], atol=0.05)
    np.testing.assert_allclose(R @ [-1, 0, 0], [0, 0, 1], atol=0.05)
