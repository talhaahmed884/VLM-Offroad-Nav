"""Tests for LiDAR-to-image projection. No dataset needed."""

import cv2
import numpy as np
import pytest

from offroad.geometry.projection import project_points_to_image, transform_points
from offroad.geometry.transforms import make_transform

# RELLIS-3D-like camera: fx, fy, cx, cy and a 1200 x 1920 image.
FX, FY, CX, CY = 2800.0, 2800.0, 960.0, 600.0
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (1200, 1920)  # (height, width)
IDENTITY = np.eye(4)


def test_point_on_optical_axis_lands_at_principal_point():
    points = np.array([[0.0, 0.0, 5.0]])
    uv, depth, kept = project_points_to_image(points, IDENTITY, K, IMAGE_SIZE)
    np.testing.assert_allclose(uv, [[CX, CY]])
    np.testing.assert_allclose(depth, [5.0])
    assert kept.tolist() == [True]


def test_off_axis_point_matches_pinhole_formula():
    # 1 m right and 0.5 m down at 10 m ahead.
    points = np.array([[1.0, 0.5, 10.0]])
    uv, _, _ = project_points_to_image(points, IDENTITY, K, IMAGE_SIZE)
    np.testing.assert_allclose(uv, [[FX * 1.0 / 10.0 + CX, FY * 0.5 / 10.0 + CY]])


def test_points_behind_or_at_camera_are_dropped():
    points = np.array(
        [
            [0.0, 0.0, -5.0],  # behind: would land at the principal point if not dropped
            [0.0, 0.0, 0.0],  # at the camera center
            [0.1, 0.0, 0.0],  # in the camera plane
            [0.0, 0.0, 5.0],  # in front: kept
        ]
    )
    _, depth, kept = project_points_to_image(points, IDENTITY, K, IMAGE_SIZE)
    assert kept.tolist() == [False, False, False, True]
    np.testing.assert_allclose(depth, [5.0])


def test_points_outside_image_are_dropped():
    height, width = IMAGE_SIZE
    z = 10.0

    def point_at_pixel(u, v):
        return [(u - CX) * z / FX, (v - CY) * z / FY, z]

    points = np.array(
        [
            point_at_pixel(0.0, 0.0),  # top-left corner: kept
            point_at_pixel(width - 0.01, height - 0.01),  # just inside bottom-right: kept
            point_at_pixel(width + 0.5, CY),  # just past the right edge: dropped
            point_at_pixel(CX, -1.0),  # above the image: dropped
            [100.0, 0.0, 1.0],  # far to the side: dropped
        ]
    )
    _, _, kept = project_points_to_image(points, IDENTITY, K, IMAGE_SIZE)
    assert kept.tolist() == [True, True, False, False, False]


def test_kept_mask_lines_up_with_outputs_and_ignores_intensity():
    # (N, 4) input like a LiDAR scan; the 4th column is intensity and must be ignored.
    points = np.array(
        [
            [0.0, 0.0, -1.0, 0.9],  # dropped
            [0.0, 0.0, 2.0, 0.1],  # kept, first
            [1000.0, 0.0, 1.0, 0.5],  # dropped
            [0.0, 0.0, 8.0, 0.3],  # kept, second
        ]
    )
    uv, depth, kept = project_points_to_image(points, IDENTITY, K, IMAGE_SIZE)
    assert kept.tolist() == [False, True, False, True]
    assert uv.shape == (2, 2)
    np.testing.assert_allclose(depth, points[kept, 2])


def test_extrinsic_is_applied_before_projection():
    # LiDAR origin sits 3 m in front of the camera, with axes aligned.
    T_cam_from_lidar = make_transform(np.eye(3), np.array([0.0, 0.0, 3.0]))
    points_lidar = np.array([[0.0, 0.0, 0.0]])
    uv, depth, _ = project_points_to_image(points_lidar, T_cam_from_lidar, K, IMAGE_SIZE)
    np.testing.assert_allclose(uv, [[CX, CY]])
    np.testing.assert_allclose(depth, [3.0])


def test_transform_points_rotates_then_translates():
    # 90 degrees about z: x -> y, then shift by (1, 2, 3).
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T = make_transform(R, np.array([1.0, 2.0, 3.0]))
    result = transform_points(T, np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(result, [[1.0, 3.0, 3.0]])


def test_matches_opencv_project_points():
    rng = np.random.default_rng(0)
    points = rng.uniform([-20, -5, -20], [20, 5, 40], size=(2000, 3))
    rvec = np.array([0.05, -0.1, 0.02])
    t = np.array([0.1, -0.2, 0.3])
    R, _ = cv2.Rodrigues(rvec)
    T_cam_from_lidar = make_transform(R, t)

    uv, _, kept = project_points_to_image(points, T_cam_from_lidar, K, IMAGE_SIZE)
    assert kept.sum() > 100  # make sure the test actually compares something

    expected, _ = cv2.projectPoints(points[kept], rvec, t, K, np.zeros(5))
    np.testing.assert_allclose(uv, expected.reshape(-1, 2), atol=1e-6)


def test_empty_input_returns_empty_outputs():
    uv, depth, kept = project_points_to_image(np.zeros((0, 4)), IDENTITY, K, IMAGE_SIZE)
    assert uv.shape == (0, 2)
    assert depth.shape == (0,)
    assert kept.shape == (0,)


@pytest.mark.parametrize("n_cols", [3, 4])
def test_accepts_three_or_four_columns(n_cols):
    points = np.zeros((1, n_cols))
    points[0, 2] = 5.0
    _, _, kept = project_points_to_image(points, IDENTITY, K, IMAGE_SIZE)
    assert kept.tolist() == [True]
