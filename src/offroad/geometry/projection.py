"""Project 3D points into a camera image with the pinhole model (pure functions, no file I/O).

Camera frame convention (OpenCV): x right, y down, z forward along the optical axis, meters.
Pixel convention: u is the column, v is the row; integer pixel (r, c) covers
u in [c, c + 1) and v in [r, r + 1), so int(u), int(v) index the image directly.
"""

import numpy as np


def transform_points(T_b_from_a: np.ndarray, points_a: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to points.

    points_a: (N, 3) in frame a, meters.
    Returns (N, 3) in frame b, meters.
    """
    R = T_b_from_a[:3, :3]
    t = T_b_from_a[:3, 3]
    return points_a @ R.T + t


def project_points_to_image(
    points_lidar: np.ndarray,
    T_cam_from_lidar: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR points into the image and keep the ones the camera can see.

    Assumes an undistorted (pinhole) camera; RELLIS-3D provides no distortion coefficients.

    points_lidar: (N, 3) or (N, 4) in the LiDAR frame, meters. Columns past x, y, z
        (e.g. intensity) are ignored.
    T_cam_from_lidar: (4, 4) maps LiDAR-frame points to the camera frame.
    K: (3, 3) camera intrinsic matrix.
    image_size: (height, width) in pixels.

    Returns:
        uv: (M, 2) float pixel coordinates (u = column, v = row) of the kept points.
        depth: (M,) distance along the camera's z axis, meters.
        kept: (N,) bool mask into points_lidar; points_lidar[kept] lines up with uv and depth.
    """
    height, width = image_size
    points_cam = transform_points(T_cam_from_lidar, points_lidar[:, :3])
    z = points_cam[:, 2]

    # Only points in front of the camera can appear in the image.
    in_front = z > 0

    # Pinhole projection: u = fx * x / z + cx, v = fy * y / z + cy.
    # Points behind the camera get NaN instead of a division warning; they are dropped below.
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels_h = points_cam @ K.T
        u = pixels_h[:, 0] / z
        v = pixels_h[:, 1] / z

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    kept = in_front & in_bounds

    uv = np.stack([u[kept], v[kept]], axis=1)
    return uv, z[kept], kept
