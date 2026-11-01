"""Parse RELLIS-3D camera intrinsics and Ouster-to-camera extrinsics.

Both files live inside each sequence folder:
- camera_info.txt: one line "fx fy cx cy" (no distortion coefficients).
- transforms.yaml: key "os1_cloud_node-pylon_camera_node" with a quaternion
  q {w, x, y, z} and translation t {x, y, z}. This is the camera pose in the
  LiDAR frame (T_lidar_from_cam), so we invert it to get T_cam_from_lidar.
  See the "Dataset on disk" notes in CLAUDE.md for how the direction was checked.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from offroad.geometry.transforms import (
    invert_transform,
    make_transform,
    quaternion_to_rotation_matrix,
)

EXTRINSIC_KEY = "os1_cloud_node-pylon_camera_node"


@dataclass(frozen=True)
class CameraCalibration:
    """Calibration for one sequence.

    K: (3, 3) camera intrinsic matrix, pixels.
    dist_coeffs: (5,) OpenCV distortion coefficients (k1, k2, p1, p2, k3).
    T_cam_from_lidar: (4, 4) maps Ouster LiDAR-frame points (meters) to the camera frame.
    """

    K: np.ndarray
    dist_coeffs: np.ndarray
    T_cam_from_lidar: np.ndarray


def load_intrinsics(path: Path) -> np.ndarray:
    """Read camera_info.txt ("fx fy cx cy") and return K, shape (3, 3)."""
    fx, fy, cx, cy = np.loadtxt(path, dtype=np.float64)
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
    )


def load_T_lidar_from_cam(path: Path) -> np.ndarray:
    """Read transforms.yaml and return the stored transform as a (4, 4) matrix.

    The stored transform is the camera pose in the LiDAR frame, i.e. T_lidar_from_cam.
    """
    with open(path) as f:
        data = yaml.safe_load(f)[EXTRINSIC_KEY]
    q, t = data["q"], data["t"]
    R = quaternion_to_rotation_matrix(q["w"], q["x"], q["y"], q["z"])
    return make_transform(R, np.array([t["x"], t["y"], t["z"]]))


def load_calibration(sequence_dir: Path) -> CameraCalibration:
    """Load the calibration files from one sequence folder (e.g. Rellis-3D/00000)."""
    sequence_dir = Path(sequence_dir)
    K = load_intrinsics(sequence_dir / "camera_info.txt")
    T_lidar_from_cam = load_T_lidar_from_cam(sequence_dir / "transforms.yaml")
    return CameraCalibration(
        K=K,
        # RELLIS-3D provides no distortion coefficients, so assume none.
        dist_coeffs=np.zeros(5),
        T_cam_from_lidar=invert_transform(T_lidar_from_cam),
    )
