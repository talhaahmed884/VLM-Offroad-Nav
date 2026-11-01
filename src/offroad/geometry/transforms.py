"""Rigid-body transform helpers (pure functions, no file I/O)."""

import numpy as np


def quaternion_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix.

    The quaternion is normalized first, so small rounding errors in stored
    values do not make the matrix slightly non-orthogonal.
    """
    q = np.array([w, x, y, z], dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from rotation R (3, 3) and translation t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T_b_from_a: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform, giving T_a_from_b.

    Uses R^T and -R^T t instead of a general matrix inverse, which is exact
    for rotations and avoids numerical noise.
    """
    R = T_b_from_a[:3, :3]
    t = T_b_from_a[:3, 3]
    return make_transform(R.T, -R.T @ t)
