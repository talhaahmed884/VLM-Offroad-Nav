"""Where the planner's target appears in the camera image, so the prompt can name its grid cell.

The baseline planner aims at a point `goal_distance` meters straight ahead on the ground
(FLU (d, 0, ground height)). Projecting that point into the image gives the cell the VLM
should drive toward, so both planners get the same goal. Pure functions, no file I/O.
"""

import numpy as np

from offroad.geometry.projection import project_points_to_image
from offroad.mapping.elevation import T_FLU_FROM_LIDAR, GridSpec
from offroad.vlm.grid import ImageGrid


def ground_height_at(ground_z: np.ndarray, spec: GridSpec, x: float, y: float, radius: float = 3.0) -> float | None:
    """Ground height near FLU (x, y), meters, or None if no ground is known nearby.

    ground_z: (nx, ny) ground height per cell (NaN where unknown), e.g. from `estimate_ground`.
    Uses the cell's own value if known, else the median of known cells within `radius` meters.
    """
    i, j = spec.cell_index(x, y)
    nx, ny = spec.shape
    if 0 <= i < nx and 0 <= j < ny and np.isfinite(ground_z[i, j]):
        return float(ground_z[i, j])

    r = int(np.ceil(radius / spec.resolution))
    window = ground_z[max(i - r, 0):max(i + r + 1, 0), max(j - r, 0):max(j + r + 1, 0)]
    known = window[np.isfinite(window)]
    return float(np.median(known)) if known.size else None


def goal_pixel(
    target_flu: tuple[float, float, float],
    T_cam_from_lidar: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    """Pixel (u, v) of an FLU point (x forward, y left, z up; meters), or None if not in view."""
    point_flu = np.array([[*target_flu, 1.0]])
    point_lidar = (np.linalg.inv(T_FLU_FROM_LIDAR) @ point_flu.T).T[:, :3]
    uv, _, kept = project_points_to_image(point_lidar, T_cam_from_lidar, K, image_size)
    return (float(uv[0, 0]), float(uv[0, 1])) if kept[0] else None


def goal_cell(
    ground_z: np.ndarray,
    spec: GridSpec,
    T_cam_from_lidar: np.ndarray,
    K: np.ndarray,
    grid: ImageGrid,
    goal_distance: float = 20.0,
) -> str | None:
    """Grid cell (e.g. "F7") showing the ground `goal_distance` m straight ahead, or None.

    None means the ground height there is unknown or the point is outside the image.
    """
    z = ground_height_at(ground_z, spec, goal_distance, 0.0)
    if z is None:
        return None
    pixel = goal_pixel((goal_distance, 0.0, z), T_cam_from_lidar, K, (grid.height, grid.width))
    if pixel is None:
        return None
    cell = grid.cell_at(*pixel)
    return grid.label(*cell) if cell is not None else None
