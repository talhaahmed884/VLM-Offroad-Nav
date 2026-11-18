"""Ground-truth class of every cell along a planned path, from two sources (pure, no file I/O).

About 40% of path cells have no LiDAR points (the blind circle around the vehicle, occlusion,
sparse far range, water that returns no points), so LiDAR labels alone leave them "unseen".
User decision (2026-10-31): per path cell,
1. use the LiDAR point labels if the cell has points;
2. else project the cell's ground point into the camera and read the camera label image there;
3. else it stays unseen.

A camera pixel shows the nearest surface, not necessarily the ground behind it. If a LiDAR
point lands near that pixel clearly closer than the ground point (e.g. a bush in front of the
cell), the camera label belongs to the bush, so the cell is marked "occluded" instead. Sky or
void labels at a ground point also count as unseen (the ground estimate is off there).
"""

from dataclasses import dataclass

import numpy as np

from offroad.fusion.grid_lookup import fill_ground
from offroad.geometry.projection import project_points_to_image
from offroad.mapping.elevation import T_FLU_FROM_LIDAR, GridSpec

UNSEEN = 0  # class ID used for "no ground truth" (same as "void" and planning.baseline.UNSEEN)

SOURCE_UNSEEN = 0
SOURCE_LIDAR = 1
SOURCE_CAMERA = 2
SOURCE_OCCLUDED = 3  # the camera pixel shows something in front of the cell
SOURCE_NAMES = {SOURCE_UNSEEN: "unseen", SOURCE_LIDAR: "lidar", SOURCE_CAMERA: "camera", SOURCE_OCCLUDED: "occluded"}

DEPTH_BLOCK_PX = 4  # LiDAR depth is checked per 4 x 4 pixel block
# A LiDAR point closer than the ground point by more than max(1 m, 10% of its depth) means occluded.
# The 10% allows for ground points in the same block being a little nearer (at 25 m, 4 px of
# image height span about 0.7 m of ground).
OCCLUSION_MARGIN_M = 1.0
OCCLUSION_MARGIN_FRACTION = 0.1


@dataclass
class PathTruth:
    """Ground truth for one path.

    labels: (M,) class ID per path cell, UNSEEN where no source gives one (unseen or occluded).
    source: (M,) SOURCE_* per path cell.
    """

    labels: np.ndarray
    source: np.ndarray

    def source_shares(self) -> dict[str, float]:
        """Share of path cells per source name, e.g. {"lidar": 0.6, "camera": 0.3, ...}."""
        n = max(len(self.source), 1)
        return {name: float((self.source == s).sum() / n) for s, name in SOURCE_NAMES.items()}

    def class_shares(self) -> dict[int, float]:
        """Share of path cells per class ID (UNSEEN included), like `BaselineResult.class_shares`."""
        if len(self.labels) == 0:
            return {}
        ids, counts = np.unique(self.labels, return_counts=True)
        return {int(i): float(c / len(self.labels)) for i, c in zip(ids, counts)}


def lidar_depth_image(
    points_lidar: np.ndarray, T_cam_from_lidar: np.ndarray, K: np.ndarray, image_size: tuple[int, int],
    block: int = DEPTH_BLOCK_PX,
) -> np.ndarray:
    """Nearest LiDAR depth per block of pixels: (ceil(H/block), ceil(W/block)) meters, inf where no point."""
    height, width = image_size
    uv, depth, _ = project_points_to_image(points_lidar, T_cam_from_lidar, K, image_size)
    nearest = np.full((-(-height // block), -(-width // block)), np.inf)
    rows, cols = (uv[:, 1] // block).astype(int), (uv[:, 0] // block).astype(int)
    np.minimum.at(nearest, (rows, cols), depth)
    return nearest


def project_cells(
    cells: list[tuple[int, int]], ground_z: np.ndarray, spec: GridSpec,
    T_cam_from_lidar: np.ndarray, K: np.ndarray, image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel (u, v) and camera depth of each cell's ground point: ((M, 2), (M,)), NaN where not in view.

    ground_z: (nx, ny) ground height in the FLU frame, meters (NaN = unknown, so not in view).
    """
    uv_out = np.full((len(cells), 2), np.nan)
    depth_out = np.full(len(cells), np.nan)
    if not cells:
        return uv_out, depth_out
    centers = np.array([spec.cell_center(i, j) for i, j in cells])
    z = np.array([ground_z[c] for c in cells])
    known = np.isfinite(z)
    points_flu = np.column_stack([centers[known], z[known], np.ones(known.sum())])
    points_lidar = (np.linalg.inv(T_FLU_FROM_LIDAR) @ points_flu.T).T[:, :3]
    uv, depth, kept = project_points_to_image(points_lidar, T_cam_from_lidar, K, image_size)
    rows = np.nonzero(known)[0][kept]
    uv_out[rows], depth_out[rows] = uv, depth
    return uv_out, depth_out


def path_truth(
    path: list[tuple[int, int]],
    lidar_labels: np.ndarray,
    ground_z: np.ndarray,
    spec: GridSpec,
    T_cam_from_lidar: np.ndarray,
    K: np.ndarray,
    label_image: np.ndarray | None,
    points_lidar: np.ndarray | None = None,
    ignore_ids: tuple[int, ...] = (UNSEEN,),
    occlusion_margin: float = OCCLUSION_MARGIN_M,
) -> PathTruth:
    """Two-source ground truth for each cell of a path (see the module docstring).

    lidar_labels: (nx, ny) majority LiDAR class per cell, UNSEEN where the cell has no points.
    ground_z: (nx, ny) estimated ground height (e.g. `CostMap.ground_z`); gaps are filled first.
    label_image: (H, W) camera class IDs, or None (then only LiDAR is used).
    points_lidar: (N, 3+) the scan, for the occlusion check; None skips the check.
    ignore_ids: camera labels that cannot be ground (void, sky); they leave the cell unseen.
    """
    labels = np.array([lidar_labels[c] for c in path], dtype=int)
    source = np.where(labels != UNSEEN, SOURCE_LIDAR, SOURCE_UNSEEN)
    todo = np.nonzero(source == SOURCE_UNSEEN)[0]
    if label_image is None or len(todo) == 0:
        return PathTruth(labels=labels, source=source)

    image_size = label_image.shape[:2]
    uv, depth = project_cells([path[k] for k in todo], fill_ground(ground_z), spec, T_cam_from_lidar, K, image_size)
    nearest = (lidar_depth_image(points_lidar, T_cam_from_lidar, K, image_size)
               if points_lidar is not None else None)
    for k, (u, v), d in zip(todo, uv, depth):
        if np.isnan(u):
            continue  # not in the camera's view
        camera_label = int(label_image[int(v), int(u)])
        if camera_label in ignore_ids:
            continue
        margin = max(occlusion_margin, OCCLUSION_MARGIN_FRACTION * d)
        if nearest is not None and nearest[int(v) // DEPTH_BLOCK_PX, int(u) // DEPTH_BLOCK_PX] < d - margin:
            source[k] = SOURCE_OCCLUDED
            continue
        labels[k], source[k] = camera_label, SOURCE_CAMERA
    return PathTruth(labels=labels, source=source)
