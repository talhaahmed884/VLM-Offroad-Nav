"""Top-down elevation map (DEM) from one LiDAR scan (pure functions, no file I/O).

Frame: forward-left-up (FLU), origin at the LiDAR, meters. x forward, y left, z up.
Grid index [i, j]: i counts cells along x (forward), j along y (left).
"""

from dataclasses import dataclass

import numpy as np

from offroad.geometry.projection import transform_points

# On the RELLIS-3D vehicle the Ouster's forward is its -x axis and left is its -y axis
# (worked out from the camera calibration, see CLAUDE.md), so LiDAR -> FLU is a
# 180-degree rotation about z.
T_FLU_FROM_LIDAR = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class GridSpec:
    """Area covered by the map and its cell size, in the FLU frame (meters)."""

    x_min: float = -5.0
    x_max: float = 35.0
    y_min: float = -15.0
    y_max: float = 15.0
    resolution: float = 0.25

    @property
    def shape(self) -> tuple[int, int]:
        """(cells along x, cells along y)."""
        return (
            round((self.x_max - self.x_min) / self.resolution),
            round((self.y_max - self.y_min) / self.resolution),
        )

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        """FLU (x, y) of the center of cell [i, j], meters."""
        return (
            self.x_min + (i + 0.5) * self.resolution,
            self.y_min + (j + 0.5) * self.resolution,
        )


@dataclass
class ElevationMap:
    """Per-cell height statistics. All arrays are (nx, ny); heights are FLU z in meters.

    count: number of points in the cell.
    min_z, max_z, mean_z: lowest, highest, and mean point height; NaN where count == 0.
    """

    spec: GridSpec
    count: np.ndarray
    min_z: np.ndarray
    max_z: np.ndarray
    mean_z: np.ndarray

    @property
    def step_height(self) -> np.ndarray:
        """Highest minus lowest point per cell, meters (NaN where empty).

        Large values mean something sticks up from the ground: a rock, a post,
        a tree, or (the hard case for geometry) tall grass and bushes.
        """
        return self.max_z - self.min_z


def lidar_to_flu(points_lidar: np.ndarray) -> np.ndarray:
    """Convert LiDAR-frame points (N, 3+) to the FLU frame, (N, 3) meters."""
    return transform_points(T_FLU_FROM_LIDAR, points_lidar[:, :3])


def points_to_cells(points_flu: np.ndarray, spec: GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the grid cell of each point.

    points_flu: (N, 3+) FLU frame, meters.
    Returns (i, j, inside): i and j are int arrays for the points inside the grid,
    and inside is an (N,) bool mask of which points those are.
    """
    i = np.floor((points_flu[:, 0] - spec.x_min) / spec.resolution).astype(int)
    j = np.floor((points_flu[:, 1] - spec.y_min) / spec.resolution).astype(int)
    nx, ny = spec.shape
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    return i[inside], j[inside], inside


def build_elevation_map(points_flu: np.ndarray, spec: GridSpec) -> ElevationMap:
    """Grid the points and compute per-cell height statistics.

    points_flu: (N, 3+) FLU frame, meters. Points outside the grid are ignored.
    """
    i, j, inside = points_to_cells(points_flu, spec)
    z = points_flu[inside, 2]
    n_cells = spec.shape[0] * spec.shape[1]
    flat = np.ravel_multi_index((i, j), spec.shape)

    count = np.bincount(flat, minlength=n_cells)
    sum_z = np.bincount(flat, weights=z, minlength=n_cells)
    min_z = np.full(n_cells, np.inf)
    max_z = np.full(n_cells, -np.inf)
    np.minimum.at(min_z, flat, z)
    np.maximum.at(max_z, flat, z)

    empty = count == 0
    min_z[empty] = np.nan
    max_z[empty] = np.nan
    mean_z = np.full(n_cells, np.nan)
    mean_z[~empty] = sum_z[~empty] / count[~empty]

    return ElevationMap(
        spec=spec,
        count=count.reshape(spec.shape),
        min_z=min_z.reshape(spec.shape),
        max_z=max_z.reshape(spec.shape),
        mean_z=mean_z.reshape(spec.shape),
    )


def majority_label_grid(
    points_flu: np.ndarray, labels: np.ndarray, spec: GridSpec, empty_label: int = 0
) -> np.ndarray:
    """Most common ground-truth class per cell.

    points_flu: (N, 3+) FLU frame, meters. labels: (N,) class IDs, one per point.
    Returns (nx, ny) class IDs; cells with no points get `empty_label` (0 = void).
    """
    i, j, inside = points_to_cells(points_flu, spec)
    cell_labels = labels[inside]
    n_cells = spec.shape[0] * spec.shape[1]
    flat = np.ravel_multi_index((i, j), spec.shape)

    best_label = np.full(n_cells, empty_label, dtype=labels.dtype)
    best_count = np.zeros(n_cells, dtype=int)
    for class_id in np.unique(cell_labels):
        class_count = np.bincount(flat[cell_labels == class_id], minlength=n_cells)
        better = class_count > best_count
        best_label[better] = class_id
        best_count[better] = class_count[better]
    return best_label.reshape(spec.shape)
