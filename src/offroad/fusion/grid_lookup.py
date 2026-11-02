"""Which image grid cell each top-down map cell appears in (pure functions, no file I/O).

The VLM talks about image grid cells ("F7 is water"); the planner works on the top-down
map. To connect them, each map cell's ground point (FLU x, y and the estimated ground
height) is projected into the camera image, and the grid cell it lands in is recorded.
This is the forward version of ray-tracing pixels onto the terrain (DEM): it gives the same
pixel <-> ground matching without marching rays, and reuses the tested projection code.

Limits:
- Map cells outside the camera's view, including the first ~5-6 m in front of the vehicle
  (below the image), get no grid cell and so no VLM information.
- Ground hidden behind something (e.g. behind a bush) still projects to the pixel that
  shows the bush, so it inherits what the VLM said about that cell.
"""

from dataclasses import dataclass

import numpy as np

from offroad.geometry.projection import project_points_to_image
from offroad.mapping.elevation import T_FLU_FROM_LIDAR, GridSpec
from offroad.vlm.grid import ImageGrid

NO_CELL = -1
FILL_STEPS = 40  # fill ground gaps up to 40 cells (10 m at 0.25 m) away from known ground


def fill_ground(ground_z: np.ndarray, max_steps: int = FILL_STEPS) -> np.ndarray:
    """Fill unknown (NaN) ground heights from their known neighbors, growing inward step by step.

    Each step gives every unknown cell that touches a known cell the mean of its known
    8-neighbors. Used for the blind circle around the vehicle and for gaps behind obstacles,
    where the LiDAR saw no ground. Cells more than `max_steps` from known ground stay NaN.
    ground_z: (nx, ny) meters. Returns a new (nx, ny) array.
    """
    filled = ground_z.astype(float)
    nx, ny = filled.shape
    for _ in range(max_steps):
        missing = np.isnan(filled)
        if not missing.any():
            break
        padded = np.pad(filled, 1, constant_values=np.nan)
        neighbors = np.stack([
            padded[1 + di:1 + di + nx, 1 + dj:1 + dj + ny]
            for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)
        ])
        known = ~np.isnan(neighbors)
        counts = known.sum(axis=0)
        sums = np.where(known, neighbors, 0.0).sum(axis=0)
        new = missing & (counts > 0)
        if not new.any():
            break
        filled[new] = sums[new] / counts[new]
    return filled


def project_map_cells(
    ground_z: np.ndarray, spec: GridSpec, T_cam_from_lidar: np.ndarray, K: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Pixel (u, v) of each map cell's ground point: (nx, ny, 2), NaN where unknown or not in view.

    ground_z: (nx, ny) ground height per cell in the FLU frame, meters (NaN = unknown).
    image_size: (height, width) in pixels.
    """
    nx, ny = spec.shape
    xs = spec.x_min + (np.arange(nx) + 0.5) * spec.resolution
    ys = spec.y_min + (np.arange(ny) + 0.5) * spec.resolution
    x, y = np.meshgrid(xs, ys, indexing="ij")
    known = np.isfinite(ground_z)

    points_flu = np.stack([x[known], y[known], ground_z[known], np.ones(known.sum())], axis=1)
    points_lidar = (np.linalg.inv(T_FLU_FROM_LIDAR) @ points_flu.T).T[:, :3]
    uv_kept, _, kept = project_points_to_image(points_lidar, T_cam_from_lidar, K, image_size)

    uv = np.full((nx, ny, 2), np.nan)
    cells = np.argwhere(known)[kept]
    uv[cells[:, 0], cells[:, 1]] = uv_kept
    return uv


def grid_cell_ids(uv: np.ndarray, grid: ImageGrid) -> np.ndarray:
    """Grid cell of each pixel as one number, row * cols + col: (nx, ny) int, NO_CELL where uv is NaN."""
    visible = np.isfinite(uv[..., 0])
    ids = np.full(uv.shape[:2], NO_CELL, dtype=int)
    rows = np.minimum((uv[..., 1][visible] // grid.cell_height).astype(int), grid.rows - 1)
    cols = np.minimum((uv[..., 0][visible] // grid.cell_width).astype(int), grid.cols - 1)
    ids[visible] = rows * grid.cols + cols
    return ids


@dataclass
class GridLookup:
    """Map cell <-> image grid cell matching for one frame.

    cell_id: (nx, ny) int, row * grid.cols + col of the grid cell each map cell appears in,
        NO_CELL if it is not in view (or its ground is unknown even after filling).
    uv: (nx, ny, 2) pixel of each map cell's ground point, NaN where not in view.
    ground_filled: (nx, ny) bool, True where the ground height was filled in, not measured.
    """

    spec: GridSpec
    grid: ImageGrid
    cell_id: np.ndarray
    uv: np.ndarray
    ground_filled: np.ndarray

    @property
    def visible(self) -> np.ndarray:
        """(nx, ny) bool, True where the map cell appears in the image."""
        return self.cell_id != NO_CELL

    def id_of(self, label: str) -> int:
        row, col = self.grid.parse(label)
        return row * self.grid.cols + col

    def label_of(self, cell_id: int) -> str:
        return self.grid.label(cell_id // self.grid.cols, cell_id % self.grid.cols)

    def mask(self, label: str) -> np.ndarray:
        """(nx, ny) bool: the map cells that appear in grid cell `label` (e.g. "F7")."""
        return self.cell_id == self.id_of(label)

    def ground_point(self, label: str) -> tuple[float, float] | None:
        """FLU (x, y) of the ground seen at the center pixel of grid cell `label`, or None if it shows no ground.

        This is the pixel-to-terrain ray trace for one pixel: among the map cells that appear in
        this grid cell, the one whose ground point projects closest to the cell's center.
        """
        mask = self.mask(label)
        if not mask.any():
            return None
        u, v = self.grid.center(*self.grid.parse(label))
        pixels = self.uv[mask]  # same (row-major) order as np.argwhere(mask)
        nearest = np.argmin((pixels[:, 0] - u) ** 2 + (pixels[:, 1] - v) ** 2)
        i, j = np.argwhere(mask)[nearest]
        return self.spec.cell_center(i, j)

    def centroid(self, label: str) -> tuple[float, float] | None:
        """FLU (x, y) center of the ground area that grid cell `label` covers, or None if it covers none."""
        cells = np.argwhere(self.mask(label))
        if len(cells) == 0:
            return None
        i, j = cells.mean(axis=0)
        return self.spec.cell_center(i, j)


def build_grid_lookup(
    ground_z: np.ndarray,
    spec: GridSpec,
    T_cam_from_lidar: np.ndarray,
    K: np.ndarray,
    grid: ImageGrid,
    fill_steps: int = FILL_STEPS,
) -> GridLookup:
    """Match every map cell to the image grid cell it appears in.

    ground_z: (nx, ny) estimated ground height (e.g. `CostMap.ground_z`), NaN where unknown.
    grid: the image grid at the full camera resolution.
    """
    filled = fill_ground(ground_z, fill_steps)
    uv = project_map_cells(filled, spec, T_cam_from_lidar, K, (grid.height, grid.width))
    return GridLookup(
        spec=spec,
        grid=grid,
        cell_id=grid_cell_ids(uv, grid),
        uv=uv,
        ground_filled=np.isnan(ground_z) & np.isfinite(filled),
    )
