"""Matplotlib helpers for drawing top-down grids (elevation, cost) in the FLU frame."""

import numpy as np
from matplotlib.patches import Rectangle

from offroad.mapping.elevation import GridSpec


def show_top_down(ax, grid: np.ndarray, spec: GridSpec, title: str, **imshow_kwargs):
    """Draw an (nx, ny) or (nx, ny, 3) grid as seen from above: forward is up, left is left.

    Returns the image artist, e.g. for a colorbar.
    """
    image = ax.imshow(
        grid,
        origin="lower",
        extent=[spec.y_min, spec.y_max, spec.x_min, spec.x_max],
        interpolation="nearest",
        **imshow_kwargs,
    )
    ax.invert_xaxis()  # +y is left, so put it on the left side of the plot
    ax.set_xlabel("left (m)")
    ax.set_ylabel("forward (m)")
    ax.set_title(title)
    return image


def draw_camera_fov(ax, K: np.ndarray, image_width: int, spec: GridSpec):
    """Dashed lines for the left and right edges of the camera's view (camera assumed at the origin)."""
    fx, cx = K[0, 0], K[0, 2]
    tan_left, tan_right = cx / fx, (image_width - cx) / fx
    ax.plot([0, spec.x_max * tan_left], [0, spec.x_max], "c--", linewidth=1)
    ax.plot([0, -spec.x_max * tan_right], [0, spec.x_max], "c--", linewidth=1)
    ax.plot(0, 0, "c^", markersize=8)  # the LiDAR / vehicle position


def label_colors(grid: np.ndarray, id_to_color: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """(nx, ny) class IDs -> (nx, ny, 3) RGB using the ontology colors; void/empty is white."""
    lookup = np.full((256, 3), 255, dtype=np.uint8)
    for class_id, rgb in id_to_color.items():
        if class_id != 0:
            lookup[class_id] = rgb
    return lookup[grid]


def add_class_legend(ax, grid: np.ndarray, id_to_name: dict[int, str], id_to_color: dict[int, tuple[int, int, int]]):
    """Legend listing the classes present in a label grid (void excluded)."""
    present = [c for c in np.unique(grid) if c != 0]
    handles = [Rectangle((0, 0), 1, 1, color=np.array(id_to_color[c]) / 255) for c in present]
    ax.legend(handles, [id_to_name[c] for c in present], loc="upper right", fontsize=8)
