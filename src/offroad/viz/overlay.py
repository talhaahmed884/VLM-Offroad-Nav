"""Draw projected LiDAR points on camera images (pure functions, no file I/O)."""

import cv2
import matplotlib
import numpy as np


def depth_to_colors(
    depth: np.ndarray,
    min_depth: float | None = None,
    max_depth: float | None = None,
    cmap: str = "turbo",
) -> np.ndarray:
    """Map depths to RGB colors with a Matplotlib colormap.

    depth: (M,) meters. Values outside [min_depth, max_depth] are clipped;
        the limits default to the smallest and largest depth given.
    With "turbo", near points are blue and far points are red.
    Returns (M, 3) uint8 RGB.
    """
    lo = depth.min() if min_depth is None else min_depth
    hi = depth.max() if max_depth is None else max_depth
    scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap](scaled)
    return (rgba[:, :3] * 255).astype(np.uint8)


def draw_points(image: np.ndarray, uv: np.ndarray, colors: np.ndarray, radius: int = 2) -> np.ndarray:
    """Draw filled dots at pixel positions on a copy of the image.

    image: (H, W, 3) uint8, RGB.
    uv: (M, 2) pixel coordinates (u = column, v = row).
    colors: (M, 3) uint8 RGB, one per point.
    Returns a new (H, W, 3) uint8 RGB image; the input is not modified.
    """
    out = image.copy()
    for (u, v), color in zip(np.round(uv).astype(int), colors):
        cv2.circle(out, (int(u), int(v)), radius, tuple(int(c) for c in color), thickness=-1)
    return out


def draw_depth_overlay(
    image: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    max_depth: float | None = None,
    radius: int = 2,
) -> np.ndarray:
    """Draw points colored by depth on a copy of the image, nearer points on top.

    image: (H, W, 3) uint8 RGB. uv: (M, 2) pixels. depth: (M,) meters.
    Returns a new (H, W, 3) uint8 RGB image.
    """
    # Points are drawn in order, so draw far ones first and let near ones cover them.
    far_to_near = np.argsort(depth)[::-1]
    colors = depth_to_colors(depth[far_to_near], max_depth=max_depth)
    return draw_points(image, uv[far_to_near], colors, radius=radius)
