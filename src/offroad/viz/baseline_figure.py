"""Picture of a geometry-only baseline result: the path on the camera image, cost map, and ground-truth classes.

Scripts using this should call matplotlib.use("Agg") before importing it if no display is available.
"""

import matplotlib.pyplot as plt
import numpy as np

from offroad.data.calibration import CameraCalibration
from offroad.geometry.projection import project_points_to_image, transform_points
from offroad.geometry.transforms import invert_transform
from offroad.mapping.elevation import T_FLU_FROM_LIDAR
from offroad.planning.baseline import BaselineResult
from offroad.viz.overlay import draw_points
from offroad.viz.topdown import (
    add_class_legend,
    add_cost_legend,
    cost_colors,
    draw_camera_fov,
    label_colors,
    show_top_down,
)

PATH_RGB = (255, 0, 0)


def draw_path_on_camera(
    image: np.ndarray, calib: CameraCalibration, result: BaselineResult, color: tuple[int, int, int] = PATH_RGB
) -> np.ndarray:
    """Copy of the (H, W, 3) RGB image with the path drawn as dots (red by default) on the estimated ground.

    Path cells with unknown ground height (e.g. the blind circle around the vehicle) take
    the height of the nearest path cells that have one, so the dots stay on the ground
    even on slopes. If no path cell has a known ground height, the image is returned unchanged.
    """
    z = np.array([result.costmap.ground_z[c] for c in result.path])
    known = ~np.isnan(z)
    if not known.any():
        return image
    steps = np.arange(len(z))
    z = np.interp(steps, steps[known], z[known])  # also holds the end values constant beyond the known ones
    points_flu = np.column_stack([result.path_xy(), z])
    points_lidar = transform_points(invert_transform(T_FLU_FROM_LIDAR), points_flu)
    uv, _, _ = project_points_to_image(points_lidar, calib.T_cam_from_lidar, calib.K, image.shape[:2])
    colors = np.tile(np.array(color, dtype=np.uint8), (len(uv), 1))
    return draw_points(image, uv, colors, radius=6)


def draw_baseline_figure(
    image: np.ndarray,
    calib: CameraCalibration,
    result: BaselineResult,
    id_to_name: dict[int, str],
    id_to_color: dict[int, tuple[int, int, int]],
    title: str,
) -> plt.Figure:
    """Three panels: camera with the path, cost map with the path, ground-truth classes with the path.

    In the top-down panels the target is a white X and the goal actually used is a red star.
    image: (H, W, 3) RGB. Returns the figure; the caller saves and closes it.
    """
    spec = result.spec
    fig, axes = plt.subplots(1, 3, figsize=(24, 9), gridspec_kw={"width_ratios": [1.6, 1, 1]})

    camera = draw_path_on_camera(image, calib, result) if result.found else image
    if not result.found:
        status = "NO PATH: vehicle cannot move"
    elif result.reached:
        status = "path in red, target reached"
    else:
        status = f"path in red, stops {result.shortfall_m:.1f} m short of target"
    axes[0].imshow(camera)
    axes[0].set_title(f"{title} ({status})")
    axes[0].axis("off")

    show_top_down(axes[1], cost_colors(result.costmap, result.params), spec, "geometric cost map + A* path")
    add_cost_legend(axes[1], result.params)
    show_top_down(axes[2], label_colors(result.labels, id_to_color), spec, "ground-truth class + A* path")
    add_class_legend(axes[2], result.labels, id_to_name, id_to_color)

    for ax in axes[1:]:
        draw_camera_fov(ax, calib.K, image.shape[1], spec)
        tx, ty = result.target_xy
        ax.plot(ty, tx, "wX", markersize=14, markeredgecolor="k")
        if result.goal is not None:
            gx, gy = spec.cell_center(*result.goal)
            ax.plot(gy, gx, "r*", markersize=16)
        if result.found:
            path_xy = result.path_xy()
            ax.plot(path_xy[:, 1], path_xy[:, 0], "r-", linewidth=2.5)

    fig.tight_layout()
    return fig
