"""Picture comparing the baseline and hybrid paths for one frame.

Panels: camera image with both paths, the VLM-adjusted cost map with both paths and the
VLM's route, and the ground-truth classes with both paths. Baseline = red, hybrid = cyan.
Scripts using this should call matplotlib.use("Agg") before importing it if no display is available.
"""

import matplotlib.pyplot as plt
import numpy as np

from offroad.data.calibration import CameraCalibration
from offroad.planning.baseline import BaselineResult
from offroad.planning.hybrid import HybridResult
from offroad.viz.baseline_figure import draw_path_on_camera
from offroad.viz.topdown import add_class_legend, draw_camera_fov, label_colors, show_cost_values, show_top_down

BASELINE_RGB = (255, 0, 0)
HYBRID_RGB = (0, 255, 255)


def path_status(result: BaselineResult) -> str:
    """Short text: "reached", "stops 3.2 m short", or "no path"."""
    if not result.found:
        return "no path"
    return "reached" if result.reached else f"stops {result.shortfall_m:.1f} m short"


def draw_hybrid_figure(
    image: np.ndarray,
    calib: CameraCalibration,
    result: HybridResult,
    id_to_name: dict[int, str],
    id_to_color: dict[int, tuple[int, int, int]],
    title: str,
) -> plt.Figure:
    """Three panels comparing the two plans (see the module docstring). Returns the figure; the caller saves and closes it."""
    base, hybrid, spec = result.baseline, result.hybrid, result.baseline.spec
    fig, axes = plt.subplots(1, 3, figsize=(26, 9), gridspec_kw={"width_ratios": [1.6, 1.1, 1]})

    camera = image
    for plan, rgb in ((base, BASELINE_RGB), (hybrid, HYBRID_RGB)):
        if plan.found:
            camera = draw_path_on_camera(camera, calib, plan, rgb)
    axes[0].imshow(camera)
    axes[0].set_title(f"{title}\nbaseline (red): {path_status(base)}   |   hybrid (cyan): {path_status(hybrid)}")
    axes[0].axis("off")

    show_cost_values(axes[1], fig, hybrid.cost, spec, "VLM-adjusted cost map (white = VLM route)")
    show_top_down(axes[2], label_colors(base.labels, id_to_color), spec, "ground-truth class (LiDAR points)")
    add_class_legend(axes[2], base.labels, id_to_name, id_to_color)

    route = np.array([(0.0, 0.0), *result.fused.route_xy]) if result.fused.route_xy else None
    for ax in axes[1:]:
        draw_camera_fov(ax, calib.K, image.shape[1], spec)
        tx, ty = base.target_xy
        ax.plot(ty, tx, "wX", markersize=14, markeredgecolor="k")
        if route is not None and ax is axes[1]:
            ax.plot(route[:, 1], route[:, 0], "w--", linewidth=1.5)
        for plan, rgb in ((base, BASELINE_RGB), (hybrid, HYBRID_RGB)):
            if plan.found:
                xy = plan.path_xy()
                ax.plot(xy[:, 1], xy[:, 0], "-", color=np.array(rgb) / 255, linewidth=2.5)

    fig.tight_layout()
    return fig
