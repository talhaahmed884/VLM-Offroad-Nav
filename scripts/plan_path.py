"""Plan a geometry-only A* path for one RELLIS-3D frame and save a picture of it.

Builds the elevation map and geometric cost map, frees the vehicle's own footprint,
and plans from the vehicle to a goal straight ahead (the nearest drivable cell to
it, if the exact spot is blocked). The picture shows the path on the camera image,
on the cost map, and on the ground-truth classes.

Usage (from the repo root):
    python scripts/plan_path.py --sequence 0 --frame 0
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file only; also works without a display (e.g. on the cluster)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.geometry.projection import project_points_to_image, transform_points  # noqa: E402
from offroad.geometry.transforms import invert_transform  # noqa: E402
from offroad.mapping.costmap import CostParams, geometric_cost_map  # noqa: E402
from offroad.mapping.elevation import (  # noqa: E402
    T_FLU_FROM_LIDAR,
    GridSpec,
    build_elevation_map,
    lidar_to_flu,
    majority_label_grid,
)
from offroad.planning.astar import astar, clear_footprint, nearest_free_cell, path_length  # noqa: E402
from offroad.viz.overlay import draw_points  # noqa: E402
from offroad.viz.topdown import (  # noqa: E402
    add_class_legend,
    add_cost_legend,
    cost_colors,
    draw_camera_fov,
    label_colors,
    show_top_down,
)

PATH_RGB = (255, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="frame index within the sequence (default: 0)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--goal-distance", type=float, default=20.0, help="goal distance straight ahead, meters (default: 20)")
    parser.add_argument("--footprint-radius", type=float, default=2.0,
                        help="cells this close to the vehicle are treated as drivable, meters (default: 2)")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: outputs/path_<seq>_<frame>.png)")
    return parser.parse_args()


def path_on_camera(path_xy: np.ndarray, ground_z: np.ndarray, path: list, frame) -> np.ndarray:
    """Project the path (on the estimated ground) into the camera image; returns the image with red dots."""
    z = np.array([ground_z[c] for c in path])
    z = np.where(np.isnan(z), np.nanmedian(ground_z), z)  # unknown ground: use the typical ground height
    points_flu = np.column_stack([path_xy, z])
    points_lidar = transform_points(invert_transform(T_FLU_FROM_LIDAR), points_flu)
    uv, _, _ = project_points_to_image(points_lidar, frame.calib.T_cam_from_lidar, frame.calib.K, frame.image.shape[:2])
    colors = np.tile(np.array(PATH_RGB, dtype=np.uint8), (len(uv), 1))
    return draw_points(frame.image, uv, colors, radius=6)


def main() -> None:
    args = parse_args()
    seq = RellisSequence(args.sequence, root=args.root)
    frame = seq[args.frame]
    id_to_name, id_to_color = load_ontology(args.root)
    spec = GridSpec()
    params = CostParams()

    points_flu = lidar_to_flu(frame.points)
    elevation = build_elevation_map(points_flu, spec)
    costmap = geometric_cost_map(points_flu, elevation, params)
    labels = majority_label_grid(points_flu, frame.point_labels, spec)

    cost = clear_footprint(costmap.cost, spec, args.footprint_radius)
    start = spec.cell_index(0.0, 0.0)
    goal = nearest_free_cell(cost, spec, args.goal_distance, 0.0)
    result = astar(cost, start, goal) if goal is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(24, 9), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    camera = frame.image
    if result is not None:
        path, total_cost = result
        path_xy = np.array([spec.cell_center(*c) for c in path])
        camera = path_on_camera(path_xy, costmap.ground_z, path, frame)
    axes[0].imshow(camera)
    axes[0].set_title(f"camera, sequence {seq.sequence} frame {args.frame} (path in red)")
    axes[0].axis("off")

    show_top_down(axes[1], cost_colors(costmap, params), spec, "geometric cost map + A* path")
    add_cost_legend(axes[1], params)
    show_top_down(axes[2], label_colors(labels, id_to_color), spec, "ground-truth class + A* path")
    add_class_legend(axes[2], labels, id_to_name, id_to_color)
    for ax in axes[1:]:
        draw_camera_fov(ax, frame.calib.K, frame.image.shape[1], spec)
        if goal is not None:
            gx, gy = spec.cell_center(*goal)
            ax.plot(gy, gx, "r*", markersize=16)
        if result is not None:
            ax.plot(path_xy[:, 1], path_xy[:, 0], "r-", linewidth=2.5)

    out_path = args.out or REPO_ROOT / "outputs" / f"path_{seq.sequence}_{args.frame:06d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)

    print(f"Sequence {seq.sequence}, frame {args.frame}: goal {args.goal_distance:.0f} m ahead", end="")
    if goal is None:
        print(" -> every cell is blocked, no goal")
    elif result is None:
        gx, gy = spec.cell_center(*goal)
        print(f" (using x={gx:.1f} m, y={gy:.1f} m) -> NO PATH: the goal is cut off by blocked cells")
    else:
        gx, gy = spec.cell_center(*goal)
        print(f" (using x={gx:.1f} m, y={gy:.1f} m) -> path found: {path_length(path, spec):.1f} m long, "
              f"total cost {total_cost:.1f}")
        classes = Counter(int(labels[c]) for c in path)
        shares = ", ".join(f"{id_to_name[c] if c else 'unseen'} {n / len(path):.0%}" for c, n in classes.most_common())
        print(f"Ground-truth classes along the path: {shares}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
