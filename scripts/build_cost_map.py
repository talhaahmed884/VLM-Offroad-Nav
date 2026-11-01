"""Build the geometry-only cost map for one RELLIS-3D frame and save a picture of it.

The picture has four panels: the camera image, obstacle height above the local ground,
the cost map (green-yellow = drivable, darker yellow = rougher, gray = unknown,
black = blocked), and the most common ground-truth class per cell for comparison.

It also prints, for each ground-truth class, how much of it the geometry calls an
obstacle. Mud and puddles at 0% and bushes near 100% show the baseline's blind spots.

Usage (from the repo root):
    python scripts/build_cost_map.py --sequence 0 --frame 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file only; also works without a display (e.g. on the cluster)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.mapping.costmap import CostMap, CostParams, geometric_cost_map  # noqa: E402
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu, majority_label_grid  # noqa: E402
from offroad.viz.topdown import add_class_legend, draw_camera_fov, label_colors, show_top_down  # noqa: E402

UNKNOWN_RGB = (0.8, 0.8, 0.8)
BLOCKED_RGB = (0.0, 0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="frame index within the sequence (default: 0)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--max-step", type=float, default=CostParams.max_step,
                        help=f"obstacle threshold in meters (default: {CostParams.max_step})")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: outputs/costmap_<seq>_<frame>.png)")
    return parser.parse_args()


def cost_colors(costmap: CostMap, params: CostParams) -> np.ndarray:
    """(nx, ny, 3) RGB picture of the cost map: drivable cells colored by cost, unknown gray, blocked black."""
    drivable_max = 1.0 + params.roughness_weight
    scaled = np.clip((costmap.cost - 1.0) / (drivable_max - 1.0), 0.0, 1.0)
    rgb = matplotlib.colormaps["summer"](scaled)[..., :3]
    rgb[costmap.unknown] = UNKNOWN_RGB
    rgb[np.isinf(costmap.cost)] = BLOCKED_RGB
    return rgb


def print_class_stats(costmap: CostMap, labels: np.ndarray, id_to_name: dict[int, str], min_cells: int = 20):
    """For each ground-truth class, the share of its cells the geometry marks as an obstacle."""
    print("Share of each ground-truth class marked as obstacle by geometry (cells with points only):")
    known = ~costmap.unknown
    for class_id in np.unique(labels[known]):
        cells = known & (labels == class_id)
        if class_id == 0 or cells.sum() < min_cells:
            continue
        share = costmap.obstacle[cells].mean()
        print(f"  {id_to_name[class_id]:9s} {share:6.0%}  of {cells.sum():5d} cells")


def main() -> None:
    args = parse_args()
    seq = RellisSequence(args.sequence, root=args.root)
    frame = seq[args.frame]
    id_to_name, id_to_color = load_ontology(args.root)
    spec = GridSpec()
    params = CostParams(max_step=args.max_step)

    points_flu = lidar_to_flu(frame.points)
    elevation = build_elevation_map(points_flu, spec)
    costmap = geometric_cost_map(points_flu, elevation, params)
    labels = majority_label_grid(points_flu, frame.point_labels, spec)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0, 0].imshow(frame.image)
    axes[0, 0].set_title(f"camera, sequence {seq.sequence} frame {args.frame}")
    axes[0, 0].axis("off")

    height_cmap = matplotlib.colormaps["magma"].copy()
    height_cmap.set_bad("lightgray")
    image = show_top_down(
        axes[0, 1], costmap.obstacle_height, spec, "obstacle height above local ground (m)", cmap=height_cmap, vmin=0, vmax=1
    )
    fig.colorbar(image, ax=axes[0, 1], shrink=0.8)

    show_top_down(axes[1, 0], cost_colors(costmap, params), spec, f"geometric cost map (obstacle > {params.max_step} m)")
    cost_legend = [
        (matplotlib.colormaps["summer"](0.0), "smooth (cost 1)"),
        (matplotlib.colormaps["summer"](1.0), f"rough (cost {1 + params.roughness_weight:.0f})"),
        (UNKNOWN_RGB, f"unknown (cost {params.unknown_cost:.0f})"),
        (BLOCKED_RGB, "blocked"),
    ]
    axes[1, 0].legend(
        [Rectangle((0, 0), 1, 1, color=color) for color, _ in cost_legend],
        [text for _, text in cost_legend],
        loc="upper right",
        fontsize=8,
    )

    show_top_down(axes[1, 1], label_colors(labels, id_to_color), spec, "most common ground-truth class")
    add_class_legend(axes[1, 1], labels, id_to_name, id_to_color)

    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        draw_camera_fov(ax, frame.calib.K, frame.image.shape[1], spec)

    out_path = args.out or REPO_ROOT / "outputs" / f"costmap_{seq.sequence}_{args.frame:06d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)

    print(f"Sequence {seq.sequence}, frame {args.frame}: {np.isinf(costmap.cost).mean():.0%} of cells blocked, "
          f"{costmap.unknown.mean():.0%} unknown")
    print_class_stats(costmap, labels, id_to_name)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
