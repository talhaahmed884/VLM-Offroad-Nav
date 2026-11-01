"""Build a top-down elevation map from one RELLIS-3D Ouster scan and save a picture of it.

The picture has four panels: the camera image, ground height (lowest point per cell),
step height (highest minus lowest point per cell), and the most common ground-truth
class per cell. The top-down panels show the vehicle at the bottom, looking up, with
the camera's field of view drawn as two dashed cyan lines.

Usage (from the repo root):
    python scripts/build_elevation_map.py --sequence 0 --frame 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file only; also works without a display (e.g. on the cluster)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.mapping.elevation import (  # noqa: E402
    GridSpec,
    build_elevation_map,
    lidar_to_flu,
    majority_label_grid,
)
from offroad.viz.topdown import add_class_legend, draw_camera_fov, label_colors, show_top_down  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="frame index within the sequence (default: 0)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--resolution", type=float, default=0.25, help="cell size in meters (default: 0.25)")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: outputs/elevation_<seq>_<frame>.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq = RellisSequence(args.sequence, root=args.root)
    frame = seq[args.frame]
    id_to_name, id_to_color = load_ontology(args.root)
    spec = GridSpec(resolution=args.resolution)

    points_flu = lidar_to_flu(frame.points)
    elevation = build_elevation_map(points_flu, spec)
    labels = majority_label_grid(points_flu, frame.point_labels, spec)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0, 0].imshow(frame.image)
    axes[0, 0].set_title(f"camera, sequence {seq.sequence} frame {args.frame}")
    axes[0, 0].axis("off")

    ground_cmap = matplotlib.colormaps["terrain"].copy()
    ground_cmap.set_bad("lightgray")
    image = show_top_down(axes[0, 1], elevation.min_z, spec, "ground height: lowest point (m)", cmap=ground_cmap)
    fig.colorbar(image, ax=axes[0, 1], shrink=0.8)

    step_cmap = matplotlib.colormaps["magma"].copy()
    step_cmap.set_bad("lightgray")
    image = show_top_down(
        axes[1, 0], elevation.step_height, spec, "step height: highest - lowest (m)", cmap=step_cmap, vmin=0, vmax=2
    )
    fig.colorbar(image, ax=axes[1, 0], shrink=0.8)

    show_top_down(axes[1, 1], label_colors(labels, id_to_color), spec, "most common ground-truth class")
    add_class_legend(axes[1, 1], labels, id_to_name, id_to_color)

    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        draw_camera_fov(ax, frame.calib.K, frame.image.shape[1], spec)

    out_path = args.out or REPO_ROOT / "outputs" / f"elevation_{seq.sequence}_{args.frame:06d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)

    has_data = elevation.count > 0
    print(f"Sequence {seq.sequence}, frame {args.frame}: {elevation.count.sum()} of {len(points_flu)} points inside the "
          f"{spec.x_max - spec.x_min:.0f} m x {spec.y_max - spec.y_min:.0f} m grid; {has_data.mean():.0%} of cells have points")
    print(f"Ground height (lowest point per cell): median {np.nanmedian(elevation.min_z):.2f} m, "
          f"5th-95th percentile {np.nanpercentile(elevation.min_z, 5):.2f} to {np.nanpercentile(elevation.min_z, 95):.2f} m")
    print(f"Cells with step height > 0.3 m: {np.mean(elevation.step_height[has_data] > 0.3):.0%}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
