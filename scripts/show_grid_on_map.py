"""Show where each image grid cell lands on the top-down map for one frame (phase 5, step 1).

Every 25 cm map cell's ground point is projected into the camera image to find the grid cell
("F7", ...) it appears in. The picture has three panels: the camera image with the grid, the
top-down map painted by grid cell (one color per row, darker on odd columns, labels at the
middle of each cell's footprint), and the geometric cost map with the grid-cell borders on top.
It also prints how far ahead each grid row reaches on the ground.

Check: the bottom row (H) should cover the ground a few meters ahead, higher rows farther
away, column 1 on the left, and the footprints should line up with the dashed camera edges.

Usage (from the repo root):
    python scripts/show_grid_on_map.py --sequence 0 --frame 120
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

from offroad.data.rellis import RellisSequence  # noqa: E402
from offroad.fusion.grid_lookup import GridLookup, build_grid_lookup  # noqa: E402
from offroad.mapping.costmap import CostParams, geometric_cost_map  # noqa: E402
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu  # noqa: E402
from offroad.viz.grid_overlay import draw_grid  # noqa: E402
from offroad.viz.topdown import cost_colors, draw_camera_fov, show_top_down  # noqa: E402
from offroad.vlm.grid import ImageGrid  # noqa: E402

NOT_VISIBLE_RGB = (1.0, 1.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="frame index within the sequence (default: 0)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: outputs/gridmap_<seq>_<frame>.png)")
    return parser.parse_args()


def grid_cell_colors(lookup: GridLookup) -> np.ndarray:
    """(nx, ny, 3) RGB: one color per grid row, darker on odd columns, white where not in view."""
    rgb = np.full((*lookup.cell_id.shape, 3), NOT_VISIBLE_RGB)
    visible = lookup.visible
    rows, cols = lookup.cell_id[visible] // lookup.grid.cols, lookup.cell_id[visible] % lookup.grid.cols
    base = np.array(matplotlib.colormaps["tab10"](rows % 10))[:, :3]
    rgb[visible] = base * np.where(cols % 2 == 1, 0.7, 1.0)[:, None]
    return rgb


def cell_borders(lookup: GridLookup) -> np.ndarray:
    """(nx, ny) bool, True where a map cell's grid cell differs from the next cell ahead or to the side."""
    ids = lookup.cell_id
    border = np.zeros(ids.shape, dtype=bool)
    border[:-1, :] |= (ids[:-1, :] != ids[1:, :]) & lookup.visible[:-1, :] & lookup.visible[1:, :]
    border[:, :-1] |= (ids[:, :-1] != ids[:, 1:]) & lookup.visible[:, :-1] & lookup.visible[:, 1:]
    return border


def label_footprints(ax, lookup: GridLookup, min_cells: int = 6):
    """Write each grid cell's name at the middle of its ground footprint."""
    for cell_id in np.unique(lookup.cell_id[lookup.visible]):
        label = lookup.label_of(int(cell_id))
        if lookup.mask(label).sum() < min_cells:
            continue
        x, y = lookup.centroid(label)
        ax.text(y, x, label, fontsize=6, ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.7, "linewidth": 0})


def print_row_ranges(lookup: GridLookup):
    """How far ahead (meters) the ground in each grid row lies."""
    spec, grid = lookup.spec, lookup.grid
    xs = spec.x_min + (np.arange(spec.shape[0]) + 0.5) * spec.resolution
    print("Ground covered by each grid row (forward distance, meters):")
    for row in range(grid.rows):
        in_row = lookup.visible & (lookup.cell_id // grid.cols == row)
        if not in_row.any():
            print(f"  row {grid.label(row, 0)[0]}: no ground in the map (sky or beyond {spec.x_max:.0f} m)")
            continue
        forward = xs[np.nonzero(in_row)[0]]
        print(f"  row {grid.label(row, 0)[0]}: {forward.min():5.1f} - {forward.max():5.1f} m  ({in_row.sum():5d} map cells)")


def main() -> None:
    args = parse_args()
    seq = RellisSequence(args.sequence, root=args.root)
    frame = seq[args.frame]
    spec = GridSpec()

    points_flu = lidar_to_flu(frame.points)
    costmap = geometric_cost_map(points_flu, build_elevation_map(points_flu, spec), CostParams())
    grid = ImageGrid(height=frame.image.shape[0], width=frame.image.shape[1])
    lookup = build_grid_lookup(costmap.ground_z, spec, frame.calib.T_cam_from_lidar, frame.calib.K, grid)

    fig, axes = plt.subplots(1, 3, figsize=(24, 9), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    axes[0].imshow(draw_grid(frame.image, grid))
    axes[0].set_title(f"camera with grid, sequence {seq.sequence} frame {args.frame}")
    axes[0].axis("off")

    show_top_down(axes[1], grid_cell_colors(lookup), spec, "ground covered by each grid cell")
    label_footprints(axes[1], lookup)
    draw_camera_fov(axes[1], frame.calib.K, grid.width, spec)

    rgb = cost_colors(costmap, CostParams())
    rgb[cell_borders(lookup)] = (1.0, 0.0, 1.0)
    show_top_down(axes[2], rgb, spec, "cost map with grid-cell borders (magenta)")
    label_footprints(axes[2], lookup)
    draw_camera_fov(axes[2], frame.calib.K, grid.width, spec)

    fig.tight_layout()
    out = args.out or REPO_ROOT / "outputs" / f"gridmap_{seq.sequence}_{args.frame:06d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)

    nx, ny = spec.shape
    print(f"Map cells in the camera's view: {lookup.visible.sum()} of {nx * ny} "
          f"({lookup.visible.mean():.0%}); {(lookup.visible & lookup.ground_filled).sum()} of them have filled-in ground")
    print_row_ranges(lookup)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
