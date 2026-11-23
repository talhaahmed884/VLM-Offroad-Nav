"""Show the path the driver actually took from one frame, from the scan poses (phase 1 check).

Three panels: the camera image with the next 20 m of the driven path (large cyan dots; the
part beyond, up to 50 m, smaller), the geometric cost map with the driven path, and the whole
recorded drive seen from above (decision frames from configs/decision_frames.csv, if it exists,
as dots; this frame as a red arrow).

Check: the dots on the camera image should lie on the trail the vehicle drove next, and the
printed sensor height should be steady along the path (the poses' heights agree with the ground
the LiDAR sees).

Usage (from the repo root):
    python scripts/show_driven_path.py --sequence 0 --frame 600
"""

import argparse
import csv
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
from offroad.mapping.costmap import CostParams, geometric_cost_map  # noqa: E402
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu  # noqa: E402
from offroad.nav.driven_path import (  # noqa: E402
    DIRECTION_DISTANCE_M,
    direction_bearing_deg,
    direction_word,
    driven_path_flu,
    path_length,
)
from offroad.nav.frames import MIN_PATH_AHEAD_M, speed_ahead  # noqa: E402
from offroad.viz.baseline_figure import draw_flu_points_on_camera  # noqa: E402
from offroad.viz.topdown import add_cost_legend, cost_colors, draw_camera_fov, show_top_down  # noqa: E402

NEAR_RGB, FAR_RGB = (0, 255, 255), (0, 140, 255)
DEFAULT_SENSOR_HEIGHT_M = 1.2  # LiDAR height above the ground, used if the map has no ground under the path
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=600, help="frame index within the sequence (default: 600)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: outputs/driven_<seq>_<frame>.png)")
    return parser.parse_args()


def decision_frames_of(sequence: str) -> list[dict]:
    path = REPO_ROOT / "configs" / "decision_frames.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["sequence"] == sequence]


def sensor_height_along(path_flu: np.ndarray, ground_z: np.ndarray, spec: GridSpec) -> np.ndarray:
    """LiDAR height above the mapped ground at each driven-path point that has known ground, meters."""
    nx, ny = spec.shape
    heights = []
    for x, y, z in path_flu:
        i, j = spec.cell_index(x, y)
        if 0 <= i < nx and 0 <= j < ny and np.isfinite(ground_z[i, j]):
            heights.append(z - ground_z[i, j])
    return np.array(heights)


def main() -> None:
    args = parse_args()
    seq = RellisSequence(args.sequence, root=args.root)
    frame = seq[args.frame]
    poses = seq.poses
    times = np.array([seq.timestamp(i) for i in seq.frame_indices])
    spec = GridSpec()

    long_path = driven_path_flu(poses, args.frame, DIRECTION_DISTANCE_M)
    near = driven_path_flu(poses, args.frame, MIN_PATH_AHEAD_M)
    bearing = direction_bearing_deg(long_path)
    word = direction_word(bearing)
    speed = speed_ahead(poses[:, :3, 3], times, args.frame)

    points_flu = lidar_to_flu(frame.points)
    costmap = geometric_cost_map(points_flu, build_elevation_map(points_flu, spec), CostParams())
    heights = sensor_height_along(near, costmap.ground_z, spec)

    # Draw the path on the ground: the LiDAR's own (smooth) height from the poses, minus its measured
    # height above the ground. The cost map's ground is noisier from cell to cell.
    sensor_height = float(np.median(heights)) if len(heights) else DEFAULT_SENSOR_HEIGHT_M
    ground_offset = np.array([0.0, 0.0, sensor_height])
    fig, axes = plt.subplots(1, 3, figsize=(25, 8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    camera = draw_flu_points_on_camera(frame.image, frame.calib, long_path - ground_offset, FAR_RGB, radius=4)
    camera = draw_flu_points_on_camera(camera, frame.calib, near - ground_offset, NEAR_RGB, radius=8)
    axes[0].imshow(camera)
    axes[0].set_title(f"sequence {seq.sequence} frame {args.frame}: driven path (next {MIN_PATH_AHEAD_M:.0f} m large, "
                      f"to {DIRECTION_DISTANCE_M:.0f} m small); Mode B: {word or '-'}")
    axes[0].axis("off")

    show_top_down(axes[1], cost_colors(costmap, CostParams()), spec, "cost map + driven path")
    add_cost_legend(axes[1], CostParams())
    draw_camera_fov(axes[1], frame.calib.K, frame.image.shape[1], spec)
    axes[1].plot(long_path[:, 1], long_path[:, 0], "-", color=np.array(FAR_RGB) / 255, linewidth=2)
    axes[1].plot(near[:, 1], near[:, 0], "-", color=np.array(NEAR_RGB) / 255, linewidth=3.5)

    world = poses[:, :3, 3]
    ax = axes[2]
    ax.plot(world[:, 0], world[:, 1], "-", color="#c3c2b7", linewidth=1.5, label="whole drive")
    decisions = decision_frames_of(seq.sequence)
    if decisions:
        idx = np.array([int(r["frame"]) for r in decisions])
        ax.scatter(world[idx, 0], world[idx, 1], s=6, color=SPLIT_COLORS.get(decisions[0]["split"], "k"),
                   label=f"decision frames ({decisions[0]['split']})")
    forward = poses[args.frame, :3, :3] @ np.array([-1.0, 0.0, 0.0])  # LiDAR forward is -x
    ax.annotate("", xy=world[args.frame, :2] + 8 * forward[:2], xytext=world[args.frame, :2],
                arrowprops={"arrowstyle": "-|>", "color": "red", "lw": 2})
    ax.plot(*world[0, :2], "ko", markersize=6, label="start")
    ax.set_aspect("equal")
    ax.set_title(f"whole drive, {path_length(world[:, :2]):.0f} m (world frame, from above)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = args.out or REPO_ROOT / "outputs" / f"driven_{seq.sequence}_{args.frame:06d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)

    print(f"Sequence {seq.sequence}, frame {args.frame}: driven path continues {path_length(long_path):.1f} m "
          f"(up to {DIRECTION_DISTANCE_M:.0f} m checked); speed {speed:.2f} m/s over the next second")
    print(f"Mode B direction: bearing {'-' if bearing is None else f'{bearing:+.1f} deg'} -> {word or 'none'}")
    if len(heights):
        print(f"LiDAR height above the mapped ground along the next {MIN_PATH_AHEAD_M:.0f} m: median "
              f"{np.median(heights):.2f} m, spread (5-95%) {np.percentile(heights, 5):.2f} to "
              f"{np.percentile(heights, 95):.2f} m over {len(heights)} points")
    else:
        print("No mapped ground under the driven path (all in the blind circle or unknown)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
