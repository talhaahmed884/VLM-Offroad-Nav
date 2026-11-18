"""Check the two-source ground truth along the baseline paths (phase 6, step 1).

Each path cell gets its class from the LiDAR points if it has any, else from the camera
label image at the cell's ground pixel, else it stays unseen ("occluded" when something
stands in front of it in the camera). This prints, for each evaluation frame, how much of
the baseline path each source covers, and the water / mud / bush shares from LiDAR alone
vs from both sources. With --sequence and --frame it checks one frame and also saves
outputs/path_truth_<seq>_<frame>.png (path dots colored by source, on the camera image and
on the label image).

Usage (from the repo root):
    python scripts/check_path_truth.py                          # all evaluation frames
    python scripts/check_path_truth.py --sequence 0 --frame 120 # one frame, with a picture
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.eval.path_truth import (  # noqa: E402
    SOURCE_CAMERA,
    SOURCE_LIDAR,
    SOURCE_OCCLUDED,
    SOURCE_UNSEEN,
    UNSEEN,
    fill_ground,
    path_truth,
    project_cells,
)
from offroad.planning.baseline import run_geometric_baseline  # noqa: E402
from offroad.viz.overlay import draw_points  # noqa: E402
from offroad.viz.vlm_answer import colorize_labels  # noqa: E402

GROUPS = {"water": ("water", "puddle"), "mud": ("mud",), "bush": ("bush",)}
SOURCE_RGB = {SOURCE_LIDAR: (0, 220, 0), SOURCE_CAMERA: (0, 120, 255), SOURCE_OCCLUDED: (255, 150, 0),
              SOURCE_UNSEEN: (128, 128, 128)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=None, help="check one frame: sequence number, 0-4")
    parser.add_argument("--frame", type=int, default=None, help="check one frame: frame index")
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "configs" / "eval_frames.csv")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    args = parser.parse_args()
    if (args.sequence is None) != (args.frame is None):
        parser.error("give both --sequence and --frame, or neither")
    return args


def group_share(shares: dict[int, float], names: tuple[str, ...], id_to_name: dict[int, str]) -> float:
    return sum(v for c, v in shares.items() if id_to_name.get(c) in names)


def save_picture(frame, result, truth, id_to_color, out: Path):
    """Camera image and label image side by side, with the path dots colored by their source."""
    uv, _ = project_cells(result.path, fill_ground(result.costmap.ground_z), result.spec,
                          frame.calib.T_cam_from_lidar, frame.calib.K, frame.image.shape[:2])
    visible = ~np.isnan(uv[:, 0])
    colors = np.array([SOURCE_RGB[s] for s in truth.source], dtype=np.uint8)[visible]
    panels = [draw_points(frame.image, uv[visible], colors, radius=7)]
    if frame.label_image is not None:
        panels.append(draw_points(colorize_labels(frame.label_image, id_to_color), uv[visible], colors, radius=7))
    picture = np.concatenate(panels, axis=1)
    cv2.imwrite(str(out), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    id_to_name, id_to_color = load_ontology(args.root)
    ignore_ids = tuple(c for c, name in id_to_name.items() if name in ("void", "sky")) + (UNSEEN,)
    if args.sequence is not None:
        entries = [{"category": "single", "sequence": f"{args.sequence:05d}", "frame": str(args.frame)}]
    else:
        with open(args.frames, newline="") as f:
            entries = list(csv.DictReader(f))

    sequences: dict[str, RellisSequence] = {}
    rows = []
    print(f"{'frame':20s} {'lidar':>6s} {'camera':>7s} {'occl.':>6s} {'unseen':>7s}   "
          + "  ".join(f"{g + ' L->L+C':>14s}" for g in GROUPS))
    for entry in entries:
        seq_id, frame_index = entry["sequence"], int(entry["frame"])
        if seq_id not in sequences:
            sequences[seq_id] = RellisSequence(seq_id, root=args.root)
        frame = sequences[seq_id][frame_index]
        result = run_geometric_baseline(frame.points, frame.point_labels)
        if not result.found:
            print(f"{entry['category']:6s} {seq_id}/{frame_index:<6d} no path")
            continue
        truth = path_truth(result.path, result.labels, result.costmap.ground_z, result.spec,
                           frame.calib.T_cam_from_lidar, frame.calib.K, frame.label_image,
                           points_lidar=frame.points, ignore_ids=ignore_ids)
        sources = truth.source_shares()
        lidar_only, both = result.class_shares(), truth.class_shares()
        row = {"category": entry["category"], **sources,
               **{f"{g}_lidar": group_share(lidar_only, n, id_to_name) for g, n in GROUPS.items()},
               **{f"{g}_both": group_share(both, n, id_to_name) for g, n in GROUPS.items()}}
        rows.append(row)
        print(f"{entry['category']:6s} {seq_id}/{frame_index:<6d}  {sources['lidar']:6.0%} {sources['camera']:7.0%} "
              f"{sources['occluded']:6.0%} {sources['unseen']:7.0%}   "
              + "  ".join(f"{row[g + '_lidar']:6.0%} -> {row[g + '_both']:4.0%}" for g in GROUPS))
        if args.sequence is not None:
            out = REPO_ROOT / "outputs" / f"path_truth_{seq_id}_{frame_index:06d}.png"
            save_picture(frame, result, truth, id_to_color, out)
            print(f"Saved {out} (dots: green = LiDAR, blue = camera, orange = occluded, gray = unseen)")

    if len(rows) > 1:
        print("\nMean per category:")
        for category in [*dict.fromkeys(r["category"] for r in rows), "all"]:
            group = rows if category == "all" else [r for r in rows if r["category"] == category]
            mean = {k: np.mean([r[k] for r in group]) for k in group[0] if k != "category"}
            print(f"{category:13s} {mean['lidar']:6.0%} {mean['camera']:7.0%} {mean['occluded']:6.0%} "
                  f"{mean['unseen']:7.0%}   "
                  + "  ".join(f"{mean[g + '_lidar']:6.0%} -> {mean[g + '_both']:4.0%}" for g in GROUPS))
        print("(shares of the baseline path's cells; 'L->L+C' = class share from LiDAR alone -> LiDAR + camera)")


if __name__ == "__main__":
    main()
