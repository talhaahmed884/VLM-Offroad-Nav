"""Run the geometry-only baseline on every evaluation frame and save pictures plus a summary table.

Reads configs/eval_frames.csv. For each frame it plans the A* path (see plan_path.py),
saves outputs/baseline/<category>_<seq>_<frame>.png, and writes one row to
outputs/baseline/summary.csv: whether the target was reached and how far short the path
stops, path length, detour ratio, cost, and the share of the path crossing each
ground-truth class ("unseen" = cells with no LiDAR points). At the end it prints a
short summary per category.

Usage (from the repo root):
    python scripts/run_baseline.py
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

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.planning.baseline import REACHED_TOLERANCE_M, UNSEEN, run_geometric_baseline  # noqa: E402
from offroad.viz.baseline_figure import draw_baseline_figure  # noqa: E402

# Class groups reported in the printed per-category summary.
SUMMARY_GROUPS = {
    "water": ("puddle", "water"),
    "mud": ("mud",),
    "bush": ("bush",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "configs" / "eval_frames.csv")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--goal-distance", type=float, default=20.0, help="target distance straight ahead, meters (default: 20)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "baseline")
    return parser.parse_args()


def fmt(value: float | None, digits: int = 2) -> str:
    """Number as text for the CSV; empty if None."""
    return "" if value is None else f"{value:.{digits}f}"


def mean_of(values: list[str]) -> float:
    """Mean of the non-empty numbers in a CSV column; nan if there are none."""
    numbers = [float(v) for v in values if v != ""]
    return float(np.mean(numbers)) if numbers else float("nan")


def print_category_summary(rows: list[dict]):
    print(f"\n{'category':9s} {'frames':>6s} {'reached':>7s} {'short m':>7s} {'detour':>7s}  "
          + "  ".join(f"{name + ' %':>8s}" for name in SUMMARY_GROUPS) + f"  {'unseen %':>8s}")
    for category in dict.fromkeys(row["category"] for row in rows):
        group = [r for r in rows if r["category"] == category]
        found = [r for r in group if r["path_found"] == "yes"]
        reached = sum(r["reached"] == "yes" for r in group)
        shortfall = mean_of([r["shortfall_m"] for r in group])
        detour = mean_of([r["detour_ratio"] for r in found])
        shares = []
        for names in list(SUMMARY_GROUPS.values()) + [("unseen",)]:
            per_frame = [sum(float(r[f"share_{n}"]) for n in names) for r in found]
            shares.append(np.mean(per_frame) if per_frame else float("nan"))
        print(f"{category:9s} {len(group):6d} {reached:7d} {shortfall:7.1f} {detour:7.2f}  "
              + "  ".join(f"{s:8.0%}" for s in shares))
    print(f"(reached = path ends within {REACHED_TOLERANCE_M:.0f} m of the target; short m = mean distance it "
          "stops short; detour and class shares are averaged over the paths)")


def main() -> None:
    args = parse_args()
    id_to_name, id_to_color = load_ontology(args.root)
    class_ids = sorted(c for c in id_to_name if c != UNSEEN)
    with open(args.frames, newline="") as f:
        eval_frames = list(csv.DictReader(f))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sequences: dict[str, RellisSequence] = {}
    rows = []
    for k, entry in enumerate(eval_frames, start=1):
        seq_id, frame_index = entry["sequence"], int(entry["frame"])
        if seq_id not in sequences:
            sequences[seq_id] = RellisSequence(seq_id, root=args.root)
        frame = sequences[seq_id][frame_index]

        result = run_geometric_baseline(frame.points, frame.point_labels, goal_distance=args.goal_distance)
        title = f"{entry['category']}: sequence {seq_id} frame {frame_index}"
        fig = draw_baseline_figure(frame.image, frame.calib, result, id_to_name, id_to_color, title)
        fig.savefig(args.out_dir / f"{entry['category']}_{seq_id}_{frame_index:06d}.png", dpi=72)
        plt.close(fig)

        shares = result.class_shares()
        row = {
            "category": entry["category"],
            "sequence": seq_id,
            "frame": frame_index,
            "path_found": "yes" if result.found else "no",
            "reached": "yes" if result.reached else "no",
            "shortfall_m": fmt(result.shortfall_m),
            "path_length_m": fmt(result.length_m),
            "straight_m": fmt(result.straight_m),
            "detour_ratio": fmt(result.detour_ratio, 3),
            "total_cost": fmt(result.total_cost, 1),
            "share_unseen": fmt(shares.get(UNSEEN, 0.0), 3),
        }
        row.update({f"share_{id_to_name[c]}": fmt(shares.get(c, 0.0), 3) for c in class_ids})
        rows.append(row)

        if not result.found:
            status = "NO PATH (vehicle cannot move)"
        else:
            detour = f"{result.detour_ratio:.2f}" if result.detour_ratio is not None else "n/a"
            reach = "reached" if result.reached else f"stops {result.shortfall_m:.1f} m short"
            status = f"{reach:18s} path {result.length_m:4.1f} m, detour {detour}"
        print(f"[{k:2d}/{len(eval_frames)}] {entry['category']:6s} {seq_id}/{frame_index:<5d} {status}")

    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print_category_summary(rows)
    print(f"\nSaved {len(rows)} pictures and summary.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
