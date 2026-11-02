"""Plan with and without the VLM on every evaluation frame, and compare (phase 5, step 4).

For each frame in configs/eval_frames.csv it loads the VLM answer saved by run_vlm.py,
plans the geometry-only baseline and the hybrid path (same A*, VLM-adjusted costs), and
saves outputs/hybrid/<run>/<category>_<seq>_<frame>.png (baseline red, hybrid cyan). It
writes outputs/hybrid/<run>/summary.csv with both plans' results side by side, and prints a
per-category comparison. No GPU needed: it only reads the saved answers.

Path class shares use the LiDAR point labels, so cells without LiDAR points count as
"unseen" (the evaluation phase adds the camera labels as a second source).

With --oracle it uses "perfect" hints made from the camera label images instead of a VLM
answer (see src/offroad/vlm/oracle.py): the upper bound of what the hybrid can do. Results
go to outputs/hybrid/oracle/.

Usage (from the repo root):
    python scripts/run_hybrid.py                                   # 8B answers
    python scripts/run_hybrid.py --run Qwen3-VL-4B-Instruct_4bit   # 4B answers
    python scripts/run_hybrid.py --oracle                          # hints from the label images
"""

import argparse
import csv
import json
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
from offroad.planning.baseline import REACHED_TOLERANCE_M, UNSEEN, BaselineResult  # noqa: E402
from offroad.planning.hybrid import run_hybrid_planner  # noqa: E402
from offroad.viz.hybrid_figure import draw_hybrid_figure, path_status  # noqa: E402
from offroad.vlm.grid import ImageGrid  # noqa: E402
from offroad.vlm.oracle import oracle_answer  # noqa: E402

DEFAULT_RUN = "Qwen3-VL-8B-Instruct_4bit"
NO_ANSWER = {"waypoints": [], "hints": []}  # plans exactly like the baseline
SUMMARY_GROUPS = {"water": ("puddle", "water"), "mud": ("mud",), "bush": ("bush",)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=DEFAULT_RUN, help=f"VLM run folder in outputs/vlm_runs (default: {DEFAULT_RUN})")
    parser.add_argument("--oracle", action="store_true", help="use hints from the camera label images instead of the VLM")
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "configs" / "eval_frames.csv")
    parser.add_argument("--goal-distance", type=float, default=20.0, help="target distance straight ahead, meters (default: 20)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output folder (default: outputs/hybrid/<run>)")
    return parser.parse_args()


def fmt(value: float | None, digits: int = 2) -> str:
    """Number as text for the CSV; empty if None."""
    return "" if value is None else f"{value:.{digits}f}"


def plan_columns(prefix: str, plan: BaselineResult, id_to_name: dict[int, str]) -> dict:
    """One plan's results as CSV columns, e.g. baseline_reached, hybrid_share_water."""
    shares = plan.class_shares()
    columns = {
        "reached": "yes" if plan.reached else "no",
        "shortfall_m": fmt(plan.shortfall_m),
        "length_m": fmt(plan.length_m),
        "detour_ratio": fmt(plan.detour_ratio, 3),
    }
    for group, names in SUMMARY_GROUPS.items():
        share = sum(v for c, v in shares.items() if id_to_name.get(c) in names) if plan.found else None
        columns[f"share_{group}"] = fmt(share, 3)
    columns["share_unseen"] = fmt(shares.get(UNSEEN, 0.0) if plan.found else None, 3)
    return {f"{prefix}_{key}": value for key, value in columns.items()}


def mean_of(rows: list[dict], column: str) -> float:
    values = [float(r[column]) for r in rows if r[column] != ""]
    return float(np.mean(values)) if values else float("nan")


def print_comparison(rows: list[dict]):
    """Per category: baseline -> hybrid for reached, shortfall, detour, and the class shares."""
    columns = [("reached", None), ("short m", "shortfall_m"), ("detour", "detour_ratio"),
               *[(f"{g} %", f"share_{g}") for g in SUMMARY_GROUPS]]
    print(f"\n{'category':9s} {'frames':>6s} {'apart m':>7s} {'max m':>6s}  "
          + "  ".join(f"{name:>13s}" for name, _ in columns))
    for category in [*dict.fromkeys(r["category"] for r in rows), "all"]:
        group = rows if category == "all" else [r for r in rows if r["category"] == category]
        cells = []
        for name, column in columns:
            if column is None:
                b = sum(r["baseline_reached"] == "yes" for r in group)
                h = sum(r["hybrid_reached"] == "yes" for r in group)
                cells.append(f"{b:d} -> {h:d}")
            elif name.endswith("%"):
                cells.append(f"{mean_of(group, 'baseline_' + column):.0%} -> {mean_of(group, 'hybrid_' + column):.0%}")
            else:
                cells.append(f"{mean_of(group, 'baseline_' + column):.2f} -> {mean_of(group, 'hybrid_' + column):.2f}")
        apart, max_apart = mean_of(group, "mean_separation_m"), mean_of(group, "max_separation_m")
        print(f"{category:9s} {len(group):6d} {apart:7.2f} {max_apart:6.2f}  " + "  ".join(f"{c:>13s}" for c in cells))
    print(f"(apart m / max m = mean and largest distance of the hybrid path from the baseline path. baseline -> "
          f"hybrid: reached = ends within {REACHED_TOLERANCE_M:.0f} m of the target; short m = mean distance it stops "
          "short; detour and class shares are averaged over frames, from LiDAR labels)")


def main() -> None:
    args = parse_args()
    run_dir = REPO_ROOT / "outputs" / "vlm_runs" / args.run
    source = "oracle" if args.oracle else args.run
    out_dir = args.out_dir or REPO_ROOT / "outputs" / "hybrid" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_name, id_to_color = load_ontology(args.root)
    with open(args.frames, newline="") as f:
        eval_frames = list(csv.DictReader(f))

    sequences: dict[str, RellisSequence] = {}
    rows = []
    for k, entry in enumerate(eval_frames, start=1):
        seq_id, frame_index, category = entry["sequence"], int(entry["frame"]), entry["category"]
        stem = f"{category}_{seq_id}_{frame_index:06d}"
        if seq_id not in sequences:
            sequences[seq_id] = RellisSequence(seq_id, root=args.root)
        frame = sequences[seq_id][frame_index]
        grid = ImageGrid(height=frame.image.shape[0], width=frame.image.shape[1])
        answer_path = run_dir / f"{stem}.json"
        if args.oracle:
            answer = oracle_answer(frame.label_image, grid, id_to_name) if frame.label_image is not None else NO_ANSWER
        elif answer_path.exists():
            answer = json.loads(answer_path.read_text(encoding="utf-8"))["answer"]
        else:
            answer = NO_ANSWER

        result = run_hybrid_planner(frame.points, frame.point_labels, frame.calib.T_cam_from_lidar, frame.calib.K,
                                    grid, answer, goal_distance=args.goal_distance)
        title = f"{category}: sequence {seq_id} frame {frame_index} ({source})"
        fig = draw_hybrid_figure(frame.image, frame.calib, result, id_to_name, id_to_color, title)
        fig.savefig(out_dir / f"{stem}.png", dpi=72)
        plt.close(fig)

        base_cost, fused = result.baseline.costmap.cost, result.fused
        finite = np.isfinite(base_cost) & np.isfinite(fused.cost)
        separation = result.path_separation()
        if answer is NO_ANSWER:
            status = "missing"
        elif args.oracle:
            status = "oracle"
        else:
            status = "valid" if answer["waypoints"] else "not usable"
        row = {
            "category": category,
            "sequence": seq_id,
            "frame": frame_index,
            "answer": status,
            "path_changed": "yes" if result.path_changed else "no",
            "mean_separation_m": fmt(separation[0] if separation else None),
            "max_separation_m": fmt(separation[1] if separation else None),
            "cells_hinted": int(fused.hints.hinted.sum()),
            "cells_unblocked": int(fused.unblocked.sum()),
            "cells_dearer": int((finite & (fused.cost > base_cost)).sum()),
            "cells_cheaper": int((finite & (fused.cost < base_cost)).sum()),
            "cells_blocked_by_vlm": int((np.isfinite(base_cost) & np.isinf(fused.cost)).sum()),
            **plan_columns("baseline", result.baseline, id_to_name),
            **plan_columns("hybrid", result.hybrid, id_to_name),
        }
        rows.append(row)
        apart = f"paths up to {separation[1]:4.1f} m apart" if separation else "no path to compare  "
        print(f"[{k:2d}/{len(eval_frames)}] {category:6s} {seq_id}/{frame_index:<5d} {apart}  "
              f"baseline {path_status(result.baseline):18s} hybrid {path_status(result.hybrid):18s} "
              f"(unblocked {row['cells_unblocked']}, dearer {row['cells_dearer']})", flush=True)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print_comparison(rows)
    print(f"\nSaved {len(rows)} pictures and summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
