"""Evaluate the baseline and every hybrid condition on all evaluation frames (phase 6, step 3).

Conditions: the geometry-only baseline, the hybrid planner with each saved VLM run (default:
the 4B and 8B answers in outputs/vlm_runs/), and the hybrid with oracle hints from the camera
labels (the upper bound). Every path is scored with the proposal's metrics, on two-source
(LiDAR + camera) ground truth. No GPU needed: it only reads saved answers.

Writes outputs/eval/:
    per_frame.csv   one row per frame and condition, with every metric
    summary.csv     mean of each metric per category and condition (plus "all")
and prints three tables: the overall comparison, a per-category view, and frame-by-frame
wins / losses / ties of each hybrid against the baseline.

Usage (from the repo root):
    python scripts/evaluate.py
    python scripts/evaluate.py --runs Qwen3-VL-8B-Instruct_4bit --no-oracle
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.eval.compare import BASELINE, evaluate_frame, paired_comparison  # noqa: E402
from offroad.vlm.grid import ImageGrid  # noqa: E402
from offroad.vlm.oracle import oracle_answer  # noqa: E402

DEFAULT_RUNS = ["Qwen3-VL-4B-Instruct_4bit", "Qwen3-VL-8B-Instruct_4bit"]
ORACLE = "oracle"
NO_ANSWER = {"waypoints": [], "hints": []}  # plans exactly like the baseline

METRIC_FIELDS = ["success", "reached", "goal_blocked", "shortfall_m", "length_m", "detour_ratio",
                 "share_water", "share_mud", "share_compliant", "share_noncompliant", "share_unseen",
                 "hard_dist_mean_m", "hard_dist_min_m", "slope_cm_per_m", "bump_height_m",
                 "compute_s", "vlm_s", "apart_mean_m", "apart_max_m"]
# (column, lower is better, tolerance for a tie) for the wins/losses table.
PAIRED = [("share_water", True, 0.01), ("share_mud", True, 0.01), ("shortfall_m", True, 0.25),
          ("hard_dist_min_m", False, 0.1), ("detour_ratio", True, 0.02)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, help="VLM run folders in outputs/vlm_runs")
    parser.add_argument("--no-oracle", action="store_true", help="leave out the oracle condition")
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "configs" / "eval_frames.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "eval")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    return parser.parse_args()


def short_name(run: str) -> str:
    """'Qwen3-VL-8B-Instruct_4bit' -> 'VLM-8B_4bit', so tables stay narrow."""
    return run.replace("Qwen3-VL-", "VLM-").replace("-Instruct", "")


def load_answer(run_dir: Path, stem: str) -> tuple[dict, float | None, str]:
    """(answer, VLM seconds, status) for one frame of a run; a missing answer plans like the baseline."""
    path = run_dir / f"{stem}.json"
    if not path.exists():
        return NO_ANSWER, None, "missing"
    record = json.loads(path.read_text(encoding="utf-8"))
    answer = record["answer"]
    return answer, record.get("seconds"), "valid" if answer["waypoints"] else "not usable"


def row_for(entry: dict, result, status: str) -> dict:
    m = result.metrics
    values = {**asdict(m), "success": m.success, "compute_s": result.compute_seconds, "vlm_s": result.vlm_seconds,
              "apart_mean_m": result.separation[0] if result.separation else None,
              "apart_max_m": result.separation[1] if result.separation else None}
    return {"category": entry["category"], "sequence": entry["sequence"], "frame": int(entry["frame"]),
            "condition": result.name, "answer": status, **{k: values[k] for k in METRIC_FIELDS}}


def mean(rows: list[dict], column: str) -> float | None:
    values = [float(r[column]) for r in rows if r[column] is not None]
    return float(np.mean(values)) if values else None


def fmt(value, kind: str = "num") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "s":
        return f"{value:.2f}" if value < 10 else f"{value:.1f}"
    return f"{value:.2f}"


def print_overall(rows: list[dict], conditions: list[str]):
    columns = [("success", "success", None), ("reached", "reached", None), ("short m", "shortfall_m", "num"),
               ("detour", "detour_ratio", "num"), ("water", "share_water", "pct"), ("mud", "share_mud", "pct"),
               ("veg ok", "share_compliant", "pct"), ("veg bad", "share_noncompliant", "pct"),
               ("unseen", "share_unseen", "pct"), ("hard avg", "hard_dist_mean_m", "num"),
               ("hard min", "hard_dist_min_m", "num"), ("slope", "slope_cm_per_m", "num"),
               ("bump m", "bump_height_m", "num"), ("plan s", "compute_s", "s"), ("vlm s", "vlm_s", "s"),
               ("apart m", "apart_mean_m", "num")]
    print(f"\nAll {len({(r['sequence'], r['frame']) for r in rows})} frames (means over frames):")
    print(f"{'condition':18s}" + "".join(f"{name:>9s}" for name, _, _ in columns))
    for condition in conditions:
        group = [r for r in rows if r["condition"] == condition]
        cells = []
        for _, column, kind in columns:
            if kind is None:
                cells.append(f"{sum(bool(r[column]) for r in group)}/{len(group)}")
            else:
                cells.append(fmt(mean(group, column), kind))
        print(f"{condition:18s}" + "".join(f"{c:>9s}" for c in cells))
    print("(success = reached the target, or stopped within 2.5 m of a target blocked by a hard obstacle. Shares are of path "
          "cells from LiDAR + camera labels; veg ok = grass/bush, veg bad = tree/log. hard = distance to hard obstacles, "
          "m; slope = cm of height change per m; plan s = geometry compute per frame; vlm s = VLM answer time.)")


def print_by_category(rows: list[dict], conditions: list[str]):
    print("\nPer category (success / water / mud / veg ok / shortfall m / hard min m):")
    categories = list(dict.fromkeys(r["category"] for r in rows))
    print(f"{'category':10s}" + "".join(f"{c:>34s}" for c in conditions))
    for category in categories:
        cells = []
        for condition in conditions:
            group = [r for r in rows if r["condition"] == condition and r["category"] == category]
            success = sum(bool(r["success"]) for r in group)
            cells.append(f"{success}/{len(group)} {fmt(mean(group, 'share_water'), 'pct'):>4s} "
                         f"{fmt(mean(group, 'share_mud'), 'pct'):>4s} {fmt(mean(group, 'share_compliant'), 'pct'):>4s} "
                         f"{fmt(mean(group, 'shortfall_m')):>5s} {fmt(mean(group, 'hard_dist_min_m')):>5s}")
        print(f"{category:10s}" + "".join(f"{c:>34s}" for c in cells))


def print_paired(rows: list[dict], conditions: list[str]):
    print("\nFrame by frame against the baseline (wins / losses / ties, and the mean change):")
    keyed = {(r["sequence"], r["frame"], r["condition"]): r for r in rows}
    frames = list(dict.fromkeys((r["sequence"], r["frame"]) for r in rows))
    header = f"{'condition':18s} {'success':>15s}" + "".join(f"{column:>24s}" for column, _, _ in PAIRED)
    print(header)
    for condition in conditions:
        if condition == BASELINE:
            continue
        base = [keyed[(s, f, BASELINE)] for s, f in frames]
        other = [keyed[(s, f, condition)] for s, f in frames]
        gained = sum(not b["success"] and o["success"] for b, o in zip(base, other))
        lost = sum(b["success"] and not o["success"] for b, o in zip(base, other))
        cells = []
        for column, lower, tolerance in PAIRED:
            c = paired_comparison([b[column] for b in base], [o[column] for o in other], lower, tolerance)
            cells.append(f"{c['wins']}/{c['losses']}/{c['ties']} ({c['mean_diff']:+.2f})")
        print(f"{condition:18s} {f'+{gained} / -{lost}':>15s}" + "".join(f"{c:>24s}" for c in cells))
    print("(success: frames gained / lost. A win means better than the baseline by more than the tie tolerance: "
          + ", ".join(f"{column} {tol}" for column, _, tol in PAIRED) + ". Lower is better except hard_dist_min_m.)")


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    id_to_name, _ = load_ontology(args.root)
    with open(args.frames, newline="") as f:
        entries = list(csv.DictReader(f))
    runs = {short_name(run): REPO_ROOT / "outputs" / "vlm_runs" / run for run in args.runs}
    for name, run_dir in runs.items():
        if not run_dir.is_dir():
            sys.exit(f"VLM run folder not found: {run_dir}")
    conditions = [BASELINE, *runs, *([] if args.no_oracle else [ORACLE])]

    sequences: dict[str, RellisSequence] = {}
    rows = []
    for k, entry in enumerate(entries, start=1):
        seq_id, frame_index = entry["sequence"], int(entry["frame"])
        stem = f"{entry['category']}_{seq_id}_{frame_index:06d}"
        if seq_id not in sequences:
            sequences[seq_id] = RellisSequence(seq_id, root=args.root)
        frame = sequences[seq_id][frame_index]
        grid = ImageGrid(height=frame.image.shape[0], width=frame.image.shape[1])

        answers, statuses = {}, {BASELINE: "-"}
        for name, run_dir in runs.items():
            answer, seconds, statuses[name] = load_answer(run_dir, stem)
            answers[name] = (answer, seconds)
        if not args.no_oracle:
            has_labels = frame.label_image is not None
            answers[ORACLE] = (oracle_answer(frame.label_image, grid, id_to_name) if has_labels else NO_ANSWER, None)
            statuses[ORACLE] = "oracle" if has_labels else "missing"

        results = evaluate_frame(frame.points, frame.point_labels, frame.calib.T_cam_from_lidar, frame.calib.K,
                                 grid, frame.label_image, answers, id_to_name)
        rows += [row_for(entry, results[name], statuses[name]) for name in conditions]
        status = "  ".join(f"{name} {'ok' if results[name].metrics.success else 'X'}" for name in conditions)
        print(f"[{k:2d}/{len(entries)}] {entry['category']:6s} {seq_id}/{frame_index:<5d} {status}", flush=True)

    write_csv(args.out_dir / "per_frame.csv", rows)
    summary = []
    for category in [*dict.fromkeys(r["category"] for r in rows), "all"]:
        for condition in conditions:
            group = [r for r in rows if r["condition"] == condition and (category == "all" or r["category"] == category)]
            summary.append({"category": category, "condition": condition, "frames": len(group),
                            **{c: fmt(mean(group, c)) if c not in ("success", "reached", "goal_blocked")
                               else sum(bool(r[c]) for r in group) for c in METRIC_FIELDS}})
    write_csv(args.out_dir / "summary.csv", summary)

    print_overall(rows, conditions)
    print_by_category(rows, conditions)
    print_paired(rows, conditions)
    print(f"\nSaved per_frame.csv and summary.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
