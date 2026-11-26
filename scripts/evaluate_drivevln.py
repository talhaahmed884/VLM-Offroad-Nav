"""Offline evaluation for the DriveVLN-style selector, matching scripts/evaluate.py

Reads saved VLM selections from `outputs/drivevln_runs/<run_name>/`, scores each chosen
candidate path on two-source (LiDAR + camera) ground truth with `compute_path_metrics`,
merges the rows with `outputs/eval/per_frame.csv`, and prints the overall, per-category,
and paired wins/losses tables.
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate import (  # noqa: E402
    METRIC_FIELDS,
    print_by_category,
    print_overall,
    print_paired,
    short_name,
    write_csv,
)
from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.eval.metrics import (  # noqa: E402
    HARD_OBSTACLE_CLASSES,
    compute_path_metrics,
    hard_obstacle_xy,
    ids_of,
)
from offroad.eval.path_truth import path_truth  # noqa: E402
from offroad.mapping.costmap import CostParams, geometric_cost_map  # noqa: E402
from offroad.mapping.elevation import (  # noqa: E402
    GridSpec,
    build_elevation_map,
    lidar_to_flu,
    majority_label_grid,
)
from offroad.planning.baseline import UNSEEN, plan_on_cost  # noqa: E402
from offroad.planning.candidates import (  # noqa: E402
    CandidateParams,
    candidate_to_baseline_result,
    generate_candidates,
)

DEFAULT_RUN = "Qwen3-VL-8B-Instruct_4bit"


def path_separation(path_a_xy: np.ndarray, path_b_xy: np.ndarray) -> tuple[float, float] | None:
    """(mean, max) nearest-point distance (m) from path_a to path_b."""
    if len(path_a_xy) == 0 or len(path_b_xy) == 0:
        return None
    d = np.hypot(
        path_a_xy[:, None, 0] - path_b_xy[None, :, 0],
        path_a_xy[:, None, 1] - path_b_xy[None, :, 1],
    )
    nearest = d.min(axis=1)
    return float(nearest.mean()), float(nearest.max())


def _parse_existing_row(row: dict[str, str]) -> dict:
    """Convert CSV strings from per_frame.csv back to bool/float/None for evaluate.py tables."""
    out: dict = {
        "category": row["category"],
        "sequence": row["sequence"],
        "frame": int(row["frame"]),
        "condition": row["condition"],
        "answer": row["answer"],
    }
    for k in METRIC_FIELDS:
        v = row.get(k, "")
        if v == "" or v is None:
            out[k] = None
        elif k in ("success", "reached", "goal_blocked"):
            out[k] = v.lower() == "true"
        else:
            out[k] = float(v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=DEFAULT_RUN, help=f"Run folder in outputs/drivevln_runs/ (default: {DEFAULT_RUN})")
    parser.add_argument("--condition-name", default=None, help="Condition label (default: DriveVLN-<short_name>)")
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "configs" / "eval_frames.csv")
    parser.add_argument("--base-csv", type=Path, default=REPO_ROOT / "outputs" / "eval" / "per_frame.csv")
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "outputs" / "eval" / "per_frame_with_drivevln.csv")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--goal-distance", type=float, default=20.0)
    args = parser.parse_args()

    run_dir = REPO_ROOT / "outputs" / "drivevln_runs" / args.run
    if not run_dir.is_dir():
        sys.exit(f"DriveVLN run folder not found: {run_dir}")

    condition_name = args.condition_name or f"DriveVLN-{short_name(args.run).replace('VLM-', '')}"
    id_to_name, _ = load_ontology(args.root)
    hard_ids = ids_of(id_to_name, HARD_OBSTACLE_CLASSES)

    with open(args.frames, newline="", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))

    spec = GridSpec()
    cost_params = CostParams()
    cand_params = CandidateParams()
    sequences: dict[str, RellisSequence] = {}

    drivevln_rows: list[dict] = []
    soft_chosen = 0
    fallbacks = 0

    for k, entry in enumerate(entries, start=1):
        seq_id, frame_idx = entry["sequence"], int(entry["frame"])
        stem = f"{entry['category']}_{seq_id}_{frame_idx:06d}"
        json_path = run_dir / f"{stem}.json"
        if not json_path.exists():
            sys.exit(f"Missing saved DriveVLN answer: {json_path}")
        record = json.loads(json_path.read_text(encoding="utf-8"))

        if seq_id not in sequences:
            sequences[seq_id] = RellisSequence(seq_id, root=args.root)
        frame = sequences[seq_id][frame_idx]

        t0 = time.perf_counter()
        points_flu = lidar_to_flu(frame.points)
        elevation = build_elevation_map(points_flu, spec)
        costmap = geometric_cost_map(points_flu, elevation, cost_params)
        labels = majority_label_grid(points_flu, frame.point_labels, spec, empty_label=UNSEEN)
        candidates = generate_candidates(
            costmap.cost,
            spec,
            obstacle_height=costmap.obstacle_height,
            goal_distance=args.goal_distance,
            params=cand_params,
        )
        compute_s = time.perf_counter() - t0

        selected_color = record["answer"].get("selected_color")
        chosen = next((c for c in candidates if c.color == selected_color), None)
        if chosen is not None:
            status = "valid"
        elif candidates:
            chosen = candidates[0]
            status = "fallback"
            fallbacks += 1
        else:
            status = "no_candidates"

        if chosen is not None and chosen.kind == "soft_obstacle":
            soft_chosen += 1

        baseline_plan = plan_on_cost(costmap.cost, costmap, labels, cost_params, goal_distance=args.goal_distance)
        plan = (
            candidate_to_baseline_result(chosen, costmap, labels, cost_params, goal_distance=args.goal_distance)
            if chosen is not None
            else baseline_plan
        )
        truth = (
            path_truth(
                plan.path,
                labels,
                costmap.ground_z,
                spec,
                frame.calib.T_cam_from_lidar,
                frame.calib.K,
                frame.label_image,
                points_lidar=frame.points,
            )
            if plan.found
            else None
        )
        obs_xy = hard_obstacle_xy(frame.points, frame.point_labels, hard_ids, costmap.ground_z, spec)
        m = compute_path_metrics(plan, truth, obs_xy, id_to_name)
        sep = (
            path_separation(plan.path_xy(), baseline_plan.path_xy())
            if (plan.found and baseline_plan.found)
            else None
        )

        values = {
            **asdict(m),
            "success": m.success,
            "compute_s": compute_s,
            "vlm_s": float(record.get("vlm_seconds", 0.0)),
            "apart_mean_m": sep[0] if sep else None,
            "apart_max_m": sep[1] if sep else None,
        }
        drivevln_rows.append(
            {
                "category": entry["category"],
                "sequence": seq_id,
                "frame": frame_idx,
                "condition": condition_name,
                "answer": status,
                **{field: values[field] for field in METRIC_FIELDS},
            }
        )
        print(
            f"[{k:2d}/{len(entries)}] {entry['category']:6s} {seq_id}/{frame_idx:<5d} "
            f"{condition_name} {'ok' if m.success else 'X'} ({selected_color}, {chosen.kind if chosen else 'none'})",
            flush=True,
        )

    print(f"\nSoft-obstacle candidate chosen: {soft_chosen}/{len(entries)} (fallbacks: {fallbacks})")

    # Merge with existing per_frame.csv if present.
    base_rows: list[dict] = []
    if args.base_csv.exists():
        with open(args.base_csv, newline="", encoding="utf-8") as f:
            base_rows = [_parse_existing_row(r) for r in csv.DictReader(f) if r.get("condition") != condition_name]

    merged = base_rows + drivevln_rows
    conditions = list(dict.fromkeys(r["condition"] for r in merged))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_csv, merged)

    print_overall(merged, conditions)
    print_by_category(merged, conditions)
    if "baseline" in conditions:
        print_paired(merged, conditions)
    print(f"\nSaved merged results to {args.out_csv}")


if __name__ == "__main__":
    main()