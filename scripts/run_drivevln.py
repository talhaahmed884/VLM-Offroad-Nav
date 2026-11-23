"""Run the DriveVLN-style candidate trajectory selector on one frame or all evaluation frames.

Generates geometric + soft-obstacle candidate paths, projects them onto the camera image
in distinct colors, asks the VLM to choose the best colored path, and saves the overlay
image, raw/parsed JSON, and latency to `outputs/drivevln_runs/<run_name>/`.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence
from offroad.mapping.costmap import CostParams, geometric_cost_map
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu
from offroad.planning.candidates import CandidateParams, generate_candidates
from offroad.viz.candidate_overlay import draw_candidates_on_camera
from offroad.vlm.backend import TransformersVlm
from offroad.vlm.prompt import VEHICLE_BRIEFING
from offroad.vlm.runner import downscale, strip_thinking
from offroad.vlm.selector import build_selector_prompt, parse_selector_answer

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def default_run_name(model_id: str, quantize: str, quantize_vision: bool, max_side: int, briefing: bool) -> str:
    short = model_id.split("/")[-1]
    parts = [short, quantize]
    if quantize_vision:
        parts.append("qvision")
    if max_side != 1280:
        parts.append(f"{max_side}px")
    if not briefing:
        parts.append("nobriefing")
    return "_".join(parts)


def load_eval_frames(csv_path: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["category"], f"{int(row['sequence']):05d}", int(row["frame"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriveVLN-style colored-trajectory VLM selector.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--quantize-vision", action="store_true", help="Also quantize the vision encoder")
    parser.add_argument("--sequence", type=int, default=None, help="Run a single sequence ID (e.g. 0)")
    parser.add_argument("--frame", type=int, default=None, help="Run a single frame index (e.g. 120)")
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "configs" / "eval_frames.csv")
    parser.add_argument("--max-side", type=int, default=1280, help="Max image side sent to VLM (default: 1280)")
    parser.add_argument("--goal-distance", type=float, default=20.0)
    parser.add_argument("--no-briefing", action="store_true", help="Omit the vehicle traversability briefing")
    parser.add_argument("--name", default=None, help="Custom run folder name")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "drivevln_runs")
    parser.add_argument("--overwrite", action="store_true", help="Re-run frames that already have a JSON")
    args = parser.parse_args()

    briefing = None if args.no_briefing else VEHICLE_BRIEFING
    run_name = args.name or default_run_name(
        args.model, args.quantize, args.quantize_vision, args.max_side, briefing is not None
    )
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.sequence is not None or args.frame is not None:
        if args.sequence is None or args.frame is None:
            parser.error("Provide both --sequence and --frame for single-frame mode.")
        targets = [("single", f"{args.sequence:05d}", args.frame)]
    else:
        targets = load_eval_frames(args.frames)

    spec = GridSpec()
    cost_params = CostParams()
    cand_params = CandidateParams()
    sequences: dict[str, RellisSequence] = {}

    vlm: TransformersVlm | None = None
    warmed_up = False
    summary_rows: list[dict[str, object]] = []

    for category, seq_id, frame_idx in targets:
        stem = f"{category}_{seq_id}_{frame_idx:06d}"
        json_path = run_dir / f"{stem}.json"
        img_path = run_dir / f"{stem}.png"

        if json_path.exists() and not args.overwrite:
            record = json.loads(json_path.read_text(encoding="utf-8"))
            print(f"[skip] {stem}: selected={record['answer']['selected_color']} ({record['vlm_seconds']:.1f}s)")
            summary_rows.append(
                {
                    "category": category,
                    "sequence": seq_id,
                    "frame": frame_idx,
                    "num_candidates": len(record["candidates"]),
                    "visible_colors": ",".join(record["visible_colors"]),
                    "selected_color": record["answer"]["selected_color"] or "",
                    "selected_kind": record["selected_kind"] or "",
                    "valid": record["answer"]["valid"],
                    "issues": len(record["answer"]["issues"]),
                    "vlm_seconds": record["vlm_seconds"],
                    "candidate_seconds": record["candidate_seconds"],
                }
            )
            continue

        if seq_id not in sequences:
            sequences[seq_id] = RellisSequence(seq_id)
        frame = sequences[seq_id][frame_idx]

        # 1. Build geometric costmap and candidate trajectories.
        t0 = time.perf_counter()
        points_flu = lidar_to_flu(frame.points)
        elevation = build_elevation_map(points_flu, spec)
        costmap = geometric_cost_map(points_flu, elevation, cost_params)
        candidates = generate_candidates(
            costmap.cost,
            spec,
            obstacle_height=costmap.obstacle_height,
            goal_distance=args.goal_distance,
            params=cand_params,
        )
        overlay_rgb, visible_colors = draw_candidates_on_camera(
            frame.image, frame.calib, costmap.ground_z, candidates
        )
        candidate_seconds = time.perf_counter() - t0

        cv2.imwrite(str(img_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        # 2. Query the VLM selector (if more than 0 visible candidates).
        prompt = build_selector_prompt(visible_colors, goal_distance_m=args.goal_distance, vehicle_briefing=briefing)
        if not visible_colors:
            raw_text = ""
            vlm_seconds = 0.0
            prompt_tokens = 0
            new_tokens = 0
            parsed = parse_selector_answer("", [])
        else:
            if vlm is None:
                vlm = TransformersVlm(
                    model_id=args.model,
                    quantize=args.quantize,
                    quantize_vision=args.quantize_vision,
                )
            vlm_image = downscale(overlay_rgb, args.max_side)
            if not warmed_up:
                _ = vlm.generate(vlm_image, prompt)
                warmed_up = True
            reply = vlm.generate(vlm_image, prompt)
            raw_text = reply.text
            vlm_seconds = reply.seconds
            prompt_tokens = reply.prompt_tokens
            new_tokens = reply.new_tokens
            parsed = parse_selector_answer(strip_thinking(raw_text), visible_colors)

        chosen = next((c for c in candidates if c.color == parsed.selected_color), None)
        record = {
            "category": category,
            "sequence": seq_id,
            "frame": frame_idx,
            "model": args.model,
            "quantize": args.quantize,
            "goal_distance": args.goal_distance,
            "candidate_seconds": round(candidate_seconds, 4),
            "vlm_seconds": round(vlm_seconds, 4),
            "prompt_tokens": prompt_tokens,
            "new_tokens": new_tokens,
            "visible_colors": visible_colors,
            "candidates": [
                {
                    "color": c.color,
                    "kind": c.kind,
                    "lateral_offset_m": c.lateral_offset_m,
                    "length_m": round(c.length_m, 3),
                    "shortfall_m": round(c.shortfall_m, 3),
                    "reached": c.reached,
                }
                for c in candidates
            ],
            "prompt": prompt,
            "raw_text": raw_text,
            "answer": {
                "valid": parsed.valid,
                "selected_color": parsed.selected_color,
                "reasoning": parsed.reasoning,
                "issues": parsed.issues,
            },
            "selected_kind": chosen.kind if chosen is not None else None,
        }
        json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        status = "OK" if parsed.valid else "INVALID"
        print(
            f"[{status}] {stem}: cands={visible_colors} -> {parsed.selected_color} "
            f"({record['selected_kind']}) in {vlm_seconds:.1f}s (issues={len(parsed.issues)})"
        )
        summary_rows.append(
            {
                "category": category,
                "sequence": seq_id,
                "frame": frame_idx,
                "num_candidates": len(candidates),
                "visible_colors": ",".join(visible_colors),
                "selected_color": parsed.selected_color or "",
                "selected_kind": record["selected_kind"] or "",
                "valid": parsed.valid,
                "issues": len(parsed.issues),
                "vlm_seconds": round(vlm_seconds, 3),
                "candidate_seconds": round(candidate_seconds, 3),
            }
        )

    # Write summary.csv when running the eval set.
    if targets and targets[0][0] != "single":
        summary_path = run_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved summary to {summary_path}")

    if vlm is not None:
        print(f"Peak GPU memory: {vlm.peak_memory_gb():.2f} GB")


if __name__ == "__main__":
    main()