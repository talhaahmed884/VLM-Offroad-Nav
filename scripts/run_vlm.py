"""Run a local VLM on the evaluation frames and save every answer (phase 4, step 3).

For each frame: draw the grid, find the grid cell of the 20 m target (the same target the
baseline planner uses), build the prompt, shrink the image, ask the model, and check its
answer. Saves to outputs/vlm_runs/<run name>/:
    <category>_<seq>_<frame>.json   prompt, raw answer, checked answer, timing
    <category>_<seq>_<frame>.png    the exact image sent to the model
    summary.csv                     one row per frame
Frames that already have a .json are skipped (so an interrupted run can continue); use
--overwrite to redo them.

The first run downloads the model from Hugging Face (about 9 GB for the 4B model) into the
Hugging Face cache, outside the repo. Check the setup first with scripts/check_gpu.py.

Usage (from the repo root):
    python scripts/run_vlm.py --sequence 0 --frame 120     # one frame, to try it out
    python scripts/run_vlm.py                              # all frames in configs/eval_frames.csv
    python scripts/run_vlm.py --model Qwen/Qwen3-VL-2B-Instruct --quantize none
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence  # noqa: E402
from offroad.mapping.costmap import CostParams, estimate_ground  # noqa: E402
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu  # noqa: E402
from offroad.viz.grid_overlay import draw_grid  # noqa: E402
from offroad.vlm.answer import parse_answer  # noqa: E402
from offroad.vlm.backend import DEFAULT_MODEL, QUANTIZE_OPTIONS, TransformersVlm  # noqa: E402
from offroad.vlm.goal import goal_cell  # noqa: E402
from offroad.vlm.grid import ImageGrid  # noqa: E402
from offroad.vlm.prompt import VEHICLE_BRIEFING, build_prompt  # noqa: E402
from offroad.vlm.runner import answer_to_dict, downscale, strip_thinking  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--quantize", choices=QUANTIZE_OPTIONS, default="4bit", help="weight precision (default: 4bit)")
    parser.add_argument("--quantize-vision", action="store_true",
                        help="also quantize the vision encoder (saves ~1 GB; try if the GPU runs out of memory)")
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "configs" / "eval_frames.csv")
    parser.add_argument("--sequence", type=int, default=None, help="run one frame instead: sequence number, 0-4")
    parser.add_argument("--frame", type=int, default=None, help="run one frame instead: frame index")
    parser.add_argument("--max-side", type=int, default=1280, help="shrink images to this many px on the long side (default: 1280)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="longest answer allowed, tokens (default: 512)")
    parser.add_argument("--goal-distance", type=float, default=20.0, help="target distance straight ahead, meters (default: 20)")
    parser.add_argument("--no-briefing", action="store_true", help="leave the vehicle briefing out of the prompt")
    parser.add_argument("--no-goal-cell", action="store_true", help="say 'straight ahead' instead of naming the goal cell")
    parser.add_argument("--name", default=None, help="run name, i.e. the output folder (default: from model and settings)")
    parser.add_argument("--overwrite", action="store_true", help="redo frames that already have an answer")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    args = parser.parse_args()
    if (args.sequence is None) != (args.frame is None):
        parser.error("give both --sequence and --frame, or neither")
    return args


def run_name(args: argparse.Namespace) -> str:
    name = f"{args.model.split('/')[-1]}_{args.quantize}"
    name += "_qvision" if args.quantize_vision and args.quantize != "none" else ""
    name += f"_{args.max_side}px" if args.max_side != 1280 else ""
    return name + ("_nobriefing" if args.no_briefing else "")


def frames_to_run(args: argparse.Namespace) -> list[dict]:
    if args.sequence is not None:
        return [{"category": "single", "sequence": f"{args.sequence:05d}", "frame": str(args.frame)}]
    with open(args.frames, newline="") as f:
        return list(csv.DictReader(f))


def find_goal_cell(frame, grid: ImageGrid, goal_distance: float) -> str | None:
    """Grid cell of the ground `goal_distance` m ahead, from this frame's LiDAR ground estimate."""
    spec = GridSpec()
    elevation = build_elevation_map(lidar_to_flu(frame.points), spec)
    ground = estimate_ground(elevation.min_z, spec, CostParams())
    return goal_cell(ground, spec, frame.calib.T_cam_from_lidar, frame.calib.K, grid, goal_distance)


def summary_row(record: dict) -> dict:
    answer = record["answer"]
    return {
        "category": record["category"],
        "sequence": record["sequence"],
        "frame": record["frame"],
        "goal_cell": record["goal_cell"] or "",
        "valid": "yes" if answer["valid"] else "no",
        "waypoints": " ".join(answer["waypoints"]),
        "n_hints": len(answer["hints"]),
        "n_issues": len(answer["issues"]),
        "route_conflicts": " ".join(answer.get("route_conflicts", [])),  # older records lack this field
        "seconds": f"{record['seconds']:.2f}",
        "prompt_tokens": record["prompt_tokens"],
        "new_tokens": record["new_tokens"],
    }


def print_summary(rows: list[dict]):
    # valid: answers with a usable route; issues: mean format problems per answer;
    # conflicted: answers whose route enters cells the model itself marked avoid/impassable.
    print(f"\n{'category':8s} {'frames':>6s} {'valid':>6s} {'issues':>7s} {'conflicted':>10s} {'seconds':>8s} {'tokens out':>10s}")
    for category in [*dict.fromkeys(row["category"] for row in rows), "all"]:
        group = rows if category == "all" else [row for row in rows if row["category"] == category]
        n = len(group)
        valid = sum(row["valid"] == "yes" for row in group)
        issues = sum(row["n_issues"] for row in group) / n
        conflicted = sum(bool(row["route_conflicts"]) for row in group)
        seconds = sum(float(row["seconds"]) for row in group) / n
        tokens = sum(row["new_tokens"] for row in group) / n
        print(f"{category:8s} {n:6d} {valid:6d} {issues:7.1f} {conflicted:10d} {seconds:8.2f} {tokens:10.0f}")


def main() -> None:
    args = parse_args()
    entries = frames_to_run(args)
    out_dir = REPO_ROOT / "outputs" / "vlm_runs" / (args.name or run_name(args))
    out_dir.mkdir(parents=True, exist_ok=True)
    briefing = None if args.no_briefing else VEHICLE_BRIEFING

    print(f"Loading {args.model} ({args.quantize}) ...", flush=True)
    vlm = TransformersVlm(args.model, quantize=args.quantize, max_new_tokens=args.max_new_tokens,
                          quantize_vision=args.quantize_vision)
    print(f"Loaded in {vlm.load_seconds:.0f} s on {'GPU' if vlm.on_gpu else 'CPU (this will be very slow)'}")

    sequences: dict[str, RellisSequence] = {}
    rows, warmed_up = [], False
    for k, entry in enumerate(entries, start=1):
        seq_id, frame_index = entry["sequence"], int(entry["frame"])
        stem = f"{entry['category']}_{seq_id}_{frame_index:06d}"
        json_path = out_dir / f"{stem}.json"
        if json_path.exists() and not args.overwrite:
            rows.append(summary_row(json.loads(json_path.read_text(encoding="utf-8"))))
            print(f"[{k:2d}/{len(entries)}] {entry['category']:6s} {seq_id}/{frame_index:<5d} already done, skipped")
            continue

        if seq_id not in sequences:
            sequences[seq_id] = RellisSequence(seq_id, root=args.root)
        frame = sequences[seq_id][frame_index]
        grid = ImageGrid(height=frame.image.shape[0], width=frame.image.shape[1])
        goal = None if args.no_goal_cell else find_goal_cell(frame, grid, args.goal_distance)
        prompt = build_prompt(grid, goal_cell=goal, goal_distance_m=args.goal_distance, vehicle_briefing=briefing)
        sent = downscale(draw_grid(frame.image, grid), args.max_side)

        if not warmed_up:
            # The first call on a GPU is slow (kernels get compiled), so do a short untimed one first.
            vlm.generate(sent, "Describe this image in one word.", max_new_tokens=4)
            warmed_up = True

        reply = vlm.generate(sent, prompt)
        answer = parse_answer(strip_thinking(reply.text), grid)
        record = {
            "category": entry["category"],
            "sequence": seq_id,
            "frame": frame_index,
            "model": args.model,
            "quantize": args.quantize,
            "image_size_sent": list(sent.shape[:2]),
            "briefing": briefing is not None,
            "goal_cell": goal,
            "prompt": prompt,
            "raw_answer": reply.text,
            "seconds": reply.seconds,
            "prompt_tokens": reply.prompt_tokens,
            "new_tokens": reply.new_tokens,
            "answer": answer_to_dict(answer),
        }
        json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        cv2.imwrite(str(out_dir / f"{stem}.png"), cv2.cvtColor(sent, cv2.COLOR_RGB2BGR))
        rows.append(summary_row(record))

        status = "VALID" if answer.valid else "NOT USABLE"
        route = " -> ".join(answer.waypoints) or "(no waypoints)"
        conflicts = f", route enters own avoid cells {' '.join(answer.route_conflicts)}" if answer.route_conflicts else ""
        print(f"[{k:2d}/{len(entries)}] {entry['category']:6s} {seq_id}/{frame_index:<5d} goal {goal or '-':3s} "
              f"{reply.seconds:5.1f} s  {status:10s} {route}  ({len(answer.issues)} issues{conflicts})", flush=True)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    print(f"\nPeak GPU memory: {vlm.peak_memory_gb():.1f} GB")
    print(f"Saved answers and summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
