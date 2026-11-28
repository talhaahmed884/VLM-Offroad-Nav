"""Phase 6: videos of saved closed-loop runs (chase view + the chooser's picture + caption) -> MP4.

Each run is driven again in MAVS with its logged choices (`ReplayChooser`): the loop and MAVS are
deterministic, so the replay drives exactly the saved path (checked at every decision: the same
candidates; at the end: the same outcome, progress and number of decisions). No VLM is needed, so
a replay takes minutes. A chase camera renders the MRZR4 from behind (its body and wheels drawn as
MAVS actors), with the chosen path drawn on the ground in its color; beside it is the picture the
chooser saw (for a VLM: exactly its input), and below it the time, progress, Mode B's word and
the answer. A title card opens the video and the outcome closes it.

Videos: outputs/sim/videos/<run name>.mp4, 10 frames per simulated second (real time; --speed 2
is twice as fast). Rendering takes ~0.3 s per frame, so a 2-minute run takes ~6-8 minutes.

Usage (from the repo root; needs MAVS and the runs from step 6.5):
    python scripts/sim_record.py                                   # the highlight runs (HIGHLIGHTS)
    python scripts/sim_record.py fork-right_finetuned_modeB        # chosen runs
    python scripts/sim_record.py --all                             # every saved run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.sim.choosers import ReplayChooser  # noqa: E402
from offroad.sim.loop import LoopParams, run_episode  # noqa: E402
from offroad.sim.mavs_world import MavsWorld  # noqa: E402
from offroad.sim.rig import RigSpec, camera_calibration  # noqa: E402
from offroad.sim.video import FPS, Recorder, title_card  # noqa: E402

SCENES_DIR = REPO_ROOT / "outputs" / "sim" / "scenes"
RUNS_DIR = REPO_ROOT / "outputs" / "sim" / "runs"
OUT_DIR = REPO_ROOT / "outputs" / "sim" / "videos"

# The runs that show step 6.5's findings (see docs/approach-2-plan.md, 6.5 status).
HIGHLIGHTS = {
    "fork-right_finetuned_modeB": "Told 'slight right' at the fork, the fine-tuned model takes the left branch: "
                                  "it ignores the direction word",
    "fork-right_zeroshot_modeB": "Told 'slight right' at the fork, the zero-shot model takes the right branch",
    "curves_zeroshot_modeA": "The zero-shot model drifts to the outside of the right-hand bend until no "
                             "candidate is left",
    "curves_finetuned_modeA": "The fine-tuned model follows both bends",
    "halfblocked_finetuned_modeA": "The fine-tuned model passes the half-wall 6 cm away (the oracle: 80 cm)",
    "blocked_finetuned_modeA": "The fine-tuned model never says stop; the planner stops when no path is left",
}
CHOOSER_NAMES = {"straightest": "straightest candidate (baseline)", "cheapest": "cheapest candidate (LiDAR only)",
                 "oracle": "oracle (candidate closest to the route; uses the answer key)",
                 "zeroshot": "Qwen3-VL-4B zero-shot", "finetuned": "Qwen3-VL-4B fine-tuned (skip, epoch 4)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run folder names in outputs/sim/runs (default: the highlights)")
    parser.add_argument("--all", action="store_true", help="every saved run")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    parser.add_argument("--fresh", action="store_true", help="redo videos that already exist")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    return parser.parse_args()


def open_writer(path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
    """An MP4 writer with MPEG-4 part 2 ("mp4v"): plays in VLC and Windows' Media Player. The pip
    OpenCV has no H.264 encoder (it looks for an OpenH264 DLL and fails, checked 2026-11-28)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV can't write {path}")
    return writer


def record(name: str, args, rig: RigSpec, calib) -> dict:
    run_dir = args.runs_dir / name
    saved = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    decisions = [json.loads(line) for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    scene, mode, chooser_name = saved["scene"], saved["mode"], saved["chooser"]
    world = MavsWorld(args.scenes_dir / scene / f"{scene}_scene.json", rig, labeling=False, route=saved.get("route"))
    world.show_vehicle()
    chooser = ReplayChooser(decisions, chooser_name, calib)

    tmp = args.out / f"{name}.part.mp4"
    writer, size = None, None

    def write(frame_rgb):
        nonlocal writer, size
        if writer is None:
            size = (frame_rgb.shape[1], frame_rgb.shape[0])
            writer = open_writer(tmp, size)
            describe = HIGHLIGHTS.get(name, "")
            card = title_card(size[0], size[1], [
                f"{scene}{'-' + saved['route'] if saved.get('route') and scene == 'fork' else ''}: "
                f"{CHOOSER_NAMES.get(chooser_name, chooser_name)}, Mode {mode}",
                describe, "Closed loop in MAVS (MRZR4, RELLIS-like camera and LiDAR). A decision every 0.5 s;",
                "simulated time pauses while the chooser answers. Replayed from the saved run."])
            for _ in range(3 * FPS):
                writer.write(card[:, :, ::-1])
        writer.write(frame_rgb[:, :, ::-1])

    header = (f"{name}  |  {CHOOSER_NAMES.get(chooser_name, chooser_name)}  |  Mode {mode}"
              f"{' (keep going + direction word)' if mode == 'B' else ' (keep going)'}")
    title = "what the VLM saw (its input)" if chooser_name in ("zeroshot", "finetuned") else "the candidates"
    recorder = Recorder(world, write, header, title, speed=args.speed)
    started = time.perf_counter()
    episode = run_episode(recorder, chooser, calib, scene, mode, LoopParams(), on_decision=recorder.on_decision)
    s = episode.summary
    reason = f" ({s['stop_reason']})" if s.get("stop_reason") else ""
    recorder.finish(f"{s['outcome']}{reason}: {round(100 * s['completion'])}% of the route")
    writer.release()
    out = args.out / f"{name}.mp4"
    tmp.replace(out)
    same = {k: (s[k], saved[k]) for k in ("outcome", "decisions", "progress_m") if s[k] != saved[k]}
    return {"video": str(out), "frames": recorder.frames, "minutes": round((time.perf_counter() - started) / 60, 1),
            "same_as_saved_run": not same, **({"differences": same} if same else {})}


def main() -> None:
    args = parse_args()
    names = sorted(p.name for p in args.runs_dir.iterdir() if (p / "summary.json").exists()) if args.all \
        else (args.runs or list(HIGHLIGHTS))
    missing = [n for n in names if not (args.runs_dir / n / "summary.json").exists()]
    if missing:
        sys.exit(f"No saved run for {', '.join(missing)} in {args.runs_dir}; run step 6.5 (sim_run_all.py) first")
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if args.fresh or not (args.out / f"{n}.mp4").exists()]
    print(f"{len(names)} videos, {len(names) - len(todo)} already made, {len(todo)} to record", flush=True)
    rig = RigSpec()
    calib = camera_calibration(rig)
    for k, name in enumerate(todo, 1):
        print(f"\n[{k}/{len(todo)}] {name}", flush=True)
        result = record(name, args, rig, calib)
        print("  -> " + json.dumps(result), flush=True)
        if not result["same_as_saved_run"]:
            print("  WARNING: the replay ended differently from the saved run", flush=True)
    print(f"\nVideos in {args.out}")


if __name__ == "__main__":
    main()
