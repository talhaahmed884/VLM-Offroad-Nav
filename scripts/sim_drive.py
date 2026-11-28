"""Phase 6, step 6.3: drive one closed-loop run in MAVS with one chooser.

The vehicle starts on the trail of a scene (from step 6.1) and keeps going: every 0.5 s of
simulated time it captures a frame, plans up to 4 candidates from the LiDAR, the chooser picks one
(or stops), and the path follower drives it until the next decision (sim/loop.py). The run ends
when the vehicle reaches the trail's end, stops, leaves the trail (> 3 m from its centre), gets
stuck, or hits the time limit.

Choosers: straightest, cheapest, oracle (the candidate closest to the route: a reference that
uses the answer key; no GPU), zeroshot (Qwen3-VL-4B), finetuned (+ phase 5's adapter,
skip epoch 4). The VLMs are asked without the reasoning field, as in phase 5.

Saved to outputs/sim/runs/<scene>_<chooser>_mode<A|B>/:
- decisions.jsonl: one line per decision (time, pose, trail progress and offset, candidates,
  the choice, and the VLM's full answer);
- summary.json: outcome, progress along the trail, completion, mean / max distance from the
  trail centre, decisions, seconds per decision;
- trajectory.png: the driven path over the trail (from above) and the distance from the trail
  centre along the run;
- pictures/: every --pictures-every-th decision, the picture with the candidates (what a VLM sees).

Usage (from the repo root; needs MAVS, and a GPU for the VLM choosers):
    python scripts/sim_drive.py --chooser straightest
    python scripts/sim_drive.py --chooser finetuned --mode B
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.sim.choosers import CheapestChooser, OracleChooser, StraightestChooser, VlmChooser  # noqa: E402
from offroad.sim.loop import LoopParams  # noqa: E402
from offroad.sim.mavs_world import MavsWorld  # noqa: E402
from offroad.sim.rig import RigSpec, camera_calibration  # noqa: E402
from offroad.sim.runs import run_and_save, run_name  # noqa: E402

SCENES_DIR = REPO_ROOT / "outputs" / "sim" / "scenes"
DEFAULT_ADAPTER = REPO_ROOT / "outputs" / "adapters" / "nomatch-skip" / "epoch4"
CHOOSERS = ("straightest", "cheapest", "oracle", "zeroshot", "finetuned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="trail01")
    parser.add_argument("--chooser", choices=CHOOSERS, default="straightest")
    parser.add_argument("--mode", choices=("A", "B"), default="A")
    parser.add_argument("--route", default=None,
                        help="the route that scores the run and gives Mode B its word (fork: left / right)")
    parser.add_argument("--start", type=int, default=10, help="trail waypoint to start at")
    parser.add_argument("--max-time", type=float, default=240.0, help="simulated seconds")
    parser.add_argument("--pictures-every", type=int, default=10, help="save every Nth decision's picture (0 = none)")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/sim/runs/<scene>_<chooser>_mode<M>")
    return parser.parse_args()


def make_chooser(args, calib, world):
    if args.chooser == "oracle":
        return OracleChooser(world.trail)
    if args.chooser == "straightest":
        return StraightestChooser()
    if args.chooser == "cheapest":
        return CheapestChooser()
    from offroad.vlm.backend import TransformersVlm

    adapter = None
    if args.chooser == "finetuned":
        if not (args.adapter / "adapter_config.json").exists():
            sys.exit(f"No adapter in {args.adapter}; copy phase 5's outputs/adapters/nomatch-skip/epoch4 there")
        adapter = str(args.adapter)
    vlm = TransformersVlm(args.model, quantize="4bit", max_new_tokens=200, adapter=adapter)
    print(f"Loaded {args.chooser} in {vlm.load_seconds:.0f} s", flush=True)
    return VlmChooser(vlm, args.chooser, calib)


def main() -> None:
    args = parse_args()
    scene_json = args.scenes_dir / args.scene / f"{args.scene}_scene.json"
    if not scene_json.exists():
        sys.exit(f"{scene_json} is missing; run scripts/sim_check_rig.py first (step 6.1)")
    out = args.out or REPO_ROOT / "outputs" / "sim" / "runs" / run_name(args.scene, args.route, args.chooser, args.mode)
    rig = RigSpec()
    calib = camera_calibration(rig)
    world = MavsWorld(scene_json, rig, labeling=False, route=args.route)
    chooser = make_chooser(args, calib, world)
    params = LoopParams(start_waypoint=args.start, max_time_s=args.max_time)
    summary = run_and_save(world, chooser, calib, args.scene, args.mode, params, out, args.pictures_every,
                           extra={"adapter": str(args.adapter) if args.chooser == "finetuned" else None})
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=2))
    print(f"Saved decisions.jsonl, summary.json, trajectory.png and pictures/ to {out}")


if __name__ == "__main__":
    main()
