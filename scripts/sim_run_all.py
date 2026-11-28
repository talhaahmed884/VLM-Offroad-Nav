"""Phase 6, step 6.5: drive every chooser on every scene (and route) in Modes A and B.

The runs (sim/evaluation.planned_runs): straightest, cheapest and oracle in Mode A only (they
ignore the direction word, so Mode B would be identical), zero-shot and fine-tuned Qwen3-VL-4B in
both modes; the fork once per route. Each run is saved like scripts/sim_drive.py does
(outputs/sim/runs/<scene>[-<route>]_<chooser>_mode<A|B>/). A run whose summary.json already
exists is skipped, so a stopped batch continues where it left off (--fresh redrives). Each VLM is
loaded once for all its runs. Afterwards, scripts/sim_evaluate.py makes the tables.

Time on the laptop: baselines ~2-3 min per run; VLM runs ~3.5-4 s per decision, about 3-4 hours
for both models and modes. Choose a subset with --choosers / --modes / --scenes.

Usage (from the repo root; needs MAVS, and the GPU for the VLMs):
    python scripts/sim_run_all.py --dry-run                  # list the runs
    python scripts/sim_run_all.py --choosers straightest cheapest oracle
    python scripts/sim_run_all.py                            # everything
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.sim.choosers import CheapestChooser, OracleChooser, StraightestChooser, VlmChooser  # noqa: E402
from offroad.sim.evaluation import CHOOSER_ORDER, MODES, SCENE_ROUTES, planned_runs  # noqa: E402
from offroad.sim.loop import LoopParams  # noqa: E402
from offroad.sim.mavs_world import MavsWorld  # noqa: E402
from offroad.sim.rig import RigSpec, camera_calibration  # noqa: E402
from offroad.sim.runs import run_and_save, run_name  # noqa: E402

SCENES_DIR = REPO_ROOT / "outputs" / "sim" / "scenes"
RUNS_DIR = REPO_ROOT / "outputs" / "sim" / "runs"
DEFAULT_ADAPTER = REPO_ROOT / "outputs" / "adapters" / "nomatch-skip" / "epoch4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--choosers", nargs="+", choices=CHOOSER_ORDER, default=list(CHOOSER_ORDER))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--scenes", nargs="+", choices=sorted({s for s, _ in SCENE_ROUTES}), default=None)
    parser.add_argument("--fresh", action="store_true", help="redrive runs that already exist")
    parser.add_argument("--dry-run", action="store_true", help="only list the runs")
    parser.add_argument("--pictures-every", type=int, default=10)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def load_vlm(name: str, args):
    import torch  # noqa: F401  (fail early if the VLM libraries are missing)

    from offroad.vlm.backend import TransformersVlm

    adapter = None
    if name == "finetuned":
        if not (args.adapter / "adapter_config.json").exists():
            sys.exit(f"No adapter in {args.adapter}; copy phase 5's outputs/adapters/nomatch-skip/epoch4 there")
        adapter = str(args.adapter)
    vlm = TransformersVlm(args.model, quantize="4bit", max_new_tokens=200, adapter=adapter)
    print(f"Loaded {name} in {vlm.load_seconds:.0f} s", flush=True)
    return vlm


def main() -> None:
    args = parse_args()
    runs = planned_runs(args.choosers, args.modes, args.scenes)
    todo = [r for r in runs if args.fresh or not (args.runs_dir / run_name(r.scene, r.route, r.chooser, r.mode)
                                                  / "summary.json").exists()]
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, {len(todo)} to drive", flush=True)
    for r in todo:
        print(f"  {run_name(r.scene, r.route, r.chooser, r.mode)}")
    if args.dry_run or not todo:
        return
    for r in todo:
        if not (args.scenes_dir / r.scene / f"{r.scene}_scene.json").exists():
            sys.exit(f"Scene {r.scene} is missing; run scripts/sim_build_scenes.py (step 6.4) or sim_check_rig.py")

    rig = RigSpec()
    calib = camera_calibration(rig)
    vlms, started = {}, time.perf_counter()
    for k, r in enumerate(todo, 1):
        name = run_name(r.scene, r.route, r.chooser, r.mode)
        print(f"\n[{k}/{len(todo)}] {name}  ({(time.perf_counter() - started) / 60:.0f} min so far)", flush=True)
        world = MavsWorld(args.scenes_dir / r.scene / f"{r.scene}_scene.json", rig, labeling=False, route=r.route)
        if r.chooser == "straightest":
            chooser = StraightestChooser()
        elif r.chooser == "cheapest":
            chooser = CheapestChooser()
        elif r.chooser == "oracle":
            chooser = OracleChooser(world.trail)
        else:
            for other in [v for v in vlms if v != r.chooser]:  # one VLM on the GPU at a time
                del vlms[other]
                import torch

                torch.cuda.empty_cache()
            if r.chooser not in vlms:
                vlms[r.chooser] = load_vlm(r.chooser, args)
            chooser = VlmChooser(vlms[r.chooser], r.chooser, calib)
        summary = run_and_save(world, chooser, calib, r.scene, r.mode, LoopParams(), args.runs_dir / name,
                               args.pictures_every, progress_every=40,
                               extra={"adapter": str(args.adapter) if r.chooser == "finetuned" else None})
        print("  -> " + json.dumps({k2: summary[k2] for k2 in ("outcome", "stop_reason", "progress_m", "trail_length_m",
                                                            "max_offset_m", "min_gap_m", "decisions", "wall_s")}),
              flush=True)
        del world
    print(f"\nDone in {(time.perf_counter() - started) / 60:.0f} min. Next: python scripts/sim_evaluate.py")


if __name__ == "__main__":
    main()
