"""Run the unit tests and every project script in one go, then print a pass/fail summary.

Each step runs with the same Python as this script (so the same conda env), and its
output is shown live. A failed step doesn't stop the others; the summary marks it FAILED
and the exit code is non-zero.

Usage (from the repo root):
    python scripts/run_all.py                   # tests + every script on one sample frame (quick)
    python scripts/run_all.py --full            # also the slow steps (all eval frames, label stats)
    python scripts/run_all.py --tests-only      # just the unit tests
    python scripts/run_all.py --only grid baseline   # just the named steps
    python scripts/run_all.py --list            # show the steps and their commands
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Step:
    name: str
    description: str
    args: tuple[str, ...]  # passed to Python, e.g. ("-m", "pytest") or ("scripts/x.py", "--flag")
    slow: bool = False  # only run with --full or --only


# In pipeline order. Add new scripts here as the project grows.
STEPS = [
    Step("tests", "unit tests (no dataset needed)", ("-m", "pytest", "-q")),
    Step("projection", "phase 1: LiDAR overlay on frame 00000/0",
         ("scripts/project_lidar.py", "--sequence", "0", "--frame", "0")),
    Step("label-stats", "phase 2: class shares for all 6,234 label images (several minutes)",
         ("scripts/compute_label_stats.py",), slow=True),
    Step("candidates", "phase 2: evaluation-frame candidates and contact sheets",
         ("scripts/select_eval_frames.py",), slow=True),
    Step("elevation", "phase 3: elevation map for 00000/0",
         ("scripts/build_elevation_map.py", "--sequence", "0", "--frame", "0")),
    Step("costmap", "phase 3: cost map for 00000/0",
         ("scripts/build_cost_map.py", "--sequence", "0", "--frame", "0")),
    Step("path", "phase 3: A* path for 00000/0",
         ("scripts/plan_path.py", "--sequence", "0", "--frame", "0")),
    Step("baseline", "phase 3: baseline on all 36 evaluation frames (1-2 minutes)",
         ("scripts/run_baseline.py",), slow=True),
    Step("grid", "phase 4: VLM grid on 00000/120",
         ("scripts/draw_grid.py", "--sequence", "0", "--frame", "120")),
    Step("grid-eval", "phase 4: VLM grid on all 36 evaluation frames",
         ("scripts/draw_grid.py", "--eval"), slow=True),
    Step("prompt", "phase 4: VLM prompt and gridded image for 00000/120",
         ("scripts/vlm_prompt.py", "--sequence", "0", "--frame", "120")),
    # The VLM steps need requirements-vlm.txt and a GPU, so they only run with --full or --only.
    Step("gpu", "phase 4: check PyTorch, the GPU, and the VLM libraries",
         ("scripts/check_gpu.py",), slow=True),
    Step("vlm-one", "phase 4: VLM answer for 00000/120 (downloads the model the first time)",
         ("scripts/run_vlm.py", "--sequence", "0", "--frame", "120"), slow=True),
    Step("vlm", "phase 4: VLM answers for all 36 evaluation frames",
         ("scripts/run_vlm.py",), slow=True),
    Step("vlm-score", "phase 4: score saved VLM answers against the label images, draw them",
         ("scripts/score_vlm.py",), slow=True),
    Step("gridmap", "phase 5: image grid cells on the top-down map for 00000/120",
         ("scripts/show_grid_on_map.py", "--sequence", "0", "--frame", "120")),
    # Needs a saved VLM answer for the frame (outputs/vlm_runs/...), so it runs with --full or --only.
    Step("fused", "phase 5: cost map re-weighted by the VLM hints for 00001/2132 (tall grass)",
         ("scripts/show_fused_costmap.py", "--sequence", "1", "--frame", "2132"), slow=True),
    Step("hybrid", "phase 5: baseline vs hybrid planner on all 36 evaluation frames (needs saved VLM answers)",
         ("scripts/run_hybrid.py",), slow=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="also run the slow steps")
    group.add_argument("--tests-only", action="store_true", help="run only the unit tests")
    group.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps (see --list)")
    group.add_argument("--list", action="store_true", help="list the steps and exit")
    return parser.parse_args()


def choose_steps(args: argparse.Namespace) -> list[Step]:
    if args.tests_only:
        return [s for s in STEPS if s.name == "tests"]
    if args.only:
        known = {s.name for s in STEPS}
        unknown = [name for name in args.only if name not in known]
        if unknown:
            sys.exit(f"Unknown step(s): {', '.join(unknown)}. Known: {', '.join(known)}")
        return [s for s in STEPS if s.name in args.only]
    return [s for s in STEPS if args.full or not s.slow]


def run_step(step: Step) -> tuple[bool, float]:
    """Run one step with this Python from the repo root; returns (passed, seconds)."""
    print(f"\n{'=' * 70}\n>>> {step.name}: {step.description}\n{'=' * 70}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, *step.args], cwd=REPO_ROOT)
    return completed.returncode == 0, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    if args.list:
        for step in STEPS:
            marker = " (slow, --full only)" if step.slow else ""
            print(f"{step.name:12s} {step.description}{marker}\n{'':12s} python {' '.join(step.args)}")
        return

    results = [(step, *run_step(step)) for step in choose_steps(args)]

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for step, passed, seconds in results:
        print(f"  {'ok    ' if passed else 'FAILED'}  {step.name:12s} {seconds:6.1f} s")
    failed = [step.name for step, passed, _ in results if not passed]
    if failed:
        print(f"\n{len(failed)} of {len(results)} steps FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"\nAll {len(results)} steps passed. Pictures and tables are in outputs/.")


if __name__ == "__main__":
    main()
