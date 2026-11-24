"""Run the unit tests and every project script in one go, then print a pass/fail summary.

Each step runs with the same Python as this script (so the same conda env), and its
output is shown live. A failed step doesn't stop the others; the summary marks it FAILED
and the exit code is non-zero.

Usage (from the repo root):
    python scripts/run_all.py                   # tests + every script on one sample frame (quick)
    python scripts/run_all.py --full            # also the slow steps (e.g. the GPU check)
    python scripts/run_all.py --tests-only      # just the unit tests
    python scripts/run_all.py --only costmap path    # just the named steps
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
    # Shared building blocks (kept from approach 1).
    Step("projection", "LiDAR overlay on frame 00000/0",
         ("scripts/project_lidar.py", "--sequence", "0", "--frame", "0")),
    Step("elevation", "elevation map for 00000/0",
         ("scripts/build_elevation_map.py", "--sequence", "0", "--frame", "0")),
    Step("costmap", "cost map for 00000/0",
         ("scripts/build_cost_map.py", "--sequence", "0", "--frame", "0")),
    Step("path", "geometric A* path for 00000/0",
         ("scripts/plan_path.py", "--sequence", "0", "--frame", "0")),
    # Needs requirements-vlm.txt and a GPU, so it only runs with --full or --only.
    Step("gpu", "check PyTorch, the GPU, and the VLM / fine-tuning libraries",
         ("scripts/check_gpu.py",), slow=True),
    # Approach 2 (see docs/approach-2-plan.md).
    Step("frames", "phase 1: choose the decision frames -> configs/decision_frames.csv",
         ("scripts/build_decision_frames.py",)),
    Step("driven", "phase 1: driven path from the poses for 00000/600",
         ("scripts/show_driven_path.py", "--sequence", "0", "--frame", "600")),
    Step("candidates", "phase 2: candidates + labels for every decision frame, strict vs relaxed (~2 min)",
         ("scripts/build_candidates.py",)),
    Step("show", "phase 2: candidates of 00000/601 on the camera image and the cost grids",
         ("scripts/show_candidates.py", "--sequence", "0", "--frame", "601")),
    Step("prompt", "phase 3: the VLM's picture and Mode B prompt for 00000/601",
         ("scripts/show_prompt.py", "--sequence", "0", "--frame", "601", "--mode", "B")),
    # Needs the VLM on the GPU (~1-2 min with loading), so it only runs with --full or --only.
    Step("selector", "phase 3: zero-shot Qwen3-VL-4B on 3 validation frames, both modes",
         ("scripts/run_selector.py", "--limit", "3", "--tag", "smoke", "--fresh"), slow=True),
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
