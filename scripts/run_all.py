"""Run the unit tests and every project script in one go, then print a pass/fail summary.

Each step runs with the same Python as this script (so the same conda env), and its
output is shown live. A failed step doesn't stop the others; the summary marks it FAILED
and the exit code is non-zero.

Usage (from the repo root):
    python scripts/run_all.py                   # the quick steps (unit tests)
    python scripts/run_all.py --full            # also the slow steps (GPU check, MAVS)
    python scripts/run_all.py --tests-only      # just the unit tests
    python scripts/run_all.py --only sim_rig    # just the named steps
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
    Step("tests", "unit tests (no dataset, GPU or MAVS needed)", ("-m", "pytest", "-q")),
    # Needs requirements-vlm.txt and a GPU, so it only runs with --full or --only.
    Step("gpu", "check PyTorch, the GPU, and the VLM libraries",
         ("scripts/check_gpu.py",), slow=True),
    # Phase 6 (closed loop in MAVS, see docs/approach-2-plan.md). Needs MAVS, so --full or --only.
    Step("sim_rig", "6.1: trail scene + Warthog with a RELLIS-like rig, one frame -> outputs/sim/rig_check",
         ("scripts/sim_check_rig.py",), slow=True),
    Step("sim_candidates", "6.2: candidates + the VLM's picture on 6 MAVS frames, zero-shot and fine-tuned answers "
         "-> outputs/sim/candidates (needs the GPU)",
         ("scripts/sim_show_candidates.py", "--ask", "zeroshot", "finetuned"), slow=True),
    Step("sim_drive", "6.3: one closed-loop run on trail01 with the straightest candidate (~3 min, no GPU) "
         "-> outputs/sim/runs/trail01_straightest_modeA",
         ("scripts/sim_drive.py", "--chooser", "straightest"), slow=True),
    Step("sim_scenes", "6.4: build and check the test scenes (curves, fork, blocked, halfblocked) "
         "-> outputs/sim/scenes", ("scripts/sim_build_scenes.py",), slow=True),
    Step("sim_oracle", "6.4: the oracle (candidate closest to the route) on the curves and the half-blocked trail "
         "-> outputs/sim/runs/{curves,halfblocked}_oracle_modeA",
         ("scripts/sim_drive.py", "--chooser", "oracle", "--scene", "curves"), slow=True),
    Step("sim_oracle_wall", "6.4: the oracle on the half-blocked trail",
         ("scripts/sim_drive.py", "--chooser", "oracle", "--scene", "halfblocked"), slow=True),
    Step("sim_runs", "6.5: every chooser on every scene, Modes A and B (resumable; ~3-4 h with the VLMs) "
         "-> outputs/sim/runs", ("scripts/sim_run_all.py",), slow=True),
    Step("sim_eval", "6.5: tables of the closed-loop runs -> outputs/sim/evaluation.md",
         ("scripts/sim_evaluate.py",), slow=True),
    Step("sim_videos", "6.5: videos of the highlight runs (replayed, chase view + the VLM's picture; ~45 min, "
         "no GPU) -> outputs/sim/videos", ("scripts/sim_record.py",), slow=True),
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
