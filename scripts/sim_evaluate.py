"""Phase 6, step 6.5: tables of the closed-loop runs (outputs/sim/runs -> outputs/sim/evaluation.md).

Reads every run of the plan (sim/evaluation.planned_runs) that exists and writes:
1. per chooser and mode: tests passed (sim/evaluation.py says what success is per scene),
   collisions, off-trail, wrong stops, mean completion of the driving scenes, mean distance from
   the route, the closest gap to an obstacle;
2. per scene: each chooser's outcome and how far it got;
3. the blocked trail: who stopped (the chooser saying "stop", or no candidates left) and how
   far before the wall;
4. what the choosers did: how often the straightest candidate, and for the VLMs valid answers,
   fallbacks, "stop" answers, "all" for every path, seconds per answer.
The word-blind choosers (straightest, cheapest, oracle) are driven in Mode A only; their Mode A
runs stand for Mode B (marked "= A").

Usage (from the repo root; no MAVS or GPU needed):
    python scripts/sim_evaluate.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.sim.evaluation import (CHOOSER_ORDER, MODES, SCENE_ROUTES, WORD_BLIND, chooser_summary,  # noqa: E402
                                    fork_success, run_row)
from offroad.sim.metrics import project_on_trail  # noqa: E402
from offroad.sim.runs import run_name  # noqa: E402
from offroad.sim.scenes import Blockade  # noqa: E402

RUNS_DIR = REPO_ROOT / "outputs" / "sim" / "runs"
SCENES_DIR = REPO_ROOT / "outputs" / "sim" / "scenes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "sim" / "evaluation.md")
    return parser.parse_args()


def load_run(runs_dir: Path, scene, route, chooser, mode):
    folder = runs_dir / run_name(scene, route, chooser, mode)
    if not (folder / "summary.json").exists():
        return None, None
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    decisions = [json.loads(line) for line in open(folder / "decisions.jsonl", encoding="utf-8") if line.strip()]
    return summary, decisions


def table(header: list[str], rows: list[list]) -> list[str]:
    fmt = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]


def main() -> None:
    args = parse_args()
    rows, raw = [], {}
    for chooser in CHOOSER_ORDER:
        for mode in MODES:
            for scene, route in SCENE_ROUTES:
                drive_mode = "A" if chooser in WORD_BLIND else mode
                summary, decisions = load_run(args.runs_dir, scene, route, chooser, drive_mode)
                if summary is None:
                    continue
                raw[(scene, route, chooser, mode)] = summary
                rows.append(run_row(scene, route, chooser, mode, summary, decisions,
                                    stands_for_b=chooser in WORD_BLIND and mode == "B"))
    if not rows:
        sys.exit(f"No runs in {args.runs_dir}; run scripts/sim_run_all.py first")

    lines = ["## Phase 6: closed-loop runs in MAVS", "",
             "Success: trail01 / curves / halfblocked = reach the end; fork = reach the end of the branch the "
             "word names (Mode B), either branch (Mode A); blocked = stop before the wall. Word-blind choosers "
             "(straightest, cheapest, oracle) are driven in Mode A; their Mode B is the same run (= A). The oracle "
             "uses the answer key (the candidate closest to the route): a reference, not a real chooser.", ""]

    summary_rows = []
    for chooser in CHOOSER_ORDER:
        for mode in MODES:
            rs = [r for r in rows if r["chooser"] == chooser and r["mode"] == mode]
            if not rs:
                continue
            s = chooser_summary(rs)
            summary_rows.append([chooser, mode + (" (= A)" if chooser in WORD_BLIND and mode == "B" else ""),
                                 f"{s['successes']} / {s['tests']}", s["collisions"], s["off_trail"],
                                 s["stopped_wrongly"], s["mean_completion"], s["mean_offset_m"], s["min_gap_m"]])
    lines += ["### 1. Per chooser", ""] + table(
        ["chooser", "mode", "tests passed", "collisions", "off trail", "stopped wrongly",
         "mean completion (driving scenes)", "mean distance from route (m)", "closest gap to an obstacle (m)"],
        summary_rows) + [""]

    scene_rows = []
    for scene, route in SCENE_ROUTES:
        name = scene + (f"-{route}" if route else "")
        row = [name]
        for chooser in CHOOSER_ORDER:
            for mode in MODES:
                r = next((x for x in rows if x["scene"] == name and x["chooser"] == chooser and x["mode"] == mode), None)
                row.append("-" if r is None else f"{'OK' if r['success'] else 'x'} {r['outcome']} "
                                                 f"{100 * (r['completion'] or 0):.0f}%")
        scene_rows.append(row)
    fork_row = ["fork (Mode A: either branch)"]
    for chooser in CHOOSER_ORDER:
        for mode in MODES:
            left, right = raw.get(("fork", "left", chooser, mode)), raw.get(("fork", "right", chooser, mode))
            fork_row.append("-" if mode == "B" or (left is None and right is None) else
                            ("OK" if fork_success(left, right) else "x"))
    scene_rows.append(fork_row)
    lines += ["### 2. Per scene (success, outcome, share of the route driven)", ""] + table(
        ["scene"] + [f"{c} {m}" for c in CHOOSER_ORDER for m in MODES], scene_rows) + [""]

    design = args.scenes_dir / "blocked" / "blocked_design.json"
    blocked_rows = []
    if design.exists():
        d = json.loads(design.read_text(encoding="utf-8"))
        route = np.array(d["routes"][d["default_route"]])
        wall = np.mean([Blockade(b["x"], b["y"], b["yaw"]).footprint_points().mean(axis=0) for b in d["blockades"]], axis=0)
        wall_s, _ = project_on_trail(route, wall)
        for chooser in CHOOSER_ORDER:
            for mode in MODES:
                summary, decisions = load_run(args.runs_dir, "blocked", None, chooser, "A" if chooser in WORD_BLIND else mode)
                if summary is None or (chooser in WORD_BLIND and mode == "B"):
                    continue
                last = decisions[-1]["progress_m"] if decisions else None
                first_stop = next((d for d in decisions if d.get("answer", {}).get("action") == "stop"), None)
                blocked_rows.append([chooser, mode, summary["outcome"], summary.get("stop_reason"),
                                     None if last is None else round(wall_s - last, 1),
                                     None if first_stop is None else round(wall_s - first_stop["progress_m"], 1)])
    lines += ["### 3. Blocked trail: who stopped, and how far before the wall", ""] + table(
        ["chooser", "mode", "outcome", "why it stopped", "stopped this far before the wall (m)",
         "first 'stop' answer this far before the wall (m)"], blocked_rows) + [""]

    behaviour = []
    for chooser in CHOOSER_ORDER:
        for mode in MODES:
            rs = [r for r in rows if r["chooser"] == chooser and r["mode"] == mode and not r["from_mode_a"]]
            if not rs:
                continue
            asked = sum(r["asked"] for r in rs)
            w = lambda key: (sum((r.get(key) or 0) * r["asked"] for r in rs) / asked) if asked else None  # noqa: E731
            vlm = any("valid" in r for r in rs)
            behaviour.append([chooser, mode, sum(r["decisions"] for r in rs), asked, w("chose_straightest"),
                              w("valid") if vlm else None, sum(r.get("fallbacks", 0) for r in rs) if vlm else None,
                              sum(r.get("said_stop", 0) for r in rs) if vlm else None,
                              w("all_ticks_all") if vlm else None, w("seconds") if vlm else None])
    lines += ["### 4. What the choosers did", ""] + table(
        ["chooser", "mode", "decisions", "with candidates", "chose the straightest", "valid answers", "fallbacks",
         "'stop' answers", "every path 'all'", "s per answer"], behaviour) + [""]

    lines += ["### 5. All runs", ""] + table(
        ["scene", "chooser", "mode", "outcome", "why stopped", "completion", "max dist. (m)", "closest gap (m)",
         "held", "decisions"],
        [[r["scene"], r["chooser"], r["mode"] + (" (= A)" if r["from_mode_a"] else ""), r["outcome"], r["stop_reason"],
          r["completion"], r["max_offset_m"], r["min_gap_m"], r["held"], r["decisions"]] for r in rows]) + [""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + len(summary_rows) + 6]))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
