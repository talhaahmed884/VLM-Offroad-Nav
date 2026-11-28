"""Phase 6, step 6.5: which runs to drive, and how to score them (pure, no file I/O).

Runs: every chooser on every scene and route, in Mode A ("keep going") and Mode B ("keep going"
+ the direction word from the route ~50 m ahead). The choosers that ignore the word
(straightest, cheapest, oracle) drive the same in both modes, so they are driven in Mode A only
and their Mode A runs stand for Mode B too. The fork is driven once per route: in Mode B the word
names that branch; in Mode A there is no instruction, so taking either branch is right (success
if one of the two Mode A runs, which drive identically until the junction, finishes).

Success per scene:
- trail01, curves, halfblocked: "finished" (reached the end of the route);
- fork, Mode B: "finished" on the route the word named; Mode A: finished on either route;
- blocked: "stopped" before the wall without a collision (by the chooser saying stop, or by the
  vehicle running out of candidates: both are logged).
"""

from dataclasses import dataclass

import numpy as np

SCENE_ROUTES: tuple[tuple[str, str | None], ...] = (
    ("trail01", None), ("curves", None), ("fork", "left"), ("fork", "right"), ("blocked", None), ("halfblocked", None))
CHOOSER_ORDER = ("straightest", "cheapest", "oracle", "zeroshot", "finetuned")
WORD_BLIND = ("straightest", "cheapest", "oracle")  # ignore the word: Mode B = Mode A
MODES = ("A", "B")


@dataclass(frozen=True)
class PlannedRun:
    scene: str
    route: str | None
    chooser: str
    mode: str


def planned_runs(choosers=CHOOSER_ORDER, modes=MODES, scenes=None) -> list[PlannedRun]:
    """The runs to drive, baselines first, each VLM's runs together (so it is loaded once)."""
    runs = []
    for chooser in [c for c in CHOOSER_ORDER if c in choosers]:
        for mode in modes:
            if chooser in WORD_BLIND and mode == "B":
                continue
            for scene, route in SCENE_ROUTES:
                if scenes is None or scene in scenes:
                    runs.append(PlannedRun(scene, route, chooser, mode))
    return runs


def is_success(scene: str, summary: dict) -> bool:
    """Success of one run (for the fork: on its own route; see fork_success for Mode A)."""
    if summary.get("outcome") == "collision":
        return False
    if scene == "blocked":
        return summary.get("outcome") == "stopped"
    return summary.get("outcome") == "finished"


def fork_success(left: dict | None, right: dict | None) -> bool:
    """Mode A at the fork: either branch is right."""
    return any(s is not None and is_success("fork", s) for s in (left, right))


def decision_stats(decisions: list[dict]) -> dict:
    """What the chooser did over a run: how often it picked the straightest candidate, and for a VLM
    its answers (valid, fell back to straightest, said stop, called every path 'all')."""
    asked = [d for d in decisions if d.get("n_candidates")]
    straightest = [d for d in asked if d.get("chosen") is not None
                   and d["chosen"] == int(np.argmin(np.abs(d["end_bearings_deg"])))]
    answers = [d["answer"] for d in asked if "valid" in d.get("answer", {})]
    out = {"decisions": len(decisions), "asked": len(asked),
           "chose_straightest": round(len(straightest) / len(asked), 3) if asked else None,
           "no_candidates": len(decisions) - len(asked)}
    if answers:
        out.update({
            "valid": round(float(np.mean([a["valid"] for a in answers])), 3),
            "fallbacks": int(sum(a.get("fallback_straightest", False) for a in answers)),
            "said_stop": int(sum(a.get("action") == "stop" for a in answers)),
            "all_ticks_all": round(float(np.mean([bool(a.get("safe_until")) and all(
                v == "all" for v in a["safe_until"].values()) for a in answers])), 3),
            "seconds": round(float(np.mean([a.get("seconds", 0.0) for a in answers])), 2),
        })
    return out


def run_row(scene: str, route: str | None, chooser: str, mode: str, summary: dict, decisions: list[dict],
            stands_for_b: bool = False) -> dict:
    """One line of the per-run table."""
    stats = decision_stats(decisions)
    return {"scene": scene + (f"-{route}" if route else ""), "chooser": chooser, "mode": mode,
            "outcome": summary.get("outcome"), "stop_reason": summary.get("stop_reason"),
            "success": is_success(scene, summary), "completion": summary.get("completion"),
            "progress_m": summary.get("progress_m"), "mean_offset_m": summary.get("mean_offset_m"),
            "max_offset_m": summary.get("max_offset_m"), "min_gap_m": summary.get("min_gap_m"),
            "held": summary.get("held_decisions"), "from_mode_a": stands_for_b, **stats}


def chooser_summary(rows: list[dict]) -> dict:
    """Totals for one chooser and mode over its runs (the fork in Mode A counted once, either route)."""
    by_scene: dict[str, list[dict]] = {}
    for r in rows:
        by_scene.setdefault(r["scene"].split("-")[0] if r["scene"].startswith("fork") and r["mode"] == "A"
                            else r["scene"], []).append(r)
    successes, tests = 0, 0
    for scene, rs in by_scene.items():
        tests += 1
        successes += any(r["success"] for r in rs) if scene == "fork" else rs[0]["success"]
    driving = [r for r in rows if not r["scene"].startswith("blocked")]
    return {
        "tests": tests, "successes": successes,
        "collisions": sum(r["outcome"] == "collision" for r in rows),
        "off_trail": sum(r["outcome"] == "off_trail" for r in rows),
        "stopped_wrongly": sum(r["outcome"] == "stopped" for r in driving),
        "mean_completion": round(float(np.mean([r["completion"] for r in driving])), 3) if driving else None,
        "mean_offset_m": round(float(np.mean([r["mean_offset_m"] for r in rows if r["mean_offset_m"] is not None])), 3)
        if rows else None,
        "min_gap_m": min((r["min_gap_m"] for r in rows if r["min_gap_m"] is not None), default=None),
    }
