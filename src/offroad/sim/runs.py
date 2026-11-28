"""Drive one closed-loop run and save it (shared by scripts/sim_drive.py and scripts/sim_run_all.py).

A run folder (outputs/sim/runs/<scene>[-<route>]_<chooser>_mode<A|B>/) holds:
- decisions.jsonl: one line per decision (time, pose, trail progress and offset, candidates,
  the choice, and the VLM's full answer);
- summary.json: the episode summary (sim/loop.py) plus wall time, adapter and route;
- trajectory.npy and trajectory.png: the driven path over the route (from above) and the
  distance from the route centre along the run;
- pictures/: every pictures_every-th decision, the picture with the candidates (what a VLM sees).
"""

import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from offroad.mapping.elevation import GridSpec  # noqa: E402
from offroad.nav.selector import prepare_selector_input  # noqa: E402
from offroad.sim.loop import Episode, LoopParams, run_episode  # noqa: E402


def run_name(scene: str, route: str | None, chooser: str, mode: str) -> str:
    return f"{scene}{f'-{route}' if route else ''}_{chooser}_mode{mode}"


def save_trajectory_figure(path: Path, trail: np.ndarray, episode: Episode) -> None:
    traj = episode.trajectory
    fig, (top, side) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.2, 1]})
    top.plot(trail[:, 0], trail[:, 1], "-", color="tab:cyan", linewidth=8, alpha=0.35, label="route (trail 4 m wide)")
    if len(traj):
        top.plot(traj[:, 0], traj[:, 1], "-", color="tab:red", linewidth=1.5, label="driven")
        top.plot(traj[0, 0], traj[0, 1], "o", color="green", label="start")
        top.plot(traj[-1, 0], traj[-1, 1], "X", color="black", markersize=10, label=f"end: {episode.outcome}")
    top.set_aspect("equal", adjustable="datalim")
    span = max(30.0, float(np.ptp(trail[:, 1])) + 10.0)  # a straight trail would be a thin strip
    top.set_ylim(float(np.mean(trail[:, 1])) - span / 2, float(np.mean(trail[:, 1])) + span / 2)
    top.set_xlabel("x (m, east)")
    top.set_ylabel("y (m, north)")
    top.legend(loc="best", fontsize=8)
    s = episode.summary
    top.set_title(f"{s['scene']} / {s['chooser']} / mode {s['mode']}: {episode.outcome}, "
                  f"{s['progress_m']:.0f} of {s['trail_length_m']:.0f} m")
    prog = [d["progress_m"] for d in episode.decisions]
    off = [d["offset_m"] for d in episode.decisions]
    side.plot(prog, off, "-", color="tab:red")
    side.axhline(0, color="gray", linewidth=0.8)
    for y in (2.0, -2.0):
        side.axhline(y, color="tab:cyan", linestyle="--", linewidth=1)
    side.axhspan(-2, 2, color="tab:cyan", alpha=0.1)
    side.set_xlabel("progress along the route (m)")
    side.set_ylabel("distance from the route (m, + = left)")
    side.set_title(f"mean {s['mean_offset_m']} m, max {s['max_offset_m']} m (trail edges dashed)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def run_and_save(world, chooser, calib, scene: str, mode: str, params: LoopParams, out: Path,
                 pictures_every: int = 10, spec: GridSpec = GridSpec(), extra: dict | None = None,
                 progress_every: int = 20) -> dict:
    """Drive one run with `chooser` in `world` and save it to `out`; returns the summary."""
    (out / "pictures").mkdir(parents=True, exist_ok=True)
    log = open(out / "decisions.jsonl", "w", encoding="utf-8")
    started = time.perf_counter()

    def on_decision(number, record, capture, plan, decision):
        log.write(json.dumps(record) + "\n")
        log.flush()
        if pictures_every and number % pictures_every == 0 and plan.candidates:
            picture = decision.picture if decision is not None and decision.picture is not None else \
                prepare_selector_input(capture.image, calib, plan.candidates, plan.costmap.ground_z, spec,
                                       scene, number, mode, record["word"], reasoning=False).picture
            cv2.imwrite(str(out / "pictures" / f"{number:04d}.png"), picture[:, :, ::-1])
        if progress_every and number % progress_every == 0:
            answer = record.get("answer", {})
            print(f"  t {record['time_s']:6.1f} s  progress {record['progress_m']:6.1f} m  offset "
                  f"{record['offset_m']:+.2f} m  {record['n_candidates']} candidates  chosen "
                  f"{record['chosen_bearing_deg']} deg{'  ' + str(answer.get('choice')) if answer.get('choice') else ''}"
                  f"  ({time.perf_counter() - started:.0f} s wall)", flush=True)

    try:
        episode = run_episode(world, chooser, calib, scene, mode, params, spec, on_decision=on_decision)
    finally:
        log.close()
    summary = {**episode.summary, "wall_s": round(time.perf_counter() - started, 1), "route": world.route,
               **(extra or {})}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.save(out / "trajectory.npy", episode.trajectory)
    save_trajectory_figure(out / "trajectory.png", np.asarray(world.trail), episode)
    return summary
