"""The closed loop: decide, drive half a second, decide again from wherever the vehicle ended up.

One run ("episode") on one trail with one chooser:
1. capture the camera image and the LiDAR scan at the vehicle's pose;
2. plan the candidates from the scan (`sim.planning.plan_frame`);
3. the chooser picks one (or stops);
4. the chosen path is fixed in the world and followed (`sim.control.follow`) for
   `decision_period_s` of simulated time in steps of `control_dt_s`; then back to 1.
Simulated time stands still while the chooser thinks, so a slow VLM doesn't crash the vehicle
(as if the vehicle waited for each answer).

A decision without any candidate (e.g. every path swings out of the camera's narrow view to get
round tall grass) keeps following the previous path, for at most `max_hold` decisions in a row
(1 s); then the vehicle stops. On RELLIS, a frame without candidates meant "stop" at once; in
closed loop one empty frame on a grassy trail would end most runs within ~10 m (2026-11-26).

The run ends with one outcome:
- "finished": within end_margin_m of the trail's end;
- "stopped": the chooser said stop, or there was no candidate for more than max_hold decisions;
- "off_trail": more than off_trail_m from the trail centre (counted as an intervention);
- "stuck": less than stuck_min_progress_m along the trail in stuck_window_s;
- "collision": the vehicle's footprint (vehicle_length_m x vehicle_width_m) touched a tree trunk
  or a blockade (world.obstacles; checked at every control step, since MAVS's vehicle drives
  through them: its physics runs on the bare ground);
- "time_limit": max_time_s of simulated time.
The closest gap between the footprint and any obstacle over the run is logged (min_gap_m).
The trail is the answer key for these checks only; the chooser never sees it.

Mode B's direction word comes from the trail ~50 m ahead (`trail_direction_word`). In the trail's
last 20 m there is no word (as on RELLIS near a drive's end), so the previous word is kept: an
instruction stays in force until a new one is given (2026-11-27; the first Mode B run crashed there).

`world` is a `sim.mavs_world.MavsWorld` (or a stand-in with the same methods, for tests):
trail, place(x, y, yaw), capture() -> Capture, step(throttle, steering, braking, dt),
heading_and_speed() -> (heading, forward speed).
"""

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from offroad.data.calibration import CameraCalibration
from offroad.mapping.elevation import GridSpec
from offroad.sim.choosers import Chooser, DecisionContext
from offroad.sim.control import FollowerParams, follow, to_world
from offroad.sim.metrics import cumulative_length, footprint_gaps, is_stuck, project_on_trail
from offroad.sim.planning import plan_frame, trail_direction_word

OUTCOMES = ("finished", "stopped", "off_trail", "stuck", "collision", "time_limit")


@dataclass(frozen=True)
class LoopParams:
    decision_period_s: float = 0.5  # 2 Hz, as the decision frames on RELLIS
    control_dt_s: float = 0.02
    max_time_s: float = 240.0
    start_waypoint: int = 10
    end_margin_m: float = 5.0
    off_trail_m: float = 3.0  # trail half-width (2 m) plus 1 m
    stuck_window_s: float = 10.0
    stuck_min_progress_m: float = 0.5
    max_hold: int = 2  # decisions in a row without candidates that keep the previous path
    vehicle_length_m: float = 3.5  # MAVS MRZR4 chassis
    vehicle_width_m: float = 1.6  # track 1.34 m + tyres
    follower: FollowerParams = field(default_factory=FollowerParams)


@dataclass
class Episode:
    outcome: str
    decisions: list[dict]  # one record per decision (see run_episode)
    trajectory: np.ndarray  # (T, 3) world x, y and time of the vehicle at every control step
    summary: dict


def start_pose(trail: np.ndarray, waypoint: int, look_ahead: int = 10) -> tuple[float, float, float]:
    """(x, y, heading) on the trail at a waypoint, facing along it."""
    p0, p1 = trail[waypoint], trail[min(waypoint + look_ahead, len(trail) - 1)]
    return float(p0[0]), float(p0[1]), math.atan2(p1[1] - p0[1], p1[0] - p0[0])


def run_episode(world, chooser: Chooser, calib: CameraCalibration, scene: str, mode: str = "A",
                params: LoopParams = LoopParams(), spec: GridSpec = GridSpec(), on_decision=None) -> Episode:
    """Drive one run; on_decision(number, record, capture, plan, decision) is called after each decision."""
    trail = np.asarray(world.trail, dtype=float)
    trail_end = float(cumulative_length(trail)[-1])
    world.place(*start_pose(trail, params.start_waypoint))
    chooser.reset()

    sim_t, decisions, trajectory = 0.0, [], []
    times, progress_log = [], []
    outcome, think_s, stop_reason = None, 0.0, None
    path_world, held_in_a_row, held_total = None, 0, 0
    steps_per_decision = max(1, round(params.decision_period_s / params.control_dt_s))
    obstacle_points, obstacle_radii = getattr(world, "obstacles", (np.zeros((0, 2)), np.zeros(0)))
    min_gap, hit = float("inf"), None
    last_word = "straight"  # only used if the run starts within 20 m of the trail's end

    def check_collision(xy, heading) -> bool:
        nonlocal min_gap, hit
        gaps = footprint_gaps(obstacle_points, obstacle_radii, xy, heading,
                              params.vehicle_length_m, params.vehicle_width_m)
        if len(gaps):
            k = int(np.argmin(gaps))
            min_gap = min(min_gap, float(gaps[k]))
            if gaps[k] < 0:
                hit = [round(float(v), 2) for v in obstacle_points[k]]
                return True
        return False

    while outcome is None:
        capture = world.capture()
        position, yaw = capture.position, capture.yaw
        progress, offset = project_on_trail(trail, position)
        times.append(sim_t)
        progress_log.append(progress)
        if trail_end - progress <= params.end_margin_m:
            outcome = "finished"
        elif abs(offset) > params.off_trail_m:
            outcome = "off_trail"
        elif is_stuck(times, progress_log, params.stuck_window_s, params.stuck_min_progress_m):
            outcome = "stuck"
        elif sim_t >= params.max_time_s:
            outcome = "time_limit"
        if outcome is not None:
            break

        plan = plan_frame(capture.points_lidar, calib, capture.image.shape[1], spec)
        if mode == "B":  # the trail's last 20 m give no word: the last instruction stays in force
            word = trail_direction_word(trail, position, yaw) or last_word
            last_word = word
        else:
            word = None
        ctx = DecisionContext(time_s=sim_t, number=len(decisions), scene=scene, mode=mode, word=word,
                              position_xy=(float(position[0]), float(position[1])), yaw=float(yaw))
        started = time.perf_counter()
        decision = chooser.decide(capture.image, plan, ctx) if plan.candidates else None
        think_s += time.perf_counter() - started
        index = None if decision is None else decision.index
        cand = plan.candidates[index] if index is not None else None
        held = not plan.candidates and path_world is not None and held_in_a_row < params.max_hold
        record = {"number": ctx.number, "time_s": round(sim_t, 2), "x": round(float(position[0]), 3),
                  "y": round(float(position[1]), 3), "yaw_deg": round(math.degrees(yaw), 2),
                  "progress_m": round(progress, 2), "offset_m": round(offset, 3), "word": word,
                  "n_candidates": len(plan.candidates), "hidden_only": plan.hidden_only, "held": held,
                  "end_bearings_deg": [round(c.end_bearing_deg, 1) for c in plan.candidates],
                  "lengths_m": [round(c.length_m, 2) for c in plan.candidates],
                  "chosen": index, "chosen_bearing_deg": None if cand is None else round(cand.end_bearing_deg, 1),
                  **({} if decision is None else {"answer": decision.record})}
        decisions.append(record)
        if on_decision is not None:
            on_decision(ctx.number, record, capture, plan, decision)
        if cand is not None:
            path_world = to_world(cand.xy, position[:2], yaw)
            held_in_a_row = 0
        elif held:  # no candidate this time: keep driving the previous path
            held_in_a_row += 1
            held_total += 1
        else:
            outcome = "stopped"
            stop_reason = "chooser said stop" if plan.candidates else "no candidates"
            break

        for _ in range(steps_per_decision):
            pos_now, _ = world.pose()
            heading, speed = world.heading_and_speed()
            throttle, steering, braking, _ = follow(path_world, pos_now[:2], heading, speed, params.follower)
            world.step(throttle, steering, braking, params.control_dt_s)
            sim_t += params.control_dt_s
            trajectory.append((float(pos_now[0]), float(pos_now[1]), sim_t))
            if check_collision(pos_now[:2], heading):
                outcome = "collision"
                break

    traj = np.array(trajectory).reshape(-1, 3)
    offsets = [abs(d["offset_m"]) for d in decisions]
    summary = {
        "scene": scene, "chooser": chooser.name, "mode": mode, "outcome": outcome, "stop_reason": stop_reason,
        "held_decisions": held_total, "collision_with": hit,
        "min_gap_m": None if math.isinf(min_gap) else round(min_gap, 3),
        "no_candidate_decisions": sum(1 for d in decisions if d["n_candidates"] == 0),
        "progress_m": round(progress_log[-1] - progress_log[0], 2) if progress_log else 0.0,
        "trail_length_m": round(trail_end - progress_log[0], 2) if progress_log else trail_end,
        "sim_time_s": round(sim_t, 2), "decisions": len(decisions),
        "mean_offset_m": round(float(np.mean(offsets)), 3) if offsets else None,
        "max_offset_m": round(float(np.max(offsets)), 3) if offsets else None,
        "mean_think_s": round(think_s / max(1, len(decisions)), 2),
        "params": {k: v for k, v in asdict(params).items() if k != "follower"},
    }
    summary["completion"] = round(summary["progress_m"] / max(summary["trail_length_m"], 1e-6), 3)
    return Episode(outcome=outcome, decisions=decisions, trajectory=traj, summary=summary)
