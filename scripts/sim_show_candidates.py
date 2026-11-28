"""Phase 6, step 6.2: candidate paths and the selector's picture on MAVS frames (and, optionally, the VLM's answers).

For a few places along the trail of a scene (from step 6.1), the vehicle is put on the trail
facing along it, one frame is captured, and the phase 2-3 pipeline runs unchanged: LiDAR ->
cost map -> up to 4 candidates -> the picture with colored paths and ticks. Saved per frame
(outputs/sim/candidates/<scene>_<waypoint>_*.png):
- picture: exactly what the VLM sees (800 px wide, random colors as in phases 3-5);
- top: the relaxed cost grid from above with the candidates and the trail ahead (the trail is
  the scene's answer key, never shown to the VLM).

With --ask, the VLM answers each picture (Mode A, and Mode B with the word from the trail
~50 m ahead): the zero-shot 4B and / or the fine-tuned adapter (phase 5's skip epoch 4), both
asked without the reasoning field as in phase 5. This is the general-ability check moved here
from phase 5: does the RELLIS-trained model still answer sensibly on simulated images?
Answers go to answers.jsonl and a summary table to summary.md.

Usage (from the repo root; needs MAVS, and a GPU for --ask):
    python scripts/sim_show_candidates.py                                # pictures only
    python scripts/sim_show_candidates.py --ask zeroshot finetuned       # + both models' answers
    python scripts/sim_show_candidates.py --scene trail01 --waypoints 30 120 210
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.mapping.elevation import GridSpec  # noqa: E402
from offroad.nav.candidates import camera_half_fov_deg  # noqa: E402
from offroad.nav.driven_path import point_at_distance  # noqa: E402
from offroad.nav.labels import TICKS_M  # noqa: E402
from offroad.nav.prompts import parse_answer  # noqa: E402
from offroad.nav.selector import prepare_selector_input  # noqa: E402
from offroad.nav.visual_prompt import assign_colors, frame_seed  # noqa: E402
from offroad.sim.mavs_world import MavsWorld  # noqa: E402
from offroad.sim.planning import cheapest_index, plan_frame, straightest_index, trail_ahead_flu, trail_direction_word  # noqa: E402
from offroad.sim.rig import RigSpec, camera_calibration  # noqa: E402
from offroad.viz.topdown import show_cost_values  # noqa: E402
from offroad.vlm.runner import strip_thinking  # noqa: E402

SCENES_DIR = REPO_ROOT / "outputs" / "sim" / "scenes"
DEFAULT_ADAPTER = REPO_ROOT / "outputs" / "adapters" / "nomatch-skip" / "epoch4"
MAX_NEW_TOKENS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="trail01", help="scene name under outputs/sim/scenes (from step 6.1)")
    parser.add_argument("--waypoints", type=int, nargs="+", default=[30, 90, 150, 210, 270, 330],
                        help="trail waypoints to stand at (~0.4 m apart)")
    parser.add_argument("--ask", nargs="*", choices=("zeroshot", "finetuned"), default=[],
                        help="also ask these models (needs the GPU)")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "sim" / "candidates")
    return parser.parse_args()


def tick_points(xy: np.ndarray) -> np.ndarray:
    points = [point_at_distance(xy, d) for d in TICKS_M]
    return np.array([p for p in points if p is not None]).reshape(-1, 2)


def save_top_view(path: Path, plan, colors, trail_flu, spec: GridSpec, calib, image_width: int, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 9))
    show_cost_values(ax, fig, plan.cost, spec, title)
    half = camera_half_fov_deg(calib.K, image_width)[0]
    for sign in (1, -1):  # camera view edges
        ax.plot([0, 30 * math.tan(math.radians(half)) * sign], [0, 30], "--", color="white", linewidth=1)
    ax.plot(trail_flu[:, 1], trail_flu[:, 0], ":", color="cyan", linewidth=3, label="trail (answer key)")
    rgb_of = {name: np.array(rgb) / 255 for name, rgb in colors}
    for (name, _), c in zip(colors, plan.candidates):
        ax.plot(c.xy[:, 1], c.xy[:, 0], "-", color=rgb_of[name], linewidth=2.5,
                label=f"{name}: {c.end_bearing_deg:+.0f} deg, {c.length_m:.1f} m, cost/m {c.cost_per_m:.2f}")
        t = tick_points(c.xy)
        ax.plot(t[:, 1], t[:, 0], "o", color="white", markeredgecolor="black", markersize=5)
    ax.set_xlim(12, -12)
    ax.set_ylim(spec.x_min, 25)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scene_json = args.scenes_dir / args.scene / f"{args.scene}_scene.json"
    if not scene_json.exists():
        sys.exit(f"{scene_json} is missing; run scripts/sim_check_rig.py first (step 6.1)")
    if "finetuned" in args.ask and not (args.adapter / "adapter_config.json").exists():
        sys.exit(f"No adapter in {args.adapter}; copy phase 5's outputs/adapters/nomatch-skip/epoch4 there")
    args.out.mkdir(parents=True, exist_ok=True)
    rig, spec = RigSpec(), GridSpec()
    calib = camera_calibration(rig)
    world = MavsWorld(scene_json, rig, labeling=False)

    frames = []
    for wp in args.waypoints:
        if wp + 10 >= len(world.trail):
            print(f"waypoint {wp}: beyond the trail ({len(world.trail)} waypoints), skipped")
            continue
        p0, p1 = world.trail[wp], world.trail[wp + 10]
        world.place(p0[0], p0[1], math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        capture = world.capture()
        started = time.perf_counter()
        plan = plan_frame(capture.points_lidar, calib, capture.image.shape[1], spec)
        plan_s = time.perf_counter() - started
        word = trail_direction_word(world.trail, capture.position, capture.yaw)
        trail_flu = trail_ahead_flu(world.trail, capture.position, capture.yaw)
        trail_flu = trail_flu[np.hypot(trail_flu[:, 0], trail_flu[:, 1]) < 30]
        name = f"{args.scene}_{wp:04d}"
        # The same colors as in the picture (prepare_selector_input draws with this palette).
        palette = assign_colors(len(plan.candidates), frame_seed(args.scene, wp)) if plan.candidates else []
        if plan.candidates:
            sel = prepare_selector_input(capture.image, calib, plan.candidates, plan.costmap.ground_z, spec,
                                         args.scene, wp, "A", reasoning=False)
            cv2.imwrite(str(args.out / f"{name}_picture.png"), sel.picture[:, :, ::-1])
        save_top_view(args.out / f"{name}_top.png", plan, palette, trail_flu, spec, calib, capture.image.shape[1],
                      f"{name}: {len(plan.candidates)} candidate(s), trail word '{word}'")
        s, c = straightest_index(plan.candidates), cheapest_index(plan.candidates)
        print(f"{name}: {len(plan.candidates)} candidate(s) in {plan_s:.2f} s, trail word {word!r}; "
              + ", ".join(f"{palette[k][0]} {cand.end_bearing_deg:+.0f} deg {cand.length_m:.1f} m"
                          for k, cand in enumerate(plan.candidates))
              + (f"; straightest {palette[s][0]}, cheapest {palette[c][0]}" if plan.candidates else ""), flush=True)
        frames.append({"name": name, "waypoint": wp, "image": capture.image, "plan": plan, "word": word,
                       "palette": palette, "straightest": s, "cheapest": c})

    answers = []
    for which in args.ask:
        from offroad.vlm.backend import TransformersVlm

        adapter = str(args.adapter) if which == "finetuned" else None
        vlm = TransformersVlm(args.model, quantize="4bit", max_new_tokens=MAX_NEW_TOKENS, adapter=adapter)
        print(f"Loaded {which} in {vlm.load_seconds:.0f} s", flush=True)
        for f in frames:
            if not f["plan"].candidates:
                continue
            for mode in ("A", "B"):
                word = f["word"] if mode == "B" else None
                if mode == "B" and word is None:
                    continue
                sel = prepare_selector_input(f["image"], calib, f["plan"].candidates, f["plan"].costmap.ground_z,
                                             spec, args.scene, f["waypoint"], mode, word, reasoning=False)
                reply = vlm.generate(sel.picture, sel.prompt)
                text = strip_thinking(reply.text)
                answer = parse_answer(text, sel.colors)
                k = sel.colors.index(answer.choice) if answer.choice in sel.colors else None
                record = {"frame": f["name"], "model": which, "mode": mode, "word": word, "colors": sel.colors,
                          "valid": answer.valid, "choice": answer.choice, "action": answer.action,
                          "safe_until": answer.safe_until, "problems": answer.problems,
                          "choice_bearing_deg": None if k is None else round(f["plan"].candidates[k].end_bearing_deg, 1),
                          "is_straightest": k == f["straightest"], "is_cheapest": k == f["cheapest"],
                          "seconds": round(reply.seconds, 2), "reply": text}
                answers.append(record)
                print(f"  {which:9s} {f['name']} mode {mode}{' ' + word if word else ''}: choice {answer.choice} "
                      f"({record['choice_bearing_deg']} deg), action {answer.action}, ticks {answer.safe_until}, "
                      f"valid {answer.valid}, {reply.seconds:.1f} s", flush=True)
        del vlm
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    if answers:
        with open(args.out / "answers.jsonl", "w", encoding="utf-8") as fh:
            for r in answers:
                fh.write(json.dumps(r) + "\n")
        lines = ["| model | mode | answers | valid | stop | choice = straightest | choice = cheapest | all ticks 'all' | s/answer |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for which in args.ask:
            for mode in ("A", "B"):
                rs = [r for r in answers if r["model"] == which and r["mode"] == mode]
                if not rs:
                    continue
                pct = lambda xs: f"{100 * np.mean(xs):.0f}%"  # noqa: E731
                lines.append(f"| {which} | {mode} | {len(rs)} | {pct([r['valid'] for r in rs])} | "
                             f"{pct([r['action'] == 'stop' for r in rs])} | {pct([r['is_straightest'] for r in rs])} | "
                             f"{pct([r['is_cheapest'] for r in rs])} | "
                             f"{pct([bool(r['safe_until']) and all(v == 'all' for v in r['safe_until'].values()) for r in rs])} | "
                             f"{np.mean([r['seconds'] for r in rs]):.1f} |")
        (args.out / "summary.md").write_text(f"## {args.scene}: VLM answers on MAVS frames\n\n" + "\n".join(lines) + "\n",
                                             encoding="utf-8")
        print("\n".join(lines))
    print(f"Saved the pictures{' and answers' if answers else ''} to {args.out}")


if __name__ == "__main__":
    main()
