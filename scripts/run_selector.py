"""Ask the VLM to choose among the candidate paths on decision frames, and score the answers (phase 3).

For each decision frame (in time order per sequence), it builds the relaxed candidates, draws
them on the camera image (random colors), and asks the VLM in Mode A ("keep going") and / or
Mode B ("keep going + direction"). The history is the model's own earlier answers in the same
run (as it would be in closed loop), or nothing (--history none). Frames without candidates
aren't asked: the vehicle would stop on its own.

Phase 4 tests: --history wrong replaces the latest history entry with a sharp turn the vehicle
didn't take (does a wrong history pull the choice?); --direction "bear left" gives Mode B the
same word in every frame instead of the driver's (does the choice follow the word?).
scripts/evaluate.py compares these runs with each other and with simple baselines.

Each answer is checked (valid JSON, a known color, a tick for every candidate) and scored against
the labels: does the choice match the driver, is it safe, does it go through a puddle when a
dry path existed, are the ticks right (and how often too optimistic), is the action right.

Output: outputs/selector/<tag>_mode<A|B>.jsonl (one line per frame, with the raw reply) and
<tag>_summary.md. Already answered frames are skipped when the files exist, so a stopped run
can be continued with the same command (use --fresh to start over).

Usage (from the repo root; needs requirements-vlm.txt and a GPU):
    python scripts/run_selector.py --limit 5                      # quick look: 5 validation frames
    python scripts/run_selector.py --split val --every 4          # ~84 frames x 2 modes, ~20-30 min
    python scripts/run_selector.py --split val --every 4 --history none --tag val_nohist
    python scripts/run_selector.py --split test --every 1 --modes B --direction "bear left"
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.mapping.elevation import GridSpec  # noqa: E402
from offroad.nav.candidates import CHOSEN_VARIANT  # noqa: E402
from offroad.nav.frame_candidates import candidates_for_frame  # noqa: E402
from offroad.nav.labels import has_real_choice  # noqa: E402
from offroad.nav.prompts import (  # noqa: E402
    DIRECTION_MEANINGS,
    MODES,
    HistoryEntry,
    chosen_bearing,
    history_text,
    parse_answer,
    wrong_history,
)
from offroad.nav.visual_prompt import frame_seed  # noqa: E402
from offroad.nav.scoring import score_answer  # noqa: E402
from offroad.nav.selector import prepare_selector_input  # noqa: E402
from offroad.nav.visual_prompt import PICTURE_VERSION  # noqa: E402
from offroad.vlm.backend import DEFAULT_MODEL, QUANTIZE_OPTIONS  # noqa: E402
from offroad.vlm.runner import strip_thinking  # noqa: E402

FRAMES_CSV = REPO_ROOT / "configs" / "decision_frames.csv"
MAX_NEW_TOKENS = 300
HISTORY_SETTINGS = ("own", "none", "wrong")


def default_tag(args: argparse.Namespace) -> str:
    """<split>_<model short name>, plus _nohist / _wronghist and _dir-<word> for the phase 4 tests."""
    tag = f"{args.split}_{args.model.split('/')[-1]}"
    tag += {"own": "", "none": "_nohist", "wrong": "_wronghist"}[args.history]
    return tag + (f"_dir-{args.direction.replace(' ', '-')}" if args.direction else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val", help="default: val")
    parser.add_argument("--every", type=int, default=4, help="use every Nth decision frame (default: 4, ~2 s apart)")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many frames")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="default: A B")
    parser.add_argument("--history", choices=HISTORY_SETTINGS, default="own",
                        help="own = the model's own last 3 answers (default); none = no history; wrong = own, "
                             "with the latest one replaced by a sharp turn it didn't take (phase 4 test)")
    parser.add_argument("--direction", choices=list(DIRECTION_MEANINGS), default=None,
                        help="give Mode B this direction word in every frame instead of the driver's "
                             "(phase 4 test: does the choice follow the word?); use with --modes B")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quantize", choices=QUANTIZE_OPTIONS, default="4bit")
    parser.add_argument("--tag", default=None, help="output name (default: <split>_<model short name>)")
    parser.add_argument("--save-images", type=int, default=5, help="save the first N pictures (default: 5)")
    parser.add_argument("--fresh", action="store_true", help="ignore earlier answers in the output files")
    parser.add_argument("--frames", type=Path, default=FRAMES_CSV)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "selector")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    return parser.parse_args()


def read_frames(path: Path, split: str, every: int, limit: int | None) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} is missing; run scripts/build_decision_frames.py first")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    rows.sort(key=lambda r: (r["sequence"], int(r["frame"])))
    rows = rows[::every]
    return rows[:limit] if limit else rows


def read_done(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {(r["sequence"], r["frame"]): r for r in records}


# --- the summary ---


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f}% ({int(part)}/{int(whole)})" if whole else "-"


def summarize(records: list[dict]) -> dict[str, str]:
    """One summary column for one mode (metric -> text)."""
    asked = [r for r in records if r["asked"]]
    scores = [r["score"] for r in asked]
    chosen = [s for s in scores if s["choice_matches_driver"] is not None]
    real = [r["score"] for r in asked if r["real_choice"] and r["score"]["choice_matches_driver"] is not None]
    puddle = [s for s in chosen if s["puddle_frame"]]
    ticks = {k: sum(s[f"ticks_{k}"] for s in scores) for k in ("answered", "exact", "within_one", "over", "under")}
    n_candidates = sum(len(r["colors"]) for r in asked)
    stop_label = [s for s in scores if s["stop_label"]]
    said_stop = [s for s in scores if s["action"] == "stop"]
    actions = [s for s in scores if s["action"] is not None]
    seconds = [r["seconds"] for r in asked]
    return {
        "frames (asked / no candidates)": f"{len(asked)} / {len(records) - len(asked)}",
        "valid answers": pct(sum(s["valid"] for s in scores), len(scores)),
        "choice matches the driver (1 m)": pct(sum(s["choice_matches_driver"] for s in chosen), len(chosen)),
        "... in frames with a real choice": pct(sum(s["choice_matches_driver"] for s in real), len(real)),
        "choice is the driver's closest candidate": pct(sum(s["choice_is_closest"] for s in chosen), len(chosen)),
        "chosen candidate safe to tick 1 / to the end": (
            f"{pct(sum(s['choice_safe_until'] != 'none' for s in chosen), len(chosen))} / "
            f"{pct(sum(s['choice_safe_until'] == 'all' for s in chosen), len(chosen))}"),
        "chose a puddle when a dry path existed": pct(sum(s["chose_puddle"] for s in puddle), len(puddle)),
        "ticks answered": pct(ticks["answered"], n_candidates),
        "ticks exact / within one": f"{pct(ticks['exact'], ticks['answered'])} / {pct(ticks['within_one'], ticks['answered'])}",
        "ticks too optimistic / too cautious": f"{pct(ticks['over'], ticks['answered'])} / {pct(ticks['under'], ticks['answered'])}",
        "action correct": pct(sum((s["action"] == "stop") == s["stop_label"] for s in actions), len(actions)),
        "said stop / stop is right": f"{len(said_stop)} / {len(stop_label)}",
        "seconds per answer (mean)": f"{np.mean(seconds):.1f}" if seconds else "-",
    }


def by_direction(records: list[dict]) -> dict[str, str]:
    """Mode B: driver match per direction word."""
    groups = defaultdict(list)
    for r in records:
        if r["asked"] and r["score"]["choice_matches_driver"] is not None:
            groups[r.get("driver_direction", r["direction"])].append(r["score"]["choice_matches_driver"])
    return {word: pct(sum(v), len(v)) for word, v in sorted(groups.items())}


def problem_kind(problem: str) -> str:
    """A short name for a checker problem, for counting."""
    if problem.startswith("no JSON"):
        return "no JSON"
    if problem.startswith("no safe_until"):
        return "missing ticks"
    return f"bad {problem.split(' ')[0]}"  # "bad choice" / "bad action"


def write_summary(path: Path, title: str, per_mode: dict[str, list[dict]]) -> str:
    columns = {f"Mode {m}": summarize(recs) for m, recs in per_mode.items()}
    metrics = list(next(iter(columns.values())))
    lines = [f"## {title}", "", "| metric | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    lines += [f"| {m} | " + " | ".join(col[m] for col in columns.values()) + " |" for m in metrics]
    if "B" in per_mode:
        lines += ["", "Mode B, choice matches the driver per direction word:", ""]
        lines += [f"- {word}: {text}" for word, text in by_direction(per_mode["B"]).items()]
    problems = Counter(problem_kind(p) for recs in per_mode.values() for r in recs if r["asked"] for p in r["problems"])
    if problems:
        lines += ["", "Answer problems: " + ", ".join(f"{k} x{v}" for k, v in problems.most_common())]
    report = "\n".join(lines) + "\n"
    path.write_text(report, encoding="utf-8")
    return report


# --- main loop ---


def main() -> None:
    args = parse_args()
    rows = read_frames(args.frames, args.split, args.every, args.limit)
    tag = args.tag or default_tag(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {m: args.out_dir / f"{tag}_mode{m}.jsonl" for m in args.modes}
    done = {m: ({} if args.fresh else read_done(paths[m])) for m in args.modes}
    for m in args.modes:
        old = {r.get("picture_version", 1) for r in done[m].values()} - {PICTURE_VERSION}
        if old:  # answers to an older drawing of the candidates: don't mix them with new ones
            sys.exit(f"{paths[m].name} holds answers to picture version {sorted(old)}, this code draws version "
                     f"{PICTURE_VERSION}. Use another --tag (keeps the old answers) or --fresh (overwrites them).")
    todo = [r for r in rows if any((r["sequence"], int(r["frame"])) not in done[m] for m in args.modes)]
    print(f"{len(rows)} {args.split} frames (every {args.every}), modes {' '.join(args.modes)}, "
          f"history {args.history}; {len(rows) - len(todo)} already answered", flush=True)

    vlm = None
    if todo:
        from offroad.vlm.backend import TransformersVlm

        vlm = TransformersVlm(args.model, quantize=args.quantize, max_new_tokens=MAX_NEW_TOKENS)
        print(f"Loaded {args.model} ({args.quantize}) in {vlm.load_seconds:.0f} s", flush=True)

    id_to_name, _ = load_ontology(args.root)
    spec = GridSpec()
    sequences: dict[str, RellisSequence] = {}
    history: dict[str, list[HistoryEntry]] = {m: [] for m in args.modes}
    current_sequence, saved_images, started = None, 0, time.perf_counter()
    files = {m: open(paths[m], "w" if args.fresh else "a") for m in args.modes}
    try:
        for count, row in enumerate(rows, 1):
            key = (row["sequence"], int(row["frame"]))
            if row["sequence"] != current_sequence:  # a new drive: no history yet
                current_sequence = row["sequence"]
                history = {m: [] for m in args.modes}
            timestamp = float(row["timestamp"])
            if all(key in done[m] for m in args.modes):  # answered in an earlier run: rebuild the history only
                for m in args.modes:
                    r = done[m][key]
                    if r["asked"] and r["action"] is not None:
                        history[m].append(HistoryEntry(timestamp, r["action"], r["chosen_bearing_deg"]))
                continue

            seq = sequences.setdefault(row["sequence"], RellisSequence(row["sequence"], root=args.root))
            frame = seq[key[1]]
            result = candidates_for_frame(frame.points, frame.point_labels, frame.label_image,
                                          frame.calib.T_cam_from_lidar, frame.calib.K, seq.poses, key[1], id_to_name,
                                          image_width=frame.image.shape[1], variants=(CHOSEN_VARIANT,))
            fc = result.variants[CHOSEN_VARIANT]
            for mode in args.modes:
                if key in done[mode]:
                    r = done[mode][key]
                    if r["asked"] and r["action"] is not None:
                        history[mode].append(HistoryEntry(timestamp, r["action"], r["chosen_bearing_deg"]))
                    continue
                record = {"split": row["split"], "sequence": key[0], "frame": key[1], "mode": mode,
                          "direction": (args.direction or row["direction"]) if mode == "B" else row["direction"],
                          "driver_direction": row["direction"], "direction_forced": bool(args.direction),
                          "history_setting": args.history, "history_injected": None,
                          "picture_version": PICTURE_VERSION,
                          "real_choice": has_real_choice(fc.labels), "asked": bool(fc.candidates),
                          "colors": [], "labels": [asdict(lab) for lab in fc.labels]}
                if not fc.candidates:
                    record.update(action="stop", chosen_bearing_deg=None, problems=[], seconds=0.0)
                    files[mode].write(json.dumps(record) + "\n")
                    continue
                if args.history == "none":
                    hist = "none."
                elif args.history == "wrong":
                    entries, record["history_injected"] = wrong_history(history[mode],
                                                                        frame_seed(key[0], key[1], salt=2))
                    hist = history_text(entries, timestamp)
                else:
                    hist = history_text(history[mode], timestamp)
                selector_input = prepare_selector_input(frame.image, frame.calib, fc.candidates,
                                                        result.costmap.ground_z, spec, key[0], key[1], mode,
                                                        record["direction"] or None, hist)
                reply = vlm.generate(selector_input.picture, selector_input.prompt)
                text = strip_thinking(reply.text)
                answer = parse_answer(text, selector_input.colors)
                score = score_answer(answer, selector_input.colors, fc.labels, fc.closest_index)
                bearing = chosen_bearing(answer, selector_input.colors, [c.end_bearing_deg for c in fc.candidates])
                record.update(
                    colors=selector_input.colors, listed=selector_input.listed, visible=selector_input.visible,
                    end_bearings_deg=[round(c.end_bearing_deg, 1) for c in fc.candidates], history=hist,
                    reply=text, choice=answer.choice, action=answer.action, safe_until=answer.safe_until,
                    reasoning=answer.reasoning, problems=answer.problems, chosen_bearing_deg=bearing,
                    score=asdict(score), seconds=round(reply.seconds, 2), prompt_tokens=reply.prompt_tokens,
                    new_tokens=reply.new_tokens,
                )
                files[mode].write(json.dumps(record) + "\n")
                files[mode].flush()
                if answer.action is not None:
                    history[mode].append(HistoryEntry(timestamp, answer.action, bearing))
                if saved_images < args.save_images:
                    image_path = args.out_dir / f"{tag}_{key[0]}_{key[1]:06d}_mode{mode}.png"
                    cv2.imwrite(str(image_path), cv2.cvtColor(selector_input.picture, cv2.COLOR_RGB2BGR))
                    saved_images += 1
                tick_text = ", ".join(f"{c} {answer.safe_until.get(c, '?')}" for c in selector_input.colors)
                print(f"[{count}/{len(rows)}] {key[0]}/{key[1]} mode {mode}: {answer.choice} {answer.action} "
                      f"({tick_text}) driver={score.choice_matches_driver} valid={answer.valid} "
                      f"{reply.seconds:.1f} s", flush=True)
    finally:
        for f in files.values():
            f.close()

    per_mode = {m: [r for r in read_done(paths[m]).values() if (r["sequence"], r["frame"]) in
                    {(row["sequence"], int(row["frame"])) for row in rows}] for m in args.modes}
    title = (f"Selector: {args.model} ({args.quantize}), {args.split} every {args.every}, "
             f"history {args.history}, {CHOSEN_VARIANT} candidates, picture version {PICTURE_VERSION}")
    report = write_summary(args.out_dir / f"{tag}_summary.md", title, per_mode)
    print("\n" + report)
    if vlm is not None:
        print(f"Peak GPU memory {vlm.peak_memory_gb():.1f} GB; {time.perf_counter() - started:.0f} s in total")
    print(f"Saved {', '.join(p.name for p in paths.values())} and {tag}_summary.md to {args.out_dir}")


if __name__ == "__main__":
    main()
