"""Evaluate selector runs against simple baselines on the same frames (phase 4).

Reads the candidates (outputs/candidates/relaxed.jsonl, with labels and terrain numbers) and any
number of selector runs (<tag>_mode{A,B}.jsonl, found anywhere under outputs/selector, which is
sorted into one subfolder per collection; see outputs/selector/README.md), and writes one report:
- path choice, per mode: baselines (random, straightest, cheapest, "toward the word" in Mode B,
  and the driver as the oracle), the driven path itself as a reference (e.g. how often the
  driver passed close to an obstacle), and each run: matches the driver (all frames, frames with
  a real choice, straight and turn frames, with 95% intervals), the side of wrong choices and
  the chosen position, safety, puddles, terrain on the chosen path, closest hard obstacle, and
  the distance from the driven path (at 5 / 10 / 15 m; at 15 m with interval, median and 90th
  percentile; the mean over the three, ADE);
- safe-distance ticks: "always all" next to each run, with how often each true answer is given,
  the mean tick error, how often an unsafe stretch is called safe, and the ticks as hazard flags
  ("not safe to the end": precision / recall / F1);
- continue / stop: "always continue" next to each run (stop precision / recall / F1);
- paired comparisons: every run against the straightest and the cheapest candidate, plus any
  --paired pairs, frame by frame (where only one is right, exact McNemar p-value);
- speed: seconds per answer of each run (mean / median / 95th percentile);
- optional behaviour tests: --direction-test (two Mode B runs with forced words, e.g. "bear left"
  and "bear right": does the choice move with the word?) and --history-test (a run with its own
  history and one with --history wrong: does a made-up turn in the history pull the choice?).

All columns use the same frames: those with candidates that every given run answered.

Usage (from the repo root):
    python scripts/evaluate.py --split val                                  # baselines only
    python scripts/evaluate.py --split val --runs val_4B_v2 val_8B_v2
    python scripts/evaluate.py --split test --runs test_Qwen3-VL-4B-Instruct test_Qwen3-VL-4B-Instruct_nohist \\
        --direction-test test_Qwen3-VL-4B-Instruct_dir-bear-left test_Qwen3-VL-4B-Instruct_dir-bear-right \\
        --history-test test_Qwen3-VL-4B-Instruct test_Qwen3-VL-4B-Instruct_wronghist
    python scripts/evaluate.py --split test --out outputs/eval/test_finetuned.md \\
        --runs test_ft-nomatch-skip-epoch4 test_Qwen3-VL-4B-Instruct_noreason \\
        --paired test_ft-nomatch-skip-epoch4 test_Qwen3-VL-4B-Instruct_noreason
"""

import argparse
import json
import sys

import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.nav.candidates import CHOSEN_VARIANT  # noqa: E402
from offroad.nav.evaluation import (  # noqa: E402
    BASELINES,
    WORD_BEARING_DEG,
    answer_weights,
    direction_test,
    driven_as_frame,
    history_test,
    one_hot,
    paired_comparison,
    summarize_actions,
    summarize_choices,
    summarize_speed,
    summarize_ticks,
)
from offroad.nav.run_files import answer_file_name, find_run_dir  # noqa: E402

# Baselines every run is compared with frame by frame (the strongest simple rules).
AUTO_PAIRED = ("straightest", "cheapest")
LEGEND = ("[a, b] = 95% confidence interval (Wilson for shares, normal approximation for means): with "
          "this many frames the true value is probably in that range. ADE = mean distance from the "
          "driven path over 5 / 10 / 15 m. For the random baseline the distances are averages over "
          "all candidates of a frame.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="default: test")
    parser.add_argument("--runs", nargs="*", default=[], help="selector run tags to evaluate")
    parser.add_argument("--direction-test", nargs=2, metavar=("TAG1", "TAG2"), default=None,
                        help="two Mode B runs made with --direction (e.g. bear left, bear right)")
    parser.add_argument("--history-test", nargs=2, metavar=("OWN", "WRONG"), default=None,
                        help="a run with its own history and the same with --history wrong")
    parser.add_argument("--paired", nargs=2, action="append", metavar=("FIRST", "SECOND"), default=[],
                        help="compare two choosers frame by frame (run tags or baseline names, e.g. "
                             "'straightest'); can be given several times. Every run is compared with "
                             f"{', '.join(AUTO_PAIRED)} anyway")
    parser.add_argument("--candidates", type=Path,
                        default=REPO_ROOT / "outputs" / "candidates" / f"{CHOSEN_VARIANT}.jsonl")
    parser.add_argument("--selector-dir", type=Path, default=REPO_ROOT / "outputs" / "selector")
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/eval/<split>_evaluation.md")
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[tuple[str, int], dict]:
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {(r["sequence"], r["frame"]): r for r in records}


def read_run(selector_dir: Path, tag: str) -> dict[str, dict[tuple[str, int], dict]]:
    """{mode: {frame key: answer record}} for the modes the run has (looked up in all subfolders)."""
    try:
        folder = find_run_dir(selector_dir, tag)
    except ValueError as err:
        sys.exit(str(err))
    if folder is None:
        sys.exit(f"No {tag}_modeA.jsonl or {tag}_modeB.jsonl in {selector_dir} or its subfolders")
    return {m: read_jsonl(folder / answer_file_name(tag, m)) for m in "AB"
            if (folder / answer_file_name(tag, m)).exists()}


def table(columns: dict[str, dict[str, str]]) -> list[str]:
    """Markdown table with metrics as rows and the given columns."""
    names = list(columns)
    metrics = list(dict.fromkeys(m for col in columns.values() for m in col))
    lines = ["| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    lines += [f"| {m} | " + " | ".join(columns[n].get(m, "") for n in names) + " |" for m in metrics]
    return lines


def main() -> None:
    args = parse_args()
    if not args.candidates.exists():
        sys.exit(f"{args.candidates} is missing; run scripts/build_candidates.py first")
    all_frames = {k: f for k, f in read_jsonl(args.candidates).items() if f["split"] == args.split}
    if not all_frames:
        sys.exit(f"No {args.split} frames in {args.candidates}")
    if any("metrics" not in c for f in all_frames.values() for c in f["candidates"]):
        sys.exit(f"{args.candidates} has no terrain numbers (older version); rerun scripts/build_candidates.py")

    baseline_names = {name for mode in "AB" for name in BASELINES[mode]}
    paired_tags = [name for pair in args.paired for name in pair if name not in baseline_names]
    tags = list(dict.fromkeys(args.runs + list(args.direction_test or []) + list(args.history_test or [])
                              + paired_tags))
    runs = {tag: read_run(args.selector_dir, tag) for tag in tags}
    keys = sorted(k for k, f in all_frames.items() if f["candidates"])
    no_candidates = sum(1 for f in all_frames.values() if not f["candidates"])
    for run in runs.values():  # only frames every run answered
        for answers in run.values():
            keys = [k for k in keys if k in answers]
    if not keys:
        sys.exit("No frames in common between the candidates and the runs (same split? same --every?)")
    frames = [all_frames[k] for k in keys]

    title = f"Evaluation: {args.split}, {len(frames)} frames with candidates"
    lines = [f"## {title}", "",
             f"Frames without any candidate (the vehicle stops without asking): {no_candidates}. "
             "Baselines and runs are scored on the same frames.", "", LEGEND, ""]

    def choices(name: str, mode: str) -> list[np.ndarray] | None:
        """One weight vector per frame for a baseline or a run in this mode (None if it has no such mode)."""
        if name in BASELINES[mode]:
            return [BASELINES[mode][name](f, f["direction"]) for f in frames]
        if name in runs and mode in runs[name]:
            return [answer_weights(f, runs[name][mode][k]) for f, k in zip(frames, keys)]
        return None

    for mode in "AB":
        columns = {name: summarize_choices(frames, [chooser(f, f["direction"]) for f in frames])
                   for name, chooser in BASELINES[mode].items()}
        columns["driven path (reference)"] = summarize_choices([driven_as_frame(f) for f in frames],
                                                               [one_hot(1, 0)] * len(frames))
        for tag in args.runs:
            if mode in runs[tag]:
                answers = runs[tag][mode]
                columns[tag] = summarize_choices(frames, [answer_weights(f, answers[k]) for f, k in zip(frames, keys)])
        what = "keep going" if mode == "A" else "keep going + the driver's direction word"
        lines += [f"### Path choice, Mode {mode} ({what})", ""] + table(columns) + [""]

    tick_columns = {'always "all"': summarize_ticks(frames, [{k: "all" for k in range(len(f["candidates"]))}
                                                            for f in frames])}
    action_columns = {"always continue": summarize_actions(frames, ["continue"] * len(frames))}
    for tag in args.runs:
        for mode, answers in runs[tag].items():
            said = []
            for f, k in zip(frames, keys):
                a = answers[k]
                colors = a.get("colors", [])
                said.append({colors.index(c): t for c, t in a.get("safe_until", {}).items() if c in colors})
            tick_columns[f"{tag} {mode}"] = summarize_ticks(frames, said)
            action_columns[f"{tag} {mode}"] = summarize_actions(frames, [answers[k].get("action") for k in keys])
    lines += ["### Safe distance (ticks)", ""] + table(tick_columns) + [""]
    lines += ["### Continue / stop", ""] + table(action_columns) + [""]

    pairs = [(tag, base) for tag in args.runs for base in AUTO_PAIRED] + [tuple(p) for p in args.paired]
    for mode in "AB":
        columns = {}
        for first, second in dict.fromkeys(pairs):
            a, b = choices(first, mode), choices(second, mode)
            if a is None or b is None:
                continue
            if first == "random" or second == "random":
                sys.exit("--paired needs one-hot choosers; the random baseline is an average, not one choice")
            columns[f"{first} vs {second}"] = paired_comparison(frames, a, b)
        if columns:
            lines += [f"### Paired comparison, Mode {mode} (driver match on the same frames)", "",
                      "Only the frames where the two disagree decide; a p-value below 0.05 means the "
                      "difference is unlikely to be luck.", ""] + table(columns) + [""]

    if args.runs:
        speed = {f"{tag} {mode}": summarize_speed([answers[k] for k in keys])
                 for tag in args.runs for mode, answers in runs[tag].items()}
        lines += ["### Speed (as measured on the machine that ran the selector)", ""] + table(speed) + [""]

    if args.direction_test:
        first, second = (runs[t].get("B") for t in args.direction_test)
        if first is None or second is None:
            sys.exit("--direction-test needs two runs with Mode B answers")
        words = [next(iter(r.values()))["direction"] for r in (first, second)]
        if WORD_BEARING_DEG[words[0]] < WORD_BEARING_DEG[words[1]]:  # put the more-left word first
            first, second, words = second, first, words[::-1]
        result = direction_test(frames, [first[k] for k in keys], [second[k] for k in keys], *words)
        lines += [f'### Does the choice follow the word? ("{words[0]}" vs "{words[1]}", Mode B)', ""]
        lines += table({"": result}) + [""]

    if args.history_test:
        own_tag, wrong_tag = args.history_test
        for mode in "AB":
            if mode in runs[own_tag] and mode in runs[wrong_tag]:
                result = history_test(frames, [runs[own_tag][mode][k] for k in keys],
                                      [runs[wrong_tag][mode][k] for k in keys])
                lines += [f"### Does a wrong history pull the choice? (Mode {mode}: {own_tag} vs {wrong_tag})", ""]
                lines += table({"": result}) + [""]

    report = "\n".join(lines)
    print(report)
    out = args.out or REPO_ROOT / "outputs" / "eval" / f"{args.split}_evaluation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
