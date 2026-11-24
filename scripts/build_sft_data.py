"""Build the phase 5 fine-tuning data: pictures, prompts and target answers (phase 5).

For every training frame (sequences 00000-00002) with candidates, it makes about
frame_weight(frame) examples per epoch, each with fresh random colors, listing order, wording,
mode (A or B), mirroring, history kind and photometric change (nav/sft.py), and draws the
picture exactly like the selector does. Validation frames (00003) get one plain example per
mode (evaluation wording, no mirroring or photometric change, the driver's history) for the
validation loss during training.

The pictures are rendered here, so training needs only this folder, not the RELLIS sequences:
build on the laptop, copy the folder to Polaris, train there.

Output (outputs/sft/<name>/): images/{train,val}/*.jpg, train.jsonl, val.jsonl (one example per
line: image, prompt, target, mask_choice, and what kind of example it is), meta.json (settings
and counts; train.jsonl is in epoch order, so the trainer can save a checkpoint per epoch).

Usage (from the repo root; needs the RELLIS sequences 00000-00003 and outputs/candidates/relaxed.jsonl):
    python scripts/build_sft_data.py --no-match skip      # -> outputs/sft/nomatch-skip (~10 min)
    python scripts/build_sft_data.py --no-match keep      # -> outputs/sft/nomatch-keep
    python scripts/build_sft_data.py --no-match keep --limit-frames 6 --epochs 1 --name smoke
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence  # noqa: E402
from offroad.mapping.costmap import CostParams, geometric_cost_map  # noqa: E402
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu  # noqa: E402
from offroad.nav.candidates import CHOSEN_VARIANT  # noqa: E402
from offroad.nav.prompts import HISTORY_LENGTH  # noqa: E402
from offroad.nav.sft import (  # noqa: E402
    NO_MATCH_POLICIES,
    copies_this_epoch,
    draw_settings,
    frame_weight,
    is_uncertain,
    make_example,
    photometric,
)
from offroad.nav.visual_prompt import PICTURE_VERSION, assign_colors, draw_visual_prompt, frame_seed  # noqa: E402

FRAMES_CSV = REPO_ROOT / "configs" / "decision_frames.csv"
JPEG_QUALITY = 95
DEFAULT_EPOCHS = 5  # setting 11 (DriveVLN's); the best epoch is picked afterwards on validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-match", choices=NO_MATCH_POLICIES, required=True,
                        help="frames where no candidate matches the driver: skip them, or keep them (the experiment)")
    parser.add_argument("--name", default=None, help="dataset name (default: nomatch-<policy>)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"default: {DEFAULT_EPOCHS}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=None, help="only the first N frames per split (smoke test)")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--candidates", type=Path,
                        default=REPO_ROOT / "outputs" / "candidates" / f"{CHOSEN_VARIANT}.jsonl")
    parser.add_argument("--frames", type=Path, default=FRAMES_CSV)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "sft")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    return parser.parse_args()


# --- rendering (runs in worker processes) ---

_cache: dict = {}


def render_frame(task: tuple) -> int:
    """Draw all pictures of one frame. task = (sequence, frame, root, candidate xys, [(path, palette, mirror, photo)])."""
    sequence, index, root, xys, renders = task
    if sequence not in _cache:
        _cache.clear()  # one sequence at a time keeps memory small
        _cache[sequence] = RellisSequence(sequence, root=root)
    frame = _cache[sequence][index]
    points_flu = lidar_to_flu(frame.points)
    spec = GridSpec()
    ground_z = geometric_cost_map(points_flu, build_elevation_map(points_flu, spec), CostParams()).ground_z
    for path, palette, mirror, photo in renders:
        image = photometric(frame.image, **photo) if photo else frame.image
        picture, _ = draw_visual_prompt(image, frame.calib, xys, palette, ground_z, spec, mirror=mirror)
        cv2.imwrite(str(path), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return len(renders)


# --- planning the examples ---


def read_frames(args: argparse.Namespace) -> dict[str, list[tuple[dict, float]]]:
    """{split: [(frame record, timestamp)]} in time order per sequence, train and val only."""
    if not args.candidates.exists():
        sys.exit(f"{args.candidates} is missing; run scripts/build_candidates.py first")
    with open(args.frames, newline="") as f:
        times = {(r["sequence"], int(r["frame"])): float(r["timestamp"]) for r in csv.DictReader(f)}
    by_split = defaultdict(list)
    with open(args.candidates) as f:
        for line in f:
            r = json.loads(line)
            if r["split"] in ("train", "val"):
                by_split[r["split"]].append((r, times[(r["sequence"], r["frame"])]))
    for split in by_split:
        by_split[split].sort(key=lambda item: (item[0]["sequence"], item[1]))
    if set(by_split) != {"train", "val"}:
        sys.exit(f"{args.candidates} has no train or val frames; rebuild it with scripts/build_candidates.py")
    return by_split


def previous_decisions(frames: list[tuple[dict, float]]) -> list[list[tuple[float, float]]]:
    """For each frame: (timestamp, driver's candidate end bearing) of up to 3 earlier frames with candidates, same drive."""
    out, recent, sequence = [], [], None
    for record, t in frames:
        if record["sequence"] != sequence:
            sequence, recent = record["sequence"], []
        out.append(list(recent[-HISTORY_LENGTH:]))
        if record["candidates"]:
            recent.append((t, float(record["candidates"][record["closest_to_driver"]]["end_bearing_deg"])))
    return out


def main() -> None:
    args = parse_args()
    name = args.name or f"nomatch-{args.no_match}"
    out = args.out_dir / name
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
    by_split = read_frames(args)
    rng = np.random.default_rng(args.seed)

    lines = {"train": [], "val": []}
    tasks = defaultdict(list)  # (sequence, frame) -> renders
    xys = {}
    skipped = Counter()
    for split, frames in by_split.items():
        history = previous_decisions(frames)
        frames = list(zip(frames, history))
        if args.limit_frames:
            frames = [f for f in frames if f[0][0]["candidates"]][:args.limit_frames]
        for (record, t), previous in frames:
            key = (record["sequence"], record["frame"])
            if not record["candidates"]:
                skipped[f"{split}: no candidates"] += 1
                continue
            if is_uncertain(record):
                skipped[f"{split}: uncertain ground"] += 1
                continue
            n = len(record["candidates"])
            if split == "val":  # plain: like the selector at evaluation, both modes
                palette = assign_colors(n, frame_seed(record["sequence"], record["frame"]))
                plan = [(0, mode, {"mode": mode, "mirror": False, "history_kind": "driver", "wording": 0,
                                   "color_seed": None, "list_seed": frame_seed(record["sequence"], record["frame"], 1),
                                   "wrong_seed": 0, "photometric": None}) for mode in "AB"]
            else:
                plan = []
                for epoch in range(args.epochs):
                    for _ in range(copies_this_epoch(frame_weight(record), rng)):
                        plan.append((epoch, None, draw_settings(rng)))
            for k, (epoch, _, s) in enumerate(plan):
                pal = palette if split == "val" else assign_colors(n, s["color_seed"])
                example = make_example(record, t, previous, pal, s["list_seed"], s["mode"], s["mirror"],
                                       s["history_kind"], s["wording"], args.no_match, s["wrong_seed"])
                if example is None:
                    skipped[f"{split}: no candidate matches the driver"] += 1
                    break  # the policy leaves out the whole frame
                image = Path("images") / split / f"{key[0]}_{key[1]:06d}_e{epoch}_{k}.jpg"
                tasks[key].append((out / image, pal, s["mirror"], s["photometric"]))
                xys[key] = [np.asarray(c["xy"]) for c in record["candidates"]]
                lines[split].append({
                    "image": image.as_posix(), "prompt": example.prompt, "target": example.target,
                    "mask_choice": example.mask_choice, "mode": example.mode, "sequence": key[0], "frame": key[1],
                    "epoch": epoch, "mirror": example.mirror, "history_kind": example.history_kind,
                    "wording": example.wording, "frame_class": example.frame_class,
                })

    shuffled_train = []  # epoch order, so the trainer can save a checkpoint per epoch; shuffled within each epoch
    for epoch in range(args.epochs):
        group = [x for x in lines["train"] if x["epoch"] == epoch]
        shuffled_train += [group[j] for j in rng.permutation(len(group))]
    lines["train"] = shuffled_train

    started = time.perf_counter()
    jobs = [(k[0], k[1], args.root, xys[k], renders) for k, renders in tasks.items()]
    print(f"Drawing {sum(len(j[4]) for j in jobs)} pictures for {len(jobs)} frames with {args.workers} workers...",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, _ in enumerate(pool.map(render_frame, jobs, chunksize=4), 1):
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} frames", flush=True)

    for split in ("train", "val"):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in lines[split])
    per_epoch = Counter(x["epoch"] for x in lines["train"])
    meta = {
        "name": name, "no_match": args.no_match, "epochs": args.epochs, "seed": args.seed,
        "picture_version": PICTURE_VERSION, "reasoning": False, "candidates": str(args.candidates),
        "train_examples": len(lines["train"]), "val_examples": len(lines["val"]),
        "train_examples_per_epoch": [per_epoch[e] for e in range(args.epochs)],
        "skipped_frames": dict(skipped),
        "train_counts": {
            "mode": dict(Counter(x["mode"] for x in lines["train"])),
            "frame_class": dict(Counter(x["frame_class"] for x in lines["train"])),
            "mirror": dict(Counter(str(x["mirror"]) for x in lines["train"])),
            "history_kind": dict(Counter(x["history_kind"] for x in lines["train"])),
            "mask_choice": dict(Counter(str(x["mask_choice"]) for x in lines["train"])),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Saved {out} ({time.perf_counter() - started:.0f} s for the pictures)")


if __name__ == "__main__":
    main()
