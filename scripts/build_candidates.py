"""Build the candidate paths and their labels for every decision frame, and compare strict vs relaxed (phase 2).

For each frame in configs/decision_frames.csv it builds the cost map, generates up to 4
candidates on the strict and on the relaxed cost grid, and labels each candidate (matches the
driver? how far is it safe?) from the two-source ground truth. The driven path is labelled the
same way, as a check on the labels. It writes one JSON line per frame per variant to
outputs/candidates/<variant>.jsonl and prints (and saves to outputs/candidates/comparison.md)
a comparison table per split.

What to look for: the better variant gives more frames a candidate close to the driven path
("driver match"), more frames with a real choice (2+ candidates safe to the first tick), and
rarely calls the driver's own path unsafe. Puddles are "caution", not unsafe: the table shows
how often the candidates and the driver go through one.

Usage (from the repo root):
    python scripts/build_candidates.py                       # all decision frames (several minutes)
    python scripts/build_candidates.py --every 10            # every 10th frame (quick look)
    python scripts/build_candidates.py --split val --workers 6
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.nav.candidates import VARIANTS  # noqa: E402
from offroad.nav.frame_candidates import candidates_for_frame  # noqa: E402
from offroad.nav.labels import SAFE_ANSWERS  # noqa: E402

FRAMES_CSV = REPO_ROOT / "configs" / "decision_frames.csv"
SPLIT_ORDER = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLIT_ORDER, default=None, help="only this split (default: all)")
    parser.add_argument("--every", type=int, default=1, help="use every Nth decision frame (default: 1 = all)")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="parallel processes (default: 4); each needs about 1 GB of RAM")
    parser.add_argument("--frames", type=Path, default=FRAMES_CSV, help="decision frames CSV (default: configs/decision_frames.csv)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "candidates")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    return parser.parse_args()


def read_decision_frames(path: Path, split: str | None, every: int) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} is missing; run scripts/build_decision_frames.py first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if "labelled" not in rows[0]:
        sys.exit(f"{path} is from an older version (no 'labelled' column); rerun scripts/build_decision_frames.py")
    rows = [r for r in rows if split is None or r["split"] == split]
    return rows[::every]


# --- one frame (runs in a worker process) ---

_cache: dict = {}


def _sequence(sequence: str, root: Path | None) -> tuple[RellisSequence, np.ndarray]:
    if sequence not in _cache:
        seq = RellisSequence(sequence, root=root)
        _cache[sequence] = (seq, seq.poses)
    return _cache[sequence]


def _round(xy: np.ndarray) -> list[list[float]]:
    return np.round(xy, 2).tolist()


def _labels_dict(lab, id_to_name: dict[int, str]) -> dict:
    return {
        "driven_mean_m": round(lab.driven_mean_m, 3), "matches_driver": lab.matches_driver,
        "first_unsafe_m": None if lab.first_unsafe_m is None else round(lab.first_unsafe_m, 2),
        "first_unsafe_class": None if lab.first_unsafe_class is None else id_to_name.get(lab.first_unsafe_class),
        "first_caution_m": None if lab.first_caution_m is None else round(lab.first_caution_m, 2),
        "safe_m": round(lab.safe_m, 2), "safe_until": lab.safe_until, "unknown_share": round(lab.unknown_share, 3),
    }


def process_frame(task: tuple[dict, Path | None]) -> dict[str, dict]:
    """{variant: JSON record} for one decision frame."""
    row, root = task
    if "ontology" not in _cache:
        _cache["ontology"] = load_ontology(root)[0]
    id_to_name = _cache["ontology"]
    seq, poses = _sequence(row["sequence"], root)
    index = int(row["frame"])
    frame = seq[index]
    result = candidates_for_frame(frame.points, frame.point_labels, frame.label_image, frame.calib.T_cam_from_lidar,
                                  frame.calib.K, poses, index, id_to_name, image_width=frame.image.shape[1])
    common = {
        "split": row["split"], "sequence": row["sequence"], "frame": index, "direction": row["direction"] or None,
        "has_label_image": frame.label_image is not None,
        "driven": {"xy": _round(result.driven_xy), **_labels_dict(result.driven, id_to_name)},
    }
    records = {}
    for variant, fc in result.variants.items():
        records[variant] = {
            **common, "variant": variant, "n_candidates": len(fc.candidates), "stop": fc.stop,
            "real_choice": fc.real_choice, "any_matches_driver": fc.any_matches_driver,
            "best_driven_m": None if fc.best_driven_m is None else round(fc.best_driven_m, 3),
            "closest_to_driver": fc.closest_index,
            "candidates": [{
                "xy": _round(c.xy), "target_bearing_deg": c.target_bearing_deg,
                "end_bearing_deg": round(c.end_bearing_deg, 1), "length_m": round(c.length_m, 2),
                "cost_per_m": round(c.cost_per_m, 3), **_labels_dict(lab, id_to_name),
            } for c, lab in zip(fc.candidates, fc.labels)],
        }
    return records


# --- the comparison table ---


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def summarize(records: list[dict]) -> dict[str, str]:
    """One table column (metric name -> text) for a group of frame records of one variant."""
    n = len(records)
    counts = Counter(r["n_candidates"] for r in records)
    best = np.array([r["best_driven_m"] for r in records if r["best_driven_m"] is not None])
    candidates = [c for r in records for c in r["candidates"]]
    safe = Counter(c["safe_until"] for c in candidates)
    matched = [c for r in records for c in r["candidates"] if c["matches_driver"]]
    unsafe_classes = Counter(c["first_unsafe_class"] for c in candidates if c["first_unsafe_class"])
    # Frames where the driver's (closest) candidate crosses a puddle, and of those, frames with a dry alternative.
    driver_puddle = [r for r in records if r["candidates"]
                     and r["candidates"][r["closest_to_driver"]]["first_caution_m"] is not None]
    dry_alternative = [r for r in driver_puddle if any(c["first_caution_m"] is None and c["safe_until"] != "none"
                                                       for c in r["candidates"])]
    return {
        "frames": str(n),
        "candidates per frame (mean)": f"{np.mean([r['n_candidates'] for r in records]):.2f}" if n else "-",
        "frames with 0 / 1 / 2 / 3 / 4": " / ".join(percent(counts[k], n) for k in range(5)),
        "best candidate to driven path, median (m)": f"{np.median(best):.2f}" if len(best) else "-",
        "best candidate to driven path, 90th pct (m)": f"{np.percentile(best, 90):.2f}" if len(best) else "-",
        "driver match (a candidate within 1 m)": percent(sum(r["any_matches_driver"] for r in records), n),
        "matching candidates per frame (mean)": f"{len(matched) / n:.2f}" if n else "-",
        "real choice (2+ safe to tick 1)": percent(sum(r["real_choice"] for r in records), n),
        "stop (none safe to tick 1)": percent(sum(r["stop"] for r in records), n),
        "candidates safe: none / tick 1 / tick 2 / all": " / ".join(percent(safe[a], len(candidates)) for a in SAFE_ANSWERS),
        "driver-matching candidates safe 'all'": percent(sum(c["safe_until"] == "all" for c in matched), len(matched)),
        "full length (>= 14 m)": percent(sum(c["length_m"] >= 14.0 for c in candidates), len(candidates)),
        "unknown ground share (mean)": f"{np.mean([c['unknown_share'] for c in candidates]):.2f}" if candidates else "-",
        "first unsafe class (top 3)": ", ".join(f"{k} {v}" for k, v in unsafe_classes.most_common(3)) or "-",
        "candidates through a puddle (caution)": percent(sum(c["first_caution_m"] is not None for c in candidates),
                                                         len(candidates)),
        "driver's candidate through a puddle": percent(len(driver_puddle), n),
        "... of those, with a dry alternative": percent(len(dry_alternative), len(driver_puddle)),
    }


def driven_summary(records: list[dict]) -> dict[str, str]:
    """The labels' verdict on the driver's own path (same for both variants)."""
    n = len(records)
    safe = Counter(r["driven"]["safe_until"] for r in records)
    classes = Counter(r["driven"]["first_unsafe_class"] for r in records if r["driven"]["first_unsafe_class"])
    return {
        "frames": str(n),
        "with a camera label image": percent(sum(r["has_label_image"] for r in records), n),
        "driven path safe: none / tick 1 / tick 2 / all": " / ".join(percent(safe[a], n) for a in SAFE_ANSWERS),
        "first unsafe class on the driven path (top 3)": ", ".join(f"{k} {v}" for k, v in classes.most_common(3)) or "-",
        "driven path through a puddle (caution)": percent(sum(r["driven"]["first_caution_m"] is not None
                                                              for r in records), n),
    }


def markdown_table(title: str, columns: dict[str, dict[str, str]]) -> str:
    names = list(columns)
    metrics = list(next(iter(columns.values())))
    lines = [f"### {title}", "", "| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    lines += [f"| {m} | " + " | ".join(columns[c][m] for c in names) + " |" for m in metrics]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = read_decision_frames(args.frames, args.split, args.every)
    print(f"{len(rows)} decision frames, {args.workers} worker(s)", flush=True)
    records: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    started = time.perf_counter()
    tasks = [(row, args.root) for row in rows]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = pool.map(process_frame, tasks, chunksize=8)
            for k, result in enumerate(results, 1):
                for variant, record in result.items():
                    records[variant].append(record)
                if k % 100 == 0 or k == len(tasks):
                    print(f"  {k}/{len(tasks)} frames, {time.perf_counter() - started:.0f} s", flush=True)
    else:
        for k, task in enumerate(tasks, 1):
            for variant, record in process_frame(task).items():
                records[variant].append(record)
            if k % 100 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} frames, {time.perf_counter() - started:.0f} s", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant, group in records.items():
        with open(args.out_dir / f"{variant}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in group)

    splits = [s for s in SPLIT_ORDER if any(r["split"] == s for r in records[VARIANTS[0]])]
    sections = [f"## Candidate comparison: strict vs relaxed ({len(rows)} decision frames, every {args.every})"]
    for split in splits + ["all"]:
        columns = {v: summarize([r for r in records[v] if split == "all" or r["split"] == split]) for v in VARIANTS}
        sections.append(markdown_table(f"{split}", columns))
    sections.append(markdown_table("the driver's own path, labelled like a candidate", {
        split: driven_summary([r for r in records[VARIANTS[0]] if split == "all" or r["split"] == split])
        for split in splits + ["all"]}))
    report = "\n\n".join(sections) + "\n"
    (args.out_dir / "comparison.md").write_text(report)
    print("\n" + report)
    print(f"Saved {', '.join(f'{v}.jsonl' for v in VARIANTS)} and comparison.md to {args.out_dir} "
          f"({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
