"""Score saved VLM answers against the camera label images and draw each answer (phase 4, step 4).

Reads the answers that scripts/run_vlm.py saved in outputs/vlm_runs/<run>/. For each frame
it compares the answer's cells with the RELLIS-3D label image (is a "water" cell really
water? does the route cross puddles?) and draws the answer next to the labels. Writes:
    outputs/vlm_runs/<run>/answers/<category>_<seq>_<frame>.png   picture per answer
    outputs/vlm_runs/<run>/scores.csv                             scores per frame
    outputs/vlm_runs/comparison.csv                               one column per run
and prints the comparison table. No GPU or model needed.

Usage (from the repo root):
    python scripts/score_vlm.py                                    # every run in outputs/vlm_runs
    python scripts/score_vlm.py --run Qwen3-VL-8B-Instruct_4bit    # just this run
    python scripts/score_vlm.py --no-pictures                      # scores only (faster)
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.viz.vlm_answer import draw_answer_figure  # noqa: E402
from offroad.vlm.grid import ImageGrid  # noqa: E402
from offroad.vlm.scoring import CLASS_GROUPS, TERRAIN_CLASSES, cell_fractions, score_answer, summarize  # noqa: E402

RUNS_DIR = REPO_ROOT / "outputs" / "vlm_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", default=None, help="run folder name(s) in outputs/vlm_runs (default: all)")
    parser.add_argument("--no-pictures", action="store_true", help="only compute scores, don't draw the answers")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    return parser.parse_args()


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def main() -> None:
    args = parse_args()
    runs = args.run or sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir() and any(p.glob("*.json")))
    if not runs:
        sys.exit(f"No runs found in {RUNS_DIR}. Run scripts/run_vlm.py first.")
    id_to_name, id_to_color = load_ontology(args.root)

    sequences: dict[str, RellisSequence] = {}
    frames: dict[tuple[str, int], tuple] = {}  # (seq, frame) -> (image, label image, grid, groups, terrains)

    def load(seq_id: str, frame_index: int):
        key = (seq_id, frame_index)
        if key not in frames:
            if seq_id not in sequences:
                sequences[seq_id] = RellisSequence(seq_id, root=args.root)
            seq = sequences[seq_id]
            image = cv2.cvtColor(cv2.imread(str(seq.image_path(frame_index)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            labels = cv2.imread(str(seq.label_image_path(frame_index)), cv2.IMREAD_UNCHANGED) \
                if seq.has_label_image(frame_index) else None
            grid = ImageGrid(height=image.shape[0], width=image.shape[1])
            groups = cell_fractions(labels, grid, id_to_name, CLASS_GROUPS) if labels is not None else None
            terrains = cell_fractions(labels, grid, id_to_name, TERRAIN_CLASSES) if labels is not None else None
            frames[key] = (image, labels, grid, groups, terrains)
        return frames[key]

    summaries, by_category = {}, {}
    for run in runs:
        run_dir = RUNS_DIR / run
        records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("*.json"))]
        records = [r for r in records if r["category"] != "single"]  # one-off tries; the eval set covers them
        print(f"\n{run}: {len(records)} answers", flush=True)
        (run_dir / "answers").mkdir(exist_ok=True)

        scores, rows = [], []
        for record in records:
            seq_id, frame_index = record["sequence"], int(record["frame"])
            image, labels, grid, groups, terrains = load(seq_id, frame_index)
            stem = f"{record['category']}_{seq_id}_{frame_index:06d}"
            if not args.no_pictures:
                title = f"{run}  |  {record['category']} {seq_id}/{frame_index}  |  goal {record['goal_cell'] or '-'}"
                figure = draw_answer_figure(image, labels, grid, record["answer"], record["goal_cell"], id_to_color, title)
                cv2.imwrite(str(run_dir / "answers" / f"{stem}.png"), cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
            if groups is None:
                print(f"  {stem}: no label image, not scored")
                continue
            score = score_answer(record["answer"], record["goal_cell"], grid, groups, terrains)
            scores.append((record["category"], score))
            rows.append({"category": record["category"], "sequence": seq_id, "frame": frame_index, **asdict(score)})

        with open(run_dir / "scores.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries[run] = summarize([s for _, s in scores])
        for category in dict.fromkeys(c for c, _ in scores):
            group = [s for c, s in scores if c == category]
            by_category[(run, category)] = (summarize(group)["route water share"][0],
                                            summarize(group)["straight-ahead water share"][0])

    # Comparison table: one row per metric, one column per run.
    metrics = list(next(iter(summaries.values())))
    width = max(len(run) for run in runs) + 2
    print(f"\n{'metric':32s}" + "".join(f"{run:>{width}s}" for run in runs))
    for metric in metrics:
        cells = [f"{fmt(summaries[run][metric][0])} ({summaries[run][metric][1]})" for run in runs]
        print(f"{metric:32s}" + "".join(f"{cell:>{width}s}" for cell in cells))

    print("\nRoute water share vs. driving straight ahead, per category (lower is better):")
    categories = list(dict.fromkeys(c for _, c in by_category))
    print(f"{'category':10s}{'straight':>10s}" + "".join(f"{run:>{width}s}" for run in runs))
    for category in categories:
        straight = by_category.get((runs[0], category), (None, None))[1]
        print(f"{category:10s}{fmt(straight):>10s}"
              + "".join(f"{fmt(by_category.get((run, category), (None, None))[0]):>{width}s}" for run in runs))

    with open(RUNS_DIR / "comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *runs])
        for metric in metrics:
            writer.writerow([metric, *[fmt(summaries[run][metric][0]) for run in runs]])
    print(f"\nSaved pictures to outputs/vlm_runs/<run>/answers/, per-frame scores.csv, and {RUNS_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    main()
