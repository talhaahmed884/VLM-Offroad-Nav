"""Propose evaluation-frame candidates per category and draw contact sheets for review.

Reads outputs/label_stats.csv (from compute_label_stats.py), scores every frame for each
category by how much of the "ahead" region is that category's classes, and picks the best
frames that are at least --min-gap frames apart. The final choice is made by a person
looking at the contact sheets.

Usage (from the repo root):
    python scripts/select_eval_frames.py
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.eval.curation import AHEAD_COLS, AHEAD_ROWS, Candidate, pick_spaced  # noqa: E402

# Everything that is not open, drivable ground. Used to find "easy" control frames.
HAZARD_CLASSES = [
    "mud", "puddle", "water", "bush", "rubble", "tree", "log", "pole",
    "fence", "barrier", "vehicle", "person", "object", "building",
]


@dataclass(frozen=True)
class Category:
    """A kind of evaluation frame.

    classes: ontology class names whose share of the ahead region is the score.
    min_score: frames scoring below this are not candidates.
    max_hazard: if set, frames whose ahead region has more than this share of
        HAZARD_CLASSES are not candidates (used for the easy control set).
    """

    name: str
    classes: tuple[str, ...]
    min_score: float
    max_hazard: float | None = None


CATEGORIES = [
    Category("mud", ("mud",), min_score=0.03),
    Category("water", ("puddle", "water"), min_score=0.03),
    Category("bush", ("bush",), min_score=0.10),
    Category("rubble", ("rubble",), min_score=0.03),
    Category("easy", ("grass", "dirt", "asphalt", "concrete"), min_score=0.80, max_hazard=0.01),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, default=REPO_ROOT / "outputs" / "label_stats.csv")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--per-category", type=int, default=12, help="candidates to propose per category (default: 12)")
    parser.add_argument("--min-gap", type=int, default=100, help="minimum frames between picks in a sequence (default: 100, about 10 s)")
    parser.add_argument("--max-per-sequence", type=int, default=4, help="most candidates per category from one sequence (default: 4)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "eval_candidates")
    return parser.parse_args()


def read_stats(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def score_frames(rows: list[dict], category: Category) -> list[Candidate]:
    """All frames that qualify for a category, scored by the ahead-region share of its classes."""
    candidates = []
    for row in rows:
        score = sum(float(row[f"ahead_{name}"]) for name in category.classes)
        if score < category.min_score:
            continue
        if category.max_hazard is not None:
            hazard = sum(float(row[f"ahead_{name}"]) for name in HAZARD_CLASSES)
            if hazard > category.max_hazard:
                continue
        candidates.append(Candidate(row["sequence"], int(row["frame"]), score))
    return candidates


def draw_thumbnail(seq: RellisSequence, frame: int, class_ids: list[int], caption: str, width: int = 480) -> np.ndarray:
    """BGR thumbnail (OpenCV order, ready for imwrite) with the category's pixels tinted
    magenta and the ahead region outlined in white."""
    image = cv2.imread(str(seq.image_path(frame)), cv2.IMREAD_COLOR)
    label = cv2.imread(str(seq.label_image_path(frame)), cv2.IMREAD_UNCHANGED)
    highlight = np.isin(label, class_ids)
    image[highlight] = (0.5 * image[highlight] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)

    h, w = label.shape
    top_left = (round(AHEAD_COLS[0] * w), round(AHEAD_ROWS[0] * h))
    bottom_right = (round(AHEAD_COLS[1] * w) - 1, round(AHEAD_ROWS[1] * h) - 1)
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), thickness=4)

    thumb = cv2.resize(image, (width, round(width * h / w)), interpolation=cv2.INTER_AREA)
    cv2.rectangle(thumb, (0, 0), (width, 28), (0, 0, 0), thickness=-1)
    cv2.putText(thumb, caption, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return thumb


def contact_sheet(thumbs: list[np.ndarray], columns: int = 4) -> np.ndarray:
    """Tile equally sized thumbnails into a grid, padding the last row with black."""
    h, w = thumbs[0].shape[:2]
    rows = -(-len(thumbs) // columns)
    sheet = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for k, thumb in enumerate(thumbs):
        r, c = divmod(k, columns)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = thumb
    return sheet


def main() -> None:
    args = parse_args()
    rows = read_stats(args.stats)
    id_to_name, _ = load_ontology(args.root)
    name_to_id = {name: class_id for class_id, name in id_to_name.items()}

    # Pick the rarest category first, so common ones cannot crowd out its few good frames.
    scored = {cat.name: score_frames(rows, cat) for cat in CATEGORIES}
    order = sorted(CATEGORIES, key=lambda cat: len(scored[cat.name]))

    picks: dict[str, list[Candidate]] = {}
    all_picked: list[Candidate] = []
    for cat in order:
        picks[cat.name] = pick_spaced(
            scored[cat.name], args.per_category, args.min_gap, all_picked, args.max_per_sequence
        )
        all_picked += picks[cat.name]
        print(f"{cat.name:7s}: {len(scored[cat.name]):5d} qualifying frames -> {len(picks[cat.name])} candidates")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sequences = {s: RellisSequence(s, root=args.root) for s in sorted({c.sequence for c in all_picked})}
    with open(args.out_dir / "candidates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "rank", "sequence", "frame", "image_name", "score"])
        for cat in CATEGORIES:
            class_ids = [name_to_id[name] for name in cat.classes]
            thumbs = []
            for rank, cand in enumerate(picks[cat.name], start=1):
                seq = sequences[cand.sequence]
                writer.writerow([cat.name, rank, cand.sequence, cand.frame, seq.image_path(cand.frame).stem, f"{cand.score:.3f}"])
                caption = f"#{rank}  {cand.sequence}/{cand.frame}  {cat.name} {cand.score:.0%}"
                thumbs.append(draw_thumbnail(seq, cand.frame, class_ids, caption))
            if thumbs:
                cv2.imwrite(str(args.out_dir / f"{cat.name}.jpg"), contact_sheet(thumbs))
    print(f"Saved candidates.csv and contact sheets to {args.out_dir}")


if __name__ == "__main__":
    main()
