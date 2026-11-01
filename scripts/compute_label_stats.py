"""Measure how much of each annotated frame is each class, and save it to a CSV.

For every frame with a label image, writes one row with the fraction of pixels per
ontology class, over the whole image ("whole_<class>") and over the "ahead" region
in front of the vehicle ("ahead_<class>"). Takes a few minutes; run it once.

Usage (from the repo root):
    python scripts/compute_label_stats.py
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.eval.curation import ahead_region_mask, class_pixel_fractions  # noqa: E402

SEQUENCES = ["00000", "00001", "00002", "00003", "00004"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "label_stats.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    id_to_name, _ = load_ontology(args.root)
    class_ids = sorted(id_to_name)
    unlisted = np.ones(256, dtype=bool)
    unlisted[class_ids] = False

    header = ["sequence", "frame", "image_name"]
    for region in ("whole", "ahead"):
        header += [f"{region}_{id_to_name[i]}" for i in class_ids] + [f"{region}_unlisted"]

    unlisted_pixels_seen = np.zeros(256, dtype=np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for seq_id in SEQUENCES:
            seq = RellisSequence(seq_id, root=args.root)
            labelled = [i for i in seq.frame_indices if seq.has_label_image(i)]
            mask = None
            for i in labelled:
                label = cv2.imread(str(seq.label_image_path(i)), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    mask = ahead_region_mask(label.shape)
                row = [seq.sequence, i, seq.label_image_path(i).stem]
                for region_mask in (None, mask):
                    fractions = class_pixel_fractions(label, region_mask)
                    row += [f"{fractions[c]:.5f}" for c in class_ids] + [f"{fractions[unlisted].sum():.5f}"]
                writer.writerow(row)
                unlisted_pixels_seen += np.bincount(label.ravel(), minlength=256) * unlisted
            print(f"{seq.sequence}: {len(labelled)} labelled frames")

    seen = np.flatnonzero(unlisted_pixels_seen)
    if seen.size:
        print("Class IDs found in label images but not in ontology.yaml:", seen.tolist())
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
