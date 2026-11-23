"""Choose the decision frames for approach 2 and write configs/decision_frames.csv (phase 1).

For every sequence it reads the scan poses and the image timestamps (no scans or images, so it
is quick), samples about 2 frames per second (moving each to a neighbour with a camera label
image when it has none), skips frames where the vehicle stands still or the recording ends
within 20 m, and labels each kept frame with its split (by sequence) and Mode B's direction
word (from where the driver was ~50 m later). It prints a summary per sequence and per split.

Usage (from the repo root):
    python scripts/build_decision_frames.py
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence  # noqa: E402
from offroad.nav.driven_path import DIRECTION_WORDS  # noqa: E402
from offroad.nav.frames import DECISION_RATE_HZ, SPLITS, build_decision_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rate", type=float, default=DECISION_RATE_HZ, help="decision frames per second (default: 2)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "configs" / "decision_frames.csv")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    print(f"{'sequence':9s} {'split':6s} {'frames':>7s} {'minutes':>8s} {'driven m':>9s} {'sampled':>8s} "
          f"{'stopped':>8s} {'near end':>9s} {'kept':>6s} {'labelled':>9s}")
    for sequence in SPLITS:
        seq = RellisSequence(sequence, root=args.root)
        times = np.array([seq.timestamp(i) for i in seq.frame_indices])
        if list(seq.frame_indices) != list(range(len(seq))):
            sys.exit(f"{sequence}: frame indices are not 0..N-1, so poses[i] can't be matched to frame i")
        poses = seq.poses
        labelled = np.array([seq.has_label_image(i) for i in seq.frame_indices])
        frames, counts = build_decision_frames(sequence, poses, times, rate_hz=args.rate, labelled=labelled)
        driven = np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1).sum()
        print(f"{sequence:9s} {SPLITS[sequence]:6s} {len(seq):7d} {(times[-1] - times[0]) / 60:8.1f} {driven:9.0f} "
              f"{counts['sampled']:8d} {counts['stopped']:8d} {counts['near_end']:9d} {len(frames):6d} "
              f"{sum(f.labelled for f in frames):9d}")
        rows += [{
            "split": f.split, "sequence": f.sequence, "frame": f.frame, "timestamp": f"{f.timestamp:.3f}",
            "path_ahead_m": f"{f.path_ahead_m:.1f}", "speed_mps": f"{f.speed_mps:.2f}",
            "bearing_deg": "" if f.bearing_deg is None else f"{f.bearing_deg:.1f}", "direction": f.direction or "",
            "labelled": "yes" if f.labelled else "no",
        } for f in frames]

    print(f"\n{'split':6s} {'frames':>7s}  " + "  ".join(f"{w:>12s}" for w in DIRECTION_WORDS))
    for split in ("train", "val", "test"):
        group = [r for r in rows if r["split"] == split]
        words = Counter(r["direction"] for r in group)
        print(f"{split:6s} {len(group):7d}  " + "  ".join(f"{words[w]:12d}" for w in DIRECTION_WORDS))
    print("(direction = Mode B's word from the driven path ~50 m ahead; it's straight on most trails)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} decision frames to {args.out}")


if __name__ == "__main__":
    main()
