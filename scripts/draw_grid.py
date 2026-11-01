"""Draw the labelled VLM grid (A1 ... H12) on camera images and save them.

These gridded images are what the VLM will be shown in phase 4.

Usage (from the repo root):
    python scripts/draw_grid.py --sequence 0 --frame 120    # one frame -> outputs/grid_<seq>_<frame>.png
    python scripts/draw_grid.py --eval                      # all frames in configs/eval_frames.csv -> outputs/grid/
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence  # noqa: E402
from offroad.viz.grid_overlay import draw_grid  # noqa: E402
from offroad.vlm.grid import ImageGrid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="frame index within the sequence (default: 0)")
    parser.add_argument("--eval", action="store_true", help="draw all frames in configs/eval_frames.csv instead")
    parser.add_argument("--rows", type=int, default=8, help="grid rows, lettered from the top (default: 8)")
    parser.add_argument("--cols", type=int, default=12, help="grid columns, numbered from the left (default: 12)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    return parser.parse_args()


def save_gridded(seq: RellisSequence, frame_index: int, rows: int, cols: int, out_path: Path):
    """Load one camera image, draw the grid, and save it (no LiDAR needed)."""
    image_bgr = cv2.imread(str(seq.image_path(frame_index)), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    grid = ImageGrid(height=image.shape[0], width=image.shape[1], rows=rows, cols=cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(draw_grid(image, grid), cv2.COLOR_RGB2BGR))
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    if not args.eval:
        seq = RellisSequence(args.sequence, root=args.root)
        out_path = REPO_ROOT / "outputs" / f"grid_{seq.sequence}_{args.frame:06d}.png"
        save_gridded(seq, args.frame, args.rows, args.cols, out_path)
        return

    with open(REPO_ROOT / "configs" / "eval_frames.csv", newline="") as f:
        eval_frames = list(csv.DictReader(f))
    sequences: dict[str, RellisSequence] = {}
    for entry in eval_frames:
        seq_id, frame_index = entry["sequence"], int(entry["frame"])
        if seq_id not in sequences:
            sequences[seq_id] = RellisSequence(seq_id, root=args.root)
        out_path = REPO_ROOT / "outputs" / "grid" / f"{entry['category']}_{seq_id}_{frame_index:06d}.png"
        save_gridded(sequences[seq_id], frame_index, args.rows, args.cols, out_path)


if __name__ == "__main__":
    main()
