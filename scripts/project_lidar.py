"""Project one RELLIS-3D Ouster scan into its camera image and save a depth-colored overlay.

Usage (from the repo root):
    python scripts/project_lidar.py --sequence 0 --frame 0

If the dots line up with trees, bushes, and terrain edges, the calibration is correct.
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence  # noqa: E402
from offroad.geometry.projection import project_points_to_image  # noqa: E402
from offroad.viz.overlay import draw_depth_overlay  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="frame index within the sequence (default: 0)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--max-depth", type=float, default=None, help="depth in meters shown as the farthest color (default: farthest point)")
    parser.add_argument("--radius", type=int, default=2, help="dot radius in pixels (default: 2)")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: outputs/overlay_<seq>_<frame>.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq = RellisSequence(args.sequence, root=args.root)
    frame = seq[args.frame]

    uv, depth, kept = project_points_to_image(
        frame.points, frame.calib.T_cam_from_lidar, frame.calib.K, frame.image.shape[:2]
    )
    overlay = draw_depth_overlay(frame.image, uv, depth, max_depth=args.max_depth, radius=args.radius)

    out_path = args.out or REPO_ROOT / "outputs" / f"overlay_{seq.sequence}_{args.frame:06d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Sequence {seq.sequence}, frame {args.frame}: {len(frame.points)} LiDAR points, "
          f"{kept.sum()} visible in the image, depth {depth.min():.1f}-{depth.max():.1f} m")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
