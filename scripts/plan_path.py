"""Plan a geometry-only A* path for one RELLIS-3D frame and save a picture of it.

Builds the elevation map and geometric cost map, frees the vehicle's own footprint,
and plans from the vehicle toward a target straight ahead. If the target is blocked
or cut off, the path goes to the closest cell it can reach and the shortfall is
reported. The picture shows the path on the camera image, on the cost map, and on
the ground-truth classes.

Usage (from the repo root):
    python scripts/plan_path.py --sequence 0 --frame 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file only; also works without a display (e.g. on the cluster)
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.planning.baseline import BaselineResult, run_geometric_baseline  # noqa: E402
from offroad.viz.baseline_figure import draw_baseline_figure  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="frame index within the sequence (default: 0)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--goal-distance", type=float, default=20.0, help="target distance straight ahead, meters (default: 20)")
    parser.add_argument("--footprint-radius", type=float, default=2.0,
                        help="cells this close to the vehicle are treated as drivable, meters (default: 2)")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: outputs/path_<seq>_<frame>.png)")
    return parser.parse_args()


def describe(result: BaselineResult, id_to_name: dict[int, str]) -> str:
    """One or two lines summarizing a baseline result."""
    if not result.found:
        return "NO PATH: the vehicle cannot move (its start cell is blocked)"
    gx, gy = result.spec.cell_center(*result.goal)
    if result.reached:
        text = f"target reached (goal x={gx:.1f} m, y={gy:.1f} m)"
    else:
        text = f"target NOT reached: stops {result.shortfall_m:.1f} m short, at x={gx:.1f} m, y={gy:.1f} m"
    detour = f"{result.detour_ratio:.2f}" if result.detour_ratio is not None else "n/a"
    shares = ", ".join(
        f"{id_to_name[c] if c else 'unseen'} {share:.0%}" for c, share in result.class_shares().items()
    )
    return (f"{text}; path {result.length_m:.1f} m long (detour ratio {detour}), "
            f"total cost {result.total_cost:.1f}\nGround-truth classes along the path: {shares}")


def main() -> None:
    args = parse_args()
    seq = RellisSequence(args.sequence, root=args.root)
    frame = seq[args.frame]
    id_to_name, id_to_color = load_ontology(args.root)

    result = run_geometric_baseline(
        frame.points, frame.point_labels, goal_distance=args.goal_distance, footprint_radius=args.footprint_radius
    )
    title = f"camera, sequence {seq.sequence} frame {args.frame}"
    fig = draw_baseline_figure(frame.image, frame.calib, result, id_to_name, id_to_color, title)

    out_path = args.out or REPO_ROOT / "outputs" / f"path_{seq.sequence}_{args.frame:06d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)

    print(f"Sequence {seq.sequence}, frame {args.frame}: {describe(result, id_to_name)}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
