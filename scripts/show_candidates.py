"""Show the candidate paths of one frame for both cost-grid variants, with their labels (phase 2 check).

Four panels: the camera image with the strict candidates, the same with the relaxed candidates
(each candidate in its color, white dots at the 5 / 10 / 15 m ticks, the driven path in
cyan), and the two cost grids from above with the same paths. It prints each candidate's labels.

This is not yet the picture the VLM will see (that's phase 3); it is a check that the
candidates lie on drivable ground and that the labels make sense.

Usage (from the repo root):
    python scripts/show_candidates.py --sequence 0 --frame 601
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file only; also works without a display (e.g. on the cluster)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.mapping.costmap import CostParams  # noqa: E402
from offroad.mapping.elevation import GridSpec  # noqa: E402
from offroad.nav.candidates import CANDIDATE_COLORS, VARIANTS, relaxed_cost, strict_cost  # noqa: E402
from offroad.nav.driven_path import point_at_distance  # noqa: E402
from offroad.nav.frame_candidates import candidates_for_frame  # noqa: E402
from offroad.nav.labels import TICKS_M  # noqa: E402
from offroad.viz.baseline_figure import draw_xy_path_on_camera  # noqa: E402
from offroad.viz.topdown import draw_camera_fov, show_cost_values  # noqa: E402

DRIVEN_RGB = (0, 255, 255)
TICK_RGB = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=601, help="frame index within the sequence (default: 601)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: outputs/candidates_<seq>_<frame>.png)")
    return parser.parse_args()


def tick_points(xy: np.ndarray) -> np.ndarray:
    """(T, 2) points at the tick distances along a path (ticks beyond its end are left out)."""
    points = [point_at_distance(xy, d) for d in TICKS_M]
    return np.array([p for p in points if p is not None]).reshape(-1, 2)


def main() -> None:
    args = parse_args()
    seq = RellisSequence(args.sequence, root=args.root)
    frame = seq[args.frame]
    id_to_name, _ = load_ontology(args.root)
    spec, params = GridSpec(), CostParams()
    result = candidates_for_frame(frame.points, frame.point_labels, frame.label_image, frame.calib.T_cam_from_lidar,
                                  frame.calib.K, seq.poses, args.frame, id_to_name, image_width=frame.image.shape[1],
                                  spec=spec, params=params)
    ground_z = result.costmap.ground_z
    grids = {"strict": strict_cost(result.costmap), "relaxed": relaxed_cost(result.costmap, params)}

    fig, axes = plt.subplots(2, 2, figsize=(22, 16), gridspec_kw={"height_ratios": [1, 1.1]})
    for column, variant in enumerate(VARIANTS):
        fc = result.variants[variant]
        camera = draw_xy_path_on_camera(frame.image, frame.calib, result.driven_xy, ground_z, spec, DRIVEN_RGB, radius=5)
        top = axes[1, column]
        show_cost_values(top, fig, grids[variant], spec, f"{variant} cost grid (log scale)")
        draw_camera_fov(top, frame.calib.K, frame.image.shape[1], spec)
        top.plot(result.driven_xy[:, 1], result.driven_xy[:, 0], "-", color=np.array(DRIVEN_RGB) / 255, linewidth=4,
                 label="driven path")
        for (name, rgb), candidate, lab in zip(CANDIDATE_COLORS, fc.candidates, fc.labels):
            camera = draw_xy_path_on_camera(camera, frame.calib, candidate.xy, ground_z, spec, rgb, radius=8)
            camera = draw_xy_path_on_camera(camera, frame.calib, tick_points(candidate.xy), ground_z, spec, TICK_RGB,
                                            radius=11)
            top.plot(candidate.xy[:, 1], candidate.xy[:, 0], "-", color=np.array(rgb) / 255, linewidth=2.5,
                     label=f"{name}: safe {lab.safe_until}{' (driver)' if lab.matches_driver else ''}")
            ticks = tick_points(candidate.xy)
            top.plot(ticks[:, 1], ticks[:, 0], "o", color="white", markeredgecolor="black", markersize=6)
        top.legend(loc="lower left", fontsize=9)
        top.set_ylim(spec.x_min, 25)
        top.set_xlim(12, -12)
        axes[0, column].imshow(camera)
        axes[0, column].set_title(f"{variant}: {len(fc.candidates)} candidate(s); stop={fc.stop}, "
                                  f"real choice={fc.real_choice}; driven path in cyan")
        axes[0, column].axis("off")

    fig.suptitle(f"sequence {seq.sequence} frame {args.frame} "
                 f"({'with' if frame.label_image is not None else 'no'} camera label image)", fontsize=14)
    fig.tight_layout()
    out = args.out or REPO_ROOT / "outputs" / f"candidates_{seq.sequence}_{args.frame:06d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)

    def describe(lab) -> str:
        unsafe = (f"first unsafe {lab.first_unsafe_m:4.1f} m ({id_to_name.get(lab.first_unsafe_class)})"
                  if lab.first_unsafe_m is not None else "nothing unsafe")
        if lab.first_caution_m is not None:
            unsafe += f", puddle at {lab.first_caution_m:.1f} m"
        return (f"to driven path {lab.driven_mean_m:5.2f} m{' (match)' if lab.matches_driver else '        '}  "
                f"safe {lab.safe_until:6s}  {unsafe}; unknown ground {100 * lab.unknown_share:.0f}%")

    print(f"Sequence {seq.sequence}, frame {args.frame}")
    print(f"  driven path: safe {result.driven.safe_until}, "
          + (f"first unsafe {result.driven.first_unsafe_m:.1f} m ({id_to_name.get(result.driven.first_unsafe_class)})"
             if result.driven.first_unsafe_m is not None else "nothing unsafe")
          + ("" if result.driven.first_caution_m is None else f", puddle at {result.driven.first_caution_m:.1f} m"))
    for variant in VARIANTS:
        fc = result.variants[variant]
        print(f"  {variant}: {len(fc.candidates)} candidate(s), stop={fc.stop}, real choice={fc.real_choice}")
        for (name, _), c, lab in zip(CANDIDATE_COLORS, fc.candidates, fc.labels):
            print(f"    {name:8s} bearing {c.end_bearing_deg:+6.1f} deg, {c.length_m:4.1f} m, cost/m {c.cost_per_m:5.2f}; "
                  + describe(lab))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
