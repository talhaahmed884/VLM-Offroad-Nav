"""Show exactly what the VLM is given for one frame: the picture and the text prompt (phase 3 check).

Saves the picture (800 x 500, the candidates drawn as colored lines with white dots at 5 / 10 /
15 m) to outputs/prompt_<seq>_<frame>.png and prints the prompt, followed by the frame's labels
(the answer key, which the model never sees).

Usage (from the repo root):
    python scripts/show_prompt.py --sequence 0 --frame 601              # Mode A
    python scripts/show_prompt.py --sequence 0 --frame 601 --mode B     # Mode B, direction from the driven path
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.mapping.elevation import GridSpec  # noqa: E402
from offroad.nav.candidates import CHOSEN_VARIANT  # noqa: E402
from offroad.nav.driven_path import DIRECTION_DISTANCE_M, direction_bearing_deg, direction_word, driven_path_flu  # noqa: E402
from offroad.nav.frame_candidates import candidates_for_frame  # noqa: E402
from offroad.nav.prompts import MODES  # noqa: E402
from offroad.nav.selector import prepare_selector_input  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=601, help="frame index within the sequence (default: 601)")
    parser.add_argument("--mode", choices=MODES, default="A", help="A = keep going, B = keep going + direction")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: outputs/prompt_<seq>_<frame>.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq = RellisSequence(args.sequence, root=args.root)
    frame = seq[args.frame]
    id_to_name, _ = load_ontology(args.root)
    spec = GridSpec()
    result = candidates_for_frame(frame.points, frame.point_labels, frame.label_image, frame.calib.T_cam_from_lidar,
                                  frame.calib.K, seq.poses, args.frame, id_to_name,
                                  image_width=frame.image.shape[1], variants=(CHOSEN_VARIANT,))
    fc = result.variants[CHOSEN_VARIANT]
    if not fc.candidates:
        sys.exit("This frame has no candidates; the vehicle would stop without asking the VLM.")
    direction = direction_word(direction_bearing_deg(driven_path_flu(seq.poses, args.frame, DIRECTION_DISTANCE_M)))
    selector_input = prepare_selector_input(frame.image, frame.calib, fc.candidates, result.costmap.ground_z, spec,
                                            seq.sequence, args.frame, args.mode, direction)

    out = args.out or REPO_ROOT / "outputs" / f"prompt_{seq.sequence}_{args.frame:06d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(selector_input.picture, cv2.COLOR_RGB2BGR))

    print(f"--- prompt (Mode {args.mode}) ---\n{selector_input.prompt}\n")
    print(f"--- answer key ({CHOSEN_VARIANT} candidates, left to right; the model never sees this) ---")
    for k, (color, c, lab) in enumerate(zip(selector_input.colors, fc.candidates, fc.labels)):
        notes = [f"safe {lab.safe_until}"]
        notes += ["driver match"] if lab.matches_driver else []
        notes += ["closest to the driver"] if k == fc.closest_index else []
        notes += [] if lab.first_caution_m is None else [f"puddle at {lab.first_caution_m:.1f} m"]
        print(f"  {color:8s} bearing {c.end_bearing_deg:+6.1f} deg: {', '.join(notes)}")
    print(f"  action: {'stop' if fc.stop else 'continue'}; Mode B direction: {direction}")
    print(f"Saved {out} ({selector_input.picture.shape[1]} x {selector_input.picture.shape[0]})")


if __name__ == "__main__":
    main()
