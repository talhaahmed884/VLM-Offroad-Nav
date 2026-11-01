"""Build the VLM prompt for one frame, and optionally check a VLM answer saved in a text file.

Saves the gridded image and the prompt text to outputs/vlm/ and prints the prompt. With
--answer, it runs the answer checker on that file and prints the cleaned-up result and
every problem it found. This lets you try the prompt by hand in any chat VLM before a
model is wired up: give it the image and the prompt, save its reply, and check it here.

Usage (from the repo root):
    python scripts/vlm_prompt.py --sequence 0 --frame 120
    python scripts/vlm_prompt.py --sequence 0 --frame 120 --answer my_answer.txt
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence  # noqa: E402
from offroad.viz.grid_overlay import draw_grid  # noqa: E402
from offroad.vlm.answer import VlmAnswer, parse_answer  # noqa: E402
from offroad.vlm.grid import ImageGrid  # noqa: E402
from offroad.vlm.prompt import VEHICLE_BRIEFING, build_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="frame index within the sequence (default: 0)")
    parser.add_argument("--goal-cell", default=None, help="grid cell of the goal, e.g. F7 (default: just 'straight ahead')")
    parser.add_argument("--no-briefing", action="store_true", help="leave out the sentence about what the vehicle can drive through")
    parser.add_argument("--answer", type=Path, default=None, help="text file with a VLM answer to check")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    return parser.parse_args()


def print_answer(answer: VlmAnswer):
    print(f"\nAnswer is {'VALID' if answer.valid else 'NOT USABLE'}")
    print(f"Waypoints (near to far): {' -> '.join(answer.waypoints) or '(none)'}")
    groups = defaultdict(list)
    for cell, hint in answer.hints.items():
        groups[(hint.terrain, hint.traversability)].append(cell)
    print("Cost hints:" if groups else "Cost hints: (none)")
    for (terrain, level), cells in groups.items():
        print(f"  {terrain:20s} {level:11s} {', '.join(cells)}")
    print(f"Reasoning: {answer.reasoning or '(none)'}")
    print("Problems found:" if answer.issues else "Problems found: none")
    for issue in answer.issues:
        print(f"  - {issue}")


def main() -> None:
    args = parse_args()
    if args.answer is not None and not args.answer.is_file():
        sys.exit(f"Answer file not found: {args.answer.resolve()}\n"
                 "Save the VLM's reply in that file first, or leave out --answer to only build the prompt.")
    seq = RellisSequence(args.sequence, root=args.root)
    image_bgr = cv2.imread(str(seq.image_path(args.frame)), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    grid = ImageGrid(height=image.shape[0], width=image.shape[1])
    briefing = None if args.no_briefing else VEHICLE_BRIEFING
    prompt = build_prompt(grid, goal_cell=args.goal_cell, vehicle_briefing=briefing)

    out_dir = REPO_ROOT / "outputs" / "vlm"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"prompt_{seq.sequence}_{args.frame:06d}"
    cv2.imwrite(str(out_dir / f"{stem}.png"), cv2.cvtColor(draw_grid(image, grid), cv2.COLOR_RGB2BGR))
    (out_dir / f"{stem}.txt").write_text(prompt, encoding="utf-8")

    print(prompt)
    print(f"\nSaved the gridded image and prompt to {out_dir / stem}.png / .txt")

    if args.answer is not None:
        print_answer(parse_answer(args.answer.read_text(encoding="utf-8"), grid))


if __name__ == "__main__":
    main()
