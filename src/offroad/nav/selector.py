"""Everything the path selector is given for one frame: the picture and the text prompt (pure, no file I/O).

Shared by scripts/show_prompt.py (look at it) and scripts/run_selector.py (ask the VLM). Uses
only run-time information: the camera image, the calibration, the candidates and the cost
map's ground (plus Mode B's direction word and the history of earlier decisions).
"""

from dataclasses import dataclass

import numpy as np

from offroad.data.calibration import CameraCalibration
from offroad.mapping.elevation import GridSpec
from offroad.nav.candidates import Candidate
from offroad.nav.prompts import build_prompt, shuffled
from offroad.nav.visual_prompt import assign_colors, draw_visual_prompt, frame_seed


@dataclass
class SelectorInput:
    """The selector's input for one frame.

    picture: (H, W, 3) RGB image with the candidates drawn on it.
    colors: color name of each candidate, in candidate order (left to right).
    listed: the same names in the (random) order the prompt lists them.
    visible: whether each candidate shows up in the picture.
    """

    picture: np.ndarray
    colors: list[str]
    listed: list[str]
    prompt: str
    visible: list[bool]


def prepare_selector_input(
    image: np.ndarray,
    calib: CameraCalibration,
    candidates: list[Candidate],
    ground_z: np.ndarray,
    spec: GridSpec,
    sequence: str,
    frame: int,
    mode: str = "A",
    direction: str | None = None,
    history: str = "none.",
    reasoning: bool = True,
) -> SelectorInput:
    """Random (but repeatable) colors and listing order, the drawn picture, and the prompt for one frame.

    reasoning: ask for the one-sentence reasoning (zero-shot); the fine-tuned model is asked without it.
    """
    palette = assign_colors(len(candidates), frame_seed(sequence, frame))
    colors = [name for name, _ in palette]
    listed = shuffled(colors, frame_seed(sequence, frame, salt=1))
    picture, visible = draw_visual_prompt(image, calib, [c.xy for c in candidates], palette, ground_z, spec)
    prompt = build_prompt(listed, mode, direction, history, reasoning=reasoning)
    return SelectorInput(picture, colors, listed, prompt, visible)
