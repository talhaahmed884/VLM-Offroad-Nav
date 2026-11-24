"""Pieces of the QLoRA fine-tuning that can be tested without a GPU (phase 5).

The trainer (scripts/train_selector.py) builds each training sequence exactly as the selector
asks at run time: the chat prompt with the picture (`apply_chat_template(...,
add_generation_prompt=True)`, as in vlm/backend.py), followed by the target answer and the
end-of-answer token. Only the answer is learned: the prompt's tokens get the label IGNORE, and
so does the choice's value when an example says so (Mode A frames where no candidate matches
the driver, nav/sft.py).
"""

import numpy as np

IGNORE = -100  # label value the loss skips
# LoRA goes on the language model's linear layers only; the vision encoder stays as it is.
LORA_TARGET = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def answer_labels(target_ids: list[int], offsets: list[tuple[int, int]], masked_span: tuple[int, int] | None,
                  end_id: int) -> tuple[list[int], list[int]]:
    """(ids, labels) for the answer part: the target's tokens plus the end token.

    offsets: (start, end) characters of each target token; tokens overlapping masked_span (a
    character range of the target) get IGNORE as label.
    """
    labels = list(target_ids)
    if masked_span is not None:
        lo, hi = masked_span
        labels = [IGNORE if start < hi and end > lo else t for t, (start, end) in zip(target_ids, offsets)]
    return list(target_ids) + [end_id], labels + [end_id]


def epoch_end_steps(examples_per_epoch: list[int], grad_accum: int) -> list[int]:
    """Optimizer step at which each epoch of the data ends (batch size 1, grad_accum examples per step)."""
    ends = np.cumsum(examples_per_epoch)
    return [int(np.ceil(e / grad_accum)) for e in ends]
