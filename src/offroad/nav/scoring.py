"""Score one selector answer against the frame's automatic labels (pure, no file I/O).

The labels are only the answer key here; the model never sees them. Phase 3 uses this to judge
the zero-shot answers; phase 4 adds baselines and the full evaluation.
"""

from dataclasses import dataclass

from offroad.nav.labels import SAFE_ANSWERS, CandidateLabels, should_stop
from offroad.nav.prompts import ParsedAnswer


@dataclass
class FrameScore:
    """How one answer compares with the labels. Fields are None when the answer gives nothing to score.

    choice_*: about the chosen candidate (true labels). chose_puddle: the chosen candidate
    crosses a puddle. puddle_frame: some candidate crosses a puddle and some candidate is dry and
    safe to tick 1, so avoiding the puddle was possible.
    ticks_*: counts over the candidates the answer gave a tick for; "over" = said safer than it
    is (the dangerous mistake), "under" = said less safe than it is.
    """

    valid: bool
    choice_matches_driver: bool | None
    choice_is_closest: bool | None
    choice_safe_until: str | None
    chose_puddle: bool | None
    puddle_frame: bool
    action: str | None
    stop_label: bool
    ticks_answered: int
    ticks_exact: int
    ticks_within_one: int
    ticks_over: int
    ticks_under: int

    @property
    def action_correct(self) -> bool | None:
        return None if self.action is None else (self.action == "stop") == self.stop_label

    @property
    def choice_safe(self) -> bool | None:
        return None if self.choice_safe_until is None else self.choice_safe_until != "none"


def score_answer(answer: ParsedAnswer, colors: list[str], labels: list[CandidateLabels],
                 closest_index: int | None) -> FrameScore:
    """Score an answer. colors[k] is candidate k's color; labels[k] its labels; closest_index the driver's candidate."""
    k = colors.index(answer.choice) if answer.choice in colors else None
    chosen = labels[k] if k is not None else None
    puddle_frame = (any(lab.first_caution_m is not None for lab in labels)
                    and any(lab.first_caution_m is None and lab.safe_until != "none" for lab in labels))

    exact = within_one = over = under = 0
    for color, predicted in answer.safe_until.items():
        truth = SAFE_ANSWERS.index(labels[colors.index(color)].safe_until)
        said = SAFE_ANSWERS.index(predicted)
        exact += said == truth
        within_one += abs(said - truth) <= 1
        over += said > truth
        under += said < truth

    return FrameScore(
        valid=answer.valid,
        choice_matches_driver=None if chosen is None else chosen.matches_driver,
        choice_is_closest=None if k is None or closest_index is None else k == closest_index,
        choice_safe_until=None if chosen is None else chosen.safe_until,
        chose_puddle=None if chosen is None else chosen.first_caution_m is not None,
        puddle_frame=puddle_frame,
        action=answer.action,
        stop_label=should_stop(labels),
        ticks_answered=len(answer.safe_until),
        ticks_exact=exact,
        ticks_within_one=within_one,
        ticks_over=over,
        ticks_under=under,
    )
