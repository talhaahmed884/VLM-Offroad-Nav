"""Who picks the path at each decision: the baselines and the VLM, behind one small interface.

A chooser sees only what a vehicle would have at run time: the camera image, the candidates
(from the LiDAR), and for Mode B the direction word. `decide` returns the chosen candidate's
index (or None to stop) and a record for the log. The VLM keeps its own history of earlier
decisions (as in phases 3-5: the last 3, as text), reset at the start of every run.
"""

from dataclasses import dataclass, field

import numpy as np

from offroad.data.calibration import CameraCalibration
from offroad.mapping.elevation import GridSpec
from offroad.nav.prompts import HistoryEntry, chosen_bearing, history_text, parse_answer
from offroad.nav.selector import prepare_selector_input
from offroad.sim.planning import FramePlan, cheapest_index, straightest_index


@dataclass
class DecisionContext:
    """What the chooser is told besides the picture: simulated time (s), decision number, mode, word."""

    time_s: float
    number: int
    scene: str
    mode: str = "A"
    word: str | None = None
    position_xy: tuple[float, float] = (0.0, 0.0)  # world pose: only the oracle may use it
    yaw: float = 0.0


@dataclass
class Decision:
    index: int | None  # the chosen candidate, or None = stop
    record: dict = field(default_factory=dict)  # what to log (answer, colors, timing, ...)
    picture: np.ndarray | None = None  # what the VLM saw (None for the baselines)


class Chooser:
    name = "chooser"

    def reset(self) -> None:
        """Called at the start of every run."""

    def decide(self, image: np.ndarray, plan: FramePlan, ctx: DecisionContext) -> Decision:
        raise NotImplementedError


class StraightestChooser(Chooser):
    """The candidate ending closest to straight ahead (phase 4's strongest simple baseline)."""

    name = "straightest"

    def decide(self, image, plan, ctx):
        return Decision(straightest_index(plan.candidates))


class CheapestChooser(Chooser):
    """The candidate with the lowest cost per meter: the LiDAR planner on its own."""

    name = "cheapest"

    def decide(self, image, plan, ctx):
        return Decision(cheapest_index(plan.candidates))


class OracleChooser(Chooser):
    """The candidate closest to the route (the answer key): the best the candidates allow.

    Like phase 4's "driver (oracle)", a reference, not a chooser a vehicle could have: it marks how
    well any chooser could do with these candidates. Mean distance to the route over the
    candidate's points from 3 m on (the first meters lie in the vehicle's cleared footprint).
    """

    name = "oracle"

    def __init__(self, route_xy: np.ndarray):
        self.route = np.asarray(route_xy, dtype=float)

    def decide(self, image, plan, ctx):
        from offroad.sim.control import to_world

        best, best_d = None, float("inf")
        for k, c in enumerate(plan.candidates):
            pts = to_world(c.xy[12:] if len(c.xy) > 12 else c.xy, np.array(ctx.position_xy), ctx.yaw)
            d = float(np.mean(np.min(np.linalg.norm(pts[:, None, :] - self.route[None, :, :], axis=2), axis=1)))
            if d < best_d:
                best, best_d = k, d
        return Decision(best, {"route_distance_m": round(best_d, 3)})


class VlmChooser(Chooser):
    """Asks the VLM (zero-shot or with the fine-tuned adapter) with the phase 3-5 picture and prompt.

    vlm: a loaded `vlm.backend.TransformersVlm` (or anything with .generate(image, prompt)).
    An answer without a valid choice falls back to the straightest candidate and is logged as such;
    "stop" stops the vehicle.
    """

    def __init__(self, vlm, name: str, calib: CameraCalibration, spec: GridSpec = GridSpec(),
                 reasoning: bool = False):
        self.vlm, self.name, self.calib, self.spec, self.reasoning = vlm, name, calib, spec, reasoning
        self.history: list[HistoryEntry] = []

    def reset(self) -> None:
        self.history = []

    def decide(self, image, plan, ctx):
        from offroad.vlm.runner import strip_thinking

        hist = history_text(self.history, ctx.time_s)
        sel = prepare_selector_input(image, self.calib, plan.candidates, plan.costmap.ground_z, self.spec,
                                     ctx.scene, ctx.number, ctx.mode, ctx.word, hist, reasoning=self.reasoning)
        reply = self.vlm.generate(sel.picture, sel.prompt)
        text = strip_thinking(reply.text)
        answer = parse_answer(text, sel.colors)
        bearings = [c.end_bearing_deg for c in plan.candidates]
        if answer.action == "stop":
            index, fallback = None, False
        elif answer.choice in sel.colors:
            index, fallback = sel.colors.index(answer.choice), False
        else:
            index, fallback = straightest_index(plan.candidates), True
        action = "stop" if index is None else "continue"
        self.history.append(HistoryEntry(ctx.time_s, action, None if index is None else bearings[index]))
        record = {"colors": sel.colors, "history": hist, "reply": text, "valid": answer.valid,
                  "choice": answer.choice, "action": answer.action, "safe_until": answer.safe_until,
                  "problems": answer.problems, "fallback_straightest": fallback,
                  "chosen_bearing_deg": chosen_bearing(answer, sel.colors, bearings),
                  "seconds": round(reply.seconds, 2)}
        return Decision(index, record, sel.picture)
