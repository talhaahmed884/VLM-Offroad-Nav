"""Videos of closed-loop runs: a chase camera behind the vehicle next to the picture the chooser saw.

A video is made by driving a saved run again (`choosers.ReplayChooser` repeats its logged choices;
the loop and MAVS are deterministic, so it drives exactly the same path in a few minutes, without
the VLM). `Recorder` wraps the world: every few control steps it renders the chase view
(`MavsWorld.chase_view`, the vehicle's body and wheels drawn as MAVS actors), draws the chosen path
on it, and puts the chooser's picture and a caption beside it.

Pure geometry and drawing (no MAVS), apart from what `Recorder` asks of its world.

Chase camera: `back_m` behind and `up_m` above the vehicle, looking `pitch_deg` down, horizontal
field of view `fov_deg`. Its heading follows the vehicle's with a lag (`yaw_follow`, the share of
the difference closed per frame), so the picture doesn't shake with every steering correction.
Frames: world ENU (x east, y north, z up); the camera looks along its own +x (as MAVS's cameras).
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from offroad.nav.candidates import CANDIDATE_COLORS
from offroad.sim.control import to_vehicle, to_world

COLOR_RGB = dict(CANDIDATE_COLORS)
# The chosen path is drawn from this far ahead of the vehicle's centre: the overlay has no depth,
# so nearer points would be drawn over the vehicle's body (its front is 1.75 m from the centre).
PATH_FROM_M = 4.0
FPS = 10
CONTROL_DT_S = 0.02
TEXT_RGB = (235, 235, 235)
STRIP_RGB = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class ChaseParams:
    size: tuple[int, int] = (960, 540)  # (width, height), pixels
    fov_deg: float = 60.0
    # High enough that the vehicle's roof (~2 m) hides little of the ground ahead (from 3.5 m up
    # it hid the first ~10 m) and a fork's branches are in view (2026-11-28).
    back_m: float = 10.0
    up_m: float = 6.0
    pitch_deg: float = 22.0
    yaw_follow: float = 0.2


def quat_mul(a, b) -> list[float]:
    """Product of two [w, x, y, z] quaternions (a then b, in the body frame)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def quat_yaw_pitch(yaw: float, pitch: float) -> list[float]:
    """[w, x, y, z] of a heading yaw (about z, + = left) then a pitch about the new y (+ = nose down)."""
    return quat_mul([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                    [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])


def quat_about_y(angle: float) -> list[float]:
    return [math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]


def quat_matrix(q) -> np.ndarray:
    """(3, 3) rotation matrix of a [w, x, y, z] quaternion."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def follow_yaw(previous: float | None, target: float, share: float) -> float:
    """Move an angle `share` of the way toward target, the short way round (radians)."""
    if previous is None:
        return target
    diff = (target - previous + math.pi) % (2 * math.pi) - math.pi
    return previous + share * diff


def chase_pose(vehicle_xyz: np.ndarray, camera_yaw: float, params: ChaseParams) -> tuple[list[float], list[float]]:
    """(camera position [x, y, z], orientation [w, x, y, z]) behind and above the vehicle."""
    x, y, z = (float(v) for v in vehicle_xyz[:3])
    position = [x - params.back_m * math.cos(camera_yaw), y - params.back_m * math.sin(camera_yaw), z + params.up_m]
    return position, quat_yaw_pitch(camera_yaw, math.radians(params.pitch_deg))


def chase_K(params: ChaseParams) -> np.ndarray:
    """(3, 3) pinhole intrinsics of the chase camera (square pixels, centred)."""
    w, h = params.size
    f = (w / 2) / math.tan(math.radians(params.fov_deg) / 2)
    return np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])


def project_to_chase(points_world: np.ndarray, cam_position, cam_quat, params: ChaseParams) -> np.ndarray:
    """(N, 2) pixels of world points in the chase image; NaN for points behind the camera."""
    p = (np.asarray(points_world, dtype=float)[:, :3] - np.asarray(cam_position, dtype=float)) @ quat_matrix(cam_quat)
    # Camera axes x forward, y left, z up -> optical x right, y down, z forward.
    x_opt, y_opt, z_opt = -p[:, 1], -p[:, 2], p[:, 0]
    K = chase_K(params)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack([K[0, 0] * x_opt / z_opt + K[0, 2], K[1, 1] * y_opt / z_opt + K[1, 2]])
    uv[z_opt <= 0.1] = np.nan
    return uv


def draw_path(image: np.ndarray, uv: np.ndarray, rgb: tuple[int, int, int], thickness: int = 5) -> np.ndarray:
    """The path as a polyline with a dark outline (as in the chooser's picture); NaN points are skipped."""
    out = image.copy()
    pts = uv[np.all(np.isfinite(uv), axis=1)]
    if len(pts) >= 2:
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.polylines(out, [pts], False, tuple(int(c) for c in rgb), thickness, cv2.LINE_AA)
    return out


def put_lines(image: np.ndarray, lines: list[str], origin: tuple[int, int], scale: float = 0.6,
              line_px: int = 28, rgb=TEXT_RGB) -> None:
    x, y = origin
    for k, line in enumerate(lines):
        cv2.putText(image, line, (x, y + k * line_px), FONT, scale, rgb, 1, cv2.LINE_AA)


def banner(image: np.ndarray, text: str, top: int = 50) -> np.ndarray:
    """A dark band with large text across the image, `top` px from its top edge."""
    out = image.copy()
    w = out.shape[1]
    cv2.rectangle(out, (0, top), (w, top + 64), STRIP_RGB, -1)
    (tw, _), _ = cv2.getTextSize(text, FONT, 1.0, 2)
    cv2.putText(out, text, ((w - tw) // 2, top + 44), FONT, 1.0, TEXT_RGB, 2, cv2.LINE_AA)
    return out


def compose_frame(chase: np.ndarray, picture: np.ndarray | None, picture_title: str, caption: list[str]) -> np.ndarray:
    """(H, W, 3) RGB frame: chase view | the chooser's picture (scaled to the same height), caption below."""
    h, w = chase.shape[:2]
    pw = round(h * 4 / 3) // 2 * 2  # the picture is 4:3 (800 x 600)
    right = np.full((h, pw, 3), 50, np.uint8)
    if picture is not None:
        right = cv2.resize(picture, (pw, h), interpolation=cv2.INTER_AREA)
    else:
        put_lines(right, ["no candidates this decision"], (20, h // 2))
    cv2.rectangle(right, (0, 0), (pw, 34), STRIP_RGB, -1)
    put_lines(right, [picture_title], (12, 24))
    cv2.rectangle(chase, (0, 0), (w, 34), STRIP_RGB, -1)
    put_lines(chase, ["chase view (not seen by the chooser)"], (12, 24))
    strip = np.full((22 + 30 * len(caption), w + pw, 3), STRIP_RGB, np.uint8)
    put_lines(strip, caption, (14, 30), scale=0.62, line_px=30)
    frame = np.vstack([np.hstack([chase, right]), strip])
    return frame[: frame.shape[0] // 2 * 2]


def title_card(width: int, height: int, lines: list[str]) -> np.ndarray:
    card = np.full((height, width, 3), STRIP_RGB, np.uint8)
    put_lines(card, lines[:1], (60, height // 2 - 60), scale=1.1, line_px=44)
    put_lines(card, lines[1:], (60, height // 2 - 10), scale=0.75, line_px=36)
    return card


def answer_text(record: dict, answer: dict) -> str:
    """One caption line for a decision record (see sim/loop.py) and the chooser's answer record
    (a VLM's parsed answer, or for a baseline just the candidates' colors)."""
    if record.get("n_candidates", 0) == 0:
        return ("no candidates: keeps the previous path" if record.get("held") else "no candidates: stops")
    colors = answer.get("colors")
    chosen = record.get("chosen")
    if "valid" not in answer:  # a baseline: no answer, just a rule
        name = colors[chosen] if colors and chosen is not None else str(chosen)
        return f"picked: {name} (end bearing {record.get('chosen_bearing_deg')} deg)"
    safe = ", ".join(f"{k} {v}" for k, v in (answer.get("safe_until") or {}).items())
    return (f"answer: {answer.get('choice')}  |  safe until: {safe}  |  {answer.get('action')}"
            f"  |  {answer.get('seconds', 0):.1f} s to answer{'  (invalid: fell back to straightest)' if answer.get('fallback_straightest') else ''}")


class Recorder:
    """Wraps a world for `loop.run_episode` and writes video frames as it drives.

    world: a MavsWorld with `show_vehicle()` done (or a stand-in with chase_view and ground_height).
    write(frame): takes one (H, W, 3) RGB frame. header: the caption's first line.
    speed: 1 = real time at FPS frames per simulated second; 2 = twice as fast, etc.
    Use `recorder` as the world and `recorder.on_decision` as run_episode's on_decision.
    """

    def __init__(self, world, write, header: str, picture_title: str = "what the chooser saw",
                 params: ChaseParams = ChaseParams(), speed: float = 1.0):
        self.world, self.write, self.header, self.params = world, write, header, params
        self.picture_title = self.title_now = picture_title
        self.picture_number = None
        self.every = max(1, round(speed / (FPS * CONTROL_DT_S)))
        self.steps, self.frames, self.cam_yaw = 0, 0, None
        self.sim_t, self.picture, self.path_world, self.path_rgb = 0.0, None, None, None
        self.lines = ["", ""]
        self.last_frame = None

    def __getattr__(self, name):  # trail, capture, pose, heading_and_speed, obstacles, ...
        return getattr(self.world, name)

    def place(self, x, y, yaw):
        self.world.place(x, y, yaw)
        self.steps, self.sim_t, self.cam_yaw = 0, 0.0, None

    def step(self, throttle, steering, braking, dt):
        self.world.step(throttle, steering, braking, dt)
        self.steps += 1
        self.sim_t += dt
        if self.steps % self.every == 0:
            self.frame()

    def on_decision(self, number, record, capture, plan, decision):
        """Called by run_episode after each decision: updates the picture, the path and the caption."""
        progress = f"progress {record['progress_m']:.1f} m  |  offset {record['offset_m']:+.2f} m from the route centre"
        word = f"  |  word: '{record['word']}'" if record.get("word") else ""
        self.lines = [f"t {record['time_s']:5.1f} s  |  decision {number}  |  {record['n_candidates']} candidates  |  "
                      f"{progress}{word}", answer_text(record, decision.record if decision is not None else {})]
        if decision is not None and decision.picture is not None:
            self.picture, self.picture_number = decision.picture, number
            self.title_now = self.picture_title
        elif not plan.candidates and self.picture is not None:  # keep the last picture, say so
            self.title_now = f"no candidates now: last picture (decision {self.picture_number})"
        if decision is not None and decision.index is not None:
            cand = plan.candidates[decision.index]
            xy = to_world(cand.xy, capture.position[:2], capture.yaw)
            ground = getattr(self.world, "ground_height", lambda x, y: 0.0)
            self.path_world = np.column_stack([xy, [ground(x, y) + 0.05 for x, y in xy]])
            colors = decision.record.get("colors") or []
            self.path_rgb = COLOR_RGB.get(colors[decision.index], (255, 255, 255)) if colors else (255, 255, 255)
        if number == 0:
            self.frame()

    def frame(self, extra_banner: str | None = None) -> np.ndarray:
        position, _ = self.world.pose()
        yaw, _ = self.world.heading_and_speed()
        self.cam_yaw = follow_yaw(self.cam_yaw, yaw, self.params.yaw_follow)
        cam_position, cam_quat = chase_pose(position, self.cam_yaw, self.params)
        chase = np.ascontiguousarray(self.world.chase_view(cam_position, cam_quat), dtype=np.uint8)
        if self.path_world is not None:  # the part still ahead of the front bumper
            ahead = self.path_world[to_vehicle(self.path_world[:, :2], position[:2], yaw)[:, 0] > PATH_FROM_M]
            chase = draw_path(chase, project_to_chase(ahead, cam_position, cam_quat, self.params), self.path_rgb)
        if extra_banner:
            chase = banner(chase, extra_banner)
        frame = compose_frame(chase, self.picture, self.title_now, [self.header, *self.lines])
        self.write(frame)
        self.frames += 1
        self.last_frame = frame
        return frame

    def finish(self, outcome_text: str, hold_s: float = 3.0) -> None:
        """The last view with the outcome across it, held for hold_s seconds."""
        frame = self.frame(outcome_text)
        for _ in range(round(hold_s * FPS) - 1):
            self.write(frame)
            self.frames += 1
