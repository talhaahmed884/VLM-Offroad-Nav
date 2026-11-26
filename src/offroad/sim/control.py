"""Drive the chosen candidate: pure-pursuit steering and a speed controller (pure, no MAVS needed).

A candidate is planned in the vehicle's FLU frame at the decision (x forward, y left, meters).
It is fixed in the world (`to_world`) and followed until the next decision: at every control
step the path is seen from the vehicle's current pose (`to_vehicle`), a point `lookahead_m`
ahead on it is picked, and the arc through that point gives the curvature (pure pursuit).

MAVS's steering command (-1..1, + = left) sets the MRZR4's front-wheel angle (17.5 degrees at
1), so the controller maps curvature to the command with a measured gain (`curvature_per_steer`,
1/m per unit command). In this MAVS build the brake drives backwards from standstill, so the
brake is only used while moving forward faster than wanted, and never below `brake_min_speed_mps`.

Measured in MAVS (2026-11-24, MRZR4 on bare ground, 0.02 s steps): following a straight path
and arcs of 25, 12 and 10 m radius, the vehicle stays on average 0.07-0.32 m (at most 0.46 m)
from the path at 1.3-1.5 m/s.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FollowerParams:
    speed_mps: float = 1.5  # RELLIS drives were ~1.25 m/s
    lookahead_m: float = 3.0
    # Measured for the MAVS MRZR4 at ~1.2 m/s (2026-11-24): curvature = 0.105 1/m x command,
    # linear up to full lock (9.5 m radius); candidate paths need at most ~0.07 1/m.
    curvature_per_steer: float = 0.105
    max_steer: float = 1.0
    # MAVS throttle works like an accelerator held down (0.1 reaches 3 m/s after 4 s and keeps
    # going), so speed needs feedback: a feed-forward plus a gain on the speed error.
    throttle_per_mps: float = 0.12
    throttle_gain: float = 0.3
    brake_gain: float = 0.3
    brake_min_speed_mps: float = 0.3


def to_world(path_flu: np.ndarray, position_xy: np.ndarray, yaw: float) -> np.ndarray:
    """(N, 2) FLU path seen from a pose -> world xy."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(path_flu, dtype=float)[:, :2] @ R.T + np.asarray(position_xy, dtype=float)[:2]


def to_vehicle(path_world: np.ndarray, position_xy: np.ndarray, yaw: float) -> np.ndarray:
    """(N, 2) world xy -> the FLU frame of the vehicle at (position, yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return (np.asarray(path_world, dtype=float)[:, :2] - np.asarray(position_xy, dtype=float)[:2]) @ R


def lookahead_point(path_flu: np.ndarray, lookahead_m: float) -> np.ndarray | None:
    """The first path point at least lookahead_m from the vehicle that lies ahead of it, else the path's end.

    None if the whole path is behind the vehicle (x <= 0): the path has been driven.
    """
    p = np.asarray(path_flu, dtype=float)
    ahead = p[p[:, 0] > 0.0]
    if len(ahead) == 0:
        return None
    far = ahead[np.hypot(ahead[:, 0], ahead[:, 1]) >= lookahead_m]
    return far[0] if len(far) else ahead[-1]


def pursuit_curvature(target_flu: np.ndarray) -> float:
    """Curvature (1/m, + = left) of the arc from the vehicle (heading +x) through the target point."""
    x, y = float(target_flu[0]), float(target_flu[1])
    d2 = x * x + y * y
    return 0.0 if d2 == 0 else 2.0 * y / d2


def steering_command(curvature: float, params: FollowerParams) -> float:
    return float(np.clip(curvature / params.curvature_per_steer, -params.max_steer, params.max_steer))


def speed_command(forward_speed: float, wanted: float, params: FollowerParams) -> tuple[float, float]:
    """(throttle, brake), each 0..1. The brake only acts while moving forward faster than wanted."""
    if wanted <= 0.0:
        brake = params.brake_gain * forward_speed if forward_speed > params.brake_min_speed_mps else 0.0
        return 0.0, float(np.clip(brake, 0.0, 1.0))
    throttle = params.throttle_per_mps * wanted + params.throttle_gain * (wanted - forward_speed)
    brake = 0.0
    if forward_speed > max(wanted + 0.5, params.brake_min_speed_mps):
        brake = params.brake_gain * (forward_speed - wanted)
        throttle = 0.0
    return float(np.clip(throttle, 0.0, 1.0)), float(np.clip(brake, 0.0, 1.0))


def follow(path_world: np.ndarray, position_xy: np.ndarray, yaw: float, forward_speed: float,
           params: FollowerParams) -> tuple[float, float, float, bool]:
    """(throttle, steering, brake, done) for one control step; done when the path is behind the vehicle."""
    target = lookahead_point(to_vehicle(path_world, position_xy, yaw), params.lookahead_m)
    if target is None:
        throttle, brake = speed_command(forward_speed, 0.0, params)
        return throttle, 0.0, brake, True
    throttle, brake = speed_command(forward_speed, params.speed_mps, params)
    return throttle, steering_command(pursuit_curvature(target), params), brake, False
