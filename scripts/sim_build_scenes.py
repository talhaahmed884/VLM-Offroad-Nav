"""Phase 6, step 6.4: build the test scenes (curves, fork, blocked, halfblocked) and check each one.

1. Generates one forest without a trail (outputs/sim/scenes/base, once; MAVS grows the plants).
2. Draws each design from sim/scenes.py into a copy of it: the trails as dirt, plants on them
   removed, blockades as 1.8 m walls (outputs/sim/scenes/<name>/).
3. Checks every scene in MAVS and saves outputs/sim/scenes/<name>/check_above.png (from above,
   with the routes and walls drawn) and check_camera.png (the view at the start):
   - the routes are loaded and no trunk stands on the trail (closest trunk to a route);
   - the candidates at the start exist;
   - blocked / halfblocked: standing 10 m before the wall, the LiDAR sees it taller than
     1.5 m (so the cost map blocks it), and the candidates there avoid it (blocked: none goes
     past the wall; halfblocked: some pass on the open half).
Results go to outputs/sim/scenes/scene_checks.json.

Usage (from the repo root; needs MAVS):
    python scripts/sim_build_scenes.py
    python scripts/sim_build_scenes.py --only fork
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.mapping.elevation import lidar_to_flu  # noqa: E402
from offroad.sim.control import to_vehicle  # noqa: E402
from offroad.sim.loop import start_pose  # noqa: E402
from offroad.sim.mavs_world import MavsWorld, TrailSceneSpec, build_designed_scene, generate_trail_scene  # noqa: E402
from offroad.sim.metrics import project_on_trail  # noqa: E402
from offroad.sim.planning import plan_frame  # noqa: E402
from offroad.sim.rig import RigSpec, camera_calibration  # noqa: E402
from offroad.sim.scenes import designs  # noqa: E402

SCENES_DIR = REPO_ROOT / "outputs" / "sim" / "scenes"
BASE_SPEC = TrailSceneSpec(trail_width_m=0.0, track_width_m=0.0, wheelbase_m=0.0)  # a forest, no trail
ABOVE_SIZE, ABOVE_HEIGHT_M, ABOVE_FOCAL = 900, 120.0, 0.0035


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=None, help="build only these designs")
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    return parser.parse_args()


def above_view(world, routes: dict, walls: np.ndarray) -> np.ndarray:
    """The scene from 120 m up, centred on (0, 0), with the routes (cyan) and walls (red) drawn on it."""
    cam = world.mavs.MavsCamera()
    cam.Initialize(ABOVE_SIZE, ABOVE_SIZE, 0.0035, 0.0035, ABOVE_FOCAL)
    cam.SetGammaAndGain(0.6, 1.0)
    cam.SetPose([0.0, 0.0, ABOVE_HEIGHT_M], [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])
    cam.Update(world.env, 0.1)
    img = np.ascontiguousarray(cam.GetNumpyArray())[:, :, ::-1].copy()
    m_per_px = ABOVE_HEIGHT_M * 0.0035 / ABOVE_SIZE / ABOVE_FOCAL

    def px(xy):  # looking straight down with the camera's top = +x (east), left = +y (north)
        return (int(ABOVE_SIZE / 2 - xy[1] / m_per_px), int(ABOVE_SIZE / 2 - xy[0] / m_per_px))

    for line in routes.values():
        pts = np.array([px(p) for p in line], dtype=np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 0), 1)
    for p in walls:
        cv2.circle(img, px(p), 2, (0, 0, 255), -1)
    return img


def check_scene(scene_json: Path, design, rig, calib) -> dict:
    out = {}
    for route in design.routes:
        world = MavsWorld(scene_json, rig, labeling=False, route=route)
        trunks = world.obstacles[0][world.obstacles[1] < 0.5 * 2] if len(world.obstacles[0]) else np.zeros((0, 2))
        gaps = [abs(project_on_trail(world.trail, t)[1]) for t in trunks] if len(trunks) else []
        x, y, yaw = start_pose(world.trail, 10)
        world.place(x, y, yaw)
        cap = world.capture()
        plan = plan_frame(cap.points_lidar, calib, cap.image.shape[1])
        out[route] = {"route_points": len(world.trail), "closest_obstacle_to_route_m": round(min(gaps), 2) if gaps else None,
                      "start_candidates": len(plan.candidates)}
        if route == design.default_route:
            cv2.imwrite(str(scene_json.parent / "check_camera.png"), cap.image[:, :, ::-1])
            walls = np.vstack([b.footprint_points() for b in design.blockades]) if design.blockades else np.zeros((0, 2))
            cv2.imwrite(str(scene_json.parent / "check_above.png"), above_view(world, design.routes, walls))
    if design.blockades:
        world = MavsWorld(scene_json, rig, labeling=False)
        wall = np.mean([[b.x, b.y] for b in design.blockades], axis=0)
        s_wall, _ = project_on_trail(world.trail, wall)
        k = int(np.argmin(np.abs(np.cumsum(np.r_[0, np.linalg.norm(np.diff(world.trail, axis=0), axis=1)]) - (s_wall - 10))))
        x, y, yaw = start_pose(world.trail, k)
        world.place(x, y, yaw)
        cap = world.capture()
        flu = lidar_to_flu(cap.points_lidar)
        wall_flu = to_vehicle(np.vstack([b.footprint_points() for b in design.blockades]), cap.position[:2], cap.yaw)
        # Points on the wall: close to its centre line and lower than 3 m (above it hang the trees).
        near = (np.min(np.linalg.norm(flu[:, None, :2] - wall_flu[None, :, :], axis=2), axis=1) < 0.4) \
            & (flu[:, 2] + rig.lidar_height_m < 3.0)
        plan = plan_frame(cap.points_lidar, calib, cap.image.shape[1])
        wall_x = float(np.mean(wall_flu[:, 0]))
        past = [bool(c.xy[:, 0].max() > wall_x + 0.5) for c in plan.candidates]
        out["wall"] = {"points_on_wall": int(near.sum()),
                       "wall_height_above_ground_m": round(float(np.percentile(flu[near, 2], 98) + rig.lidar_height_m), 2) if near.any() else None,
                       "candidates_10m_before": len(plan.candidates), "candidates_past_the_wall": int(sum(past))}
        cv2.imwrite(str(scene_json.parent / "check_wall_camera.png"), cap.image[:, :, ::-1])
    return out


def main() -> None:
    args = parse_args()
    base = args.scenes_dir / "base" / "base_scene.json"
    if not base.exists():
        print("Generating the base forest (no trail)...", flush=True)
        t = time.perf_counter()
        generate_trail_scene(base.parent, "base", BASE_SPEC)
        print(f"  done in {time.perf_counter() - t:.0f} s", flush=True)
    rig = RigSpec()
    calib = camera_calibration(rig)
    checks = {}
    for name, design in designs().items():
        if args.only and name not in args.only:
            continue
        t = time.perf_counter()
        scene = build_designed_scene(design, base, args.scenes_dir / name)
        checks[name] = check_scene(scene, design, rig, calib)
        print(f"{name}: {design.note}; {json.dumps(checks[name])} ({time.perf_counter() - t:.0f} s)", flush=True)
    path = args.scenes_dir / "scene_checks.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **checks}, indent=2) + "\n", encoding="utf-8")
    problems = []
    for name, c in checks.items():
        for route, r in c.items():
            if route == "wall":
                if (r["wall_height_above_ground_m"] or 0) < 1.5:
                    problems.append(f"{name}: the wall looks lower than 1.5 m")
                if name == "blocked" and r["candidates_past_the_wall"]:
                    problems.append(f"{name}: a candidate passes the wall")
                if name == "halfblocked" and not r["candidates_past_the_wall"]:
                    problems.append(f"{name}: no candidate passes on the open half")
            else:
                if r["start_candidates"] == 0:
                    problems.append(f"{name}/{route}: no candidate at the start")
                if r["closest_obstacle_to_route_m"] is not None and r["closest_obstacle_to_route_m"] < 2.5 and name != "blocked" and name != "halfblocked":
                    problems.append(f"{name}/{route}: an obstacle {r['closest_obstacle_to_route_m']} m from the trail centre")
    print("CHECK", "OK" if not problems else "FAILED: " + "; ".join(problems))
    print(f"Saved the scenes and checks to {args.scenes_dir}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
