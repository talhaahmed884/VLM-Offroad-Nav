"""Phase 6, step 6.1: check the simulated sensor rig in MAVS.

Generates a trail scene (once; it is reused afterwards), puts the MRZR4 on the trail with a
RELLIS-like LiDAR and camera (sim/rig.py), lets it settle, captures one frame, and saves:
- camera.png: what the selector would see (before the candidates are drawn);
- lidar_on_camera.png: the LiDAR points projected on the image with our calibration, colored by
  distance (blue near, red far). Points must sit on the objects they hit;
- projection_check.png: our projected points (yellow) over the pixels MAVS itself marks for the
  same points (red): they should coincide (the partner's alignment check, as a number below);
- above.png: the scene seen from 25 m above the vehicle (the trail and where the vehicle is);
- rig_check.json: the numbers below.

Checks printed (and required for "CHECK OK"): the LiDAR's height above the ground measured from
its own scan (about 1.2 m, as on RELLIS), no movement while settling, enough points, and the
median distance between our projection and MAVS's (under 2 pixels).

Usage (from the repo root; needs MAVS, see README):
    python scripts/sim_check_rig.py                       # scene outputs/sim/scenes/trail01
    python scripts/sim_check_rig.py --scene trail02 --start 60
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

from offroad.geometry.projection import project_points_to_image  # noqa: E402
from offroad.mapping.elevation import lidar_to_flu  # noqa: E402
from offroad.sim.mavs_world import MavsWorld, TrailSceneSpec, generate_trail_scene  # noqa: E402
from offroad.sim.rig import RigSpec, camera_calibration  # noqa: E402
from offroad.viz.overlay import draw_depth_overlay  # noqa: E402

SCENES_DIR = REPO_ROOT / "outputs" / "sim" / "scenes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="trail01", help="scene name under outputs/sim/scenes")
    parser.add_argument("--start", type=int, default=30, help="trail waypoint to start at (~0.5 m apart)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "sim" / "rig_check")
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    return parser.parse_args()


def nearest_distances(a: np.ndarray, b: np.ndarray, chunk: int = 500) -> np.ndarray:
    """For each row of a (N, 2), the distance to the nearest row of b (M, 2)."""
    out = [np.linalg.norm(a[i:i + chunk, None, :] - b[None, :, :], axis=2).min(axis=1)
           for i in range(0, len(a), chunk)]
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    args = parse_args()
    scene_json = args.scenes_dir / args.scene / f"{args.scene}_scene.json"
    if not scene_json.exists():
        print(f"Generating {scene_json.parent} (MAVS grows the forest; a few minutes)...", flush=True)
        started = time.perf_counter()
        generate_trail_scene(scene_json.parent, args.scene, TrailSceneSpec())
        print(f"  done in {time.perf_counter() - started:.0f} s", flush=True)

    rig = RigSpec()
    started = time.perf_counter()
    world = MavsWorld(scene_json, rig)
    if len(world.trail) < args.start + 11:
        sys.exit(f"The trail has only {len(world.trail)} waypoints; use a smaller --start")
    p0, p1 = world.trail[args.start], world.trail[args.start + 10]
    yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    world.place(p0[0], p0[1], yaw)
    capture = world.capture()
    print(f"Loaded, placed and captured in {time.perf_counter() - started:.0f} s", flush=True)

    # MAVS's own projection of the same scan: the pixels it marks on its camera image.
    world.camera.AddLidarPointsToImage(world.lidar)
    marked = np.ascontiguousarray(world.camera.GetNumpyArray()).astype(int)
    painted = np.abs(marked - capture.image.astype(int)).sum(axis=2) > 60
    rows, cols = np.nonzero(painted)
    calib = camera_calibration(rig)
    h, w = capture.image.shape[:2]
    uv, depth, _ = project_points_to_image(capture.points_lidar, calib.T_cam_from_lidar, calib.K, (h, w))
    offsets_px = nearest_distances(np.column_stack([cols, rows]).astype(float), uv)
    projection_px = float(np.median(offsets_px)) if len(offsets_px) else float("nan")

    flu = lidar_to_flu(capture.points_lidar)
    ring = np.hypot(flu[:, 0], flu[:, 1])
    near_ground = (ring > 2.5) & (ring < 6.0)
    lidar_height = -float(np.percentile(flu[near_ground, 2], 20)) if near_ground.any() else float("nan")
    moved = float(np.hypot(*(capture.position[:2] - p0)))
    labels, counts = np.unique(capture.labels, return_counts=True)
    checks = {
        "scene": args.scene, "start_waypoint": args.start, "trail_waypoints": len(world.trail),
        "cg_height_m": round(world.cg_height_m, 3),
        "lidar_height_from_scan_m": round(lidar_height, 3), "lidar_height_wanted_m": rig.lidar_height_m,
        "moved_while_settling_m": round(moved, 2),
        "points": int(len(flu)), "points_ahead_share": round(float((flu[:, 0] > 0).mean()), 3),
        "labels": {int(k): int(v) for k, v in zip(labels, counts)},
        "image_size": list(capture.image.shape[:2][::-1]),
        "projection_vs_mavs_median_px": round(projection_px, 2),
    }
    print(json.dumps(checks, indent=2))
    ok = (abs(lidar_height - rig.lidar_height_m) < 0.15 and moved < 0.5 and len(flu) > 1000
          and projection_px < 2.0)
    print("CHECK", "OK" if ok else "FAILED: LiDAR height, settling, point count or alignment is off (see above)")

    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "camera.png"), capture.image[:, :, ::-1])
    overlay = draw_depth_overlay(capture.image, uv, depth, max_depth=30.0, radius=1)
    cv2.imwrite(str(args.out / "lidar_on_camera.png"), overlay[:, :, ::-1])
    compare = capture.image.copy()
    compare[painted] = (255, 0, 0)
    for u, v in np.round(uv).astype(int):
        cv2.circle(compare, (int(u), int(v)), 1, (255, 255, 0), -1)
    cv2.imwrite(str(args.out / "projection_check.png"), compare[:, :, ::-1])

    above = world.mavs.MavsCamera()
    above.Initialize(800, 800, 0.0035, 0.0035, 0.0035)
    above.SetGammaAndGain(0.6, 1.0)
    look_down = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
    above.SetPose([capture.position[0], capture.position[1], capture.position[2] + 25.0], look_down)
    above.Update(world.env, 0.1)
    cv2.imwrite(str(args.out / "above.png"), np.ascontiguousarray(above.GetNumpyArray())[:, :, ::-1])
    (args.out / "rig_check.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(f"Saved camera.png, lidar_on_camera.png, projection_check.png, above.png and rig_check.json to {args.out}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
