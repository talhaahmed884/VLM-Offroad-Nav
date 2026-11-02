"""Show how the VLM's cost hints and route change the cost map for one frame (phase 5, steps 2-3).

Uses the answer saved by scripts/run_vlm.py for this frame. Four panels: the camera image
with the VLM's hints and route, the geometry-only cost map, the re-weighted (hybrid) cost
map, and what changed (green = cheaper, orange = more expensive, blue = unblocked by the
VLM, red = blocked by the VLM, dark gray = blocked in both). The VLM's route, traced onto
the ground, is the white line; cells along it are slightly cheaper. It also prints how many
map cells each kind of change affected and where the waypoints landed.

Usage (from the repo root):
    python scripts/show_fused_costmap.py --sequence 1 --frame 2132
    python scripts/show_fused_costmap.py --sequence 0 --frame 120 --run Qwen3-VL-4B-Instruct_4bit
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file only; also works without a display (e.g. on the cluster)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence  # noqa: E402
from offroad.fusion.grid_lookup import build_grid_lookup  # noqa: E402
from offroad.fusion.reweight import FusionParams, reweight_cost_map  # noqa: E402
from offroad.mapping.costmap import CostParams, geometric_cost_map  # noqa: E402
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu  # noqa: E402
from offroad.viz.grid_overlay import draw_grid  # noqa: E402
from offroad.viz.topdown import draw_camera_fov, show_cost_values, show_top_down  # noqa: E402
from offroad.viz.vlm_answer import draw_hints, draw_route  # noqa: E402
from offroad.vlm.grid import ImageGrid  # noqa: E402

DEFAULT_RUN = "Qwen3-VL-8B-Instruct_4bit"
CHANGE_COLORS = {  # name -> RGB
    "unchanged": (1.0, 1.0, 1.0),
    "cheaper": (0.2, 0.7, 0.2),
    "more expensive": (1.0, 0.6, 0.0),
    "unblocked by VLM": (0.1, 0.3, 1.0),
    "blocked by VLM": (0.9, 0.0, 0.0),
    "blocked in both": (0.3, 0.3, 0.3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=int, default=0, help="sequence number, 0-4 (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="frame index within the sequence (default: 0)")
    parser.add_argument("--run", default=DEFAULT_RUN, help=f"VLM run folder in outputs/vlm_runs (default: {DEFAULT_RUN})")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: outputs/fused_<seq>_<frame>.png)")
    return parser.parse_args()


def load_answer(run: str, sequence: str, frame: int) -> dict:
    """The saved answer for this frame (an evaluation-set answer if there is one, else a one-off try)."""
    run_dir = REPO_ROOT / "outputs" / "vlm_runs" / run
    matches = sorted(run_dir.glob(f"*_{sequence}_{frame:06d}.json"), key=lambda p: p.name.startswith("single"))
    if not matches:
        sys.exit(f"No saved answer for {sequence}/{frame} in {run_dir}. Run scripts/run_vlm.py for this frame first.")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def change_categories(base: np.ndarray, fused: np.ndarray) -> np.ndarray:
    """(nx, ny, 3) RGB picture of how each cell's cost changed."""
    base_blocked, fused_blocked = np.isinf(base), np.isinf(fused)
    both_finite = ~base_blocked & ~fused_blocked
    rgb = np.full((*base.shape, 3), CHANGE_COLORS["unchanged"])
    rgb[both_finite & (fused < base)] = CHANGE_COLORS["cheaper"]
    rgb[both_finite & (fused > base)] = CHANGE_COLORS["more expensive"]
    rgb[base_blocked & ~fused_blocked] = CHANGE_COLORS["unblocked by VLM"]
    rgb[~base_blocked & fused_blocked] = CHANGE_COLORS["blocked by VLM"]
    rgb[base_blocked & fused_blocked] = CHANGE_COLORS["blocked in both"]
    return rgb


def main() -> None:
    args = parse_args()
    seq = RellisSequence(args.sequence, root=args.root)
    frame = seq[args.frame]
    record = load_answer(args.run, seq.sequence, args.frame)
    spec, cost_params, fusion = GridSpec(), CostParams(), FusionParams()

    points_flu = lidar_to_flu(frame.points)
    costmap = geometric_cost_map(points_flu, build_elevation_map(points_flu, spec), cost_params)
    grid = ImageGrid(height=frame.image.shape[0], width=frame.image.shape[1])
    lookup = build_grid_lookup(costmap.ground_z, spec, frame.calib.T_cam_from_lidar, frame.calib.K, grid)
    answer = record["answer"]
    fused = reweight_cost_map(costmap, lookup, answer["hints"], fusion, cost_params, answer["waypoints"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    camera = draw_hints(draw_grid(frame.image, grid), grid, answer["hints"])
    axes[0, 0].imshow(draw_route(camera, grid, answer["waypoints"], record["goal_cell"], answer.get("route_conflicts", [])))
    axes[0, 0].set_title(f"VLM hints ({args.run}), sequence {seq.sequence} frame {args.frame}")
    axes[0, 0].axis("off")

    show_cost_values(axes[0, 1], fig, costmap.cost, spec, "geometry-only cost map")
    show_cost_values(axes[1, 0], fig, fused.cost, spec, f"with VLM hints (up to {fusion.max_hint_distance:.0f} m)")
    show_top_down(axes[1, 1], change_categories(costmap.cost, fused.cost), spec, "what the VLM changed")
    handles = [Rectangle((0, 0), 1, 1, facecolor=rgb, edgecolor="gray") for rgb in CHANGE_COLORS.values()]
    axes[1, 1].legend(handles, list(CHANGE_COLORS), loc="upper right", fontsize=8)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        draw_camera_fov(ax, frame.calib.K, grid.width, spec)
    if fused.route_xy:
        route = np.array([(0.0, 0.0), *fused.route_xy])
        for ax in (axes[1, 0], axes[1, 1]):
            ax.plot(route[:, 1], route[:, 0], "w-o", linewidth=2, markersize=4, markeredgecolor="k")

    fig.tight_layout()
    out = args.out or REPO_ROOT / "outputs" / f"fused_{seq.sequence}_{args.frame:06d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)

    base, new = costmap.cost, fused.cost
    finite = np.isfinite(base) & np.isfinite(new)
    print(f"VLM answer: {len(answer['hints'])} hinted grid cells "
          f"-> {fused.hints.hinted.sum()} map cells within {fusion.max_hint_distance:.0f} m")
    on_ground = ", ".join(f"{x:.1f} m ahead / {y:+.1f} m left" for x, y in fused.route_xy) or "none"
    print(f"VLM route {' -> '.join(answer['waypoints'])}: on the ground at {on_ground}")
    print(f"  {fused.corridor.sum()} map cells on the route corridor (cost x{fusion.corridor_factor})")
    print(f"  cheaper:          {(finite & (new < base)).sum():5d} map cells")
    print(f"  more expensive:   {(finite & (new > base)).sum():5d}")
    print(f"  unblocked by VLM: {(np.isinf(base) & np.isfinite(new)).sum():5d}  "
          f"({fused.unblocked.sum()} obstacle cells judged to be low grass/bush, plus their safety margin)")
    print(f"  blocked by VLM:   {(np.isfinite(base) & np.isinf(new)).sum():5d}  "
          f"({fused.impassable.sum()} cells marked impassable, plus their safety margin)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
