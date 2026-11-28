# VLM-Offroad-Nav: closed-loop tests in MAVS

CS5230 (Connected and Autonomous Vehicles) course project at Cal Poly Pomona.

Can a small vision-language model (VLM) guide an off-road vehicle without a map? This branch
drives the approach-2 path selector **in closed loop** in the
[MAVS](https://github.com/Mississippi-State-University-OTM/MAVS) off-road simulator: the vehicle's
own LiDAR proposes candidate paths, the VLM picks one from the camera image, the vehicle drives
it, and the next decision comes from wherever it ended up.

## Branches

| Branch | What | Status |
|---|---|---|
| [`approach-1-vlm-grid-costmap`](../../tree/approach-1-vlm-grid-costmap) | The VLM labels a grid on the camera image to re-weight a LiDAR cost map for A\*; evaluation on RELLIS-3D. | done |
| [`approach-2-mapless-vln`](../../tree/approach-2-mapless-vln) | Phases 1-5 of approach 2 (after DriveVLN, CVPR 2026) on RELLIS-3D: candidate paths, the picture and prompts, zero-shot and QLoRA fine-tuned Qwen3-VL-4B, the offline evaluation and its results. | done |
| `approach-2-closed-loop` (this branch) | Phase 6: closed loop in MAVS. Keeps only what the loop needs from approach 2. | in progress |

The full plan, all decisions and the phase-by-phase results are in
[`docs/approach-2-plan.md`](docs/approach-2-plan.md) (phase 6 at the end of section 5).

## Setup

```bash
conda create -n vlm-offroad-nav python=3.13
conda activate vlm-offroad-nav
pip install -r requirements.txt
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-vlm.txt
python scripts/check_gpu.py     # checks PyTorch sees the GPU and 4-bit layers work
```

**MAVS** is built separately (see its
[build instructions](https://github.com/Mississippi-State-University-OTM/MAVS/blob/master/docs/MavsBuildInstructions.md)).
Our code loads its Python API from the MAVS checkout: set `MAVS_ROOT` to that folder (default
`~/source/repos/MAVS`). MAVS's API calls `mavs.dll` through ctypes, so it runs in the same conda
environment as the VLM. Notes from setting it up (2026-11-24):

- Most of MAVS's tree meshes (`data/scenes/meshes/vegetation/USA_Conifers`, `Europe`,
  `Groundcover`, ...) are not in the checkout, so we use the `forest_simple.json` ecosystem, whose
  plants all exist. Scenes are cleaned of objects with missing meshes, because one missing mesh
  breaks MAVS's ground-height query and with it the vehicle's tires.
- In this MAVS build the **brake drives backwards from standstill**, so the vehicle settles with
  no pedals and our controller only brakes while moving forward.
- The simulated vehicle is MAVS's **MRZR4** with RELLIS-like sensors (LiDAR 1.2 m above the
  ground, RELLIS's camera view). MAVS's Warthog (the RELLIS vehicle) reports a heading ~15 degrees
  off its direction of travel, so it can't follow a path.
- Plants make MAVS's vehicle physics 30x slower, so the vehicle drives on a copy of the scene
  without plants while the camera and LiDAR see the full forest.

The fine-tuned adapter (`outputs/adapters/nomatch-skip/epoch4`, from the approach-2 branch) is
not in git; copy it into `outputs/adapters/` to use it.

## Run

```bash
python scripts/run_all.py             # unit tests (no MAVS or GPU needed)
python scripts/run_all.py --full      # also the GPU check and the MAVS steps
```

| Step | Script | What |
|---|---|---|
| 6.1 | `scripts/sim_check_rig.py` | generates a trail scene, places the MRZR4 with a RELLIS-like LiDAR and camera, captures one frame, and saves the image, the LiDAR projected on it, a comparison with MAVS's own projection (calibration check), and a view from above |
| 6.2 | `scripts/sim_show_candidates.py` | stands at 6 places on the trail, builds the cost map and the candidate paths from the simulated LiDAR, draws the VLM's picture, and (with `--ask zeroshot finetuned`) asks both models in Modes A and B |
| 6.3 | `scripts/sim_drive.py --chooser straightest\|cheapest\|zeroshot\|finetuned --mode A\|B` | one closed-loop run: a decision every 0.5 s of simulated time, the chosen path driven by the path follower, until the trail's end, a stop, leaving the trail, getting stuck, or the time limit; saves every decision, a summary and a trajectory plot |
| 6.4 | `scripts/sim_build_scenes.py` | builds the test scenes from one generated forest: S-curves, a fork (routes `left` / `right`), a trail blocked by a wall, a trail half-blocked by a wall; checks each (routes, candidates at the start, wall height and which candidates get past it) and saves views from above |
| 6.5 | `scripts/sim_run_all.py`, `scripts/sim_evaluate.py` | every chooser (straightest, cheapest, the oracle reference, zero-shot and fine-tuned Qwen3-VL-4B) on every scene and route, in Modes A and B (resumable), then the tables in `outputs/sim/evaluation.md`: tests passed, collisions, leaving the trail, wrong stops, completion, distance from the route, who stopped at the wall, what the choosers did |

## Layout

```
src/offroad/
  geometry/      projection and transforms
  data/          CameraCalibration
  mapping/       elevation map and LiDAR cost map
  planning/      A* and the shortest-path tree
  nav/           candidates, labels, the selector's picture and prompts, answer checking
  vlm/           Qwen3-VL backend (4-bit, optional LoRA adapter)
  sim/           phase 6: rig (sensor setup), mavs_world (MAVS wrapper), control (path follower)
  viz/           image overlays, top-down maps
scripts/         run_all.py, check_gpu.py, sim_*.py
tests/           unit tests (no dataset, GPU or MAVS needed)
docs/            approach-2-plan.md
```
