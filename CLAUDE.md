# CLAUDE.md

Context for Claude Code working in this repository (branch `approach-2-closed-loop`).

## Project overview

Graduate course project for CS5230 (Connected and Autonomous Vehicles) at Cal Poly Pomona. Team of four.

**Research question:** can an onboard vision-language model (VLM) guide an off-road vehicle without a map?

Branches:
- **Approach 1** (`approach-1-vlm-grid-costmap`, done): the VLM labels grid cells to re-weight a LiDAR cost map for A\*.
- **Approach 2, phases 1-5** (`approach-2-mapless-vln`, done 2026-11-24): mapless "keep going" navigation after **DriveVLN** (Guo et al., CVPR 2026) on RELLIS-3D. A LiDAR cost map + A\* fan proposes ≤ 4 candidate paths; they are drawn in colors with distance ticks on the camera image; Qwen3-VL-4B picks one (`choice`), gives `safe_until` per candidate (ticks 5/10/15 m) and `action` continue/stop. Mode A "keep going", Mode B + a coarse direction word. That branch's CLAUDE.md has the full phase-by-phase log.
- **Approach 2, phase 6** (this branch, started 2026-11-24): **closed loop in MAVS**. **Read `docs/approach-2-plan.md` (phase 6 is at the end of section 5) before working.**

## Working rules (from the user)

- **The user runs all scripts and tests themselves.** Write the code, then give the exact commands and work from the output they report. (Quick self-checks of new code before handing over are fine; don't write into `outputs/` on the user's behalf. Use the scratchpad.)
- **The user commits and pushes.** Suggest commit messages; don't commit unless asked.
- **One-command runner:** `python scripts/run_all.py` runs pytest plus every script and prints a pass/fail summary (`--full` adds slow steps, `--tests-only`, `--only <steps>`, `--list`). **When adding a new script, add a `Step` to `STEPS` in `scripts/run_all.py`**.
- **Finish the current phase (or step) before moving on**, including the user's own runs, reviews and choices. Don't make the user's decisions.
- **The user is new to this area.** Explain concepts in plain language, and what a step is for before doing it. When they ask only for an explanation or opinion, don't change anything.

## Approach 2 results to keep in mind (phases 1-5, RELLIS-3D test drive 00004)

- Reported model: **Qwen3-VL-4B QLoRA, "skip" data (no-match frames left out), epoch 4** (adapter `outputs/adapters/nomatch-skip/epoch4`, not in git). Asked **without** the reasoning field.
- Driver match (Mode A): fine-tuned 66% [61, 71], zero-shot 4B 44%, straightest 67%, cheapest 48%, random 43%, oracle 89%. Turn frames: 57% vs straightest 43%. ADE 0.98 m vs straightest 1.07 m. On the twistier val drive the model beats straightest (56 vs 44%, p < 0.001).
- Limitations: after fine-tuning the model **ignores the direction word** (forced word changes 5% of choices; zero-shot 89%) and the history, and says "safe all" for every path; RELLIS has almost no unsafe or stop situations, so safety and stop couldn't be measured offline. That is what closed loop should test.
- Closing decisions: no GRPO, no 8B fine-tuning; the general-ability check on non-RELLIS scenes moved to phase 6.

## Phase 6 status

- **2026-11-24, user decisions:** simulator **MAVS** (built on the user's PC: `C:\Users\talha\source\repos\MAVS`; use `MAVS_ROOT`, never hardcode); this branch was **cleaned of the RELLIS pipeline and training code** (user: "RELLIS + training"; removed: `data/rellis.py`, `nav/frames.py`, `nav/frame_candidates.py`, `nav/sft.py`, `nav/evaluation.py`, `nav/run_files.py`, `nav/candidate_metrics.py`, `vlm/finetune.py`, `eval/`, `planning/baseline.py`, `viz/baseline_figure.py`, their scripts and tests, `configs/`, the committed `outputs/`; all still on `approach-2-mapless-vln`; `unsafe_cells` moved into `nav/labels.py`); **rig matches RELLIS** (Warthog, OS1, RELLIS camera field of view); **generated trail scenes**; **about one week** for phase 6 (steps 6.1-6.3 + a small run set).
- The partner's PDF is a LiDAR-to-camera projection check; the video was a MAVS recording (chase-camera view).
- **MAVS findings (2026-11-24, my checks in the scratchpad):**
  - `mavs_interface` imports in our conda env (ctypes over `install/bin/mavs.dll`), so MAVS, the candidate pipeline and the VLM run in one process.
  - Most vegetation meshes are **missing** from the user's MAVS data (`USA_Conifers`, `Europe`, `Groundcover`, ...); only `forest_simple.json` has all its meshes. **One missing mesh in a scene breaks `GetSurfaceHeight`** (returns -FLT_MAX), and then the vehicle's physics gives NaN / an access violation. `mavs_world.drop_missing_meshes` cleans generated scenes.
  - **The brake drives backwards from standstill** (Warthog and MRZR4): settle with no pedals; brake only while moving forward; we use our own path follower, not `MavsVehicleController` (its braking would reverse).
  - **Throttle works like an accelerator held down** (speed keeps rising): speed feedback in `sim/control.py`.
  - **MAVS's Warthog is unusable** (heading ~15° off its motion, even on bare ground; very sensitive skid steering). **User decision 2026-11-24: MRZR4 with the RELLIS sensors.** MRZR4: curvature = 0.105 1/m × steering command (+ = left), follower stays 0.07–0.32 m from test paths at ~1.4 m/s.
  - **Plants make the vehicle physics 30× slower** (7.5 vs 0.26 s per sim s): `MavsWorld` drives the vehicle in `ground_env` (scene without objects, `<name>_ground.json`) and renders sensors in `env` (full scene).
- **Step 6.1 written and checked in my scratchpad** (not run by the user yet): `sim/rig.py` (`RigSpec` with the MRZR4, `camera_K`, `mavs_camera_args`, `sensor_offsets`, `camera_calibration`, `to_lidar_axes`, `yaw_from_quaternion`), `sim/mavs_world.py` (`import_mavs`, `TrailSceneSpec`, `generate_trail_scene`, `drop_missing_meshes`, `ground_only_scene`, `MavsWorld` with `place`, `step`, `pose`, `heading_and_speed`, `capture` → `Capture`), `sim/control.py` (pure pursuit + speed control, MRZR4 gains), `scripts/sim_check_rig.py` (`run_all.py` step `sim_rig`). My check: CHECK OK, LiDAR 1.11 m above ground (from its scan), projection 0.73 px from MAVS's own, ~48.5k points, 4 s. Tests `test_sim_rig.py` (5), `test_sim_control.py` (4); suite 143.

## What this branch keeps

All pure functions unless noted; frames and shapes are in the docstrings.
- `geometry/`: `transform_points(T_b_from_a, pts)`, `project_points_to_image(points_lidar, T_cam_from_lidar, K, (H, W))` → `(uv, depth, kept)`; transform helpers. `data/calibration.py`: `CameraCalibration(K, dist_coeffs, T_cam_from_lidar)`.
- `mapping/elevation.py`: `T_FLU_FROM_LIDAR = diag(-1, -1, 1, 1)` (the RELLIS Ouster's axes; MAVS points are converted to them by `sim.rig.to_lidar_axes`), `GridSpec` (x −5..35 m, y −15..15 m, 0.25 m), `lidar_to_flu`, `build_elevation_map`, `majority_label_grid`. `mapping/costmap.py`: `CostParams`, `geometric_cost_map`, `estimate_ground`, `inflate`, `fill_ground`.
- `planning/astar.py`: `astar`, `shortest_path_tree`, `path_from_tree`, `cost_to_come`, `choose_goal`.
- `nav/candidates.py` (relaxed / strict cost, `generate_candidates`, `CHOSEN_VARIANT = "relaxed"`), `nav/labels.py` (tick labels, `cells_along`, `unsafe_cells`), `nav/driven_path.py` (path helpers, `direction_bearing_deg`, `direction_word`), `nav/visual_prompt.py` (`draw_visual_prompt`, `PICTURE_VERSION = 2`), `nav/prompts.py` (`build_prompt(..., reasoning, wording)`, `parse_answer`), `nav/scoring.py`, `nav/selector.py` (`prepare_selector_input`).
- `vlm/backend.py`: `TransformersVlm(model_id, quantize="4bit", adapter=None)`; `vlm/runner.py`.
- `viz/overlay.py`, `viz/topdown.py`.

## Environment

- Windows: conda env `%USERPROFILE%\miniconda3\envs\vlm-offroad-nav` (Python 3.13). The `python` on PATH is a separate 3.15 install with no packages; use the conda env. torch 2.11.0+cu128, transformers 5.22.1, peft 0.22.0, bitsandbytes installed.
- GPU: RTX 5060 Laptop, 8 GB. Qwen3-VL-4B 4-bit answers in ~1-5 s.
- Polaris (CPP HPC): H200 GPUs; used for phase 5 training and runs (conda env with Python 3.13).
- Git: remote `origin` = github.com/talhaahmed884/VLM-Offroad-Nav. Teammate Antonio Duran-Ramos also pushes.

## Tech stack and conventions

- Python 3.13; NumPy, OpenCV, Matplotlib, PyYAML, pytest; PyTorch + Transformers + bitsandbytes + PEFT for the VLM; MAVS (own checkout, ctypes) for the simulator.
- Type hints and short docstrings; state array shapes and coordinate frames in docstrings. Keep geometry, control and evaluation functions pure (arrays in, arrays out); everything that touches MAVS lives in `sim/mavs_world.py`. Name transforms by direction (`T_cam_from_lidar`). Small, readable functions.
- Pin package versions so the teammates match.

## Key references

- Guo et al., "DriveVLN: Towards Mapless Vision-and-Language Navigation in Autonomous Driving", CVPR 2026.
- Jiang et al., "RELLIS-3D Dataset: Data, Benchmarks and Analysis" (arXiv 2011.12954).
- MAVS: Mississippi State University Autonomous Vehicle Simulator (Goodin et al.), https://github.com/Mississippi-State-University-OTM/MAVS.
