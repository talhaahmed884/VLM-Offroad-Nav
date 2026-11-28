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
  - 2026-11-26: the missing vegetation folders (`Groundcover`, `USA_Southwest`, `USA_East`, `Europe`, `Basic`, `USA_Conifers`) are **not in MAVS's public GitHub repo either** (main has the same 11 plant folders as the user's checkout); re-downloading can't fix it. **User decision: keep `forest_simple.json`** (no custom ecosystem). MAVS has no rock or log meshes; obstacles can be road blockades, walls, cubes, cylinders.
  - Most vegetation meshes are **missing** from the user's MAVS data (`USA_Conifers`, `Europe`, `Groundcover`, ...); only `forest_simple.json` has all its meshes. **One missing mesh in a scene breaks `GetSurfaceHeight`** (returns -FLT_MAX), and then the vehicle's physics gives NaN / an access violation. `mavs_world.drop_missing_meshes` cleans generated scenes.
  - **The brake drives backwards from standstill** (Warthog and MRZR4): settle with no pedals; brake only while moving forward; we use our own path follower, not `MavsVehicleController` (its braking would reverse).
  - **Throttle works like an accelerator held down** (speed keeps rising): speed feedback in `sim/control.py`.
  - **MAVS's Warthog is unusable** (heading ~15° off its motion, even on bare ground; very sensitive skid steering). **User decision 2026-11-24: MRZR4 with the RELLIS sensors.** MRZR4: curvature = 0.105 1/m × steering command (+ = left), follower stays 0.07–0.32 m from test paths at ~1.4 m/s.
  - **Plants make the vehicle physics 30× slower** (7.5 vs 0.26 s per sim s): `MavsWorld` drives the vehicle in `ground_env` (scene without objects, `<name>_ground.json`) and renders sensors in `env` (full scene).
- **Step 6.1 written and checked in my scratchpad** (not run by the user yet): `sim/rig.py` (`RigSpec` with the MRZR4, `camera_K`, `mavs_camera_args`, `sensor_offsets`, `camera_calibration`, `to_lidar_axes`, `yaw_from_quaternion`), `sim/mavs_world.py` (`import_mavs`, `TrailSceneSpec`, `generate_trail_scene`, `drop_missing_meshes`, `ground_only_scene`, `MavsWorld` with `place`, `step`, `pose`, `heading_and_speed`, `capture` → `Capture`), `sim/control.py` (pure pursuit + speed control, MRZR4 gains), `scripts/sim_check_rig.py` (`run_all.py` step `sim_rig`). My check: CHECK OK, LiDAR 1.11 m above ground (from its scan), projection 0.73 px from MAVS's own, ~48.5k points, 4 s. Tests `test_sim_rig.py` (5), `test_sim_control.py` (4); suite 143.
- **Verified 2026-11-26 (the user ran `sim_rig`): step 6.1 is done.** Output identical to my run (scenes are reproducible), CHECK OK, images look right. `trail01` is straight; turns need other scenes (6.4). Next: step 6.2 (candidates and the picture on MAVS frames) when the user says so.
- **2026-11-26: step 6.2 written and checked in my scratchpad** (the user said "Start 6.2"; not run by the user yet): `sim/planning.py` (`plan_frame`, `trail_ahead_flu`, `trail_direction_word`, `straightest_index`, `cheapest_index`), `frame_seed` accepts scene names (CRC-32; RELLIS numbers unchanged), `scripts/sim_show_candidates.py` (`run_all.py` step `sim_candidates`, `--ask zeroshot finetuned`). Tests `test_sim_planning.py` (4) + 1 in `test_visual_prompt.py`; suite 148. My run on `trail01` (6 frames): 2 candidates each (along the trail / bending into grass); both models 100% valid; fine-tuned picks the trail path 6/6 and ticks it "all", the other "tick 2"; zero-shot 5/6, over-cautious; 3-4 s per answer on the laptop.
- 2026-11-26, the user ran `sim_candidates`: top views identical to mine, but the camera pictures differed (MAVS's sun follows the wall clock, clouds random) and 3/24 answers changed. Fixed: `MavsWorld(date=(2026, 6, 21), hour=12, cloud_cover=0.0)` → `SetDate`, `SetTimeSeconds(hour, 0, 0)`, no clouds; renders are now identical between runs.
- **Verified 2026-11-26 (the user reran `sim_candidates`): step 6.2 is done.** 12/12 pictures byte-identical and 24/24 answers identical to my reference run. Next: step 6.3 (the driving loop) when the user says so.
- **2026-11-26: step 6.3 written and checked in my scratchpad** (the user said "Start 6.3"; not run by the user yet): `sim/metrics.py`, `sim/choosers.py` (straightest / cheapest / `VlmChooser`), `sim/loop.py` (`LoopParams`, `run_episode`; outcomes finished / stopped / off_trail / stuck / time_limit), `scripts/sim_drive.py` (`run_all.py` step `sim_drive`), `tests/test_sim_loop.py` (kinematic stand-in world); suite 154. Fixes found by driving: candidates shifted onto the vehicle (`centred_on_vehicle`: grid cell centres were 12.5 cm left → drift), no-candidate decisions keep the previous path ≤ 2 times (`max_hold`), overhanging foliage ignored (`without_overhangs`: an obstacle needs a point 0.3–1.0 m above ground; MAVS canopy at 1.5–1.8 m blocked the trail). First run trail01 / straightest / A: finished 193 of 197 m, 272 decisions, mean offset 0.69 m, max 1.40 m, ~3 min wall. 6.2's outputs are now slightly out of date.
- **Verified 2026-11-26 (the user ran `sim_drive`): step 6.3 is done.** 272/272 decisions and 28/28 pictures identical to my run (the loop incl. MAVS physics is reproducible). Next: step 6.4 (the scene set: curved trails, a fork, a blocked trail) when the user says so.
- **2026-11-26: step 6.4 in progress** (the user said "Start 6.4"; offered to rebuild MAVS: not needed). User decisions: scenes curves / fork / blocked / halfblocked; collisions from object positions; **keep the RELLIS camera, gentler scenes** (bend radius 40 m, fork arcs to 25°, half-wall over the left third). Code: `sim/scenes.py` (designs, `Blockade`, `vprp_text`, `near_any`), `mavs_world.build_designed_scene`, `scene_obstacles`, `MavsWorld(route=...)` reads `<name>_design.json`, `metrics.footprint_gaps`, loop outcome "collision" + `min_gap_m`, `scripts/sim_build_scenes.py` (`run_all.py` step `sim_scenes`), `sim_drive.py --route`. Walls: blockade scaled to 1.8 m (Scale [1, 1, 2.25]), "blocked" = 4 walls. **Bug fixed: `place()` twice didn't move the vehicle → 6.2's six frames were one spot; 6.2 must be rerun.** Tests `test_sim_scenes.py`; suite 161.
- 2026-11-26, more fixes found by driving the new scenes (all in `sim/`): **`pull_string`** straightens candidates (grid paths ran straight 9 m and turned late, so the vehicle never turned into bends; the picture shows the straightened path; `centred_on_vehicle` only when `straighten=False`), **`OracleChooser`** (candidate closest to the route; reference), **`SIM_COST_PARAMS` inflation 1.0 m** (MRZR4 is 1.6 m wide), **candidate ends must be in view**, **halfblocked route swerves 1.2 m round the wall** (`scenes.swerve`). Final scratchpad check: oracle finishes trail01 / curves / both fork routes / halfblocked and stops at the full wall; straightest fails curves and both forks. Suite 163. 6.2 corrected (six real places): fine-tuned picks the straightest 6/6 and says "all" for every path in 4/6; zero-shot 4/6, over-cautious. **The user has not run 6.4 yet and must rerun 6.2 and 6.3.**
- **Verified 2026-11-27: steps 6.2 (rerun), 6.3 (rerun) and 6.4 are done** (the user ran them; every picture, answer and decision identical to my runs). Next: step 6.5 (all choosers × scenes × modes + an evaluation table) when the user says so.
- **2026-11-27: step 6.5 written** (the user: "Start 6.5. Don't include the video recording yet. We'll do it later."; recordings later, as a replay of saved runs, chase view with the MAVS MRZR4 mesh + the VLM picture; the user asked about Polaris: I recommended the laptop). Code: `sim/runs.py` (`run_name`, `run_and_save`, `save_trajectory_figure`; `sim_drive.py` uses it), `sim/evaluation.py` (`planned_runs`: word-blind straightest / cheapest / oracle in Mode A only, VLMs both modes, fork per route = 42 runs; `is_success`, `fork_success` (Mode A: either branch), `decision_stats`, `run_row`, `chooser_summary`), `scripts/sim_run_all.py` (resumable, each VLM loaded once), `scripts/sim_evaluate.py` (→ `outputs/sim/evaluation.md`); `run_all.py` steps `sim_runs`, `sim_eval`. Tests `test_sim_evaluation.py`; suite 167.
- 2026-11-27, the user's batch crashed at run 22/39 (`trail01_zeroshot_modeB`, ~176 m): the trail's last 20 m give no direction word and Mode B's prompt needs one. Fix in `sim/loop.py`: the last word stays in force ("straight" if a run starts within 20 m of the end); test `test_mode_b_keeps_the_last_word_near_the_trail_end`; suite 168. The 21 finished runs (all Mode A) are unaffected; the batch resumes.
- **Verified 2026-11-28: the user's 6.5 batch is done (42 runs, `outputs/sim/evaluation.md`).** Tests passed, Mode A / B: straightest 3/5, 3/6; cheapest 5/5, 5/6; oracle 5/5, 6/6; zero-shot 4/5, 5/6 (fails curves: drifts out of the right-hand bend); fine-tuned 5/5, 5/6 (fails fork-right: identical choices in A and B, ignores the word; zero-shot follows it). No collisions; 0 'stop' answers; blocked stops are the planner's (no candidates, 8-10 m before the wall); half-wall gaps 0.06 (fine-tuned) to 0.25 m, oracle 0.80 m. Summary in the plan doc (6.5 status). Still open (user's choice): the video recordings, the 'metres in the bushes' measure, repeats for spread, the non-RELLIS general-ability check.

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
