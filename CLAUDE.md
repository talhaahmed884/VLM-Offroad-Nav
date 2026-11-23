# CLAUDE.md

Context for Claude Code working in this repository (branch `approach-2-mapless-vln`).

## Project overview

Graduate course project for CS5230 (Connected and Autonomous Vehicles) at Cal Poly Pomona. Team of four.

**Research question:** can an onboard vision-language model (VLM) guide an off-road vehicle without a map?

The repository has two approaches on separate branches:
- **Approach 1** (branch `approach-1-vlm-grid-costmap`, done): the VLM labels grid cells on the camera image to re-weight a LiDAR cost map for A\*. Its full code, phase-by-phase log (in that branch's CLAUDE.md), results, report figures, and `docs/limitations.md` live on that branch. Summary below.
- **Approach 2** (this branch, **planning done, implementation not started**): mapless "keep going" navigation adapted from **DriveVLN** (Guo et al., CVPR 2026). A geometric planner proposes up to 4 drivable candidate paths; they are drawn in colors with distance ticks on the camera image; the VLM (Qwen3-VL-4B, zero-shot and QLoRA fine-tuned) picks one, says how far each is safe, and whether to stop. **The full plan, decisions, and settings are in `docs/approach-2-plan.md`. Read it before working on approach 2.**

## Working rules (from the user)

- **The user runs all scripts and tests themselves.** Write the code, then give the exact commands and work from the output they report. (Quick self-checks of new code before handing over are fine; don't write into `outputs/` on the user's behalf. Use the scratchpad.)
- **The user commits and pushes.** Suggest commit messages; don't commit unless asked.
- **One-command runner:** `python scripts/run_all.py` runs pytest plus every script on a sample frame and prints a pass/fail summary (`--full` adds slow steps, `--tests-only`, `--only <steps>`, `--list`). **When adding a new script, add a `Step` to `STEPS` in `scripts/run_all.py`**, and prefer telling the user to run `run_all.py` (or `--only <step>`).
- **Finish the current phase before moving on.** A phase is done only when every task is complete, including the user's own runs, reviews, and choices. Don't start the next phase or make the user's decisions for them unless they explicitly say so.
- **The user is new to this area.** Explain concepts in plain language, and explain what a step is for before doing it. When they ask only for an explanation or opinion, don't change anything.

## Approach 2 status (2026-11-22)

- Plan written and all decisions made (see `docs/approach-2-plan.md`, section 6): offline open-loop evaluation on RELLIS (no simulator); **no destinations**, "keep going until stopped" (Mode A) and "keep going + a coarse direction" (Mode B), both evaluated; every answer = `choice` (a color) + `safe_until` per candidate (ticks at 5/10/15 m) + `action` continue/stop; candidates from the cost map + A\* fan (curved anchors optional later); Qwen3-VL-4B with QLoRA on the laptop (DriveVLN used Qwen2.5-VL-3B), 8B decided later; 13 smaller settings accepted as defaults (split train 00000–00002 / val 00003 / test 00004, ~2 Hz frames, ≤ 4 candidates of ~15 m, etc.).
- 2026-11-22: the branch was cleaned at the user's request ("this branch will only have the code for the new approach"). The user chose to keep the shared building blocks below and delete everything approach-1-specific (grid prompting, cost re-weighting, hybrid planner, the 36-frame set, their scripts and tests, `docs/limitations.md`, and the committed `outputs/`, which are now gitignored again). `fill_ground` moved from the deleted `fusion/grid_lookup.py` into `mapping/costmap.py`. `requirements-vlm.txt` gained `peft==0.22.0` and `trl==1.18.2` (checked compatible with the pinned torch/transformers/accelerate; not installed yet). README rewritten for approach 2, with the new poses download.
- **Next: the user downloads the "Ouster LiDAR Scan Poses files" (174 MB, one `poses.txt` per sequence) and commits the cleanup. Then phase 1 (poses loader, driven-path check, decision frames).** Before writing the loader, inspect the actual files (format, line count = scan count, frame convention), as was done for the scans in approach 1. Don't start phase 1 until the user says so.

## What this branch keeps (shared building blocks)

All pure functions unless noted; frames and shapes are in the docstrings.
- `data/rellis.py`: `RellisSequence(seq, root=None)`, `seq[i]` → `RellisFrame(image RGB, points (N,4) with empty returns dropped, point_labels uint16, label_image or None, calib)`; `frame_indices`, `image_path`, `label_image_path`, `has_label_image`; `load_ontology(root)` → `(id_to_name, id_to_color)`; `default_rellis_root()`. `data/calibration.py`: `CameraCalibration(K, dist_coeffs, T_cam_from_lidar)`.
- `geometry/`: `transform_points(T_b_from_a, pts)`, `project_points_to_image(points_lidar, T_cam_from_lidar, K, (H, W))` → `(uv, depth, kept)`; quaternion / transform helpers.
- `mapping/elevation.py`: `T_FLU_FROM_LIDAR = diag(-1, -1, 1, 1)` (forward-left-up frame at the LiDAR), `GridSpec` (x −5..35 m, y −15..15 m, 0.25 m cells; `cell_index`, `cell_center`, `shape`), `lidar_to_flu`, `build_elevation_map`, `majority_label_grid`.
- `mapping/costmap.py`: `CostParams` (max_step 0.3, vehicle_height 1.8, ground windows 1.5 / 6.0 m, hidden-ground threshold 0.5, roughness weight 4, unknown cost 3, inflation 0.75), `geometric_cost_map` → `CostMap(cost, ground_z, obstacle_height, obstacle, unknown)`, `drivable_cost`, `estimate_ground` (two-scale, handles slopes), `inflate`, `fill_ground`, `LETHAL = inf`.
- `planning/astar.py`: `astar(cost, start, goal)` (8-connected, no corner cutting), `cost_to_come` (Dijkstra), `choose_goal`, `clear_footprint`, `path_length`. `planning/baseline.py`: `run_geometric_baseline(points, labels, ...)` and `plan_on_cost(cost_grid, costmap, labels, ...)` → `BaselineResult` (`found`, `shortfall_m`, `reached`, `length_m`, `detour_ratio`, `path_xy()`, `class_shares()`, `mean_roughness_m`, `hard_obstacle_distance(ids)`).
- `vlm/backend.py`: `TransformersVlm(model_id="Qwen/Qwen3-VL-4B-Instruct", quantize="4bit"|"8bit"|"none", max_new_tokens=512, quantize_vision=False)`, `.generate(image_rgb, prompt, max_new_tokens=None)` → `VlmReply(text, seconds, prompt_tokens, new_tokens)`, `.peak_memory_gb()`; torch is imported only inside the class. `vlm/runner.py`: `downscale(image, max_side)`, `strip_thinking(text)`.
- `eval/path_truth.py`: two-source ground truth along a path (LiDAR majority label, else camera label at the cell's ground pixel, else unseen; occlusion check with LiDAR depth) → `PathTruth(labels, source)`. `eval/metrics.py`: class lists (water, mud, compliant / non-compliant vegetation, hard obstacles), `hard_obstacle_xy`, `distance_to_obstacles`, `goal_is_blocked`, `path_roughness`, `compute_path_metrics` → `PathMetrics` (success = reached, or goal blocked and stopped within 2.5 m).
- `viz/overlay.py` (depth-colored points), `viz/topdown.py` (top-down maps, legends, `show_cost_values`), `viz/baseline_figure.py` (`draw_path_on_camera(image, calib, result, color, radius)`, `draw_baseline_figure`).
- Scripts: `project_lidar.py`, `build_elevation_map.py`, `build_cost_map.py`, `plan_path.py`, `check_gpu.py`, `run_all.py`. Tests for all of the above (no dataset needed).

## Approach 1 in brief (for context and comparison)

- Pipeline: gridded camera image (8 × 12 cells) → VLM JSON hints (terrain + preferred/ok/caution/avoid/impassable) and a cell route → map cells ↔ grid cells by projecting ground points → cost multipliers (avoid ×5, impassable = blocked; low grass/bush may be unblocked) + a ×0.8 route corridor → same A\* as the geometry-only baseline.
- Final results on 36 frames (baseline / 4B / 8B / oracle hints from labels): success 30 / 30 / 31 / 32; water on path in water frames 17% / 16% / 11% / 8%; mud in mud frames 32% / 34% / 31% / 11%; bush frames successful 4 / 4 / 5 / 7 of 8; closest hard obstacle 1.55 / 1.44 / 1.47 / 1.43 m; geometry 0.06–0.09 s per frame, VLM 12.5–13.8 s per answer.
- **Lessons that matter for approach 2:**
  - Small zero-shot VLMs are weak at naming grid cells and planning routes. The 8B described only ~12% of the nearby ground, and its routes were often straight lines or went through cells it had itself marked "avoid". Picking among drawn paths should be easier; that is the reason for approach 2.
  - **Prompt pitfalls:** small models copy concrete examples (use placeholders) and fall into endless lists (cap them, put the important field first).
  - "VALID" (well-formed JSON) ≠ correct; score answers against labels.
  - **The camera can't see the first ~4.4 m** in front of the vehicle (below the image), and far image rows cover huge strips of ground.
  - RELLIS ground rises ~10% on many trails; the two-scale ground estimate in the cost map fixes the resulting false obstacles.
  - RELLIS labels plain trail ruts as "mud", puddle labels are coarse, rubble only appears far away, and there are no close rocky slopes.

## Environment

- Windows: conda env `%USERPROFILE%\miniconda3\envs\vlm-offroad-nav` (Python 3.13.17). The `python` on PATH is a separate 3.15 install with no packages; use the conda env. `requirements.txt` versions are installed. From approach 1: torch 2.11.0+cu128, transformers, accelerate, and bitsandbytes are installed; `check_gpu.py` passed. `peft` and `trl` are not installed yet.
- GPU: RTX 5060 Laptop, 8 GB (compute capability 12.0). Qwen3-VL-4B / 8B in 4-bit run locally (~10–14 s per answer at 1280 px). QLoRA of the 4B should fit with batch size 1, gradient checkpointing, and 800×600 images; fine-tuning an 8B needs the Polaris H200.
- Polaris (CPP HPC): H200 GPUs (~141 GB); a GPU is only visible inside `srun/sbatch -p gpu --gres=gpu:1`; the `basic` QOS gives 4 GB RAM and 1 GPU per job (`student-advanced`: 16 GB). The user's Polaris environment was deleted on 2026-11-03.
- Git: remote `origin` = github.com/talhaahmed884/VLM-Offroad-Nav. Teammate Antonio Duran-Ramos also pushes (approach 1 PR "obj4").

## Dataset: RELLIS-3D (verified on disk, Windows machine)

- Source: https://github.com/unmannedlab/RELLIS-3D. The data lives in `data/Rellis-3D/` (gitignored) or wherever `RELLIS_ROOT` points. Never hardcode absolute paths.
- **Layout:** `0000N/{pylon_camera_node, pylon_camera_node_label_id, os1_cloud_node_kitti_bin, os1_cloud_node_semantickitti_label_id}` plus `camera_info.txt` and `transforms.yaml` in each sequence folder; the ontology is in `Rellis_3D_ontology/`. **`poses.txt` per sequence is the new download for approach 2** (the RELLIS README calls it "file containing the poses of every scan"; format and frame not documented, so verify on download. A 2021 changelog entry says wrong poses were fixed).
- **Counts:** images == scans == per-point labels per sequence (00000: 2847, 00001: 2319, 00002: 4147, 00003: 2184, 00004: 2059), at about 10 Hz. Label images cover about every other frame.
- **Matching is by frame index:** `frame000123-<timestamp>.jpg` ↔ `000123.bin` ↔ `000123.label`. Ignore the stray `.jpg` in `00000/pylon_camera_node_label_id/`.
- **Images:** 1920 × 1200. **Intrinsics:** `fx fy cx cy` = 2813.64 2808.33 969.29 624.05, the same for all sequences, no distortion.
- **Extrinsics:** `transforms.yaml` key `os1_cloud_node-pylon_camera_node`, quaternion + translation, stored as `T_lidar_from_cam` (so it is inverted); they differ per sequence (00000/00001/00003 share one).
- **Ouster scans:** 131072 points × (x, y, z, intensity) float32; ~40% are empty (0,0,0) returns, dropped. LiDAR forward is **−x**, left is −y.
- **Labels:** uint32 per point (class IDs); ontology of 20 classes (`ontology.yaml`: [id→name, id→RGB]). Read names from the ontology; never hardcode IDs.

## Tech stack and conventions

- Python 3.13; NumPy, OpenCV, Matplotlib, PyYAML, pytest; PyTorch + Transformers + Accelerate + bitsandbytes + PEFT + TRL for the VLM and fine-tuning (`requirements-vlm.txt`). No ROS.
- Type hints and short docstrings; state array shapes and coordinate frames in docstrings. Keep geometry and evaluation functions pure (arrays in, arrays out, no file I/O). Name transforms by direction (`T_cam_from_lidar`). Small, readable functions; this is a team of students.
- Pin package versions so the teammates and the cluster match.

## Key references

- Guo et al., "DriveVLN: Towards Mapless Vision-and-Language Navigation in Autonomous Driving", CVPR 2026 (the basis of approach 2).
- Jiang et al., "RELLIS-3D Dataset: Data, Benchmarks and Analysis" (arXiv 2011.12954).
- Liao et al., "DiffusionDrive" (CVPR 2025), DriveVLN's planning module; Dauner et al., "NAVSIM" (NeurIPS 2024), the open-loop protocol DriveVLN aligns with.
- Shao et al., GRPO (DeepSeekMath, arXiv 2402.03300).
