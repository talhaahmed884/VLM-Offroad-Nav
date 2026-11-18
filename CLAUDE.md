# CLAUDE.md

Context for Claude Code working in this repository.

## Project overview

Graduate course project for CS5230 (Connected and Autonomous Vehicles) at Cal Poly Pomona. Team of four.

**Research question:** Can an onboard Vision-Language Model (VLM) act as an asynchronous, high-level guide for a low-level geometric path planner in off-road driving, by producing rough waypoints and traversability cost modifiers?

**Motivation:** Pure geometric perception fails off-road. Tall, penetrable bushes get flagged as solid obstacles (unnecessary detours), while mud pits and shallow water look flat and traversable (vehicle gets stuck). A VLM can add semantic, contextual reasoning about terrain that geometry alone misses.

## Project objectives (full scope)

1. Build an offline evaluation pipeline that synchronizes RGB images with LiDAR and calibration data from an off-road benchmark (RELLIS-3D).
2. Overlay a structured coordinate grid on camera frames so a VLM outputs 2D image-plane waypoints and semantic cost weights in a parseable format.
3. Project 2D VLM waypoints into 3D by ray-tracing onto a LiDAR-derived digital elevation model (DEM), then use them to re-weight costs in a path planner.
4. Compare the hybrid VLM approach against a pure geometric planner on traversability, safety, and latency.

**Evaluation metrics (later):** mean distance to hard obstacles, % of path through compliant vs. non-compliant vegetation, % of path through mud/water, average path roughness, end-to-end compute time per frame.

## Working rules (from the user)

- **The user runs all scripts and tests themselves.** Write the code, then give the exact commands and work from the output they report.
- **One-command runner (added 2026-10-31 at the user's request):** `python scripts/run_all.py` runs pytest plus every script on a sample frame and prints a pass/fail summary. `--full` adds the slow steps (label stats, candidates, the baseline on all frames, the grid on all frames); `--tests-only`, `--only <steps>`, and `--list` are also available. **When adding a new script, add a `Step` to `STEPS` in `scripts/run_all.py`**, and prefer telling the user to run `run_all.py` (or `--only <step>`) over listing many commands.
- **Finish the current phase before moving on.** A phase is done only when every task is complete, including the user's own steps such as runs, reviews, and choices. Don't start the next phase, and don't make the user's decisions for them, unless they explicitly say so.

## PROJECT STATUS (2026-11-18): all six phases are done; the user closed this approach

The user ran `make_report.py` (`outputs/report/` exists) and said on 2026-11-18 they want to finish this approach here and move on to another one. All planned work from the proposal is complete. Optional follow-ups that were never done: (A) better VLM hint coverage, (B) limiting "impassable" on far grid cells, (C) bigger models on the Polaris H200, (D) an open-loop time replay (needs the scan poses), (E) fine-tuning (an open question, never decided). Phase 6 code was uncommitted at that point; the user commits.

## PHASE 6 (done 2026-11-18): evaluation

Objective 4: compare the hybrid (VLM-guided) planner against the geometry-only baseline on traversability, safety, and latency, on the 36 evaluation frames, using RELLIS-3D labels as ground truth. Conditions: baseline, hybrid with the 4B answers, hybrid with the 8B answers (main), and hybrid with oracle hints (upper bound). Everything runs offline from saved answers; no GPU needed.

**Milestone:** final per-frame and per-category tables plus figures for the report, with every metric from the proposal and a short limitations list.

### Phase 6 tasks (proposed 2026-11-12; refine with the user)
1. **Two-source ground truth along a path** (user decision 2026-10-31): for each path cell use the LiDAR point labels if it has points, else the camera label image at the cell's projected ground pixel, else "unseen". Report how much of each path each source covers. Pure + tests, plus a coverage check over the 36 baseline paths.
2. **Metrics per path** (definitions need the user's OK): % water (water + puddle) and % mud; % compliant vegetation (grass, bush) vs non-compliant (tree, log); mean and minimum distance to hard obstacles; roughness along the path; reached / shortfall / detour / length; and a "goal blocked" flag for frames where stopping short is correct (e.g. a person or obstacle at the target), so shortfall isn't counted as failure there. Timing: VLM seconds (saved), fusion + planning seconds (measured).
3. **Evaluation script** over all conditions → per-frame CSV, per-category tables, and per-frame wins/losses vs baseline (n = 36 is small, so report counts and paired differences, not just means).
4. **Report figures and tables** (bar charts per category, a few example frames) and a limitations list (blind zone, coarse far cells, coarse labels, tuning on the eval frames, offline/no time replay).

### Phase 6 progress
- 2026-11-12: phase 6 started at the user's request ("Yes, start phase 6"), which also closes phase 5 with option C (A and B stay possible later).
- 2026-11-12: **task 1 (two-source path ground truth) is written. The user has not run it yet.**
  - `src/offroad/eval/path_truth.py` (pure): `SOURCE_UNSEEN/LIDAR/CAMERA/OCCLUDED`, `PathTruth(labels, source)` with `source_shares()` and `class_shares()`. `lidar_depth_image(points, T, K, size, block=4)` gives the nearest LiDAR depth per 4×4 px block. `project_cells(cells, ground_z, spec, T, K, size)` → uv and depth of each cell's ground point. `path_truth(path, lidar_labels, ground_z, spec, T, K, label_image, points_lidar=None, ignore_ids=(0,))`: LiDAR majority label first; else the camera label at the ground pixel (ground gaps filled with `fill_ground`); void/sky labels are ignored; it is occluded when a LiDAR point in that block is closer than the ground point by more than max(1 m, 10% of the depth).
  - `scripts/check_path_truth.py [--sequence S --frame F]`: prints per-frame source coverage of the baseline path and water/mud/bush shares from LiDAR alone vs both sources. Single-frame mode saves `outputs/path_truth_<seq>_<frame>.png` (dots: green LiDAR, blue camera, orange occluded, gray unseen). `run_all.py` steps: `truth` (00000/120, default) and `truth-eval` (all frames, slow).
  - `tests/test_path_truth.py`: 10 tests. The suite should be 197.
  - **My run on the 36 baseline paths (print only):** lidar 61%, camera 19%, occluded 0.4%, **unseen down from about 39% to 20%**. The rest is mostly the first ~5 m in front of the vehicle, below the camera's view. Two-source shares vs LiDAR only: water 4% → 4%, mud 8% → 11% (mud category 24% → 32%), **bush 1% → 3% (bush category 4% → 12%**; 2132 3% → 22%, 1261 0% → 20%). So LiDAR labels alone undercount bush on the path. Caveat: the camera shows front surfaces and the 64-beam LiDAR is sparse, so some "bush" may be bush in front of the ground. Only 0–5% of cells were caught as occluded.
  - Next, task 2 needs the user's OK on the metric definitions (proposed in the task list above).
- **2026-11-18: teammate commits found on main** (Antonio Duran-Ramos, PR #1 "obj4", Nov 9–16, written on Polaris): `BaselineResult.mean_roughness_m` (mean cost-map obstacle height along the path), a hard-obstacle distance function using the LiDAR majority-label grid, `HybridResult.baseline_seconds` / `hybrid_plan_seconds` (fields only, never filled), and `run_hybrid.py` columns `hard_dist_mean_m`, `hard_dist_min_m`, `roughness_m`. Also a LaTeX report skeleton that was added and then removed again (`report/`). **Two bugs made `run_hybrid.py` crash on main**: `hard_obstacle_distance` was defined at module level instead of as a `BaselineResult` method, and `mean_roughness_m` read `costmap.obstacle_heights` (the field is `obstacle_height`). I fixed both minimally (moved the function into the class; fixed the name) and told the user; their logic is unchanged.
- **2026-11-18: the user approved the proposed metric definitions ("Yes, use them"). Task 2 (metrics) is written; the user has not run it yet.**
  - `src/offroad/eval/metrics.py` (pure): class lists `WATER_CLASSES` (water, puddle), `MUD_CLASSES`, `COMPLIANT_VEGETATION` (grass, bush), `NON_COMPLIANT_VEGETATION` (tree, log), `HARD_OBSTACLE_CLASSES` (tree, pole, fence, barrier, building, vehicle, person, log, object, rubble); `GOAL_BLOCKED_RADIUS_M = 1.5`; `VEHICLE_IGNORE_RADIUS_M = 2.0`.
    - `hard_obstacle_xy(points, labels, hard_ids, ground_z, spec)`: hard-labelled LiDAR points at most vehicle_height (1.8 m) above the local ground, not on the vehicle, inside the map.
    - `distance_to_obstacles(path_xy, obstacle_xy)` → (mean, min), chunked. `goal_is_blocked(target, obstacles)`.
    - `path_roughness(path, ground_z, obstacle_height, spec)` → (slope in cm per m along the path, mean bump height in m).
    - `compute_path_metrics(plan, truth, obstacle_xy, id_to_name)` → `PathMetrics` (found, reached, goal_blocked, shortfall, length, detour, shares of water / mud / compliant / non-compliant / unseen from the two-source truth, hard distance mean/min, slope, bump height; `success` = reached or (found and goal_blocked)).
  - `tests/test_metrics.py`: 10 tests, including a regression test for the two teammate bugs. The suite should be 207.
  - **Checked on 4 frames (scratch script, baseline → 8B hybrid):** 120 water 24% → 15%, hard clearance mean 2.3 → 2.7 m; 158 water 29% → 9% (mud 16% → 24%), min clearance 0.6 → 1.2 m; 2132 unchanged (goal_blocked = true: hard points within 1.5 m of the target); 1060 unchanged. Slope is about 7–11 cm/m on these trails (it mostly measures the uphill incline, not bumpiness); bump height is about 5–9 cm. Min clearance is about 1 m, as expected from the 0.75 m inflation plus the cell size.
  - **Verified 2026-11-18 (the user ran it): tasks 1 and 2 are done.** `run_all.py`: 207 tests passed and all 9 default steps were OK, including `truth`. In `path_truth_00000_000120.png` the green LiDAR dots run through the right puddle and the blue camera dots sit on the far trail at about 15–20 m, matching the label image. `run_hybrid.py --oracle` runs again with the teammate's columns (hard m 5.13 → 5.18, rough m 0.06 → 0.06), and its results match my earlier oracle run exactly (reached 25 → 28, bush 3 → 7, water 16% → 8%, mud 24% → 9%).
- **2026-11-18: task 3 (evaluation script) is written; the user approved starting it. The user has not run it yet.**
  - `planning/hybrid.py`: `run_hybrid_planner` now fills the teammate's `baseline_seconds` (maps + baseline plan) and `hybrid_plan_seconds` (lookup + re-weighting + hybrid plan).
  - `src/offroad/eval/compare.py` (pure): `evaluate_frame(points, labels, T, K, grid, label_image, answers={name: (answer, vlm_seconds)}, id_to_name, ...)` → `{"baseline": ConditionResult, name: ...}`. All conditions share the maps, the lookup, and the hard-obstacle points. `ConditionResult(name, plan, truth, metrics, compute_seconds, vlm_seconds, separation)`. `paired_comparison(base, other, lower_is_better, tolerance)` → wins / losses / ties / frames / mean_diff.
  - **Success rule tightened (my change; told the user):** success = reached, or goal blocked **and** shortfall ≤ `BLOCKED_STOP_MAX_M` = 2.5 m (blocked radius 1.5 + reached tolerance 1.0). Without the limit, "goal blocked" was true in 11/36 frames (often a tree within 1.5 m of the target), and stops 3.4–4.4 m short counted as successes.
  - `scripts/evaluate.py [--runs ...] [--no-oracle] [--out-dir]` → `outputs/eval/per_frame.csv` (frame × condition, every metric) and `summary.csv` (category × condition means), plus printed overall, per-category, and paired wins/losses tables. Conditions default to baseline, VLM-4B_4bit, VLM-8B_4bit, and oracle. A missing answer = no hints. Added to `run_all.py` as the slow step `evaluate`.
  - `tests/test_compare.py`: 6 tests. The suite should be 213.
  - **My trial run (scratchpad output):**

    | | baseline | 4B | 8B | oracle |
    |---|---|---|---|---|
    | success (of 36) | 30 | 30 | 31 | 32 |
    | reached | 25 | 26 | 26 | 28 |
    | water on path (all / water cat.) | 4% / 17% | 4% / 16% | 3% / **11%** | 2% / 8% |
    | mud on path (mud cat.) | 32% | 34% | 31% | **11%** |
    | bush cat. success | 4/8 | 4/8 | 5/8 | **7/8** |
    | compliant veg on path | 62% | 60% | 63% | 70% |
    | hard obstacle min distance (mean over frames) | 1.55 m | 1.44 | 1.47 | 1.43 |
    | compute per frame | 0.06 s | 0.08 | 0.08 | 0.08 |
    | VLM answer | – | 12.5 s | 13.8 s | – |

    Paired against the baseline: 8B water 7 wins / 3 losses / 26 ties; success +1 / −0. 4B is all ties except for small shifts (success +0 / −0). Oracle: success +5 / −3 (it loses easy 1060 and others to far-cell over-blocking), mud 14 / 1 / 21, shortfall 9 / 3 / 24. **Safety trade-off:** the hybrids go slightly closer to hard obstacles (min distance 1.55 → 1.47 m for 8B; 4 wins / 7 losses), e.g. hugging edges to avoid water. Every hybrid keeps ≥ about 1 m clearance on average, because of the 0.75 m inflation.
  - **Verified 2026-11-18: the user ran `evaluate.py`.** `outputs/eval/per_frame.csv` (144 rows = 36 frames × 4 conditions) and `summary.csv` match my trial run exactly in every metric except the timing (mean compute per frame: baseline 0.057 s, hybrids 0.082–0.086 s). Task 3 is done once the user confirms the tightened success rule.
- **2026-11-18: the user confirmed the 2.5 m success rule ("Yes, start step 4"). Task 3 is done. Task 4 (report tables, figures, limitations) is written; the user has not run it yet.**
  - `src/offroad/eval/report_tables.py` (pure): `parse_row` (per_frame.csv strings → bools/floats/None), `select`, `mean_of`, `overall_table`, `category_table(metric, kind)` (with an "All" row), `paired_table(rows, conditions, baseline=...)`, `markdown_table`, `latex_escape`, `latex_table` (booktabs).
  - `src/offroad/viz/report_charts.py`: colors by condition (baseline gray `#898781`; 8B blue `#2a78d6`, oracle orange `#eb6834`, 4B aqua `#1baf7a`: slots 1–3 of the dataviz skill's validated default palette; the validator script couldn't run because Node.js is not installed). `display_name` ("Hybrid 8B" …), `by_category_figure` (2×3 small multiples: success, water, mud, compliant veg, shortfall, hard min), `paired_figure` (diverging better/worse bars per hybrid and metric), `latency_figure` (log axis, plain "0.1 s" ticks), and `example_figure` (camera with nested path dots + top-down labels with paths). `draw_path_on_camera` gained a `radius` parameter.
  - `scripts/make_report.py [--results] [--out-dir] [--no-examples]` → `outputs/report/{tables.md, tables.tex, fig_by_category.png, fig_paired.png, fig_latency.png, examples/<stem>.png}`. Example frames: water 158, water 120, bush 935, bush 2132, easy 1060 (re-planned with the baseline, 8B, and oracle). Figures are saved with `bbox_inches="tight"`. Added to `run_all.py` as the slow step `report`.
  - `docs/limitations.md` (committed doc): 16 limitations in 4 groups (setup/scope, VLM, fusion design, ground truth/metrics), with numbers.
  - `tests/test_report.py`: 8 tests (tables, names/colors, a chart smoke test). The suite should be 221.
  - I checked every figure by eye in the scratchpad and fixed the layout: the legend overlapped a label in the paired figure, and the example frames had clipped titles and too much white space.
  - **Correction:** earlier notes said the oracle's 120 path "stops 2.5 m short because the person stands at the target". That was wrong. The person is at about 13–14 m, and the nearest person point is 5.7 m from the target (goal_blocked = false). The oracle stops short because its "impassable" person hint on a coarse far D-row cell blocks a long strip of ground (the over-blocking limitation).

## PHASE 5 (done 2026-11-12): hybrid planner, VLM hints on the cost map

**Closed 2026-11-12** when the user chose to start phase 6 (option C). Final results: see "Oracle test" below (8B: reached 25 → 26, water on path 16% → 10%; oracle: 25 → 28, bush reached 3 → 7).

Objective 3: put the VLM's answers (cost hints and waypoints, in image grid cells) onto the top-down LiDAR map, re-weight the geometric cost map with them, and plan with the same A* as the baseline. Everything runs offline from the saved phase 4 answers (main model: `outputs/vlm_runs/Qwen3-VL-8B-Instruct_4bit/`), so no GPU is needed. Do not build the evaluation (phase 6) beyond what is needed to check the hybrid paths.

**Milestone:** for every evaluation frame, a hybrid path planned on the VLM-adjusted cost map, saved next to the baseline path in a picture, plus a summary table like `run_baseline.py`'s.

### Phase 5 tasks (proposed 2026-11-01; refine with the user)
1. Image grid cell ↔ map cell lookup: project each top-down map cell's ground point (cost map ground height, with gaps filled) into the camera image and record which grid cell it falls in. This is the forward version of "ray-tracing pixels onto the DEM": it gives the same pixel ↔ ground mapping without ray marching. Cells outside the camera's view get no VLM information. Pure + tests, plus a picture of the grid painted onto the top-down map.
2. Cost re-weighting: traversability → cost multiplier, and the rule for when the VLM may lower a LiDAR "obstacle" cell (the tall-grass/bush case). **Needs the user's decisions** (see the open questions).
3. Waypoints: put each waypoint cell on the map (the centroid of the map cells it covers) and give a small cost discount along the route corridor, as soft guidance.
4. `run_hybrid.py`: hybrid vs baseline on all 36 frames (pictures + summary.csv), using the same goal choice, footprint clearing, and A*.

### Phase 5 progress
- 2026-11-01: **task 1 (grid cell ↔ map cell lookup) is written. The user has not run it yet.**
  - `src/offroad/fusion/grid_lookup.py` (pure, new package `fusion`): `fill_ground(ground_z, max_steps=40)` (grows known ground into NaN gaps, mean of known 8-neighbors per step; 40 steps = 10 m), `project_map_cells(ground_z, spec, T_cam_from_lidar, K, image_size)` → (nx, ny, 2) uv, NaN if not in view; `grid_cell_ids(uv, grid)` → (nx, ny) int `row*cols+col`, `NO_CELL = -1`; `GridLookup(spec, grid, cell_id, uv, ground_filled)` with `visible`, `id_of`, `label_of`, `mask(label)`, `centroid(label)` (FLU x, y of the footprint, for waypoints in task 3); `build_grid_lookup(ground_z, spec, T, K, grid)`. It uses `CostMap.ground_z`.
  - `scripts/show_grid_on_map.py --sequence S --frame F [--out]` → `outputs/gridmap_<seq>_<frame>.png` (camera + grid | map painted by grid cell, one color per row | cost map with magenta grid-cell borders), plus the share of map cells in view and the forward range per grid row. Added to `run_all.py` as the default (non-slow) step `gridmap` on 00000/120.
  - `tests/test_grid_lookup.py`: 11 tests (filling, pinhole formula, behind/too-close/unknown not in view, NO_CELL, F7 straight ahead at 20 m, near = lower row, left = lower column, centroid, filled flag). The suite should be 153.
  - **Checked on 00000/120 (picture in my scratchpad only):** 34% of map cells are in view, 2049 of the 6543 visible cells use filled-in ground. Forward range per row: H 4.4–5.6 m, G 5.1–7.4, F 6.1–10.1, E 7.9–17.6, D 10.9–34.9, C 18.9–34.9, B 26–32 (a few cells), A none. Findings:
    - (1) **Nothing closer than about 4.4 m gets VLM information** (below the image).
    - (2) **Rows near the horizon cover huge ground areas.** C and D have about 2,800 map cells each, H only 50. That is normal perspective (near the horizon a small pixel step is a large distance step), and the slope uphill pushes far ground into rows C/D. So a single hint on a D cell would re-weight a strip up to ~24 m deep. **Step 2 should limit how far hints apply** (e.g. only up to the 20 m goal, or weight far rows less).
    - (3) Far footprints interleave a little (C/D islands at 20–35 m) because of noisy and filled ground estimates there.
    - (4) The footprint sits slightly right of the dashed camera-edge lines. Those lines come from the old `draw_camera_fov` helper, which ignores the camera's small yaw and offset; the lookup itself uses the full calibration, which was verified in phase 1. The left side at 25–35 m is also cut off where the ground is unknown beyond the 10 m fill.

- **Verified 2026-11-01: the user ran `run_all.py` (153 tests passed) and `show_grid_on_map.py` on 00000/120 and 00001/2132; both look right to the user and to me. Task 1 is done.** In 2132 (the tall-grass frame), rows E–H cover about 4.5–15 m, which is exactly the LiDAR-blocked "grass wall" in the cost map. So vegetation hints on E–H cells would land on the cells the hypothesis is about. The road on the left of the image (D1–D4) lands at about 25–28 m on the left. Far ground (C/D, 15–35 m) is mostly filled in, because the tall grass hides it from the LiDAR.
- Proposed defaults for task 2 (awaiting the user's OK): (a) the VLM may unblock only vegetation hints (`short_grass`/`tall_grass_or_bush`, preferred/ok/caution) where the obstacle height is ≤ 1.5 m; those cells become rough-drivable (base cost 1 + roughness_weight = 5, times the level multiplier), and the remaining obstacles are re-inflated. (b) Multipliers: preferred ×0.7, ok ×1, caution ×2, avoid ×5, impassable = blocked. (c) Hints apply only to map cells closer than 25 m (the 20 m goal + 5 m).

- **2026-11-01: the user approved the proposed defaults ("use what you recommend"). Task 2 (cost re-weighting) is written; the user has not run it yet.**
  - `costmap.py`: new `drivable_cost(obstacle_height, params)` (1 + roughness_weight·min(h/max_step, 1); unknown → unknown_cost), now used by `geometric_cost_map`. Behavior is unchanged.
  - `src/offroad/fusion/reweight.py` (pure): `FusionParams(multipliers={preferred 0.7, ok 1, caution 2, avoid 5}, unblock_terrains=(short_grass, tall_grass_or_bush), unblock_levels=(preferred, ok, caution), max_unblock_height=1.5, max_hint_distance=25)`. `paint_hints(lookup, hints, max_distance)` → `HintMap(level, terrain)` (indices, `NO_HINT`). `reweight_cost_map(costmap, lookup, hints, params, cost_params)` → `FusedCostMap(cost, hints, unblocked, impassable)`. It starts from `drivable_cost`, multiplies hinted cells, unblocks low vegetation obstacles, adds impassable cells as obstacles, then re-inflates. With no hints the result equals the baseline exactly (tested).
  - `scripts/show_fused_costmap.py --sequence S --frame F [--run NAME]` → `outputs/fused_<seq>_<frame>.png` (hints | geometric cost | fused cost, log color scale | change map) plus counts. Added to `run_all.py` as the slow step `fused` (00001/2132), because it needs saved VLM answers.
  - `tests/test_reweight.py`: 16 tests. The suite should be 169.
  - **Checked on two frames with the 8B answers (pictures in my scratchpad only):**
    - **00001/2132 (tall grass):** the VLM says "drive straight through the tall grass" with E5–E8 and F5–F8 tall_grass_or_bush ok. **107 obstacle cells were unblocked, but they form an island at about 7–12 m** inside the LiDAR grass wall. 0–4.4 m is below the camera's view, and the VLM gave no hints for rows G/H (which are also grass), so the planner can't reach the island. It also misread the road on the left (D1–D4) as a "fence line", other_obstacle avoid, raising the cost of the road at 18–25 m. The tree hints (A9–A12) are in the sky row, so they have no effect.
    - **00000/120 (puddles):** F5–F12 water avoid, G5–G12 and H1–H8 mud: 163 map cells at about 5–9 m straight ahead became more expensive (the right puddle and the ridge). The left puddle (F1–G4) was not hinted. The person is not hinted but is blocked by LiDAR anyway.
  - **Open design question raised (for the user):** how to handle the blind zone between the 2 m footprint clearing and the first visible ground (~4.4 m), plus near rows the VLM didn't hint. Options: (1) strict, the VLM only affects what it sees (the island problem stays and is reported); (2) carry the nearest visible vegetation hint straight down to the blind zone, for low obstacles only (a live system would have seen that ground a moment earlier); (3) use the previous frames' answers and the scan poses (not downloaded; bigger work).
  - **User decision (2026-11-01): leave it as it is (option 1, strict).** The VLM only affects ground it can see; no code change. The tall-grass "island" effect is a known limitation to report. The user also asked about running a "simulation". I explained that the project is an offline per-frame test bench (Objective 1); an open-loop time replay (needs the scan poses, world-frame hints, and VLM runs on many frames) is a possible later extension; and a closed-loop simulator is out of scope. No decision on that yet.
- **2026-11-01: task 3 (waypoints as soft guidance) is written. The user has not run it, or task 2, yet.**
  - `GridLookup.ground_point(label)`: the FLU (x, y) of the map cell whose ground projects closest to the grid cell's center pixel (a one-pixel ray trace onto the DEM); None for cells with no ground (the sky). Accuracy is limited by the 0.25 m cells: at about 19 m it can be about 0.5 m off in distance (one cell moves ~37 px sideways but only ~3 px vertically).
  - `src/offroad/fusion/waypoints.py` (pure): `waypoint_ground_points(lookup, waypoints)` (skips the sky), `distance_to_polyline(spec, points)`, `route_corridor(spec, points, half_width, max_distance)` (the polyline starts at the vehicle (0, 0) and stops at max_distance).
  - `FusionParams` gains `corridor_half_width = 1.0` m and `corridor_factor = 0.8`. `reweight_cost_map(..., waypoints=())` multiplies the corridor before blocking, so it never unblocks. `FusedCostMap` gains `route_xy` and `corridor`. With no hints and no waypoints the result still equals the baseline.
  - `show_fused_costmap.py` now draws the route on the camera panel and on the map panels, and prints where each waypoint landed and how many cells are in the corridor.
  - Tests: `tests/test_waypoints.py` (7) plus 1 `ground_point` test in `test_grid_lookup.py`. The suite should be 177.
  - **Checked (scratchpad pictures):**
    - 2132: route H6→…→D6 lands at 4.4, 5.1, 6.9, 9.6, 19.1 m straight ahead; 722 corridor cells, 175 of them cheaper. The unblocked island is unchanged, since the corridor doesn't unblock.
    - 120: route H6→H5→H4→G4→F4→E4→D4→D6 lands 5–17 m ahead, drifting up to 1.9 m left, so it **nudges toward the un-hinted left puddle** (a VLM route error the corridor passes on, softly). 673 corridor cells.
    - Near the vehicle one grid column is only ~0.3 m wide on the ground, so the waypoints can barely steer sideways there.
- **2026-11-01: task 4 (hybrid vs baseline on all 36 frames) is written. The user has not run it (or tasks 2–3) yet.**
  - `planning/baseline.py`: new `plan_on_cost(cost_grid, costmap, labels, params, goal_distance, footprint_radius, goal_cost_weight)` → `BaselineResult`; `run_geometric_baseline` now calls it (no behavior change). `BaselineResult` is used for both plans.
  - `src/offroad/planning/hybrid.py` (pure): `run_hybrid_planner(points_lidar, point_labels, T_cam_from_lidar, K, grid, answer, spec, params, fusion, goal_distance, footprint_radius, goal_cost_weight)` → `HybridResult(baseline, hybrid, fused, lookup)` with `path_changed`. It shares the elevation map, cost map, and labels, so the only difference between the two plans is the cost grid.
  - `viz/topdown.py`: `show_cost_values` (log-scale cost, black = blocked) moved here from `show_fused_costmap.py`. `viz/baseline_figure.draw_path_on_camera` gained a `color` parameter. New `viz/hybrid_figure.py`: `draw_hybrid_figure` (camera with both paths | VLM-adjusted cost + both paths + VLM route | LiDAR labels + both paths; baseline red, hybrid cyan) and `path_status`.
  - `scripts/run_hybrid.py [--run NAME] [--frames] [--goal-distance] [--out-dir]` → `outputs/hybrid/<run>/<stem>.png` + `summary.csv` (answer status, path_changed, cells unblocked/dearer/cheaper/blocked by VLM, and per plan: reached, shortfall, length, detour, share water/mud/bush/unseen). It prints a per-category "baseline -> hybrid" table. A missing answer means "no hints", so the plan equals the baseline. Added to `run_all.py` as the slow step `hybrid`.
  - `tests/test_hybrid.py`: 4 tests (no answer = baseline path; a grass hint opens a full-width 1 m grass wall so the hybrid reaches the target while the baseline stops short; a tree hint does not; a water hint steers the path around). The suite should be 181.
  - **First results (my run, 8B answers, saved only to my scratchpad):** reached 25 → 26 of 36 (bush 00002/935 now reached: detour 1.19 → 1.09). **Water category: LiDAR-labelled water share on the path 16% → 10%**: 120 22→13%, 590 17→8%, 1448 39→26%, 158 26→9%. 683 and 807 got slightly worse (0→1%, 0→6%). Mud in the water category 10% → 12% (it sometimes swaps water for mud ruts). Bush category: reached 3 → 4, shortfall 2.15 → 2.02 m, bush share on path 4% → 6% (it drives through bush more, as intended). **2132 is unchanged (detour 1.43)**: the unblocked grass island is unreachable, as expected with the strict blind-zone choice. Mud, rubble, and easy are essentially unchanged.
  - **The user's verdict (2026-11-01): "It doesn't change anything much."** My diagnosis (scratch script `path_diff.py`, 8B answers): the hybrid path is on average **0.39 m** from the baseline path (mean over frames of the per-frame max: 1.16 m; max > 1 m in 18 frames, > 2 m in 3). Why: (1) **coverage**: the VLM's hints cover only about 12% of the visible ground within 25 m (367 of ~3,120 cells per frame), and only 14% of the baseline path's cells get any hint. That comes from the hint cap (6 entries × 8 cells), tiny near-row footprints, and the 25 m limit. (2) **strength**: ×5 for "avoid" is often not enough to prefer a detour through unknown (3) or rough (5) cells, and puddles often span the whole trail. (3) **blind zone** 0–4.4 m (strict). (4) **hint quality**: about 38% recall of near water cells. (5) The baseline already reaches 25 of 36 targets. Proposed next diagnostic (awaiting the user): an **oracle run**, with hints made from the camera label images (perfect perception), to show whether the fusion *can* change paths when the hints are right. That separates "the VLM isn't good enough" from "the fusion design is too weak", and gives an upper bound for the report.
  - **Oracle test (user approved 2026-11-01; written, and run by me into the scratchpad only; the user has not run it).**
    - `src/offroad/vlm/oracle.py`: `oracle_answer(label_image, grid, id_to_name)` → answer dict with no waypoints and a hint for every grid cell with clear ground truth. `HAZARD_RULES` (person/vehicle, tree/log, other obstacle → impassable; water/puddle, rubble → avoid; mud → caution) win at ≥ 20% of the cell, most cautious first. Otherwise `GROUND_RULES` (dirt/asphalt/concrete → preferred; grass, bush → ok) at ≥ 50%. Otherwise no hint.
    - `HybridResult.path_separation()` → (mean, max) distance of the hybrid path from the baseline path.
    - `run_hybrid.py --oracle` → `outputs/hybrid/oracle/`; the summary gains `mean_separation_m`, `max_separation_m`, `cells_hinted`; the per-frame line and the table show how far apart the paths are.
    - Tests: `tests/test_oracle.py` (6), plus separation asserts in `test_hybrid.py`. The suite should be 187.
    - **Results, baseline → hybrid (oracle vs 8B VLM):**

      | | 8B VLM | oracle |
      |---|---|---|
      | reached (all 36) | 25 → 26 | 25 → **28** |
      | bush reached | 3 → 4 | 3 → **7** |
      | bush shortfall | 2.15 → 2.02 m | 2.15 → **0.65 m** |
      | water on path (water frames) | 16% → 10% | 16% → **8%** |
      | mud on path (mud frames) | 24% → 22% | 24% → **9%** |
      | paths apart (mean / max) | 0.39 / 1.16 m | 0.66 / 1.74 m |

      Bush 1261, 1554, 935, and 2081 go from stopping short to reaching the target.
    - **Conclusions:**
      - (1) The fusion design *can* matter, and the main bottleneck is the VLM's perception and coverage: the VLM hints about 12% of the ground, the oracle about 100%.
      - (2) The design has limits even with perfect hints. 2132 is unchanged (blind zone). "Impassable" on far, coarse cells over-blocks: easy 00003/1060 goes from reached to 1.2 m short with 823 cells blocked, and bush 1625 gets 2.1 → 3.4 m short. Water 120 now stops 2.5 m short because the person stands at the target, which is correct behavior that the "reached" metric penalizes. Phase 6 must not treat "reached" as always good.
    - Options proposed next: (A) improve VLM coverage, e.g. ask for a terrain label for every ground cell in rows E–H in a compact per-row format, then rerun the 8B; (B) limit "impassable" on far rows; (C) go to phase 6 with three conditions (8B, 4B, oracle).
  - **Status check 2026-11-12:** all phase 5 code (tasks 1–4 + oracle) is committed (phase 4 as 33c3d6b, phase 5 as ebd4898 "Added QWEN VLM Zero-shot / Added hybrid results"). The user has run `run_hybrid.py` for the 8B answers and `--oracle` (`outputs/hybrid/{Qwen3-VL-8B-Instruct_4bit,oracle}/`, 36 pictures + summary each). Phase 5 is not formally closed: the user still has to pick option A, B, or C above. Polaris (CPP HPC): H200 GPUs (~141 GB), but jobs only see a GPU inside `srun/sbatch -p gpu --gres=gpu:1`; the `basic` QOS gives 4 GB RAM and 1 GPU per job (`student-advanced`: 16 GB). The user deleted their Polaris conda env and caches on 2026-11-03 and will set it up again when needed.
  - `path_changed` is "yes" for 34/36, but most changes are tiny shifts (for example from the ×0.8 corridor). Phase 6 should measure *how much* a path changed (e.g. mean distance between the paths), not just whether it did.

### Phase 5 open questions (for the user)
- Should the VLM be allowed to make a LiDAR-blocked cell drivable? This is the core hypothesis (tall grass and soft bush look like walls), but it is the risky direction. Proposed: only for vegetation hints (`short_grass` / `tall_grass_or_bush` with preferred/ok/caution), only where the obstacle is lower than about 1.5 m, and to a finite cost (not "free"), never for trees, rocks, people, etc. Raising costs (mud/water "avoid") is always allowed.
- Multipliers, proposed: preferred ×0.7, ok ×1.0, caution ×2, avoid ×5, impassable = blocked.

## PHASE 4 (done 2026-11-01): VLM guidance with a grid overlay

**Closed by the user on 2026-11-01** after looking at the scored answer pictures: "the answers aren't as perfect as I was hoping, but we can close this phase". The user did not paste the `score_vlm.py` table; the numbers in the progress log come from my scratch scoring of the same answers. Main model: Qwen3-VL-8B-Instruct, 4-bit; the 4B is the size comparison.

Objective 2: get a vision-language model to look at a camera frame with a labelled grid drawn on it and answer, in a strict JSON format, (a) rough waypoints as grid cells and (b) traversability cost hints for grid cells (e.g. "passable vegetation", "mud/water: avoid"). The model is used zero-shot (pre-trained, no fine-tuning). Only work on this phase unless asked otherwise. Do not project VLM output into 3D or change the planner yet; that is phase 5.

**Milestone:** for every frame in `configs/eval_frames.csv`, save the gridded image, the VLM's raw answer, the parsed and validated waypoints and cost hints, and the response time, plus a picture of the answer drawn on the image.

### Phase 4 tasks (proposed 2026-10-31; refine with the user)
1. Grid overlay: draw a labelled grid on the camera image, and map cell labels ↔ pixel regions and centers. Pure functions + tests.
2. Prompt and answer format: a JSON schema for waypoints and cost hints, the prompt text, and a parser/validator that rejects or repairs malformed answers. Pure functions + tests on made-up answers (no model needed).
3. VLM runner: load the chosen model, run it on each evaluation frame, and save raw answers, parsed output, and latency. **Blocked on decisions: which model, and where it runs** (university GPU cluster with an open-weight model via Hugging Face, and/or a hosted API).
4. Pictures and a sanity check: draw the VLM's waypoints and cost hints on the image, and compare its cost hints per cell with the camera label image (does it call mud cells mud?).

### Phase 4 progress
- 2026-10-31: **task 1 (grid overlay) is written. The user has not run it yet.**
  - `src/offroad/vlm/grid.py` (pure): `ImageGrid(height, width, rows=8, cols=12)` with `label(row, col)` ("A1" = top-left; rows are letters from the top, columns are numbers from the left), `parse("F6")` (case-insensitive; ValueError on bad or out-of-grid names), `all_labels`, `bounds` (u0, v0, u1, v1), `center`, `cell_at(u, v)`, `cell_mask`. On RELLIS images, cells are 160 × 150 px.
  - `src/offroad/viz/grid_overlay.py`: `draw_grid(image, grid, line_alpha=0.6)` draws semi-transparent white lines plus a white label on a black box in each cell's top-left corner. Sizes scale with the cell size so the grid survives VLM downscaling.
  - `scripts/draw_grid.py --sequence S --frame F` → `outputs/grid_<seq>_<frame>.png`; `--eval` → `outputs/grid/<category>_<seq>_<frame>.png` for all 36 frames. `--rows`/`--cols` change the grid.
  - `tests/test_grid.py`: 23 test cases (14 functions, one of them parametrized over 10 bad names). The suite should be 88.
  - `scripts/run_all.py` added (see "Working rules").
  - **Verified 2026-10-31:** the user ran `run_all.py` (default steps): all tests passed (including the 23 grid tests) and all pictures were regenerated. In `grid_00000_000120.png` the labels are readable and each covers about 7% of its cell; the puddle spans F5–H9, the person B5–E6, the trail D7–D9. **Task 1 is done.** (`--only grid-eval` has not been run yet.)
- 2026-10-31: **task 2 (prompt + answer checker) is written. The user has not run it yet.**
  - `src/offroad/vlm/answer.py` (pure): `TERRAIN_TYPES` (9: dirt_or_gravel, short_grass, tall_grass_or_bush, mud, water, rocks, tree_or_log, other_obstacle, person_or_vehicle), `TRAVERSABILITY_LEVELS` (preferred, ok, caution, avoid, impassable; categories rather than numbers because VLMs are poor at calibrated weights; phase 5 maps them to cost multipliers), and synonym tables. `extract_json` handles code fences, surrounding text, braces inside strings, and trailing commas. `parse_answer(text, grid)` → `VlmAnswer(waypoints, hints: {cell: CellHint(terrain, traversability)}, reasoning, issues)`; `valid` means at least one waypoint. It never raises: invalid cells or words are dropped and recorded, consecutive duplicate waypoints are merged, and conflicting hints for a cell keep the more cautious one.
  - `src/offroad/vlm/prompt.py`: `build_prompt(grid, goal_cell=None, goal_distance_m=20, vehicle_briefing=VEHICLE_BRIEFING)` and the answer format shown to the model (originally a concrete `example_answer`; replaced on 2026-11-01 by the placeholder `answer_template`, see task 3). **Design choice for the user to review:** `VEHICLE_BRIEFING` tells the model what the vehicle can drive through (grass and soft bushes yes; trees, rocks, logs, fences, people, deep water no; mud and puddles risky). It is fair operator context but partly gives the answer; it can be removed with `vehicle_briefing=None` / `--no-briefing`. **User decision (2026-11-01): keep the briefing.** It is fixed vehicle knowledge (available live, not labels), and the VLM still has to recognize the terrain in each frame. The report should state that the VLM gets the vehicle's rules. Running a no-briefing comparison in task 3 stays optional. The goal cell will come from projecting the 20 m target in task 3; for now the prompt says "about 20 m straight ahead".
  - `scripts/vlm_prompt.py --sequence S --frame F [--goal-cell F7] [--no-briefing] [--answer file.txt]` saves `outputs/vlm/prompt_<seq>_<frame>.{png,txt}`, prints the prompt, and checks a saved answer. This lets the user try the prompt by hand in a chat VLM before a model is wired up. Added to `run_all.py` as step `prompt`.
  - `tests/test_vlm_answer.py`: 23 tests (an empty-answer test added 2026-11-01). The suite should be 111.
  - **2026-11-01: task 2 is done (the user confirmed).** Task 3 model choice: the user wants a VLM running **locally on their Windows machine (RTX 5060, 8 GB VRAM)** instead of the cluster for now. I proposed Ollama; **the user chose PyTorch + Transformers instead** (no Ollama; the project will later move to a Linux machine, where the same code runs).
- 2026-11-01: **task 3 (VLM runner) is written. The user has not installed the packages or run it yet.** Nothing ML was installed in the conda env at that point (no torch/transformers). GPU: RTX 5060 Laptop, 8 GB, driver 621.01.
  - `requirements-vlm.txt`: torch 2.11.0 + torchvision 0.26.0 from the cu128 index (RTX 50 needs CUDA 12.8+ builds), transformers 5.22.1, accelerate 1.15.0, bitsandbytes 0.50.2.
  - `src/offroad/vlm/backend.py`: `TransformersVlm(model_id=DEFAULT_MODEL="Qwen/Qwen3-VL-4B-Instruct", quantize="4bit"|"8bit"|"none", max_new_tokens=512)`, using `AutoModelForImageTextToText` + `AutoProcessor` and the processor's chat template. 4-bit is NF4 with bf16 compute; the vision encoder ("visual") stays bf16. Greedy decoding. `generate(image_rgb, prompt)` → `VlmReply(text, seconds, prompt_tokens, new_tokens)`; `peak_memory_gb()`. torch is imported only inside the class. Model choice reasoning: Qwen3-VL-4B bf16 is ~9 GB (too big for 8 GB), so 4-bit; 2B fits in bf16; 8B in 4-bit is ~6 GB and may fit. Qwen3.5 small models (March 2026, natively multimodal) were not chosen as default because their new architecture is less proven in Transformers on Windows; they can be tried with `--model`.
  - `src/offroad/vlm/goal.py` (pure): `ground_height_at`, `goal_pixel`, `goal_cell(ground_z, spec, T_cam_from_lidar, K, grid, goal_distance=20)`: projects FLU (20, 0, ground) into the image and returns its cell, so the VLM gets the same target as the baseline. Expected around F7 on flat ground.
  - `src/offroad/vlm/runner.py` (pure): `downscale(image, max_side)`, `strip_thinking`, `answer_to_dict`.
  - `scripts/check_gpu.py` (versions, GPU, a bf16 matmul and a 4-bit layer on the GPU) and `scripts/run_vlm.py` (`--model`, `--quantize`, `--sequence/--frame` or the eval CSV, `--max-side 1280`, `--no-briefing`, `--no-goal-cell`, `--name`, `--overwrite`). Saves `outputs/vlm_runs/<model>_<quant>[_nobriefing]/<category>_<seq>_<frame>.{json,png}` plus `summary.csv` and prints a per-category table (valid, issues, seconds, tokens) and the peak GPU memory. It skips frames that already have a JSON. One untimed warm-up call is made before the first frame.
  - `run_all.py`: new slow steps `gpu`, `vlm-one`, `vlm`.
  - `tests/test_vlm_runner.py`: 13 tests (goal cell on synthetic calibration, downscale, strip_thinking, answer_to_dict). The suite should be 124.
  - **2026-11-01, first real run (the user ran it):** `check_gpu.py` passed (RTX 5060 Laptop, compute capability 12.0, bf16 and 4-bit OK). `run_vlm.py --sequence 0 --frame 120`: the model download took about 6.6 min, loading 8 s; the answer took 7.7 s; 1517 prompt tokens (1280×800 image), 135 tokens out; peak GPU memory 3.9 GB. Harmless warnings: no HF_TOKEN, no symlink support on Windows (the cache uses more disk), triton not found.
  - **Problem found: the model copied the prompt's example.** Its cost hints are exactly the example's (G4/G5 water avoid, F10 tall grass ok), and its waypoints are the example's H7→G7→F7 plus E7. In the image, that route drives straight into the right puddle (F7–H8). The left puddle is F3–G5 (so the copied G4/G5 is right only by chance), the mud ridge between the puddles is G6/H6, and the person stands at D5–D6, on the goal cell D6. The checker said VALID with 0 issues, so **"valid" only means well-formed, not correct**; task 4 has to judge correctness.
  - **Fix (2026-11-01, the user approved; not yet rerun):** `prompt.py`'s `example_answer` was replaced by `answer_template(grid)`, which uses `<placeholders>` (no real cells), so a copied template gives an invalid answer with every placeholder listed as an issue. The keys are now ordered reasoning → cost_hints → waypoints so the model describes hazards before choosing a route. The prompt says to replace every placeholder based on the image, asks for every hazard near the way to the goal, and says waypoints should not pass through avoid/impassable cells unless there is no other way. Tests: the old example tests were replaced by "copied template is caught", key order, and "no real cells in the template". The suite should be 125. Rerun with `python scripts/run_vlm.py --sequence 0 --frame 120 --overwrite`.
  - **Rerun with the template prompt (2026-11-01): NOT USABLE, "the JSON object is not closed".** The reasoning was sensible ("muddy and wet, with a puddle in the center… proceed through the grassy area… toward D6"). But cost_hints turned into a column-by-column list of every cell (A1–H1 dirt preferred, A2–H2 short grass, … A6–H6 mud caution, …), with labels unrelated to the image (the sky in A1 called dirt). It hit the 512-token limit before writing waypoints. Time: 27 s (about 19 tokens/s), 1582 prompt tokens. Cause: "one entry per group… include every hazard" invited an exhaustive list, and a small model fell into a repeating pattern.
  - **Fix 2 (2026-11-01, the user approved; not yet rerun):** the template order is now reasoning → waypoints → cost_hints, so a cut-off answer still has a route. `MAX_HINTS = 6` and `MAX_CELLS_PER_HINT = 8` are stated in the prompt, with "only mark ground cells that matter for the route… skip the sky, distant scenery, and cells far from the route". The waypoint rule now says to steer around water, mud, and obstacles. The checker does not enforce the caps. Tests: key-order test updated, plus a caps test. The suite should be 126. Next, if the 4B model is still weak: compare Qwen3-VL-8B in 4-bit on the same frame (a ~17 GB download).
  - **Rerun with fix 2 (2026-11-01): VALID, 0 issues, 13.2 s, 241 tokens out.** Format problems are solved. Content is mixed: right puddle correctly marked (F8/G8 water avoid); F6/G6 mud avoid is roughly right (the muddy ridge and puddle edge). But it **missed the left puddle (F3–G5) and the person (D5–D6)**, called H6/H7 (mud/ruts) and D6 (the person) "dirt preferred", and its **route H6→G6→F6→E6→D6 goes through the F6/G6 cells it marked avoid**, contradicting its own reasoning ("steer left"). It ends on the person because the prompt's goal is D6. Takeaways: the planner (phase 5) makes the final path, so the hints matter more than the waypoints; a "route passes through its own avoid cells" check would measure self-consistency.
  - **User decision (2026-11-01): run the 4B model on all 36 frames first** (no download, about 8 min) to get a baseline across categories, then compare the 8B model. Added before that run: `VlmAnswer.route_conflicts` (waypoints in cells the model itself marked avoid/impassable; a content measure, not a format issue), saved in each JSON as `answer.route_conflicts`, a `route_conflicts` column in `summary.csv`, a "conflicted" column (answers with ≥ 1 conflict) and an "all" row in the printed table, and a note in each per-frame line. Tests: 2 new in `test_vlm_answer.py`. The suite should be 128.
  - **4B 4-bit on all 36 frames (2026-11-01, the user ran it; the user's verdict: "not working very well").** Format: 36/36 valid, 6 answers with issues (e.g. invented row "I"), 4 with route conflicts. Time: 7–23 s, mean about 12.6 s; 108–381 tokens out. **Content is poor:**
    - The routes carry no information. All 36 start at H6, and 32/36 are exactly 3 waypoints, H6 → H7/H5/G6 → the goal cell, jumping several rows at once. The other 4 go along row H to H10 and then along row D back to D6.
    - The hints are mostly "preferred" ground: short_grass preferred 180 cells, tall grass preferred 49, dirt preferred 43; mud avoid 28, water avoid 12. The 3-hint answer "H6, H7 dirt preferred + E6 grass preferred" repeats 5 times.
    - It ignores the caps sometimes: a 32-cell list around the image border, and 33 cells in several answers.
    - Perception is partly there: 26/36 reasonings mention water, puddles, or mud, and water 00004/1460 says "muddy puddle in H6-H7", which is roughly right (the puddle spans G5–H12). But the cells and the route don't follow.
    - Note: in mud 00004/1010 the model says "grassy, no obstacles". That's arguably right, since RELLIS "mud" is often just dark trail ruts.
  - Options discussed: (a) Qwen3-VL-8B in 4-bit (a ~17 GB download), to test whether it's the model; (b) the 4B in 8-bit (no download) to rule out 4-bit damage; (c) switch to Qwen's native grounding output (boxes/points in 0–1000 coordinates; it is trained for this) and convert to grid cells ourselves. That is a method change from the grid-label objective and is the user's call. (d) A hosted large model as an upper-bound reference for the grid method.
  - **User decision (2026-11-01): try Qwen3-VL-8B in 4-bit** (option a; the same prompt, so it's a clean model-size comparison). No code change was needed for the model itself. Added a fallback in case the 8B doesn't fit in 7.9 GB: `--quantize-vision` (`TransformersVlm(quantize_vision=True)` also quantizes the vision encoder, saving about 1 GB). Run names now include `_qvision` and `_<N>px` when those differ from the defaults, so runs don't overwrite each other. Command: `python scripts/run_vlm.py --model Qwen/Qwen3-VL-8B-Instruct --sequence 0 --frame 120`, then the full 36. Output: `outputs/vlm_runs/Qwen3-VL-8B-Instruct_4bit/`.
  - **8B 4-bit on all 36 (2026-11-01, the user ran it; it fit without `--quantize-vision`).** 36/36 valid, 2 with issues, 6/36 conflicted; mean 13.8 s per answer (4B: 12.5 s), so it costs barely any extra time.
  - **Quick scoring against the camera label images** (my scratch script, not yet a project script; this is a prototype of task 4). Per grid cell, the label-pixel fractions were: water = puddle + water; hazard = water ≥ 20% or hard obstacle ≥ 20% (tree, pole, vehicle, object, building, log, person, fence, barrier). RELLIS "mud" is not counted as a hazard because it is mostly trail ruts. 4B → 8B:
    - water hints correct (cell ≥ 10% water): 9/12 → 30/40.
    - avoid/impassable hints on real hazards (water + mud + hard + bush ≥ 30%): 72% → 81%.
    - "preferred" hints on good ground (≥ 50% dirt/grass/asphalt/concrete, < 10% water/hard): 39% (105/272) → 73% (40/55).
    - water cells in rows E–H (≥ 30% water) flagged as water/mud or caution or worse: 16% → 38% (of 170).
    - frames with a labelled person that mention a person: 0/4 → 3/4.
    - distinct routes 10 → 15; mean waypoints 3.6 → 5.1, and the routes are now continuous columns rather than jumps.
    - **Routes vs. driving straight up column 6** (mean water share of route cells after the start, water category): 4B 0.47 (worse than straight, 0.33); 8B 0.31 (about the same as straight). Per water frame, 8B beats straight on 3 (00003/1448: 0.28 vs 0.53; 00003/158: 0.11 vs 0.24; 00000/120, slightly), ties on 2, and loses on 2 (00001/590, 00002/807). Near the vehicle a whole row is often puddle, so cell-level routes can't dodge it.
  - **Conclusion:** the 8B is clearly better at *perception* (hints), but its *routes* are still weak. For phase 5 the cost hints are the more useful output; treat the waypoints as soft guidance. Proposed: make the 8B the main model, keep the 4B as the size comparison, and do task 4 next (turn this scoring into a project script plus pictures of the answers).
- **2026-11-01: the user approved: the 8B is the main model, and task 4 has started. Task 4 is written; the user has not run it yet.**
  - `src/offroad/vlm/scoring.py` (pure): `CLASS_GROUPS` and `TERRAIN_CLASSES` (by ontology name, IDs looked up via `class_ids`), named thresholds (`WATER_HINT_MIN` 0.1, `AVOID_HAZARD_MIN` 0.3, `PREFERRED_GOOD_MIN` 0.5 / `PREFERRED_BAD_MAX` 0.1, `WATER_CELL_MIN` 0.3 in the bottom half of the rows, `ROUTE_HAZARD_MIN` 0.2, `TERRAIN_MATCH_MIN` 0.2, `PERSON_PRESENT_MIN` 0.02). Functions: `cell_fractions(label_image, grid, id_to_name, groups)`, `straight_route(grid, goal)` (same column up to the goal row), `score_answer(...)` → `FrameScore` (counts), and `summarize(scores)` → metric → (value, "k/n"). `dirt_or_gravel` matches mud too, because RELLIS labels trail ruts as mud.
  - `src/offroad/viz/vlm_answer.py`: `colorize_labels`, `draw_hints` (cells tinted green→red by traversability, terrain word), `draw_route` (white line, red dots on conflicts, magenta star on the goal), `draw_answer_figure` (camera + hints | labels + route, with a header: title, reasoning, legend).
  - `scripts/score_vlm.py [--run NAME ...] [--no-pictures]`: by default scores every run folder; skips `single` records. Writes `<run>/answers/*.png`, `<run>/scores.csv`, and `outputs/vlm_runs/comparison.csv`, and prints a metric × run table plus route-vs-straight water share per category. Added to `run_all.py` as the slow step `vlm-score`.
  - `tests/test_vlm_scoring.py`: 14 tests. The suite should be 142.
  - Checked on 8B water 00000/120 (picture in my scratchpad only): the figure reads well. **Caveat for the report:** RELLIS labels the whole wet trail area of that frame as puddle, including the mud ridge and grass edges (38 of the 48 bottom-half cells are ≥ 30% water). So the water metrics are measured against coarse labels.
  - 2026-11-01: the user's first `--answer` run said "no JSON object found" because `scripts/my_answer.txt` was empty (0 bytes); the checker now reports "the answer is empty", and `vlm_prompt.py` exits with a clear message if the answer file is missing.

## PHASE 3 (done 2026-10-31): terrain map (DEM) + baseline geometric planner

Built the geometry-only baseline that the VLM approach will be compared against. Last commit 6d36d73. Details and final results are under "Phase 3 progress" below.

**Milestone (reached):** for each frame in `configs/eval_frames.csv`, build a top-down elevation map from the Ouster scan, turn it into a cost map, and plan an A* path from the vehicle to a goal ahead. Save a picture of the map with the path.

### Phase 3 tasks (all done)
1. Elevation map: grid the scan in a forward-left-up frame (per cell: lowest/highest/mean height, point count, step height = highest − lowest, majority ground-truth label). Pure functions + tests, and a script that saves a picture next to the camera image.
2. Cost map from geometry only: lethal where the step height or slope is too large, a higher cost for rough cells, and a set cost for unknown (empty) cells.
3. A* planner on the cost map, from the vehicle cell to a goal ahead. At first the goal is a fixed distance straight ahead. Later it could be where the vehicle actually went, which needs the Ouster scan poses download.
4. A script that runs this on all evaluation frames and saves the pictures.

## PHASE 2 (done 2026-10-31): evaluation frame curation

Chose the 36 annotated frames that all later work is evaluated on: `configs/eval_frames.csv`, committed in bb2f532. Details are under "Phase 2 progress" below.

## PHASE 1 (done): data loading and LiDAR-to-camera projection

**Milestone:** Load one RELLIS-3D frame (RGB image + matching Ouster LiDAR scan + calibration), project the LiDAR points into the image, and save an overlay image with points colored by depth. If the points line up with trees, rocks, and terrain edges in the photo, the geometry is correct.

This step matters because Objective 3 (image pixel to 3D terrain) uses the same matrices and conventions in reverse.

### Phase 1 tasks (all done)

1. Set up the repo skeleton (layout below), `requirements.txt`, `.gitignore`, and a short `README.md`.
2. Write a dataset loader for one RELLIS-3D sequence that returns, for a given frame index:
   - RGB image (H x W x 3, uint8)
   - LiDAR point cloud (N x 4 float32: x, y, z, intensity)
   - Semantic label image (H x W, class IDs), when available
   - Camera intrinsics K (3x3) and distortion coefficients
   - LiDAR-to-camera extrinsic transform (4x4)
3. Write the projection function: transform points to the camera frame, drop points behind the camera (z <= 0), project with K, keep points inside image bounds. Return pixel coordinates, depths, and a mask of which original points were kept.
4. Write a script that runs the loader + projection on one frame and saves the depth-colored overlay to `outputs/`.
5. Add basic unit tests for the projection math (e.g., a point on the optical axis lands at the principal point; points behind the camera are dropped).

## Progress log (as of 2026-10-25)

Kept here so work can continue on another machine (the user switches between macOS and Windows). Update this section as work progresses.

### Done
- Task 1 (repo skeleton) is complete: `README.md`, `requirements.txt`, `.gitignore`, and empty packages `src/offroad/{data,geometry,viz}/__init__.py`, plus empty `scripts/` and `tests/` folders.
- `requirements.txt` pins numpy 2.2.6, opencv-python 4.11.0.86, matplotlib 3.10.3, PyYAML 6.0.2, pytest 8.4.1. Open3D is commented out because it has no Python 3.13 wheels.
- `.gitignore` covers `data/`, `outputs/`, Python caches, venvs, `.idea/workspace.xml`, `.idea/misc.xml`, `.DS_Store`.
- Task 2 (loader) is written and smoke-tested on the first frame of all five sequences (2026-10-26):
  - `src/offroad/geometry/transforms.py`: pure helpers `quaternion_to_rotation_matrix`, `make_transform`, `invert_transform`. They sit in `geometry/` (not in the suggested layout) because Objective 3 reuses them.
  - `src/offroad/data/calibration.py`: `load_calibration(sequence_dir)` returns `CameraCalibration(K, dist_coeffs=zeros(5), T_cam_from_lidar)`.
  - `src/offroad/data/rellis.py`: `RellisSequence(seq_id, root=None)`. `seq[i]` gives a `RellisFrame` (RGB image, (N,4) points with empty returns dropped, uint16 point labels, label image or None, calib). Also provides `default_rellis_root()` and `load_ontology()`.
  - Smoke-test result: about 6–13% of the non-empty points per scan project inside the image. This is plausible for a camera with a ~38° horizontal field of view.
- Task 3 (projection) is written (2026-10-26): `src/offroad/geometry/projection.py`.
  - `transform_points(T_b_from_a, points_a)`.
  - `project_points_to_image(points_lidar, T_cam_from_lidar, K, (H, W))` returns `(uv (M,2), depth (M,), kept (N,) bool)`. It assumes a pinhole camera with no distortion. Kept pixels satisfy 0 <= u < W and 0 <= v < H, so `int(u)`, `int(v)` index the image directly.
  - Checked on frame 0 of sequences 00000 and 00002. It matches `cv2.projectPoints` to about 1e-12 px. 74% and 84% of visible points have the same class as the label image at their pixel. This is strong evidence that the extrinsic direction is right; the overlay still needs a visual check.
- Task 4 (overlay) is written (2026-10-26). **Milestone reached:** the projection lines up visually.
  - `src/offroad/viz/overlay.py` has `depth_to_colors` ("turbo": near blue, far red), `draw_points`, and `draw_depth_overlay` (draws far to near).
  - `scripts/project_lidar.py --sequence S --frame F [--max-depth M] [--radius R] [--out PATH]` saves `outputs/overlay_<seq>_<frame>.png`. It puts `src/` on `sys.path` itself.
  - Inspected 00000/0, 00002/0, and 00004/1000. Dots follow a fence rail and post top to within about 2 px. Distant tree canopies get points while the sky gaps between them get none. Depth increases smoothly up the ground plane. No sign that distortion correction is needed.
- Task 5 (tests) is written (2026-10-26). **All five tasks of this phase are done.**
  - `pyproject.toml` holds only pytest config (`pythonpath = ["src"]`, `testpaths = ["tests"]`). It is not a package build config. The script adds `src/` to `sys.path` itself.
  - `tests/test_projection.py` (11 tests) covers: principal point, the pinhole formula, dropping z <= 0, image bounds, mask order with (N,4) input, the extrinsic being applied, `transform_points`, agreement with `cv2.projectPoints`, and empty input.
  - `tests/test_transforms.py` (7 tests) covers: quaternion→rotation, normalization, `invert_transform`, calibration parsing from temp files, and a guard on the inversion direction and the RELLIS axis mapping.
  - Run with `pytest` from the repo root. 18 passed in 0.8 s. No dataset needed.
- Git: remote `origin` = github.com/talhaahmed884/VLM-Offroad-Nav. The user pushed the Task 2–5 code as commit 999c767 (its message mentions a "DEM grid", but it contains only this phase's code). Earlier uv/Docker files were added and then removed again. `CLAUDE.md` itself is still untracked.
- README setup docs (2026-10-31): the Data section lists the 7 required RELLIS-3D downloads with Google Drive links (taken from the official README), the Baidu mirror and the rate-limit workaround, unzip tips, the full expected folder tree with per-sequence frame counts, `RELLIS_ROOT` for macOS/Linux and PowerShell, and the CC BY-NC-SA 3.0 license with the citation. The per-point label download (`os1_cloud_node_semantickitti_label_id`) is required because the loader always reads it.

### Phase 2 progress (evaluation frame curation)
- 2026-10-31: tasks 1–3 and 5 are done. Task 4 (the user's final pick) is pending.
  - `src/offroad/eval/curation.py` (pure): `ahead_region_mask` (rows 55–100%, columns 25–75% of the image), `class_pixel_fractions`, `Candidate`, and `pick_spaced(candidates, n, min_gap, already_picked, max_per_sequence)`.
  - `scripts/compute_label_stats.py` writes `outputs/label_stats.csv`. It has 6234 rows, one per labelled frame, with `whole_<class>` and `ahead_<class>` fractions. It takes a few minutes. Every label ID is in `ontology.yaml`.
  - `scripts/select_eval_frames.py` writes `outputs/eval_candidates/{candidates.csv, <category>.jpg}`. The categories are mud, water (puddle+water), bush, rubble, and easy (grass/dirt/asphalt/concrete ≥ 80% with hazards ≤ 1%). Defaults: 12 per category, `--min-gap 100` frames (~10 s), `--max-per-sequence 4`. The rarest category is picked first.
  - `tests/test_curation.py`: 7 tests. The full suite is 25 passed.
- Findings from the label stats:
  - "dirt" never appears. Trail ruts are labelled **mud**, which covers ≥ 3% of the ahead region in 41% of frames. So "mud" candidates are mostly wet trails or ruts, not deep mud pits.
  - Bush covers ≥ 10% of the ahead region in 42% of frames.
  - Puddles are common in 00000/00001/00003. Real "water" (a pond) is only in 00002.
  - **Rubble is rare (135 frames), and every rubble pile is far away at the horizon.** RELLIS-3D has no close-range rocky or sloped terrain, so the "rocky slopes" category from the original plan cannot really be covered.
- Proposed: 55 candidates (12 mud, 12 water, 12 bush, 7 rubble, 12 easy).
- **Final set, approved by the user (2026-10-31): `configs/eval_frames.csv`, 36 frames.** It is the top 8 candidates of mud, water, bush, and easy, plus the top 4 rubble frames (a "far obstacle" group, not rocky terrain). Columns: `category, sequence, frame, image_name`. Edit the file to swap frames. Phase 2 is complete once this file is committed.

### Phase 3 progress (DEM + baseline planner)
- 2026-10-31: **task 1 (elevation map) is done and verified.** The user ran the tests (all passed; 8 elevation tests, 33 total) and the pictures for 00000/0 and 00001/2132.
  - Left/right is correct: frame 0's bushes and trees on the right of the photo are on the right of the top-down panels, and the puddle is straight ahead about 3–8 m out.
  - The LiDAR is level: near ground is about −1.3 m on both sides.
  - The baseline weakness is visible already: puddle cells have about 0 step height (they look safe), while bush cells have 0.5–1.5 m (they look like walls).
  - Things the cost map has to handle: (a) a blind circle of radius about 4.5 m around the vehicle with no ground points, which must not block the start; (b) under tree canopies `min_z` is a branch 2–5 m up, so the ground has to come from a wider neighborhood, and overhead-only cells should not count as walls.
  - `src/offroad/mapping/elevation.py` (pure): `T_FLU_FROM_LIDAR`, `GridSpec` (default x −5..35 m, y −15..15 m, 0.25 m cells → 160 × 120), `lidar_to_flu`, `points_to_cells`, `build_elevation_map` → `ElevationMap(count, min_z, max_z, mean_z, step_height)`, `majority_label_grid`.
  - `tests/test_elevation.py`: 8 tests.
  - `scripts/build_elevation_map.py --sequence S --frame F` saves `outputs/elevation_<seq>_<frame>.png` with 4 panels (camera, ground height, step height, majority class, plus the camera FOV lines).
  - The forward-left-up (FLU) frame has its origin at the LiDAR: x forward = LiDAR −x, y left = LiDAR −y, z up = LiDAR +z. This is a 180° rotation about z, worked out from the calibration. **Still to check against real data:** that left/right is not mirrored, and that the LiDAR is level (the ground median should be about −1 to −2 m and the ground should look flat).
  - `max_z` includes overhanging branches, so cells under trees get a large step height. The cost map handles this with a vehicle-height cap.
- 2026-10-31: **task 2 (geometric cost map) is done and verified.** The user ran it: tests passed (40), the elevation picture is unchanged after the helper refactor, and the cost maps look right.
  - Share of each class marked as obstacle, 00000/0 vs 00001/2132: bush 97% / 82%, tree 97% / 97%, puddle 0% / –, mud 0% / 3%, concrete 0% / 1%, grass 23% / 13%. This is exactly the baseline blind spot: bushes look like walls, puddles and mud look safe.
  - 00000/0: the trail and puddle ahead are drivable, and a narrow drivable strip runs along the trail to about 30 m. 00001/2132: everything ahead is blocked by tall grass, so the geometry planner will fail or detour left (the road).
  - The printed "blocked %" and "unknown %" overlap, because inflation also covers unknown cells.
  - **Issue for task 3:** in 00000/0 there is a blocked "person"-labelled blob right at the vehicle origin (vehicle self-returns or a person beside it). The planner must treat the vehicle's own footprint as drivable, or it starts inside an obstacle. Also, the ground closer than about 6 m ahead is below the camera's view (vertical half-FOV ≈ 12° from about 1.3 m up), so the VLM cannot see it.
- 2026-10-31: **task 3 (A* planner) is done and verified.** The user ran it: tests passed (52), and the pictures and printed results look right. The user pushed tasks 1–2 first; task 3 is not committed yet.
  - 00000/0: path found, 20.3 m, cost 222.8. It goes **straight through the puddle** (puddle 16%, mud 15%, grass 12%, unseen 56%). In the camera image the red dots sit on the water along the trail. This is the "gets stuck" failure.
  - 00001/2132: path found, 27.7 m (about 38% longer than the straight 20 m), cost 324.7. It **detours left** around the tall grass "wall" through unknown cells (unseen 77%, grass 11%, mud 9%, bush 2%). This is the "unnecessary detour" failure.
  - Projected path dots on 2132 appear above the grass because they mark the ground behind it, at 15–20 m. Parts of the detour more than about 19° off-axis fall outside the camera's view.
  - For evaluation later: many path cells are "unseen" (no LiDAR points), so per-path class statistics need another ground-truth source (e.g. the camera label image) or a fill-in.
- 2026-10-31: task 4 (batch run on the evaluation frames) is written. **The user has not run it yet.** The user committed task 3 first.
  - `src/offroad/planning/baseline.py` (pure): `run_geometric_baseline(points_lidar, point_labels, spec, params, goal_distance=20, footprint_radius=2)` → `BaselineResult` (costmap, cost, labels, start, goal, path, total_cost; properties `found`, `length_m`, `straight_m`, `detour_ratio`; methods `path_xy()`, `class_shares()` with `UNSEEN = 0`). This is the single entry point for the baseline; the VLM comparison should call it.
  - `src/offroad/viz/baseline_figure.py`: `draw_path_on_camera` and `draw_baseline_figure` (3 panels). `plan_path.py` is now a thin wrapper around these two modules.
  - `scripts/run_baseline.py [--frames configs/eval_frames.csv] [--goal-distance 20]` writes `outputs/baseline/<category>_<seq>_<frame>.png` and `outputs/baseline/summary.csv` (category, sequence, frame, path_found, path_length_m, straight_m, detour_ratio, total_cost, share_unseen, share_<class>…). It prints a per-category table: paths found, mean detour, mean water/mud/bush/unseen share.
  - `tests/test_baseline.py`: 5 synthetic-scan tests (flat ground gives a straight path; a wall with a gap gives a detour through the gap; a full-width wall gives no path; an obstacle at the vehicle doesn't block the start; unseen cells are labelled). The suite should be 57.
- 2026-10-31: the user ran `run_baseline.py`. Paths found: mud 7/8 (detour 1.04, 24% mud), **water 1/8**, bush 3/8 (detour 1.18), rubble 4/4, **easy 3/8** (detour 1.43). **Task 4 is not done: the results expose a cost-map problem.**
  - easy 00001/412 (an open grass trail) and water 00000/120 (a grass trail with puddles) are almost entirely blocked, including plain grass and puddles.
  - Suspected cause: `local_ground` takes the *minimum* over a 6 m window, so any low outlier (water-reflection "mirror" points under puddles, or a slope) drags the ground down for 3 m around, making grass look > 0.3 m tall. A third possibility is that the grass really is > 0.3 m tall.
  - easy 00002/102 is partly a frame-choice issue: the LiDAR shows bush from about 3 m on, so it isn't "easy" over the 20 m planning range. The easy category was chosen from the image region, which covers about 6–15 m.
  - Diagnosis (from the user's elevation and cost-map runs on 00000/120 and 00001/412): **the cause is slope.** Relative to the sensor, the ground rises about 10–12% ahead (412: −1.3 m near the vehicle → +0.5 m at 20 m; 120: −1.6 → +0.5 m by 18 m). The 6 m minimum window puts the ground about 0.3+ m too low, so everything on the slope was flagged (grass 50–85%, flat concrete road 82%). The per-cell step height of the grass is under 0.25 m, so the grass is not tall, and there are no low "mirror" outliers under the puddles.
  - **Fix (2026-10-31, not yet run by the user):** `estimate_ground` in `costmap.py` now uses two scales. Ground = lowest point in a **1.5 m** window (`ground_window`). Where that is more than `hidden_ground_threshold` = 0.5 m above the lowest point in the **6 m** window (`wide_ground_window`), the ground is hidden (canopy or bush tops), so the wide value is used. Added `window_cells()`. `local_ground`'s second parameter is renamed to `window`.
  - New tests in `test_costmap.py`: a 12% slope gives no obstacles; a canopy at 3 m with hidden ground gives unknown (not smooth); bush tops at 1 m with hidden ground give an obstacle. Costmap tests: 10. The suite should be 60.
  - **Verified 2026-10-31 (the user ran it; 60 tests passed).** 00000/120: grass 19%, concrete 1% (was 82%), puddle 0%, tree 94%, bush 91% obstacle. 00001/412: grass 13% (was 50%), bush 68%.
  - Re-run of `run_baseline.py`, paths found: mud 7/8 (detour 1.07, 27% mud), **water 8/8** (detour 1.08, 16% water + 10% mud; e.g. 120 goes straight through the big puddle), bush 6/8 (**detour 1.41**, up to 2.05), rubble 3/4, easy 7/8 (detour 1.09).
  - Remaining NO PATH frames: bush 00000/1625 and 00000/2081 (expected); easy 00002/102 (tall bush, not really easy); **mud 00000/1521 and rubble 00001/1182 are goal-choice artifacts**. `nearest_free_cell` picked an isolated pocket (in trees, or inside the rubble pile) that can't be reached, while the vehicle could drive most of the way.
  - Fixed (not yet run): `draw_path_on_camera` filled unknown path ground with the map median, which put blind-circle path cells at the wrong height on slopes (stray red dots at the image edges in 120). It now interpolates along the path from the cells that have a known ground height.
  - **User decided (2026-10-31): the goal is now the reachable cell nearest the target, and the shortfall is reported.** Implemented but not yet run:
    - `astar.py`: `reachable_cells(cost, start)` (flood fill with the same moves and corner rule as A*) and `nearest_reachable_cell(cost, spec, start, x, y)`, which replaces `nearest_free_cell`.
    - `BaselineResult`: new field `target_xy` and properties `shortfall_m` (goal cell center → target) and `reached` (shortfall ≤ `REACHED_TOLERANCE_M` = 1.0 m). `found` is now False only if the start cell is blocked, which shouldn't happen after `clear_footprint`.
    - Figure: the target is a white X, the goal a red star, and the title says "target reached" or "stops X m short". `plan_path.py` prints reached / shortfall.
    - `run_baseline.py`: `summary.csv` gains `reached` and `shortfall_m`. The per-category table now shows frames, reached, mean shortfall, detour, and class shares.
    - Tests: `test_astar.py` swaps the `nearest_free_cell` test for 4 reachability tests (astar tests: 14). In `test_baseline.py`, the full-wall test now expects a path that stops about 10.9 m short; the flat and gap tests also assert `reached`. The suite should be 63.
- **2026-10-31: task 4 is done and verified. All phase 3 tasks are complete**, pending the user's confirmation and commit. The user ran it: 63 tests passed.
  - Final `run_baseline.py` table (reached / mean shortfall m / mean detour / water% / mud% / bush% / unseen%): mud 7/8, 0.5, 1.25, 2, 24, 0, 41 · water 7/8, 0.3, 1.08, 16, 10, 0, 39 · bush 5/8, 1.1, 1.34, 0, 3, 4, 49 · rubble 1/4, 1.3, 1.36, 0, 0, 0, 50 · easy 7/8, 2.1, 1.09, 0, 0, 0, 37.
  - Water 00000/120: the path goes straight through the big puddle, and the stray camera dots are gone. Bush 00000/2081 follows a narrow trail and stops 4.4 m short.
  - **Known limitations:** (1) mud 00000/1521: the nearest reachable cell is a pocket at about 23 m, reachable only by looping out to 33 m, giving a 56.6 m path (detour 2.51) instead of stopping at the trail end (about 18 m). This inflates the mud mean detour (about 1.07 without it). **The user chose to fix this before closing phase 3, see below.** (2) 37–50% of path cells are "unseen", so class shares cover only LiDAR-seen cells; evaluation should add camera-label ground truth. (3) easy 00002/102 is mostly tall grass and bush (stops 15.3 m short); consider swapping it out of `eval_frames.csv`, which is the user's call.
- 2026-10-31: **goal-choice fix for the loop (written, not yet run by the user).**
  - `astar.py`: `cost_to_come(cost, start)` (Dijkstra from the start to every cell; inf = unreachable) replaces `reachable_cells`. `choose_goal(cost, spec, start, x, y, cost_weight)` replaces `nearest_reachable_cell`: it minimises `shortfall_m + cost_weight * cost_to_come * resolution` over reachable cells.
  - `run_geometric_baseline(..., goal_cost_weight=0.1)`: getting 1 m closer to the target is worth up to 10 m of extra smooth driving (about 3.3 m through unknown cells).
  - Tests (`test_astar.py`, 16): cost-to-come stops at a wall, respects the corner rule and a blocked start, and matches the A* cost; `choose_goal` stops in front of a wall; it skips a ~22-cost loop for a 2 m gain at weight 0.5 but takes it at weight 0; None when the start is blocked. The suite should be 65.
  - Expected: mud 1521 stops at the trail end (shortfall about 3–4 m, no loop); other frames mostly unchanged.
  - The user's first test run failed 2 `choose_goal` tests (both use `cost_weight=0.0`): `0 * inf` = NaN for unreachable cells, and `argmin` picked the NaN. Fixed by scoring only reachable cells (others get inf). The real baseline uses 0.1, so it was unaffected.
- **2026-10-31: verified. All 65 tests pass and phase 3 is complete.** It was committed and pushed as 6d36d73, and the user moved on to phase 4.
  - mud 1521: 19.4 m path that stops 3.4 m short at the trail end (no loop). bush 1554: follows the short-grass strip and stops 5.3 m short in front of the tall grass (was a 40 m detour).
  - **Final baseline table** (reached / mean shortfall m / mean detour / water% / mud% / bush% / unseen%): mud 7/8, 0.6, 1.07, 2, 24, 0, 37 · water 7/8, 0.3, 1.08, 16, 10, 0, 38 · bush 3/8, 2.1, 1.16, 0, 3, 4, 43 · rubble 1/4, 1.4, 1.10, 0, 1, 0, 41 · easy 7/8, 2.1, 1.09, 0, 0, 0, 38 (easy's shortfall is driven by 00002/102 at 15.5 m).
  - **For the comparison phase:** with `goal_cost_weight` 0.1, the bush failure shows mostly as *stopping short* (bush reached 5/8 → 3/8, shortfall 1.1 → 2.1 m) rather than *detouring* (1.41 → 1.16). Evaluate both shortfall and detour.
  - Still open (not blocking): many unseen path cells (37–43%), so add camera-label ground truth when evaluating; easy 00002/102 could be swapped out (the user's call).
  - `GridSpec.cell_index(x, y)` was added, with a test.
  - `src/offroad/planning/astar.py` (pure): `astar(cost, start, goal)` → `(path cells, total cost)` or None. It is 8-connected, a move costs step × average cell cost, there is no corner-cutting past blocked cells, and the heuristic is octile × minimum cost (admissible). Also `clear_footprint(cost, spec, radius)`, `nearest_free_cell(cost, spec, x, y)`, and `path_length`.
  - `tests/test_astar.py`: 11 tests. The suite should be 52.
  - `cost_colors` and `add_cost_legend` moved into `viz/topdown.py`. `build_cost_map.py` uses them.
  - `scripts/plan_path.py --sequence S --frame F [--goal-distance 20] [--footprint-radius 2]` saves `outputs/path_<seq>_<frame>.png` (path on the camera image, the cost map, and the classes). It prints the path length, the cost, and the classes along the path, or NO PATH. The start is the FLU origin; the goal is the nearest drivable cell to (goal_distance, 0).
  - Shared plotting helpers moved to `src/offroad/viz/topdown.py` (`show_top_down`, `draw_camera_fov`, `label_colors`, `add_class_legend`). `build_elevation_map.py` now imports them.
  - `src/offroad/mapping/costmap.py` (pure): `CostParams` (max_step 0.3, vehicle_height 1.8, ground_window 1.5, wide_ground_window 6.0, hidden_ground_threshold 0.5, roughness_weight 4, unknown_cost 3, inflation_radius 0.75; the ground windows were changed 2026-10-31, see the task 4 notes), `local_ground` (minimum of `min_z` over a window), `estimate_ground` (two-scale), `obstacle_height_map` (tallest point below vehicle height above the local ground), `inflate`, and `geometric_cost_map` → `CostMap(cost, ground_z, obstacle_height, obstacle, unknown)`. `LETHAL = inf`.
  - Cost: drivable = 1 + 4·min(h/max_step, 1); unknown (no points, or only overhead points) = 3; obstacle (h > 0.3 m) grown by 0.75 m = LETHAL. There is no separate slope term, because the height above the 6 m local ground already catches sudden rises and RELLIS is mostly flat. Revisit if needed.
  - `tests/test_costmap.py`: 7 tests. The suite should be 40.
  - `scripts/build_cost_map.py --sequence S --frame F [--max-step M]` saves `outputs/costmap_<seq>_<frame>.png` and prints, per ground-truth class, the share of cells marked as obstacle.

### Not done yet
- Nothing planned is left (see PROJECT STATUS at the top); only the optional follow-ups listed there.
- On Windows, packages are installed in the conda env `%USERPROFILE%\miniconda3\envs\vlm-offroad-nav` (Python 3.13.17). As of 2026-10-26 the versions match `requirements.txt` exactly. The `python` on PATH is a separate 3.15 install with no packages. Use the conda env. On the Mac, the env is still empty.
- Open decision: confirm all teammates and the GPU cluster can use Python 3.13. Otherwise standardize on 3.11 or 3.12 (which would also allow Open3D). **This now matters for phase 4:** PyTorch and Transformers must install on the cluster.

### Dataset on disk (verified 2026-10-25, Windows machine)
Located at `data/Rellis-3D/`. All five sequences have images, Ouster scans, per-point labels, and image labels.
- **Layout:** `0000N/{pylon_camera_node, pylon_camera_node_label_id, os1_cloud_node_kitti_bin, os1_cloud_node_semantickitti_label_id}`, plus `camera_info.txt` and `transforms.yaml` **inside each sequence folder**. The ontology is in `Rellis-3D/Rellis_3D_ontology/{ontology.yaml, ontology.csv, ontology.png}`. There is no `poses.txt` or `calib.txt` (not downloaded).
- **Counts:** images == scans == per-point labels per sequence (00000: 2847, 00001: 2319, 00002: 4147, 00003: 2184, 00004: 2059). Image labels cover about every other frame (1200/1161/2074/900/899 PNGs).
- **Filename matching is by frame index:** image `frame000123-1581624652_750.jpg` ↔ scan `000123.bin` ↔ `000123.label`. Verified that every index matches in all sequences. Label PNGs use the image's name with `.png`. A stray `frame001493-1581624802_049.jpg` sits next to its `.png` in `00000/pylon_camera_node_label_id/`. Ignore it and load only `.png`.
- **Images:** 1920 x 1200 JPG.
- **Intrinsics (`camera_info.txt`):** one line `fx fy cx cy` = `2813.643275 2808.326079 969.285772 624.049972`, identical in all sequences. **No distortion coefficients are provided.** Assume zero distortion and judge from the overlay.
- **Extrinsics (`transforms.yaml`):** key `os1_cloud_node-pylon_camera_node`, quaternion `q: {w,x,y,z}` + translation `t: {x,y,z}`. **They differ per sequence** (00000/00001/00003 share one; 00002 and 00004 each have their own), so load each sequence's own file. **Direction:** the evidence says this is the camera pose in the LiDAR frame (`T_lidar_from_cam`, the ROS parent-child convention), so `T_cam_from_lidar` = its inverse. Checks: with R^T, LiDAR +z (up) maps to camera -y (up) and LiDAR -x maps to camera +z (forward). Read the other way, LiDAR up would map to camera left. t = (-0.13, 0.04, -0.17) puts the camera 13 cm forward of and 17 cm below the LiDAR, which is plausible. The overlay is the final confirmation.
- **Ouster scans:** every `.bin` is exactly 131072 points (64 x 2048, organized) x 16 bytes, float32 x,y,z,intensity. About 40% are empty returns at (0,0,0) and must be dropped. Their labels are junk. Forward is LiDAR **-x**. Intensity is small (max about 0.012 in frame 0), probably normalized.
- **Per-point labels:** uint32, one per point, plain class IDs (no instance bits seen).
- **Ontology:** `ontology.yaml` is a 2-item list: [id→name, id→RGB]. It has 20 classes. `ontology.csv` additionally lists uphill (29) and downhill (30).

### Data plan (original download notes)
The user will store the data inside the repo at `data/Rellis-3D/` (gitignored), with sequence folders `00000/`… and the calibration/ontology files at the root keeping their original names. `RELLIS_ROOT` may also point elsewhere.

Download from the RELLIS-3D README (Google Drive can rate-limit large downloads; wait 24 h, email maskjp@tamu.edu, or use the Baidu Pan mirror):
- **Needed now:** Full Images (11 GB), Ouster LiDAR SemanticKITTI Format (14 GB), Camera Intrinsic (2 KB), Basler Camera to Ouster LiDAR (3 KB), Full Image Annotations ID Format (94 MB), Ontology Definition (18 KB).
- **Useful soon:** Ouster LiDAR Annotation SemanticKITTI Format (174 MB, per-point labels), Full Image Annotations Color Format (119 MB), Ouster LiDAR Scan Poses (174 MB), Image and Ouster Split Files (<100 KB).
- **Skip:** Ouster PLY format (26 GB, duplicates the SemanticKITTI scans), everything Velodyne, Velodyne-to-Ouster, Stereo, Calibration Raw Data, ROS bags.
- **Quick start option:** "Image with Annotation Examples" (3 MB) and "Ouster LiDAR with Annotation Examples" (24 MB), plus the two calibration files, are enough to inspect formats while the big downloads run. The sample images and scans may not be the same frames.
- **Next action once data is on disk:** inspect the files and answer the questions in the "Important" note under the Dataset section before writing the loader.

### Related work found (from search summaries only; read the full papers before citing)
- The LiDAR→camera projection is a standard technique (KITTI tools, MATLAB `projectLidarPointsOnImage`, many GitHub scripts). No RELLIS-3D Ouster-specific script was found. We write our own because Objective 3 reuses it in reverse.
- The RELLIS-3D paper is a dataset paper with segmentation baselines. It does nothing with VLMs.
- **PIVOT** (arXiv 2609.20983): the closest match. A geometric planner (C-BIT*) runs by default, and a VLM builds a semantic costmap only when planning fails.
- **Reasoning About Traversability** (arXiv 2604.21249): a fine-tuned VLM (LoRA) outputs 3D trajectories directly from one image, with no costmap or planner.
- **CATNAV** (arXiv 2603.22800): cached vision-language traversability for zero-shot navigation. Only the title has been checked.
- **Interactive Navigation with Traversable Obstacles** (arXiv 2310.08873): an LLM plus Grounding DINO build a costmap aware of traversable obstacles, then A* plans on it. Not off-road.
- **VAMOS** (arXiv 2510.20818): a hierarchical vision-language-action model for navigation.
- **Learning Traversability for Long Horizon Off-Road Navigation** (arXiv 2607.23743): learned traversability from overhead data turned into a planning cost.
- **DriveVLM** (2024, on-road): pairs a slow VLM with a fast classical planner, similar to the asynchronous-guide idea. Cited from memory, not from the search.
- **How this project differs:** the VLM runs continuously and asynchronously as a guide (not only on failure, and not replacing the planner). It outputs both waypoints and cost modifiers via a grid overlay, the waypoints are grounded in 3D on a LiDAR DEM, and evaluation is offline on RELLIS-3D using its labels as ground truth.

### Notes for working with this user
- They are new to this area. Explain concepts in plain language, and explain what a step is for before doing it.
- When they ask for an explanation only, don't make changes.

## Dataset: RELLIS-3D

- Source: https://github.com/unmannedlab/RELLIS-3D (Texas A&M + Army Research Lab, off-road data from the Rellis Campus).
- ~13,556 annotated LiDAR scans and ~6,235 annotated images, across sequences `00000`–`00004`.
- Sensors relevant to us: Basler acA1920-50gc RGB camera (`pylon_camera_node`), Ouster OS1 64-channel LiDAR (`os1_cloud_node`). The dataset also has a Velodyne Ultra Puck and Nerian stereo camera; ignore those for now.
- Expected per-sequence layout (verify on disk):
  - `pylon_camera_node/` — RGB `.png` / `.jpg` images
  - `pylon_camera_node_label_id/` — semantic label images (class IDs)
  - `pylon_camera_node_label_color/` — color-coded label images
  - `os1_cloud_node_kitti_bin/` — Ouster point clouds as KITTI-style `.bin` (float32 x, y, z, intensity)
  - `os1_cloud_node_semantickitti_label_id/` — per-point labels
  - `poses.txt`, `calib.txt` (note: `calib.txt` is Velodyne-vs-camera, not Ouster)
- Calibration is downloaded separately from the README: "Camera Intrinsic" and "Basler Camera to Ouster LiDAR". Use the **Ouster** extrinsics with the Ouster point clouds.
- Classes include grass, tree, bush, mud, puddle, water, rubble, dirt, concrete, asphalt, etc. Read the official ontology file for the exact ID-to-name mapping rather than hardcoding it.

**Important:** Do not assume file names, matrix formats, or frame conventions. Before writing loader code, inspect the actual downloaded files (list directories, open the calibration files, check `.bin` sizes divide evenly by 16 bytes) and confirm:
- Whether the extrinsic is stored as a quaternion + translation or a matrix, and its direction (LiDAR→camera or camera→LiDAR).
- How image and LiDAR filenames correspond (frame index vs. timestamp).
- Whether images need undistortion before projection.

If anything is ambiguous, stop and ask rather than guessing.

## Data location

Raw data lives outside the repo and is never committed. Read its path from an environment variable `RELLIS_ROOT` (fall back to a `data/` folder that is gitignored). Never hardcode absolute paths.

## Tech stack

- Python 3.10+
- NumPy (geometry, point clouds), OpenCV (images, projection, undistortion), Matplotlib (overlays), PyYAML (calibration)
- Open3D optional, for 3D viewing only
- pytest for tests
- Later phases: PyTorch + Hugging Face Transformers for VLM inference on the university GPU cluster
- No ROS. Use the plain file-based dataset, not the ROS bags.

## Suggested repo layout

```
vlm-offroad-nav/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore            # data/, outputs/, __pycache__/, .venv/
├── src/
│   └── offroad/
│       ├── __init__.py
│       ├── data/
│       │   ├── rellis.py         # dataset loader
│       │   └── calibration.py    # intrinsics/extrinsics parsing
│       ├── geometry/
│       │   └── projection.py     # LiDAR -> image projection
│       └── viz/
│           └── overlay.py        # depth-colored overlays
├── scripts/
│   └── project_lidar.py   # run projection on one frame, save image
├── tests/
│   └── test_projection.py
├── data/                  # gitignored
└── outputs/               # gitignored
```

## Coding conventions

- Type hints and short docstrings on public functions; state array shapes and coordinate frames in docstrings (e.g., "points_cam: (N, 3) in camera frame, meters").
- Keep geometry functions pure (arrays in, arrays out, no file I/O) so they are easy to test and reuse in Objective 3.
- Name transforms explicitly by direction, e.g. `T_cam_from_lidar`, never just `T` or `extrinsic`.
- Small, readable functions over clever one-liners. This is a team of students; clarity matters.
- Pin package versions in `requirements.txt` so all four teammates and the GPU cluster match.

## Roadmap (phases)

- Phase 1 (done): data loading and LiDAR-to-camera projection.
- Phase 2 (done): curate the evaluation frames (`configs/eval_frames.csv`, 36 frames).
- Phase 3 (done): DEM / cost grid from LiDAR and a geometry-only A* baseline.
- Phase 4 (done): VLM prompting with a grid overlay; parse structured JSON waypoints and cost weights. Local Qwen3-VL (4B and 8B, 4-bit) via Transformers.
- Phase 5 (done): put VLM waypoints and cost hints onto the DEM and re-weight the planner costs (the hybrid planner), plus the oracle upper bound.
- Phase 6 (done): evaluation, comparing geometry-only against hybrid on the evaluation frames (`evaluate.py`, `make_report.py`, `docs/limitations.md`).
  - Use RELLIS-3D semantic labels as ground truth for hazard metrics (no manual labeling needed).
  - **Ground truth along paths from two sources (user decision, 2026-10-31):** use the LiDAR point labels where a path cell has points; otherwise project the cell into the camera and use the camera label image at that pixel; otherwise "unseen". About 40% of baseline path cells have no LiDAR points (blind circle, occlusion, sparse range, water returns). Limits: the camera can't see the first ~6 m or outside ~38° FOV; a pixel shows the front surface (e.g. a bush) rather than the ground behind it, so compare against LiDAR depth if possible; far pixels are coarse.
  - Score the tall-grass failure with both shortfall and detour (see the phase 3 notes).
- Open question (raised 2026-10-31, **not decided; the user said not to change the roadmap yet**): whether to add VLM fine-tuning (e.g. LoRA on a small open VLM, with targets generated from RELLIS-3D labels and poses) as an extension after zero-shot. The user is checking class expectations.

## Key references

- DARPA RACER (Robotic Autonomy in Complex Environments with Resiliency)
- Jiang et al., "RELLIS-3D Dataset: Data, Benchmarks and Analysis" (arXiv 2011.12954)
- Sima et al., "DriveLM: Driving with Graph Visual Question Answering"
- Cui et al., "A Survey on Multimodal Large Language Models for Autonomous Driving"
