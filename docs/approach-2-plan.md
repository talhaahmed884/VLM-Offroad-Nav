# Approach 2: mapless, language-guided off-road driving (plan)

Adapting the planner + VLM-selector idea of **DriveVLN** (Guo et al., CVPR 2026, "Towards
Mapless Vision-and-Language Navigation in Autonomous Driving") from on-road parking lots to
off-road driving on RELLIS-3D. This is a plan only; nothing here is implemented yet.
Approach 1 is preserved on the branch `approach-1-vlm-grid-costmap`.

## 1. What DriveVLN does (summary)

- **Task.** The vehicle gets only a short, destination-level instruction ("Go to the exit"), no
  map and no step-by-step route. At every decision it must (a) say whether it has arrived and
  (b) pick a direction that leads toward the destination, using what it sees (signs, landmarks)
  and what it decided before.
- **Model: two parts.**
  1. A **planning module** (DiffusionDrive, from camera + LiDAR) proposes several *feasible but
     destination-agnostic* candidate trajectories (8 waypoints over 4 s). It only knows what is
     drivable, not where to go.
  2. A **navigation selector** (Qwen2.5-VL-3B, fine-tuned) sees the front camera image with the
     top-4 candidates **drawn in different colors**, plus the instruction and a rolling history
     of its earlier questions and answers. It answers in a fixed format: "not arrived; yellow".
- **Training.** Supervised fine-tuning (LoRA, rank 64, 5 epochs, 800×600 images) on single
  frames, then reinforcement learning (GRPO) in simulation. The reward combines a *local* term
  (how close the chosen trajectory is to the ground-truth one, zeroed on any collision) and a
  *global* term (how much of the shortest route it followed, plus a bonus for arriving).
- **Benchmark.** 200 CARLA scenes rebuilt from real scans; closed-loop evaluation. Metrics:
  success rate, navigation error, L2 error of the chosen trajectory at 1/2/3 s, collision rate,
  and a Driving Score = route completion × infraction penalty.
- **Key results.** Fine-tuned 3B selector: Driving Score 0.67, success 0.44. Zero-shot GPT-5: 0.48;
  Qwen2.5-VL-72B: 0.45. RL added a lot (success 0.21 → 0.44).

## 2. Our task: "keep going" off-road, with optional coarse directions

**Decision (2026-11-22): no destinations.** Off-road, destinations are arbitrary and usually
unnamed. A realistic command is "keep driving (this way) until I stop you". The stop comes from
the operator or from odometry, not from the VLM. So DriveVLN's "arrived?" question is replaced,
and the instruction becomes optional and coarse.

At each decision frame, the planner proposes K drivable candidate paths, drawn in colors with
numbered distance ticks. The VLM answers, in one call:

```json
{
  "choice": "<color of the path to drive>",
  "safe_until": {"<color>": "all" | "tick 1" | "tick 2" | "none", "...": "..."},
  "action": "continue" | "stop",
  "reasoning": "<one or two sentences>"
}
```

- **choice:** the path to drive next.
- **safe_until:** for every candidate, how far along it is safe, read off numbered ticks at 5, 10,
  15 m (VLMs are poor at meters but can point at a tick). This is the user's idea of letting the
  VLM check how far the planner's path is really drivable.
- **action:** "stop" only when no candidate is safe; this replaces DriveVLN's arrival flag.

Two prompt modes share this answer format, the frames, the candidates, and the labels:

| | Mode A: "keep going" | Mode B: "keep going + direction" |
|---|---|---|
| Instruction | none: "drive on safely; prefer trails; avoid water, mud and obstacles" | plus a coarse direction, e.g. "bear left", "keep straight", "bear right" |
| Main question | can a small VLM keep the vehicle safe? | can it also follow a coarse human direction? |
| Scored on | safety of the chosen path; safe-distance answers; stop decisions | the same, plus "did it take the path the driver took" |

Mode B's direction is generated automatically from where the driver actually went over the next
~50 m, and is kept coarse (a few discrete options) so it doesn't give the answer away. It plays
the role of DriveVLN's instruction (and of a coarse GPS heading) and gives forks a right answer.

**Both modes are evaluated.** One fine-tuned model is trained on a mix of both (some examples
with a direction, some without).

### Mapping from DriveVLN

| DriveVLN (on-road) | Off-road adaptation |
|---|---|
| Destination instruction ("Go to the exit") | None (Mode A) or a coarse direction (Mode B); the stop is external |
| "Arrived or not" | "Continue or stop", where stop = no safe candidate |
| Choose the candidate toward the destination | Choose a safe candidate (A), or the one matching the direction (B) |
| — | New: how far each candidate is safe (distance ticks) |
| DiffusionDrive planner (learned) | Our cost map + A\* toward a fan of directions (from approach 1) |
| CARLA closed loop | Offline, open-loop on recorded RELLIS drives; the driver's path is the reference |
| Collision in simulation | Path crosses LiDAR points labelled as hard obstacles |
| — | Water / mud / vegetation on the path, from approach 1's two-source ground truth |

This also fixes approach 1's main weakness: there the VLM had to name grid cells and re-weight
costs, and it described only about 12% of the ground. Choosing among a few drawn paths and
pointing at a tick is a much easier question for a VLM.

## 3. What we reuse from approach 1

- data loading, calibration, LiDAR-to-camera projection (`data/`, `geometry/`);
- the terrain map and cost map (`mapping/`), A\* and Dijkstra (`planning/astar.py`);
- drawing paths on the camera image (`viz/baseline_figure.draw_path_on_camera`);
- the local VLM runner and the GPU setup (`vlm/backend.py`, `vlm/runner.py`);
- two-source ground truth along a path, hazard and clearance metrics (`eval/path_truth.py`, `eval/metrics.py`);
- `run_all.py`, the test conventions, and the oracle idea (an upper bound with perfect choices).

These are the only parts of approach 1 kept on this branch (the branch was cleaned on
2026-11-22). Ideas from deleted approach-1 code, such as the forgiving JSON answer checker, the
paired wins/losses comparison, and the report tables and charts, will be rewritten for approach 2
as needed; the originals stay on the `approach-1-vlm-grid-costmap` branch.

## 4. New data and labels

1. **Ouster scan poses** (RELLIS-3D download "Ouster LiDAR Scan Poses", about 174 MB, skipped so
   far). One pose per scan gives the vehicle's actual driven path. Everything below depends on
   it. On download, check the format (expected: one 3×4 matrix per line, SemanticKITTI style),
   which frame it is in, and that it lines up with our forward-left-up map frame.
2. **Decision frames.** Every RELLIS frame that has a scan and a pose, sampled at about 2 Hz (as
   DriveVLN recorded), so neighbouring frames aren't near-duplicates.
3. **Labels, all generated automatically:**
   - **driven path:** the next ~20 m of the recorded trajectory, in the vehicle frame;
   - **driver's candidate:** the candidate closest to the driven path; candidates within a
     tolerance (e.g. ≤ 1 m mean distance) also count, so frames with several equivalent paths
     aren't scored as wrong;
   - **safe distance per candidate:** the first water or hard-obstacle cell along it (puddles
     are "caution" since 2026-11-22, see setting 6),
     from the two-source ground truth, rounded to a tick;
   - **stop:** true when every candidate is unsafe within the first tick;
   - **Mode B direction:** the bearing of the driven path ~50 m ahead, binned into a few words.
4. **Splits by sequence, not by frame** (neighbouring frames are correlated): for example
   train 00000–00002, validate 00003, test 00004.

## 5. Implementation phases

Each phase ends in something we can run and check, as in approach 1.

### Phase 1: poses and decision frames
- Download the poses; write a loader and a check script that draws the driven trajectory of a
  sequence top-down and projects the next 20 m of it into the camera image. The projected path
  should lie on the trail the vehicle actually drove.
- Frame builder (pure + tests): sample decision frames at ~2 Hz, compute each frame's driven path
  in the vehicle frame and its Mode B direction word; write the split files.

**Status (2026-11-22): done; the user ran it and reviewed the picture.**
- **Poses checked:** `poses.txt` has one 3×4 matrix per scan (two files lack a final newline, so
  `wc -l` shows one fewer). They are `T_world_from_lidar`: two scans 2–3 s apart overlap 59–65%
  after moving them to the world frame this way, versus 1–11% with the inverse. The vehicle moves
  along LiDAR −x at about 1.25 m/s, and the drives are 226–483 m long. Along the driven path the
  LiDAR stays about 1.2 m above the mapped ground (00000/600: 1.16–1.28 m), so the poses' heights
  agree with the LiDAR ground.
- **Decision frames:** 2,220 (train 1,553 / val 336 / test 331), after skipping frames where
  the vehicle stands still (91) or the recording ends within 20 m (308). Mode B directions are
  mostly "straight" (train 862 of 1,553; val 256 of 336; test 250 of 331). The test drive has
  only 3 "bear left" and 0 "bear right" frames, so Mode B results on test will mainly show the
  slight turns; report per word.
- **Early warning for phase 2:** in 00000/2042 the driven trail beyond ~15 m runs through cells
  the cost map marks as blocked (grass and brush along the trail edges). Candidates from the cost
  map may not follow real trails there; phase 2's candidate-quality check will tell.
- **Added in phase 2:** only about every other frame has a camera label image, and the safe-distance
  labels need it, so each sampled frame now moves to a labelled neighbour (0.1 s away) when it has
  none (`prefer_labelled`). 2,207 of the 2,220 decision frames have one (13 have no labelled
  neighbour); the CSV gained a `labelled` column. The frame count and directions are unchanged.

### Phase 2: candidate paths and labels (the planning module)
- **K diverse, feasible candidates** per frame from the LiDAR cost map: end points on an arc
  ~15 m ahead at bearings from −60° to +60°, A\* to each, near-duplicates dropped (paths less than
  ~1.5 m apart), keep up to 4 (as DriveVLN). Smooth the A\* paths so they look drivable.
- Labels: driver's candidate(s), safe distance per candidate, stop.
- **Candidate-set quality:** how close the best candidate gets to the driven path (the ceiling
  for any selector), and how often frames have a real choice (≥ 2 safe, clearly different
  candidates). Tune the generator here before involving a VLM.
- Later comparison if there's time: **curved anchors**, a fixed library of smooth, drivable path
  shapes (straight, gentle/sharp left/right) built by clustering the recorded drives, placed in
  front of the vehicle and kept only if they don't hit obstacles. This is closer to DriveVLN's planner.

**Status (2026-11-22): done and verified by the user's run (see the end of this section).** The user asked
to build two cost grids and compare them ("strict" = approach 1's cost map, where anything over
0.3 m blocks; "relaxed" = only hard obstacles over 1.5 m block, lower vegetation costs 10).
- **How candidates are made** (`nav/candidates.py`): one Dijkstra shortest-path tree per frame;
  a fan of 13 bearings (−30° … +30°, every 5°) at 15 m; for each, the goal that trades closeness
  to the target against cost (the baseline's rule); smoothed, resampled every 0.25 m, cut at 15 m.
  Changes from the first sketch above, found on real frames:
  - **Camera view:** the camera sees only about ±19°, so ±60° candidates were invisible to the
    VLM. A candidate must now keep ≥ 80% of its points from 5 m on inside the view (2° margin).
  - **Duplicates:** "mean distance < 1.5 m" merged paths that share the trail for 7 m and then
    split 3–4 m apart. Two paths are now different if they get ≥ 2 m apart somewhere (compared
    point by point at the same distance along them). On a sample this raised the strict / relaxed
    mean from 1.3 / 1.5 to 2.2 / 3.1 candidates.
- **Labels** (`nav/labels.py`, `nav/frame_candidates.py`):
  - The driver match is measured from the first tick (5 m) on; with the shared, invisible start
    included, nearly every candidate "matched". The driver's choice is the closest candidate.
  - A hard-obstacle label counts only where the LiDAR also sees something taller than 0.3 m in or
    next to the cell: RELLIS marks some flat ground as "person" (people walking near the vehicle
    smear onto the ground in the annotation). An unsafe stretch must be ≥ 0.5 m (2 points) long.
  - The driven path is labelled like a candidate, as a check on the labels.
- **Trial run, all 2,220 frames (~2 min with 4 workers):**

  | | strict | relaxed |
  |---|---|---|
  | candidates per frame (mean) | 2.20 | 3.13 |
  | frames with 0 / 1 / 2 / 3 / 4 | 4 / 22 / 36 / 29 / 10% | 1 / 4 / 17 / 36 / 42% |
  | best candidate to driven path, median / 90th pct | 0.61 / 1.14 m | 0.68 / 1.34 m |
  | driver match (a candidate within 1 m) | 83% | 83% |
  | matching candidates per frame | 1.24 | 1.34 |
  | real choice (2+ safe to tick 1) | 75% | 95% |
  | stop | 6% | 2% |
  | full length | 86% | 98% |
  | candidates through a puddle (caution) | 9% | 8% |
  | driver's candidate through a puddle / of those, with a dry alternative | 9% / 21% | 9% / 42% |

  (Numbers with puddles as caution. With puddles unsafe, the first run gave real choice 72 / 92%
  and stop 8 / 4%.)

  - The ~17% of frames with no match are mostly sharp turns: the driver's path leaves the
    camera's view (53–63 of ~95 misses in a sample); the rest miss by 1.0–1.3 m.
  - Relaxed candidates almost never cross vegetation (cost per m ≤ 3.4); they gain options
    because vegetation isn't inflated, e.g. the left rut, the trail, and a branch into tall grass.
  - **Puddles:** the driver drives through puddles. With puddles unsafe, the labels called the
    driver's own path unsafe in 6% of frames (137 puddle, 4 person), and in most "stop" frames
    the driver kept going (150 of 183 strict, 50 of 79 relaxed). **User decision 2026-11-22:
    puddles are "caution"**: they don't shorten the safe distance or cause "stop", but each
    candidate records where its first puddle is (`first_caution_m`). Now the driver's own path
    is safe to the end in 100% of frames; the driver goes through a puddle in 6% of them.
  - **"Stop" on RELLIS is only planner failure.** All remaining stop frames (129 strict, 35
    relaxed) have no candidates or only candidates that end before 5 m, and the driver kept
    going in every one. RELLIS has no situation where the driver really had to stop, so stop
    can't be evaluated meaningfully here; it needs the simulator (phase 6). Phase 3–4 should
    decide whether to leave these frames out of the stop metrics.
  - **Safe distance is nearly always "all"** (96% of relaxed candidates); the unsafe ones are
    trees (83), "person" (19) and rubble (11). Safety on RELLIS will mostly show in the caution
    metric: in relaxed, 42% of the frames where the driver's candidate crosses a puddle (~85
    frames) also have a dry candidate. This makes balancing in phase 5 important.
- **User decision 2026-11-22: the relaxed cost grid** is used from phase 3 on
  (`CHOSEN_VARIANT`): more and more distinct candidates (3.1 vs 2.2 per frame, real choice 95%
  vs 75%) with the same driver match (83%). `build_candidates.py` still builds both, for the report.
- **User decision 2026-11-22: keep the 1 m match tolerance** (setting 5, measured from 5 m on;
  ~1.3 matching candidates per frame). Training uses the single closest candidate as the
  driver's choice; the tolerance only makes scoring forgiving of near-identical choices.
- **Verified 2026-11-22: phase 2 is done.** The user ran `run_all.py`; `configs/decision_frames.csv`,
  both JSONL files and `comparison.md` are byte-identical to my trial run.
- **Notes for phase 3's picture** (from `outputs/candidates_00000_000601.png`): at 10–15 m the
  candidates are squeezed into a thin band of image rows and cross each other near the bottom;
  their far ends show small hooks and wiggles because each dot sits on its own map cell's
  ground height. Draw them as lines on a smoothed ground (e.g. a plane fitted under the path, or
  the pose height minus the sensor height as `show_driven_path.py` does), with ticks that stand out.

### Phase 3: visual prompt and zero-shot selector
- Draw the candidates on the front camera image in distinct colors that stand out from grass and
  sky, each with a name at its end and numbered ticks at 5 / 10 / 15 m. Paths start below the
  image (the ~4.4 m blind zone) and are drawn from their first visible point.
- Prompts for Mode A and Mode B, the JSON answer above, and a forgiving checker (as in approach 1).
- **History:** the last few decisions as a short text summary (cheap on memory), with earlier
  images as an option if memory allows.
- Run zero-shot with **Qwen3-VL-4B** on the laptop; the existing 8B as a reference point.
- **Against shortcuts** (user decision 2026-11-22, see section 8): assign the colors to the
  candidates at random (and list them in the prompt in random order, not left to right); draw all
  candidates in the same style (same line width and dot size); no costs or other planner numbers
  in the prompt. The model should tell the candidates apart only by color and what it sees.
- **History:** prompts are built so the history can be left out, and evaluated both with and
  without it (see phase 4).

**Status (2026-11-22): written; my trial on 12 validation frames is below; not yet run by the user.**
- **Picture** (`nav/visual_prompt.py`): the camera image shrunk to 800 × 500 (the camera's
  shape; setting 10 said 800 × 600, which would stretch it). Each candidate is a colored line
  with a dark outline (drawn side by side where candidates share a stretch, see below), white dots at 5 / 10 / 15 m (the 5 m dot is at or just below the bottom
  edge; the camera starts seeing at ~4.4 m), and its color name at the far end. The lines sit on
  a smoothed ground (a quadratic along each path fitted to the cost map's ground, one round of
  outlier removal, held constant in the blind zone) and get extra smoothing for drawing only
  (3 m window: the 0.25 m grid zigzag showed as ~20 px wiggles at 15 m). Only run-time data is
  used (image, calibration, candidates, LiDAR ground), never poses or labels.
- **Random colors and order** per frame, seeded from sequence and frame so runs repeat
  (`assign_colors`, `frame_seed`, `prompts.shuffled`).
- **Prompts** (`nav/prompts.py`): Mode A ("keep driving … no destination"; prefer the trail) and
  Mode B (plus the direction word and its meaning). The tick rules are spelled out, unsafe =
  deep water or something the vehicle would hit, and grass, low bushes, mud and puddles are
  drivable ("prefer a dry path when one is as good"). Answer: `{"safe_until": {...}, "choice",
  "action", "reasoning"}`. The per-candidate judgement comes **first**: with the choice first,
  the 4B justified its pick by calling every other candidate "none" (first try, 6 frames).
- **History** (`history_text`): the last 3 decisions without colors ("2.0 s ago: continued
  slight left"), from the model's own earlier answers in the run, as in closed loop; or none.
- **Checker** (`parse_answer`): first JSON object, case-insensitive colors ("Red path" → red;
  two colors named → no choice), tick synonyms ("tick2", "10 m", 2), action synonyms.
- **Scoring** (`nav/scoring.py`, labels only as the answer key): choice matches the driver /
  is the closest candidate, chosen candidate safe, chose a puddle when a dry path existed, ticks
  exact / within one / too optimistic / too cautious, action correct.
- **Scripts:** `show_prompt.py` (the exact picture and prompt, plus the answer key) and
  `run_selector.py` (`--split`, `--every`, `--limit`, `--modes`, `--history own|none`, `--model`,
  `--tag`; resumable; → `outputs/selector/<tag>_mode{A,B}.jsonl` + `<tag>_summary.md`).
  `run_all.py` steps `prompt` and `selector` (slow: 3 frames).
- **Trial, Qwen3-VL-4B 4-bit, first 12 validation frames (00003, every 4th):** 100% valid
  answers, ~5 s per answer, 3.4 GB GPU memory. The chosen path was always safe, but it matched
  the driver in only 1–2 of 12 (from frame 168 on, no candidate matches: the driver turns), and
  it often picked the rightmost candidate. Ticks were never too optimistic but always too
  cautious: it never answers "all", stopping at "tick 2" even on clear paths. It mentions real
  objects (the orange barriers in 00003/42) beyond the 15 m the labels cover. This is the
  zero-shot starting point; further prompt tuning on validation frames would overfit the prompt,
  so it stops here, and fine-tuning (phase 5) is where this should improve.
- **The user's run (2026-11-23), 4B and 8B, val every 4th (83 frames asked, both modes):**

  | | 4B A | 4B B | 8B A | 8B B |
  |---|---|---|---|---|
  | valid answers | 100% | 100% | 100% | 100% |
  | choice matches the driver | 25% | 28% | 19% | 27% |
  | ticks exact | 2% | 2% | 37% | 45% |
  | ticks too optimistic / too cautious | 0 / 98% | 0 / 98% | 0 / 63% | 0 / 55% |
  | seconds per answer | 5.8 | 5.5 | 4.5 | 4.2 |

  - **Choice no better than chance:** a random pick would match the driver in 32% of these
    frames, the straightest candidate in 36%, the middle one in 40%; the best possible is 82%
    (in the other frames no candidate matches). The 4B and 8B agree on only ~55% of choices.
  - **Mode B ignores the word:** in the 15 "slight left" frames, the leftmost candidate would
    match 60%, the models 27%.
  - **Safety numbers are trivial on this drive:** 280 of 284 val candidates are labelled "all"
    (train: 95%), so "chosen safe", "action correct" and "always all" (99% of ticks exact) need
    no vision. Phase 4 must report each number next to these baselines; phase 5 needs balancing.
  - **Picture problem:** the candidates branch off one shortest-path tree, so about half share
    their first 10 m with another (within 0.1 m), and the line drawn last hid the other (in 223 of
    321 val frames a candidate was hidden up to 10 m, e.g. 00003/22, where the two hidden ones
    were the driver's).
- **User decision 2026-11-23: shared stretches are drawn side by side** (picked from mock-ups:
  now / 50% transparent / side by side / both). Transparency alone doesn't help: exactly
  overlapping lines blend into a new color (red over yellow → orange) that names no candidate,
  and without the outline the lines fade on grass. `spread_overlapping` shifts lines that would
  overlap (within 8 px at the same distance along) 7 px apart, in the left-to-right order of
  where they end, smoothed over 9 points; tick dots stay in the middle of a shared stretch.
  `PICTURE_VERSION = 2` is stored with each answer, and `run_selector.py` refuses to add to a file
  with answers to another version.
- **The user's rerun with picture version 2 (2026-11-23, tags `val_4B_v2`, `val_8B_v2`):** choice
  matches the driver 4B 25% / 27%, 8B 25% / 27% (Mode A / B); ticks exact 4B 1–2%, 8B 37% / 46%;
  4.2–5.2 s per answer. Choices changed in only 12–18% of frames, and the match rate stayed the
  same (8B Mode A 19% → 25%, 5 frames). So the hidden lines were not the main reason for the weak
  choices; e.g. in 00003/22 both models now pick red (the driver's), but overall the zero-shot
  choice is still at chance (random 32%). No listing-order bias (first listed chosen as often as
  chance); a color preference (4B: yellow 33, blue 28, red 13, magenta 9 of 83), which the random
  colors turn into noise instead of a fake score. One likely reason the driver match is low: on a
  wide trail, both ruts (or the trail and its edge) are equally reasonable, and only one is "the
  driver's". **Phase 3 is done** (pending the user's review); next is phase 4 when the user says.

### Phase 4: evaluation
- **Baselines:** random choice; the geometric choice (lowest-cost candidate); the straightest
  candidate; "always all safe, continue"; "follow the direction" without vision (the candidate
  closest to the Mode B bearing); and the **oracle** (always a correct candidate, always correct
  safe distances). On trails, the straightest or cheapest candidate is often right, so the VLM
  must beat these to show it reads the image rather than a shortcut.
- **Per-frame metrics:**
  - path choice: water / mud / vegetation on the chosen path, collision proxy (crosses
    hard-obstacle points), clearance; and in Mode B, agreement with the driver and L2 error
    against the driven path at 5 / 10 / 15 m;
  - safe distance: accuracy of the tick answers (exact, and within one tick), and how often it
    calls an unsafe stretch safe (the dangerous error);
  - stop: precision and recall of "stop".
- Report results separately for **frames with a real choice**, since many off-road frames have
  only one sensible path, and **per answer type** (e.g. how many true "stop" frames it catches,
  how many unsafe stretches it calls unsafe), not just overall accuracy: "always all safe,
  continue" is right most of the time.
- **Does it follow the direction?** (Mode B) Ask the same frame with different direction words
  ("bear left", then "bear right"); where a matching candidate exists, the choice should change.
- **Does it lean on the history?** Evaluate with and without the history, and with a wrong
  earlier decision in it. In closed loop the history holds the model's own (possibly wrong)
  decisions, not the driver's.
- ~~**Comparison with approach 1:** score approach 1's baseline and hybrid paths with the same
  safety metrics (and driven-path L2) on the same test frames.~~ **User decision 2026-11-23:**
  not in phase 4; the two approaches are compared at the end, using approach 1's existing report
  (its numbers come from 36 frames with destinations, so the comparison is side by side, not on
  the same frames). The "cheapest" baseline is roughly what approach 1's geometric planner picks.
- **User decision 2026-11-23: all 331 test frames** (sequence 00004, `--every 1`).

**Status (2026-11-23): written; checked on the validation runs; the test runs are the user's.**
- **Terrain numbers per path** (`nav/candidate_metrics.py`, stored by `build_candidates.py` as
  `metrics` on every candidate and on the driven path; the labels are unchanged, checked
  byte-for-byte): shares of water / puddle / mud / grass-bush / tree-log / no ground truth,
  closest hard obstacle (`clearance_m`), near miss (< 0.75 m, half the vehicle's width), slope
  and bump height. Hard obstacles count only if they stand more than 0.3 m above the ground
  (new `min_height` in `eval.metrics.hard_obstacle_xy`; approach 1's default is unchanged):
  without it, ~1/3 of all paths, the driver's included, had a "near miss", all from "person"
  labels on flat ground (median 6 cm high). After the fix: 6–9% for candidates and 8% for the
  driven path (val), mostly branches 1.65–1.8 m up that the vehicle drove under.
- **Evaluation** (`nav/evaluation.py`, `scripts/evaluate.py`): a choice is a weight per candidate
  (one-hot; uniform for random, so random is the exact average). Baselines: random, straightest,
  cheapest, Mode B "toward the word" (end bearing closest to the word: straight 0°, slight 17.5°,
  bear 35°), driver (oracle), and the **driven path itself as a reference column**. Rows: matches
  the driver (all / real choice / turn frames), closest candidate, safe to tick 1 / end, puddle
  when a dry path existed, terrain shares, closest hard obstacle, near misses, slope, distance
  from the driven path at 5 / 10 / 15 m. Ticks: "always all" vs runs, with exact / within one /
  too optimistic / too cautious, "unsafe stretch called safe past it", and per true answer type.
  Continue / stop: "always continue" vs runs, stop recall and precision. All columns use the
  same frames (those every run answered).
  **Added 2026-11-24 (phase 5, for the final report):** 95% intervals (Wilson for shares, normal
  approximation for means), straight vs turn frames, side of wrong choices (left / right of the
  driver's candidate), chosen position, distance at 15 m (mean with interval, median, 90th
  percentile), ADE (mean over 5 / 10 / 15 m), ticks as hazard flags (precision / recall / F1)
  and mean tick error, stop F1, speed (mean / median / 95th percentile seconds), and paired
  comparisons (every run vs straightest and cheapest, plus `--paired`; exact McNemar).
  Precision / recall / F1 are used only for the yes/no detections (hazard flags, stop); the
  path choice is one pick among 2–4 candidates, where accuracy is the right measure.
  The user regenerated all five reports with them (2026-11-24; identical to my rebuild). Paired
  tests, skip epoch 4, Mode A: on **val** it beats straightest (right where straightest is wrong
  66 times vs 25, p < 0.001) and cheapest (54 vs 34, p = 0.042); on **test** it ties
  straightest (50 vs 51, p = 1.0) and beats cheapest (81 vs 22, p < 0.001) and the zero-shot 4B
  (110 vs 35, p < 0.001). ADE: val 1.23 m (straightest 1.39), test 0.98 m (straightest 1.07).
- **Behaviour tests** (`run_selector.py`): `--direction WORD` gives Mode B the same word in every
  frame (`evaluate.py --direction-test LEFT RIGHT`: followed / ignored / reversed, and how often
  the leftmost / rightmost is picked); `--history wrong` replaces the latest history entry with a
  sharp turn not taken, opposite to the real one (or a seeded random side) (`--history-test OWN
  WRONG`: moved toward / away / same). Default tags get `_nohist`, `_wronghist`, `_dir-<word>`.
- **First look, validation (val every 4th, picture v2):** the cheapest candidate matches the
  driver in 47% (turn frames 58%), straightest 36%, random 32%, "toward the word" 42%, oracle 82%;
  the 4B and 8B 25–27%. So the geometric planner's own pick is far ahead of the zero-shot VLM. The
  driver's center line runs over much less "mud" than the candidates (1.7% vs ~10%): the wheels
  are in the ruts, the center on the grass between them, while candidates follow the ruts.
- **Zero-shot results on the test drive (2026-11-23; the user ran everything on Polaris; all 331
  frames, 328 with candidates; `outputs/eval/test_evaluation.md`):**
  - **Choice matches the driver:** Mode A: random 43%, straightest 67%, cheapest 48%, oracle 89%;
    4B 43% (no history 45%, wrong history 45%), 8B 42%. Mode B: toward the word 67%; 4B 48%,
    8B 47%. In turn frames (83 of 328, mostly "slight left"): Mode A 4B 18%, 8B 19% (random 33%,
    straightest 43%); Mode B 4B 41%, 8B 35% (toward the word 45%). So zero-shot the VLM chooses
    at chance level, far below "straightest"; the test drive is mostly straight (245 of 328
    frames), which is why straightest and random score higher than on validation.
  - **Position bias:** the 8B picks the rightmost candidate in 222 of 326 Mode A frames (68%);
    the 4B prefers middle / right.
  - **The word is understood:** with "bear left" vs "bear right" the choice ends further left
    under "bear left" in 89% of 326 frames, never the reverse; rightmost under "bear right" 97%,
    leftmost under "bear left" only 48% (the right lean again). In normal Mode B, where the word
    is mostly "straight", it doesn't pick the straightest.
  - **History is ignored:** a made-up sharp turn changes the choice in only 5% (Mode A) / 8%
    (Mode B) of frames, as often away from it as toward it; no history ≈ own history.
  - **Safety and stop can't be measured on this drive:** only 6 of 1,123 candidates have an unsafe
    stretch (no model called them safe past it; "always all" got all 6 wrong), 7 are labelled
    "tick 2", and there are no stop frames. Ticks: the 4B says "all" almost never (8–17 of
    1,116), the 8B is right 40% / 50% (A / B), both never too optimistic.
  - **Puddles** (13 frames only): the 8B took the puddle when a dry path existed in 31% (4 of
    13), random 46%, the 4B 54%. Terrain, near misses and slope are about the same for every
    chooser and similar to the driven path.
  - **Target for phase 5:** beat "straightest" / "toward the word" (67%) on driver match, and
    show the ticks distinguish paths; the oracle (89%) is the ceiling.
- **Phase 4 is done** (closed by the user, 2026-11-23).

### Phase 5: fine-tuning
- **QLoRA supervised fine-tuning of Qwen3-VL-4B on the laptop** (RTX 5060, 8 GB): the base model
  stays frozen in 4-bit, only small LoRA adapters are trained. Batch size 1 with gradient
  accumulation, gradient checkpointing, 800×600 images (as DriveVLN), a mix of Mode A and Mode B
  examples, shuffled color assignment (so the model can't learn "always blue"), and paraphrased
  instructions.
- **Against overfitting** (user decision 2026-11-22, see section 8):
  - **History dropout:** in some training examples leave the history out, and in some put a
    wrong earlier decision in it, so the model doesn't learn "repeat the last answer" (in the
    training data the history comes from the driver, who is almost always right).
  - **Balance:** most frames are "straight", ~90% of candidates are "safe all", and "stop" is
    rare (~4%). Use the rare frames (turns, unsafe candidates, stop) more often and the easy
    straight ones less, so the model can't score well by always answering "all safe, continue".
  - **Mirroring:** flip the image left-right, mirror the candidates, and swap left / right in
    the Mode B word. This doubles the turn data and balances left and right (training has 382
    right-turn and 309 left-turn frames; the test drive has 20 "slight right" and no "bear right").
  - **Frames where no candidate matches the driver** (user decision 2026-11-23: an experiment).
    238 of the 1,527 training frames with candidates (15%; 44–49% of the "bear" frames, 3% of
    "straight"): the driver turned off a trail that is still open ahead (235 of them have a
    candidate safe to tick 1), usually out of the camera's view. Not dead ends: those are the 26
    frames with no candidate at all, where the vehicle stops without asking the VLM. Train with
    and without these frames and compare. Options for "with": the safety part only (no choice
    target) in Mode A, and the candidate closest to the driver in Mode B, where the direction
    word carries the intent.
  - **Photometric augmentation:** small random changes of brightness, contrast, color, and blur,
    so the model doesn't fit RELLIS's look (simulator images look different).
  - **Few epochs, early stopping:** frames 0.5 s apart on the same trail are near-duplicates, so
    the ~1,550 training frames hold far fewer distinct examples than it seems. Validate on 00003,
    stop when it stops improving, and use the test drive (00004) only for the final numbers.
  - **Skip uncertain frames:** leave out frames where most of the candidates' ground has no
    ground truth.
- Compare zero-shot vs fine-tuned on the test sequence, in both modes. Zero-shot can't have
  overfit RELLIS, so it's the reference for how much fine-tuning helps outside it.
- **Check general ability:** after fine-tuning, run a few non-RELLIS images (e.g. from the
  simulator) to make sure the model still judges drivable ground sensibly.
- **Optional afterwards:** GRPO with a per-frame reward (choice safety + agreement with the
  driver + tick accuracy), which needs no simulator. DriveVLN's route-level reward needs closed
  loop and is out of scope.
- **8B:** decide after the 4B results. Fine-tuning an 8B doesn't fit in 8 GB; it would run on the
  Polaris H200.

**User decisions (2026-11-23, at the start of phase 5):** train on **Polaris** (the laptop is for
smoke tests and for building the data); the training answers **leave out the reasoning field**
(no human-written reasons exist), so the fine-tuned model is also asked without it
(`build_prompt(..., reasoning=False)`; the zero-shot model can be asked the same way with
`run_selector.py --no-reasoning` for a like-for-like comparison).

**Status (2026-11-24): done, closed by the user.** Written and smoke-tested 2026-11-23; trained,
validated and tested on Polaris 2026-11-24 (details below). **Reported model: skip epoch 4.**
Decisions at closing (the user accepted my recommendations, 2026-11-24):
- **GRPO: not done.** It wouldn't fix the main weakness (the word is ignored), and RELLIS has
  almost no unsafe or stop situations to reward.
- **8B fine-tuning: not done.** The zero-shot 8B was no better than the 4B (42 vs 43%) and
  leaned right; the fine-tuned 4B answers the research question.
- **General-ability check: moved to phase 6.** It needs candidates (LiDAR) on non-RELLIS
  scenes, which only the simulator gives.
- **The ignored direction word is reported as a finding and limitation**, not fixed (a fix would
  be another training round with examples whose word is changed and whose target follows it).
- **Data** (`nav/sft.py`, `scripts/build_sft_data.py` → `outputs/sft/<name>/`): pictures are
  rendered on the laptop (JPEG, 800 × 500), so Polaris needs only this folder. Per training
  frame and epoch, `frame_weight` examples on average: easy straight 0.5, turn 1.5, puddle 2,
  unsafe candidate 3, stop 3 (~1,700 per epoch). Each example draws its own mode (A / B 50%),
  mirroring (50%: picture, word and history flipped; the color names stay readable), history
  (driver 55%, none 30%, wrong 15%), wording (3 wordings of the task and preference;
  evaluation uses wording 0, unchanged from phases 3–4), colors, listing order, and photometric
  change of the camera image (±15% brightness / contrast / saturation, slight blur in 30%) before
  the lines are drawn. History entries are the driver's candidate's end bearing at the earlier
  decision frames (like the chosen candidate's at run time). Frames whose candidates have on
  average more than half their points without ground truth are skipped (14 training frames).
  Validation examples: one per frame and mode, plain (as at evaluation), for the validation loss.
  `--no-match skip|keep` makes the two datasets of the experiment; in "keep", Mode A examples of
  no-match frames have the choice masked out of the loss.
- **Target:** `{"safe_until": {...in the listed order...}, "choice": ..., "action": ...}` from
  the labels; the choice is the driver's (closest) candidate; "stop" when no candidate is safe to
  tick 1.
- **Training** (`vlm/finetune.py`, `scripts/train_selector.py` → `outputs/adapters/<name>/`):
  the base model loaded in 4-bit exactly as for the selector; LoRA rank 64, alpha 128, dropout
  0.05 on the language model's linear layers (132 M trainable parameters, 2.9%); learning rate
  1e-4, cosine, 3% warm-up; batch 1 × 8 accumulation; gradient checkpointing. The sequence is
  the selector's prompt (the same chat template, checked token for token) plus the target and
  `<|im_end|>`; only the answer is learned. The data is read in order, and the adapter is saved
  at the end of each epoch (`epoch1` … `epoch5`) with the validation loss logged.
- **Early stopping by choosing the epoch:** ask the validation frames with each epoch's adapter
  (`run_selector.py --split val --adapter ...`), compare with `evaluate.py --split val`, and keep
  the best; only that one goes to the test drive.
- **Selector:** `run_selector.py --adapter <folder>` (tag `<split>_ft-<data>-<epoch>`, asked without
  reasoning) and `--no-reasoning`; each answer records `adapter` and `reasoning_asked`, and the
  script refuses to mix answers from different adapters or prompts in one file.
- **Smoke test (laptop):** 3 training steps on a 6-frame dataset ran (~6 s per step on the RTX
  5060), saved an adapter, and the selector loaded it and answered in the short format.
- **Full build check ("keep", scratchpad):** 8,522 training examples (1,686–1,727 per epoch),
  666 validation examples, 1.5 GB, ~6 min with 4 workers. Classes: turn 4,422, easy 1,865,
  unsafe 1,515, puddle 720; mirrored 51%; history driver / none / wrong 55 / 30 / 16%; choice
  masked in 829 (Mode A no-match). **No stop examples:** the only 3 training frames labelled
  "stop" with candidates are also "uncertain" (00002/92–102) and skipped, so the model can't
  learn "stop" from RELLIS (as expected from phase 2: no real stop situations here).
- **Training on Polaris (2026-11-24, the user ran both; H200):** no errors, no `nan`, gradient
  norms ≤ 0.4 after the warm-up (≤ 1.6 at the very start).

  | | skip | keep |
  |---|---|---|
  | training examples (per epoch) | 6,898 (1,366–1,398) | 8,522 (1,686–1,727) |
  | validation examples | 520 (73 no-match val frames left out) | 666 |
  | time | 32.5 min (863 steps) | 39.9 min (1,066 steps) |
  | training loss, epoch 1 → 5 (mean) | 0.063, 0.032, 0.024, 0.018, 0.012 | 0.055, 0.029, 0.021, 0.016, 0.011 |
  | validation loss, epoch 1 → 5 | 0.0364, 0.0303, **0.0298**, 0.0334, 0.0357 | 0.0297, 0.0304, 0.0287, **0.0285**, 0.0298 |

  The training loss keeps falling, the validation loss is lowest at epoch 3 (skip) / 4 (keep)
  and rises after: the first signs of memorizing the training drives. The two validation losses
  are not comparable (different validation examples), and the loss is dominated by the easy
  tokens, so the epoch is chosen with the selector on all validation frames (next step).
- **Epoch selection on validation (2026-11-24, the user ran all 10 adapters on Polaris, every
  frame; `outputs/eval/val_epochs.md`):** 333 frames with candidates per mode, 100% valid,
  ~1.1 s per answer (no reasoning). Choice matches the driver, Mode A / B:

  | epoch | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | skip | 49 / 51% | 56 / 55% | 52 / 53% | **57 / 57%** | 57 / 56% |
  | keep | 52 / 53% | 53 / 53% | 54 / 54% | 54 / 54% | 54 / 54% |

  Baselines on these frames: random 33%, straightest 44%, toward the word 47% (Mode B),
  cheapest 50%, driver oracle 78% (the 73 no-match frames can't be matched by anyone). The
  zero-shot 4B / 8B were at 25–27% on val (every 4th frame, with reasoning). Findings:
  - Skip epoch 4 is better than every keep epoch in a paired comparison (it's right where keep
    is wrong 40–61 times, the reverse 20–39 times; p ≈ 0.01–0.07). Skip epochs 4 and 5 make
    the same choice in 97% of frames; epochs 2, 4, 5 are within noise of each other.
  - **Mode B ignores the word:** A and B choose the same candidate in 96–98% of frames; turn
    frames 30–44% (cheapest 42%, oracle 49%). On val, "toward the word" (47%) is itself below
    "cheapest", so copying the driver can be learned without the word. To be measured on test
    with the forced-word test.
  - No side bias (skip epoch 4: leftmost / middle / rightmost 18 / 55 / 26%, the driver's
    20 / 58 / 22%); keep epoch 3 leans right (43% rightmost, 7% leftmost).
  - **Puddles:** "chose a puddle when a dry path existed" (43 frames) 77–88%, above the driver
    (70%) and random (49%), and rising with the epochs: the model copies the driver, who drove
    through most puddles, and puddles are only "caution" in the labels.
  - Ticks and action are trivial on val ("always all" 98%, no stop frames).
- **User decision (2026-11-24): skip epoch 4 and keep epoch 5 go to the test drive.**
- **Test results (2026-11-24; the user ran everything on Polaris; 328 frames with candidates,
  100% valid, ~1.1 s per answer on the H200; `outputs/eval/test_finetuned.md`,
  `test_ft_{skip,keep}_behaviour.md`):**

  | Mode A / B | driver match | turn frames (83) | at 15 m from the driven path |
  |---|---|---|---|
  | random | 43 / 43% | 33 / 33% | 2.85 m |
  | straightest | 67 / 67% | 43 / 43% | 1.70 m |
  | cheapest | 48 / 48% | 42 / 42% | 1.89 m |
  | toward the word (B) | – / 67% | – / 45% | 1.49 m |
  | zero-shot 4B, no reasoning | 44 / 48% | 18 / 42% | 2.89 / 2.79 m |
  | zero-shot 4B / 8B (phase 4) | 43 / 48%, 42 / 47% | 18–19 / 35–41% | 2.7–3.5 m |
  | **fine-tuned skip, epoch 4** | **66 / 66%** | **57 / 54%** | **1.43 / 1.44 m** |
  | fine-tuned keep, epoch 5 | 59 / 60% | 49 / 51% | 1.63 / 1.59 m |
  | driver (oracle) | 89 / 89% | 72 / 72% | 1.14 m |

  - **Fine-tuning works:** skip epoch 4 vs the same model zero-shot (no reasoning): right where
    zero-shot is wrong 110 times, the reverse 35 (Mode A; p < 0.001).
  - **Skip beats keep** on test too (31 vs 6 frames, p < 0.001): leaving out the frames where no
    candidate matches the driver helps.
  - **Tied with "straightest" on this drive** (66 vs 67%; they disagree in ~100 frames, each
    right in half; p ≈ 0.9). Test 00004 is mostly straight (245 of 328 frames "straight"),
    which favours "straightest"; the model is much better on turns (57 vs 43%) and worse on
    straight frames, and its choices end closer to the driven path (1.43 vs 1.70 m at 15 m).
    On val (twistier) it beat straightest clearly (57 vs 44%). Across both drives it is the
    only chooser that is good on both.
  - No side bias (skip: left / middle / right 29 / 54 / 18%, the driver's 21 / 63 / 16%; the
    zero-shot 8B picks the rightmost 68%). Fewer near misses (4.9%) than random (8.1%) and the
    driven path itself (10.1%).
  - **The direction word is ignored after fine-tuning:** forced "bear left" vs "bear right"
    changes the choice in 5% (skip) / 14% (keep) of frames, against 89% zero-shot; Mode A and
    B choose the same in 96–97%. Fine-tuning traded instruction following for imitating the
    driver from the picture.
  - **History is ignored** (wrong history changes 3–5%; no history gives the same scores), as
    zero-shot.
  - Ticks: fine-tuned skip says "all" every time (= "always all", 99%; calls all 6 unsafe
    stretches safe), keep finds 2–3 of 6; zero-shot is too cautious (99%). Too few unsafe
    labels (6) to measure safety; no stop frames. Puddles: only 13 frames.

### Phase 6: closed loop in MAVS (branch `approach-2-closed-loop`)

**Started 2026-11-24** (the user: "we now start the phase 6"; simulator: **MAVS**, the Mississippi
State University Autonomous Vehicle Simulator, already built on the user's PC at
`C:\Users\talha\source\repos\MAVS`, Python API `src/mavs_python/mavs_interface.py`, ctypes over
`install/bin/mavs.dll`). The partner's PDF is a LiDAR-to-camera projection check (the same as our
`project_lidar.py`); the shared video is a MAVS recording.

**What I checked (2026-11-24):** `mavs_interface` imports in our conda env (Python 3.13), and a
camera, an Ouster `OS1` LiDAR and the `MavsVehicleController` can be created, so the simulator,
the candidate pipeline and the VLM run in **one Python process on the laptop**. MAVS gives
everything the loop needs:
- **Scenes:** `MavsRandomScene` generates a terrain with a trail (`path_type` 'Loop', 'Ridges' or
  'Valleys'; trail width, ruts, potholes, roughness, plant density, ecosystem files such as
  `american_southeast_forest_obstacles.json`) and saves the trail as waypoints; fixed scenes
  exist too (`odoa_scene.json`: obstacle avoidance, `cavs_proving_ground*.json`).
- **Vehicle:** `MavsRp3d` with `clearpath_warthog_cpu_tires.json`, **the Warthog RELLIS was
  recorded with**; `Update(env, throttle, steering, braking, dt)`, `GetPosition`,
  `GetOrientation`, `GetVelocity`.
- **Sensors:** `MavsCamera` (`Initialize(nx, ny, detector w, h, focal length)`, `SetOffset`,
  `SetPose`, `Update`, `GetNumpyArray`); `MavsLidar('OS1')`, the RELLIS LiDAR model
  (`GetUnRegisteredPointsXYZIL`: points in the sensor frame **with labels**).
- **Control:** `MavsVehicleController` (pure pursuit: `SetDesiredPath`, `SetCurrentState`,
  `SetDesiredSpeed`, `GetDrivingCommand`), so a chosen candidate can be driven directly.

**The loop (one decision every 0.5 s of simulated time, as the 2 Hz decision frames):**
1. Camera image + LiDAR scan from MAVS at the vehicle's pose.
2. LiDAR → FLU → elevation map → relaxed cost map → candidates (phase 2 code, unchanged).
3. Picture with the candidates (phase 3 code) → the chooser answers (choice, ticks, continue/stop).
4. The chosen candidate (vehicle frame → world) goes to the pure-pursuit controller, which drives
   for 0.5 s of simulated time; then back to 1. The simulator waits while the VLM thinks
   (simulated time, not wall-clock), so a slow answer doesn't crash the vehicle.
5. Logged per decision: pose, candidates, choice, answer, the picture (every n-th), and the
   simulator's ground truth for scoring only (labels, the trail waypoints, contact).

**Choosers compared (same scenes, same seeds):** straightest, cheapest (the LiDAR planner on its
own), zero-shot 4B (asked without reasoning), **fine-tuned 4B skip epoch 4**; optionally random.

**Metrics (scored from the simulator's ground truth, never shown to the chooser):** route
completion (distance along the trail before the run ends), interventions (stuck: no progress for
N s; off the trail by more than X m; contact with an obstacle), distance from the trail
centre, time, stops (right when no candidate is safe, e.g. a blocked trail; wrong otherwise),
and the safety rows of phase 4 where MAVS labels allow. Mode B: the word comes from the trail
waypoints ~50 m ahead, as on RELLIS; a fork scene tests whether the word is followed.

**Steps (each a small, checked piece):**
- **6.1 Sensors and calibration:** a MAVS rig like RELLIS's (Warthog, OS1 LiDAR, camera with
  RELLIS's field of view), one frame saved, LiDAR projected on the camera image (the partner's
  check), and the FLU and camera transforms verified.
- **6.2 Candidates on MAVS data:** cost map and candidates on simulated scans; the picture the
  VLM would see; also the general-ability check moved here from phase 5.
- **6.3 The loop:** controller, stepping, logging, a chooser interface, one scene with
  "straightest" end to end.
- **6.4 Scenes:** a small fixed set (e.g. 3–5 trails with obstacles / a fork / a blocked end).
- **6.5 Runs and evaluation:** all choosers on all scenes, both modes; a report like phase 4.

**Branch cleanup (user's request 2026-11-24):** this branch keeps only what the closed loop needs;
the RELLIS pipeline, its results and the training code stay on `approach-2-mapless-vln`.

**User decisions (2026-11-24):** remove the RELLIS pipeline and the training code from this
branch; sensors like RELLIS's; generated trail scenes; about **one week** for phase 6, so steps
6.1-6.3 plus a small run set. **Vehicle: MRZR4** (changed from the Warthog after my tests, see
below), with the RELLIS-like sensors.

**Step 6.1 status (2026-11-24): written and checked in my scratchpad; the user has not run it.**
- Findings in MAVS (all checked with small scripts):
  - **Missing meshes:** most vegetation folders (`USA_Conifers`, `Europe`, `Groundcover`, ...)
    are not in the user's MAVS data; only `forest_simple.json` has all its plants. One missing
    mesh gives the scene an invalid bounding box, `GetSurfaceHeight` then returns -FLT_MAX and the
    vehicle physics fails (NaN, access violation). Generated scenes are cleaned
    (`drop_missing_meshes`). The missing folders are not in MAVS's public GitHub repository
    either (checked 2026-11-26), so a new download can't add them. **User decision
    (2026-11-26): keep `forest_simple.json`.** There are no rock or log meshes; obstacles for
    step 6.4 can be road blockades, walls, cubes and cylinders.
  - **The brake drives backwards** from standstill (Warthog and MRZR4): the vehicle settles with
    no pedals; the controller brakes only while moving forward.
  - **Throttle works like an accelerator held down** (0.1 reaches 3.1 m/s after 4 s and keeps
    rising; 0.25 and 0.4 both reach ~5 m/s): speed needs feedback.
  - **The Warthog model is unusable for driving:** its reported heading is ~15 degrees off the
    direction it moves, even on bare ground, and its skid steering is very sensitive at walking
    pace (command 0.1 = 2.3 m turning radius). **The MRZR4** moves where it points (within 2
    degrees), curvature = 0.105 1/m x command (linear to full lock), and with our pure-pursuit
    follower stays on average 0.07-0.32 m (at most 0.46 m) from a straight path and arcs of
    25 / 12 / 10 m radius at 1.3-1.5 m/s. **User decision: MRZR4 with the RELLIS sensors.**
  - **Plants make the vehicle physics 30x slower** (7.5 vs 0.26 s per simulated second): the
    vehicle drives in a copy of the scene without objects (`<name>_ground.json`); the sensors
    see the full scene. In MAVS plants only add rolling resistance (they are not solid), so
    little is lost; collisions will be judged from geometry, not from the physics.
- Rig: MRZR4 (CG 0.71 m above the ground), OS1 LiDAR 1.2 m above the ground (measured 1.11 m
  from its own scan on a 150 m trail scene), camera 0.1 m ahead of / 0.17 m below the LiDAR,
  level, 960 x 600 with RELLIS's field of view. **Calibration check:** our projection of the
  scan lands a median 0.73 px from the pixels MAVS marks for the same points (5 px shifts give
  3.7-5.4 px). One scan ~48,500 points; loading, placing and capturing take ~4 s.
- Code: `sim/rig.py`, `sim/mavs_world.py`, `sim/control.py`, `scripts/sim_check_rig.py`
  (`run_all.py` step `sim_rig`); tests `test_sim_rig.py`, `test_sim_control.py`.
- **Verified 2026-11-26 (the user ran `run_all.py --only tests sim_rig`): step 6.1 is done.**
  `rig_check.json` is identical to my scratchpad run (so MAVS generates the same trail every
  time: scenes are reproducible), CHECK OK, projection 0.73 px from MAVS's own (in
  `projection_check.png` our points cover every pixel MAVS marks). Note for 6.4: `trail01` is a
  straight trail ('Ridges'), so turns need other scenes.

**Step 6.2 status (2026-11-26): written and checked in my scratchpad; the user has not run it.**
- `sim/planning.py`: `plan_frame` (scan → cost map → relaxed grid → candidates, phase 2 code
  unchanged; ~0.1 s per frame), `trail_ahead_flu` / `trail_direction_word` (Mode B's word from
  the trail ~50 m ahead), `straightest_index` / `cheapest_index`. `nav.visual_prompt.frame_seed`
  now also takes scene names (CRC-32; RELLIS numbers give the same seeds as before).
  `scripts/sim_show_candidates.py` (`run_all.py` step `sim_candidates`): 6 places on `trail01`,
  the picture, a top view, and with `--ask` both models in Modes A and B.
- **Candidates on `trail01`:** 2 per frame everywhere (one along the trail, one bending left at
  ~12 m into tall grass): a straight 4 m trail lined by dense grass and trees leaves few
  distinct paths. Cost grid checked cell by cell: the trail is drivable or unknown (LiDAR rings
  ~0.5 m apart beyond ~10 m, as on RELLIS), blocked cells are at its edges.
- **General-ability check (the item moved from phase 5):** both models answer every MAVS
  picture validly (12 / 12 each). The fine-tuned 4B chooses the path along the trail in 6 / 6
  frames (both modes) and, unlike on RELLIS, does not call everything "all": the trail path
  "all", the one bending into the grass "tick 2". The zero-shot 4B chooses the trail path in
  5 / 6 and is over-cautious (the trail path "tick 2"). Laptop speed: 3-4 s per answer (first
  answer ~18 s, warm-up), models load in 13-16 s. Only 6 easy frames: no real choice yet.
- **The user's run (2026-11-26) found a reproducibility problem:** the candidates and top views
  were byte-identical to mine, but every camera picture differed (~2 grey levels brighter,
  clouds different) and 3 of 24 answers changed. MAVS's sun follows the computer's clock and
  its clouds are random: two runs seconds apart differed by 5.8 grey levels on average.
  **Fix:** `MavsWorld` sets a fixed date (2026-06-21), time (12:00:00) and no clouds; two
  separate runs now render identical pixels (12 / 12 files identical), 6.1 still passes. The
  user has to rerun 6.2. My reference answers with the fixed lighting: fine-tuned picks the
  trail path 6 / 6 in both modes, ticks it "all" and the other path "tick 2" in every frame;
  zero-shot picks the trail path 4 / 6 (Mode A) and 5 / 6 (Mode B), ticks everything "tick 1-2".
- **Verified 2026-11-26 (the user reran `sim_candidates`): step 6.2 is done.** All 12 pictures
  byte-identical to my run and all 24 answers identical (choice, ticks, action): simulator runs
  are now fully reproducible, VLM included.

**Step 6.3 status (2026-11-26): written and checked in my scratchpad; the user has not run it.**
- Code: `sim/metrics.py` (`project_on_trail` → progress along the trail and signed offset from
  its centre, `is_stuck`), `sim/choosers.py` (`StraightestChooser`, `CheapestChooser`,
  `VlmChooser` with its own history, an invalid choice falls back to straightest and is logged),
  `sim/loop.py` (`LoopParams`, `run_episode` → `Episode`), `scripts/sim_drive.py` (`run_all.py`
  step `sim_drive`). Tests `test_sim_loop.py` with a kinematic stand-in world.
- The loop: a decision every 0.5 s of simulated time (time stands still while the chooser
  thinks), the chosen path fixed in the world and driven by the follower in 0.02 s steps. Ends:
  finished (within 5 m of the trail's end), stopped, off trail (> 3 m from the centre), stuck
  (< 0.5 m progress in 10 s), time limit (240 s).
- **Three fixes found by driving (all in `sim/`, the phase 2-3 code is unchanged):**
  - **Candidates were 12.5 cm to the left:** they run through grid-cell centres and the
    vehicle's own cell centre is (0.125, 0.125) m away; offline harmless, in closed loop the
    follower aimed left at every decision and the vehicle turned off the trail. `plan_frame`
    now shifts them onto the vehicle (`centred_on_vehicle`).
  - **No candidate → keep the previous path** for at most 2 decisions (1 s), then stop
    (`LoopParams.max_hold`); counted per run (`held_decisions`). On RELLIS such a frame meant an
    immediate stop.
  - **Overhanging foliage blocked the trail:** the canopy edge hangs 1.5-1.8 m over MAVS's
    trails, just under the 1.8 m vehicle height, so the relaxed grid saw 1.6 m "obstacles"
    across the trail and the first run stopped after 12 m. A cell now counts as an obstacle only
    if something stands up from the ground in it (a point 0.3-1.0 m above the ground):
    `without_overhangs` (on by default in the simulator). Blocked cells at that spot: 2,651 →
    1,430; trunks, bushes and posts still block (test).
- **First full run, `trail01`, straightest, Mode A:** finished, 193 of 197 m in 136 s of
  simulated time, 272 decisions, never without a candidate, on average 0.69 m (at most 1.40 m)
  from the trail centre (the trail is 4 m wide), ~3 min wall time (0.7 s per decision).
- The fixes change the candidates slightly, so 6.2's pictures and answers are no longer the
  current ones (rerun if needed for the report).
- **Verified 2026-11-26 (the user ran `sim_drive`): step 6.3 is done.** All 272 decisions and
  all 28 pictures identical to my run (only the wall time differs: 198.5 vs 186 s): the whole
  closed loop, MAVS physics included, is reproducible.

**Step 6.4 status (2026-11-26): in progress (written, checks running in my scratchpad).**
- **User decisions:** all four scenes (S-curves, fork, blocked, half-blocked); collisions judged
  from object positions; then, after the first drives, **keep the RELLIS camera and design the
  scenes within its limits** ("gentler scenes"; the narrow view is reported as a limitation).
- **No MAVS rebuild needed.** MAVS's Python API can't pass a custom trail to its scene generator
  (the C++ `RandomScene::Load(json)` with "Path File" isn't exported), but a generated scene file
  lists its trail texture as path files and every plant instance, so a design is drawn by
  editing that file: `sim/scenes.py` (pure: trail pieces, designs, blockades, path files) and
  `mavs_world.build_designed_scene` (one forest without a trail, `outputs/sim/scenes/base`; per
  design: the trails as path files, plants within 2.5 m of a trail centre removed, trees within
  3.5 m, blockades added; `<name>_design.json` holds the routes = answer keys). A fork works by
  listing two path files.
- **Collisions:** `scene_obstacles` (tree trunks by mesh, radius ~0.3-0.4 m x scale; blockades
  sampled along their length) and `metrics.footprint_gaps` (MRZR4 footprint 3.5 x 1.6 m);
  outcome "collision" and `min_gap_m` per run.
- Findings while building:
  - MAVS's road blockade is only 0.8 m tall, and the relaxed grid treats anything under 1.5 m
    as passable vegetation, so the walls are scaled to 1.8 m (MAVS scales after turning the
    mesh's y up to z: height is the third scale value). Two walls (5.8 m) left room to go round
    through the grass, so "blocked" uses four (11.5 m).
  - **Bug: placing the same MAVS vehicle twice didn't move it** (MAVS only takes the initial
    position before the first update). `place()` now loads a new vehicle each time and checks
    where it settled. **This invalidates step 6.2's six frames: all six were the same spot**
    (different colors only); 6.1 and 6.3 place once and are unaffected. 6.2 must be rerun.
  - **The narrow RELLIS camera (38 degrees) limits the scenes:** with 22 m bends even the path
    along the trail ended out of view and "straightest" stopped after 35 m; passing the first
    half-wall needed a 3 m swerve within 5-8 m (out of view). Loosening the view rule didn't
    help. Redesigned: bends of radius 40 m (`BEND_RADIUS_M`), fork branches leaving through a
    40 m arc to 25 degrees, the half-wall over the left third of the trail.
- **Driving fixes found with the new scenes** (simulator only; phase 2-5 code unchanged):
  - **Grid paths turn late:** shortest paths on the 8-connected grid run straight as long as they
    can and bend only near their end (on flat ground every candidate was straight for the first
    9 m). The follower (look-ahead 3 m) then never turned, and each new plan started straight
    again: the vehicle drifted into the half-wall's shadow, and even an oracle could not follow
    a 40 m bend on open ground. Look-ahead 5 / 7 m was no fix (7 m passed the half-wall but cut
    the bends and left the curves trail). **Fix: `planning.pull_string`** straightens each
    candidate (jumps to the farthest point a straight segment reaches without crossing a
    costlier cell than the original path does there), so a side path turns from the start; the
    picture shows the straightened path, so the VLM sees what is driven. With it the
    half-cell shift (`centred_on_vehicle`) is not needed (straight paths start at the vehicle).
  - **An oracle chooser** (`OracleChooser`: the candidate closest to the route, a reference
    like phase 4's driver oracle) shows what the candidates allow.
  - **Obstacle margin 1.0 m** (`planning.SIM_COST_PARAMS`): RELLIS's 0.75 m is less than half
    the MRZR4's width (1.6 m), so a path along the margin scraped the half-wall.
  - **A candidate's end must be in view:** the phase 2 rule checks points from 5 m on, so a
    3 m path at 70 degrees passed with nothing checked.
  - **The half-blocked route swerves 1.2 m right around the wall** (`scenes.swerve`): along the
    trail centre it passed 0.76 m from the wall's end, and the oracle following it grazed it.
  - With straightened paths (before the last three fixes): oracle finished trail01, curves and
    both fork routes and stopped at the full wall; straightest left the curves trail and one
    fork route, and stopped at the wall.
- **Final check of the scene set (my scratchpad, 2026-11-26), Mode A / fork in Mode B:**

  | scene | oracle (reference) | straightest |
  |---|---|---|
  | trail01 | finished, max 0.47 m off, gap 0.93 m | finished, max 1.15 m, gap 0.59 m |
  | curves | finished, max 1.26 m | stopped at 72 of 130 m (no visible path in a bend) |
  | fork, left route | finished, max 0.65 m | stopped at 51 of 93 m |
  | fork, right route | finished, max 1.11 m | off the route at 46 m |
  | blocked | stopped ~10 m before the wall | stopped ~10 m before the wall |
  | halfblocked | finished, gap to the wall 0.80 m | finished, gap 0.25 m |

  Every scene can be driven with these candidates (the oracle), and the straight-ahead
  baseline fails the bends and the fork, so the set separates choosers. The fixes changed the
  candidates, so 6.3's trail01 run and 6.2 must be rerun.
- **Step 6.2 corrected (my rerun with the placement fix and the new candidates):** the six
  frames are now six places, with 2-3 candidates each (bearings -11 ... +16 degrees). Both
  models 12 / 12 valid. Fine-tuned: picks the straightest candidate in 6 / 6 (both modes) and
  calls every path "all" in 4 / 6 frames (as on RELLIS); zero-shot: the straightest in 4 / 6,
  over-cautious ("tick 1-2"). The earlier "fine-tuned tells the grass path apart" came from one
  spot seen six times.
- **Verified 2026-11-27 (the user ran `run_all.py --only tests sim_candidates sim_drive sim_scenes
  sim_oracle sim_oracle_wall`): steps 6.2, 6.3 (rerun) and 6.4 are done.** 6.2: 12 / 12 pictures
  and 24 / 24 answers identical to my run; 6.3 (trail01, straightest): finished, 192 of 197 m,
  decisions identical; scene checks identical (CHECK OK); oracle on curves (finished, max
  1.26 m) and halfblocked (finished, 0.80 m from the wall), decisions identical.

**Step 6.5 status (2026-11-27): written; my self-check done; the user has not run it.**
- User decisions: start 6.5 without video recording ("We'll do it later": then a replay of the
  saved runs, chase view with MAVS's MRZR4 mesh + the VLM's picture); run on the laptop (I
  advised against building MAVS on Polaris for this).
- `sim/evaluation.py` (the plan: 42 runs, word-blind choosers in Mode A only, the fork per
  route; success rules; `decision_stats`, `chooser_summary`), `sim/runs.py` (shared run saving),
  `scripts/sim_run_all.py` (resumable, each VLM loaded once), `scripts/sim_evaluate.py`
  (`outputs/sim/evaluation.md`, 5 tables).
- **Self-check (scratchpad):** the three baselines on all six scene/routes (31 min) and both VLMs
  on the blocked trail (first closed-loop VLM runs: 3.3 / 4.2 s per answer, 100% valid).
  Baselines, tests passed in Mode A: straightest 3 / 5 (fails the curves and the fork),
  cheapest 5 / 5 (closest gap to an obstacle 0.14 m on halfblocked), oracle 5 / 5. Blocked
  trail: every chooser stopped 8-10 m before the wall because the candidates ran out; neither
  VLM ever answered "stop". The fine-tuned model ticked its path "all" (safe for 15 m) right up
  to the wall at ~10 m; the zero-shot one said "tick 2" (10 m).
- Blocked trail: full-length candidates (the nearer ones bending round the wall's ends) from 25
  to 13 m before the wall, none at 10 m; so the VLM sees the wall several times and can say
  stop, otherwise the vehicle stops by itself ~9 m before it (logged as "chooser said stop" vs
  "no candidates").
- 2026-11-27: the user's batch crashed at the first Mode B run (trail01, zero-shot, ~176 m):
  the trail's last 20 m give no direction word. Fixed: the last word stays in force.

**Step 6.5 runs verified (2026-11-28, the user's batch, all 42 runs; `outputs/sim/evaluation.md`).**
The baselines match my self-check exactly (the loop is reproducible).

| chooser | Mode A tests passed (fork = either branch) | Mode B tests passed (fork per route) | mean distance from route | closest gap |
|---|---|---|---|---|
| straightest | 3 / 5 | 3 / 6 | 0.85 m | 0.25 m |
| cheapest (LiDAR alone) | 5 / 5 | 5 / 6 (fails fork-left) | 0.49 m | 0.14 m |
| oracle (answer key) | 5 / 5 | 6 / 6 | 0.30 m | 0.80 m |
| zero-shot 4B | 4 / 5 (fails curves) | 5 / 6 (fails curves) | 0.61 / 0.74 m | 0.15 m |
| fine-tuned 4B | 5 / 5 | 5 / 6 (fails fork-right) | 0.66 / 0.68 m | 0.06 m |

- **No collisions, no "stop" answers** (0 of ~3,200 VLM answers; all valid, no fallbacks). Every
  chooser stopped 8-10 m before the blocked trail's walls because the planner ran out of
  candidates. The fine-tuned model called every path "all" in 75-77% of answers, the zero-shot in 0%.
- **Direction word (the fork):** the fine-tuned model's choices on fork-right are identical in
  Modes A and B, decision by decision, although 47 of 61 words said "slight/bear right": it
  ignores the word (as in phase 5: 5% changed choices). The zero-shot model takes the left
  branch without a word and the right one when told "slight right": it follows the word.
- **Curves:** zero-shot drifted to the outside (left) of the second, right-hand bend, choosing
  left-pointing candidates in a right bend and switching between them, until no candidate was
  left (stopped at 56% / 40%). Fine-tuned finished both, as cheapest and oracle.
- **Half-wall:** every real chooser passed the wall's end within 0.06-0.25 m (fine-tuned 0.06 m,
  within the accuracy of the rectangle-footprint check); the oracle kept 0.80 m by choosing
  the paths further right. The fine-tuned model drove ~0.8 m left of the swerving route and
  rated every path "all" there. The planner's margin (inflation 1.0 m minus half the width 0.8 m)
  is thin, so the choice between offered paths decides the clearance.
- Answer time: zero-shot 3.4 s, fine-tuned 4.7-5.3 s.
- One run per combination (the simulator is deterministic), so pass counts are single
  observations; the decision-level shares rest on 700-900 decisions per chooser and mode.

### Phase 6 notes kept from the plan
- An off-road simulator (e.g. Isaac Sim, AirSim/Colosseum, BeamNG, or an off-road CARLA map)
  would allow real closed-loop driving. Revisit only if the offline results call for it.
- **What it needs:** an RGB camera (the VLM's input) and a LiDAR or other depth sensor mounted
  roughly like RELLIS's (~1.2 m up), because the candidates come from the LiDAR cost map. The
  simulator's own ground truth (collisions, water, distance covered) is used only to score runs.
- **No labels at run time:** the RELLIS labels (and ours made from them) are only training
  targets and the answer key for scoring. Candidates come from LiDAR geometry only, and the VLM
  sees only the camera image with the candidates drawn on it, so the same pipeline runs in
  closed loop without labels.
- **Closed loop tests what open loop can't:** the training frames all lie on the driver's own
  good path, so the model never sees a state where it has already drifted. In closed loop small
  errors add up and lead to such states; this is the main thing a simulator run would measure.

## 6. Decisions

**Made (2026-11-22):**
1. **Evaluation mode:** offline, open-loop on RELLIS-3D; a simulator only if needed later.
2. **Task:** no destinations. "Keep going until stopped" (Mode A) and "keep going + coarse
   direction" (Mode B), both evaluated; every answer includes a path choice, a safe distance per
   candidate (ticks), and continue/stop.
3. **Candidates:** cost-map fan + A\* first; curved anchors as an optional later comparison.
4. **Model and fine-tuning:** Qwen3-VL-4B, QLoRA on the laptop. DriveVLN used Qwen2.5-VL-3B; the
   4B is the closest current model of that size, framed as a starting point for future work.
   8B decided after the 4B results (fine-tuning it would need the H200). 2026-11-24: not
   fine-tuned; the 8B stays a zero-shot reference.
5. **Cost grid for the candidates:** relaxed (only hard obstacles over 1.5 m block; lower
   vegetation costs 10), chosen after the phase 2 comparison with the strict grid.

**Settings (the recommended defaults, accepted 2026-11-22):**

| # | Setting | Value |
|---|---|---|
| 1 | Split by sequence | train 00000–00002, validate 00003, test 00004 |
| 2 | Decision frames | about 2 Hz (≈ 2,700 frames in total) |
| 3 | Branch | `approach-2-mapless-vln`, from `main` |
| 4 | Candidates | up to 4 per frame, about 15 m long, ticks at 5 / 10 / 15 m |
| 5 | "Driver's path" | candidates within 1 m mean distance of the driven path (several may count) |
| 6 | Unsafe | water or a hard obstacle (approach 1's classes; a hard obstacle only where the LiDAR sees something standing); mud and grass count as safe. **Puddles are "caution"** (changed 2026-11-22: the driver drives through them): recorded and scored as "puddle on the chosen path when a dry candidate existed", but not unsafe |
| 7 | Stop is correct | every candidate is unsafe before its first tick (5 m) |
| 8 | Mode B directions | "bear left", "slight left", "straight", "slight right", "bear right", from the driven path ~50 m ahead |
| 9 | History | the last 3 decisions as a short text summary |
| 10 | Image size | 800×600 |
| 11 | Fine-tuning | LoRA rank 64, learning rate 1e-4, 5 epochs (DriveVLN's); GRPO decided after the SFT results. 2026-11-23: trained on Polaris; answers without the reasoning field. 2026-11-24: no GRPO; reported adapter = "skip" data, epoch 4 |
| 12 | Models in the report | Qwen3-VL-4B zero-shot and fine-tuned (main); the 8B zero-shot as a reference |
| 13 | Approach-1 comparison | approach 1's geometric baseline on the same test frames, same safety metrics; its VLM hybrid only if time allows (needs new 8B runs, ~1.5 h) |

## 7. Risks and limitations to expect

- **RELLIS isn't a navigation dataset.** The driver mostly follows trails, so real forks are
  rare; many frames have one obviously right choice. Report "real choice" frames separately.
- **Open-loop only:** a wrong choice is scored but never driven, so errors don't compound the way
  they would on a real vehicle.
- **The driver's path isn't the only good path.** That's why Mode A is scored on safety, and
  Mode B uses a tolerance and a direction.
- **The Mode B direction comes from the driver's own future path**, so it leaks a little of the
  answer; it's kept coarse, and Mode A is the leak-free comparison.
- **Small and correlated data** for fine-tuning: a few thousand frames from one campus. The test
  sequence must stay unseen.
- **Labels are only as good as the ground truth:** coarse RELLIS labels, ruts labelled as mud,
  and ~20% of path cells without ground truth (mostly the blind zone).
- **Latency:** about 10–14 s per answer on the laptop. Fine offline; a real vehicle would need a
  faster model or fewer decisions.
- **The blind zone and cost-map limits** carry over, since the candidates come from the same cost map.

## 8. Safeguards against overfitting and shortcuts (agreed 2026-11-22)

The user's concern: the model must not learn things that won't hold in a closed-loop simulator,
where there are no labels and its own mistakes shape what it sees next. Where each one goes:

| Safeguard | Why | Phase |
|---|---|---|
| Labels only as targets / answer key, never as input | They don't exist in closed loop | 2 (done: candidates use LiDAR geometry only) |
| Random color assignment and prompt order, same drawing style, no costs in the prompt | So it can't learn "red is usually right" or "pick the middle one" | 3 and 5 |
| Simple baselines: random, straightest, cheapest, always "all safe, continue" | The VLM must beat them to show it reads the image | 4 |
| Results per answer type (stop recall, unsafe calls) | Overall accuracy hides "always continue" | 4 |
| Same frame with different direction words | Checks it follows the instruction | 4 |
| History dropout and wrong history; evaluate with and without | In closed loop the history holds its own mistakes | 3, 4, 5 |
| Balanced training data (more turns, unsafe, stop) | Most frames are easy and straight | 5 |
| Mirroring with left / right swapped | Doubles turns; test has no "bear right" | 5 |
| Brightness / color / blur augmentation | Simulator images look different from RELLIS | 5 |
| Few epochs, early stopping on 00003, test 00004 only at the end | Neighbouring frames are near-duplicates | 5 |
| Skip frames with mostly unknown ground | Don't train on guesses | 5 |
| Zero-shot reported next to fine-tuned; a few non-RELLIS images after fine-tuning | Shows what fine-tuning changes outside RELLIS | 5 |
| Puddles as caution, not unsafe | Labels that contradict the driver teach RELLIS habits, not driving | 2 (done 2026-11-22) |
