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
   - **safe distance per candidate:** the first water / puddle or hard-obstacle cell along it,
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

### Phase 3: visual prompt and zero-shot selector
- Draw the candidates on the front camera image in distinct colors that stand out from grass and
  sky, each with a name at its end and numbered ticks at 5 / 10 / 15 m. Paths start below the
  image (the ~4.4 m blind zone) and are drawn from their first visible point.
- Prompts for Mode A and Mode B, the JSON answer above, and a forgiving checker (as in approach 1).
- **History:** the last few decisions as a short text summary (cheap on memory), with earlier
  images as an option if memory allows.
- Run zero-shot with **Qwen3-VL-4B** on the laptop; the existing 8B as a reference point.

### Phase 4: evaluation
- **Baselines:** random choice; the geometric choice (lowest-cost candidate); "follow the
  direction" without vision (the candidate closest to the Mode B bearing); and the **oracle**
  (always a correct candidate, always correct safe distances).
- **Per-frame metrics:**
  - path choice: water / mud / vegetation on the chosen path, collision proxy (crosses
    hard-obstacle points), clearance; and in Mode B, agreement with the driver and L2 error
    against the driven path at 5 / 10 / 15 m;
  - safe distance: accuracy of the tick answers (exact, and within one tick), and how often it
    calls an unsafe stretch safe (the dangerous error);
  - stop: precision and recall of "stop".
- Report results separately for **frames with a real choice**, since many off-road frames have
  only one sensible path.
- **Comparison with approach 1:** score approach 1's baseline and hybrid paths with the same
  safety metrics (and driven-path L2) on the same test frames.

### Phase 5: fine-tuning
- **QLoRA supervised fine-tuning of Qwen3-VL-4B on the laptop** (RTX 5060, 8 GB): the base model
  stays frozen in 4-bit, only small LoRA adapters are trained. Batch size 1 with gradient
  accumulation, gradient checkpointing, 800×600 images (as DriveVLN), a mix of Mode A and Mode B
  examples, shuffled color assignment (so the model can't learn "always blue"), and paraphrased
  instructions.
- Compare zero-shot vs fine-tuned on the test sequence, in both modes.
- **Optional afterwards:** GRPO with a per-frame reward (choice safety + agreement with the
  driver + tick accuracy), which needs no simulator. DriveVLN's route-level reward needs closed
  loop and is out of scope.
- **8B:** decide after the 4B results. Fine-tuning an 8B doesn't fit in 8 GB; it would run on the
  Polaris H200.

### Phase 6 (stretch): closed loop
- An off-road simulator (e.g. Isaac Sim, AirSim/Colosseum, BeamNG, or an off-road CARLA map)
  would allow real closed-loop driving. Revisit only if the offline results call for it.

## 6. Decisions

**Made (2026-11-22):**
1. **Evaluation mode:** offline, open-loop on RELLIS-3D; a simulator only if needed later.
2. **Task:** no destinations. "Keep going until stopped" (Mode A) and "keep going + coarse
   direction" (Mode B), both evaluated; every answer includes a path choice, a safe distance per
   candidate (ticks), and continue/stop.
3. **Candidates:** cost-map fan + A\* first; curved anchors as an optional later comparison.
4. **Model and fine-tuning:** Qwen3-VL-4B, QLoRA on the laptop. DriveVLN used Qwen2.5-VL-3B; the
   4B is the closest current model of that size, framed as a starting point for future work.
   8B decided after the 4B results (fine-tuning it would need the H200).

**Settings (the recommended defaults, accepted 2026-11-22):**

| # | Setting | Value |
|---|---|---|
| 1 | Split by sequence | train 00000–00002, validate 00003, test 00004 |
| 2 | Decision frames | about 2 Hz (≈ 2,700 frames in total) |
| 3 | Branch | `approach-2-mapless-vln`, from `main` |
| 4 | Candidates | up to 4 per frame, about 15 m long, ticks at 5 / 10 / 15 m |
| 5 | "Driver's path" | candidates within 1 m mean distance of the driven path (several may count) |
| 6 | Unsafe | water / puddle or a hard obstacle (approach 1's classes); mud and grass count as safe |
| 7 | Stop is correct | every candidate is unsafe before its first tick (5 m) |
| 8 | Mode B directions | "bear left", "slight left", "straight", "slight right", "bear right", from the driven path ~50 m ahead |
| 9 | History | the last 3 decisions as a short text summary |
| 10 | Image size | 800×600 |
| 11 | Fine-tuning | LoRA rank 64, learning rate 1e-4, 5 epochs (DriveVLN's); GRPO decided after the SFT results |
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
