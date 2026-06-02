# Handoff — Joint Memory VLA (MemER-structured) trained with GRPO

Last updated: 2026-06-01

## What this is

A **memory-augmented manipulation hierarchy** on RoboMME, mirroring MemER
(arXiv:2510.20328) but trained with **GRPO instead of imitation learning**: a
Qwen3-VL-4B high-level policy that *remembers* (keyframe memory) and *selects*
(which frames to keep) and emits a grounded subgoal to a **frozen subgoal-conditioned
π0.5 (GroundSG)**. The project's actual contribution is training **keyframe
selection + subtask jointly, one GRPO run** (see `JOINT_MEMORY_DESIGN.md`).

```
  reveal keyframes (memory) + candidate window
        │
        ▼
  Qwen3-VL-4B + LoRA  ──► {current_subtask: "...at <x,y>", keyframe_positions:[...]}
   (we train, jointly: SELECT which frames to keep + USE them to pick)
        │ grounded subgoal (coordinate)
        ▼
  GroundSG π0.5 (frozen) — config=mme_vla_suite + symbolic-grounded-subgoal
        │ action chunks
        ▼
  ManiSkill / RoboMME (ButtonUnmask: bins reveal R/G/B cubes 0–64, then cover)
```

Design docs: **`MEMORY_PIPELINE.md`** (Phase 0–5 plan), **`JOINT_MEMORY_DESIGN.md`**
(the joint select-then-use architecture). Memories in `/memory/` carry the
hard-won lessons — read `project_memory_pipeline_direction.md` first.

## MemER single-call STREAMING (current direction — branch `grpo-memer-streaming`)

**Goal:** make the VLM emit the SAME output as MemER's VLM — ONE JSON
`{current_subtask, keyframe_positions}` per call — and train it with **GRPO** instead
of imitation. This SUPERSEDES the two-call SELECT/USE split (`joint_selection`), which
was itself a divergence from MemER (two prompts, two generations) adopted only because
keyframe selection is causal only ACROSS timesteps and we queried the VLM once per pick.

**Mechanism — decision-point streaming (`--streaming-memory`, task `ButtonUnmaskSwap`):**
the VLM owns the whole episode. At each pick it makes ONE MemER call over
`[keyframe_buffer + broad recent window]`; the nominated `keyframe_positions` accumulate
into a persistent buffer (MemER clustering, `KeyframeBuffer`, d=8/cap=8). What it keeps at
pick 0 (whose window spans reveal+swaps) is the ONLY memory of pick 1's now-occluded
target → selection causally moves the episode reward → **GRPO trains selection**. The
oracle drives only the deterministic scaffolding (presses, put-downs); the VLM owns each
grounded pick. Trajectory = the per-pick calls, all sharing the one episode advantage
(the existing trajectory loss in `_accumulate_gradients` handles this unchanged).

**New/changed code (all additive + gated; one-shot/joint paths and their parity tests
stay green):**
- `grpo/keyframe_buffer.py` (NEW) — `KeyframeBuffer`/`TaggedFrame`, MemER d=8/cap=8 merge.
- `grpo/rollout.py` — `rollout_streaming` GENERATOR (yields `DecisionPoint`, `.send()`s
  `(subtask, keyframe_positions)`, yields `RolloutResult`); abs-step frame tagging;
  `from_qwen_xy` applied at the single executor point, NOT to oracle scaffolding.
- `grpo/trainer.py` — `GRPOConfig.streaming_memory`/`streaming_max_picks`; `_streaming_group`
  (K full episodes/group under CRN seed, k=1 per call); dispatch in `_rollout_group`;
  asserts `snapshot_branching` off (streaming branches the whole episode, not one point).
- `grpo/selection.py` — `valid_keyframe_positions` (shared by `apply_selection` + the
  buffer step-tagging).
- `grpo/state_dataset.py` — `streaming` flag → ONE state per episode.
- `data/build_memory_sft_dataset.py` — `streaming` mode: SINGLE-call rows (BOTH fields
  populated), keyframe buffer threaded across picks.
- `modal_train.py` — `build_memory_dataset(streaming=…)`, `grpo(streaming_memory=…,
  streaming_max_picks=…)`; `validate_memory_checkpoints` now reads the key/recent split
  from each row's prompt (so the variable streaming buffer validates correctly).
- Tests: `test_keyframe_buffer.py`, `test_streaming_rollout.py` (generator protocol +
  buffer accumulation + the conversion point); all prior tests still pass.

**RUN IT (this is the GRPO run the branch was built for):**
```bash
# 1. SINGLE-call MemER streaming SFT data on the Swap task (both fields per row)
modal run modal_train.py::build_memory_dataset --only-tasks ButtonUnmaskSwap --streaming
# 2. warm-start SFT (lr 2e-5, label_smoothing 0.0; same stage as before)
modal run modal_train.py::sft_warmstart
# 3. gate: coord px-dist small + varying (validator auto-splits key/recent per row)
modal run modal_train.py::validate_memory_checkpoints
# 4. STREAMING GRPO (single-call MemER output, selection trained by reward) — SMOKE FIRST
modal run modal_train.py::grpo --streaming-memory --only-tasks ButtonUnmaskSwap \
  --num-steps 8 --debug-subgoals \
  --sft-adapter-path <…>/permanence_grounded/<v*>/checkpoint-<best>
```
Smoke goal (step 0–8): `--debug-subgoals` shows each pick's `kf=` VARIES, the buffer grows
(`buf=` on pick 1), generations terminate, and Swap groups are NON-degenerate (reward
varies with selection — unlike vanilla ButtonUnmask's 11/12-dropped). Then scale
`--num-steps`. **Cost note:** streaming runs K FULL episodes/group (no snapshot shortcut —
each candidate's pick-0 choice changes the whole episode), so it is the dominant
wall-clock; keep `--episodes-per-task`/`--group-size` modest for the smoke run.

## Where we are (2026-06-01)

**Foundation VALIDATED.** The original `pi05_baseline` *discards the subgoal*
(`project_pi05_baseline_ignores_subgoal`); we swapped to **GroundSG**, and the
causality probe confirmed **GroundSG + online oracle = ~83% success on
ButtonUnmask** and that a single held subgoal == the online oracle. The
executor, env, serving, reward, and rollout are all sound. We are NOT debugging a
deaf low-level policy anymore.

**Joint pipeline BUILT** (code-complete, compiles, 25 tests pass; one-shot path
preserved behind `joint_selection=False`):
- `grpo/selection.py` `apply_selection` (causal keyframe selection) + candidate
  window capture in `rollout.peek`.
- `trainer.py` trajectory GRPO: SELECT → `apply_selection` → USE → rollout, loss
  sums logp over both generations with the shared episode advantage.
- `build_memory_sft_dataset.py` emits USE rows + SELECT rows; `--joint-selection`
  wired through `main.py` + modal `grpo()`.

**SFT — the long pole — just had its breakthrough.** Journey:
1. Collapse (token_acc→0) → fixed with **lr 4e-6 + `label_smoothing 0.0`** (no
   collapse, token_acc 0.89). See `project_sft_v4_adapter_degenerate`.
2. Model copied the task goal (which *gives* the cube colour) and dodged the
   coordinate → fixed the target to be **coordinate-only-variable**
   ("pick up the container at <y,x>", colour stays in prompt as the task spec).
3. Then it emitted a **constant `<100,100>`** (dataset mean) — input-blind.
4. **Cube-visibility probe (the key finding):** the *base* VLM localizes the cubes
   correctly but in **Qwen-native (x,y) 0–1000**, not our raw `<y,x>` 0–256. The
   mismatch was the whole problem. Fix applied: builder now rescales the target to
   `<x,y>` 0–1000 (`_to_qwen_xy`); `<85,155>` → `<605,332>` ≈ the VLM's observed
   `(610,348)`.
5. **RESULT (2026-06-01): rescale was necessary but NOT sufficient — STILL
   input-blind. TODO #1 gate FAILED.** Best ckpt (checkpoint-300) emits a
   near-constant `~<315,305>` regardless of target (mean coord px-dist **298**,
   0–1000 space). Proof: targets `<605,332>` and `<578,504>` (Δy=172) both →
   `~<312,~324>` (Δy≈4). The constant just moved from `<100,100>` to `~<315,305>`,
   and it is NOT the model's own localization (`~<610,348>` per probe) — SFT pulled
   it toward a low-loss shortcut → **underfitting**, not perception. Caveat: the
   validation set was degenerate (5/6 rows were one episode) — fix to ≥10 distinct
   episodes before trusting the next read. → Execute the TODO #1 multi-task branch
   (below), and raise LR with it.
6. **SUCCESS (2026-06-01): the multi-task + LR fix CURED the input-blindness. TODO
   #1 PASSED.** Multi-task (ButtonUnmask+VideoUnmask) + lr 2e-5 + augment 2 →
   `checkpoint-150` **mean coord px-dist 10.4 (0–1000 space) ≈ 2.7px at 256**
   (ckpt-100 was 15px; still descending, no collapse). Outputs now VARY per episode
   and TRACK targets across a wide spread (x∈[344,668], y∈[266,517], each within
   ~4–24px). **Best ckpt so far = 150.** CAVEAT: the validator reads the TRAIN jsonl,
   so this proves the model CONDITIONS on the keyframes (the broken thing), NOT yet
   that it generalizes to unseen layouts vs. memorizing train episode→coord maps —
   that's settled by a rollout on unseen seeds (now the gating check before GRPO).
   **TODO #2 (coordinate conversion) is now also DONE** (see below).
7. **PROBE PASSED (2026-06-01): the full hierarchy works end-to-end on HELD-OUT
   seeds.** `probe_vlm_rollout` (ckpt-150, 12 unseen seeds): **VLM 0.750 progress /
   67% success vs ORACLE 0.833 / 75%** (VLM matches the oracle on 8 of its 9 winnable
   seeds; ep4/ep11 are hard seeds the oracle also fails). **SHIFTED 0.396** (gap
   +0.354, 10/1 wins) → the `from_qwen_xy` conversion + grounding are load-bearing and
   correct end-to-end. The flatline saga is RESOLVED: VLM reads spatial memory →
   grounded subgoal → conversion → GroundSG executes → task completes, on unseen seeds.
   **Implication:** the coordinate head is ~solved by SFT alone (as in RoboMME/MemER);
   one-shot ButtonUnmask GRPO has only ~8% headroom → it's a LOOP smoke test, not a
   results-driver. GRPO's real value is keyframe SELECTION on the Swap variants (TODO #4+).

## Plan review (2026-06-01, research-grounded)

Reviewed against how RoboMME + MemER actually train the memory module
(`reference_robomme_memer_training`, `project_sft_plan_adjustments`). **The spine
is correct — keep SFT-warm-up → GRPO on GroundSG.** Both reference papers train the
memory module SFT-only (no RL at all), proving SFT yields a competent subgoal +
keyframe policy; and the VLA-RL literature (SimpleVLA-RL, RIPT-VLA, VLA-RFT) is
unanimous that cold-start GRPO needs a NON-ZERO rollout success rate to bootstrap.
So **"skip the warm-up / go straight to GRPO" is settled: no.** Three adjustments
fold into the TO-DO below (multi-task as a stability lever, Swap variant for the
selection claim, SELECT/USE schema collision). Our "GRPO instead of IL" is the
novel and riskiest bet — RL-for-keyframe-selection is unprecedented, so a fallback
is named in the Deferred section.

## TO-DO (prioritized)

1. **Confirm the coordinate-rescaling fix** ← *immediate (safe to let a running SFT finish)*
   ```bash
   modal run modal_train.py::build_memory_dataset --no-joint   # USE rows, now <x,y> 0–1000 targets
   modal run modal_train.py --stage sft                         # lr 4e-6, label_smoothing 0.0
   modal run modal_train.py::validate_memory_checkpoints         # expect: coords MATCH targets, token_acc climbs
   ```
   Success = `mean_coord_px_dist` small **and varying per episode** (not a constant);
   token_acc climbs past the prefix-only floor. That proves memory works.

   **STATUS: single-task gate FAILED (2026-06-01, see "Where we are" #5). The
   corrective multi-task run is now IMPLEMENTED in `modal_train.py` defaults — run:**
   ```bash
   modal run modal_train.py::build_memory_dataset --no-joint   # ButtonUnmask+VideoUnmask, augment 2, USE rows
   modal run modal_train.py::sft_warmstart                      # lr 2e-5 → ckpts/.../permanence_grounded
   modal run modal_train.py::validate_memory_checkpoints        # 12 DISTINCT episodes, coord-px-dist
   ```
   Why these changes go together (all attack the same underfit/constant-output, ONE run):
   - **Multi-task ButtonUnmask + VideoUnmask** (`only_tasks` default). Narrow
     single-task data makes a constant a winning shortcut and CAPS the safe LR
     (`project_sft_v4_adapter_degenerate`, gotcha #5); RoboMME ran lr 1e-4 multi-task
     across 1,600 demos with no collapse. VideoUnmask is builder-compatible (phase0
     "static" vs "press", same `pick up the container at <>` target, reveal=64).
     **Swap variants are EXCLUDED here** — the cube moves mid-episode so reveal-window
     keyframes go stale; Swap is for the keyframe-SELECTION work (TODO #4+), not the
     coordinate warm-up.
   - **LR 4e-6 → 2e-5.** The constant output is UNDERFITTING, but single-task lr 1e-5
     already collapsed (token_acc 0.65→0.50 at ~step 80), which is why 4e-6 was the
     ceiling. The higher LR is only safe BECAUSE multi-task lifts the collapse
     threshold — they are coupled, not independent. Back off toward 1e-5 if it
     collapses; raise toward 4e-5 if it still underfits.
   - **`augment_factor` 5 → 2.** `=5` (same target, different keyframes) over-teaches
     keyframe-INVARIANCE → input-blindness; multi-task already 2×'s distinct episodes.
   - **Validation fixed** — `validate_memory_checkpoints` now keeps ONE row per
     DISTINCT episode (was 5/6 the same episode), skips SELECT rows, spreads across
     both tasks; `n_samples` 6 → 12. Watch **coord-px-dist**, NOT token_acc
     (template-inflated); `save_total_limit` already 20; pick the best pre-collapse ckpt.
   - **If still a constant after this → extractability probe:** give the VLM the
     reveal keyframes and ask it to locate "the container hiding the {color} cube."
     If even that fails, the memory isn't recoverable from the keyframes (need richer
     keyframes/representation) — no LR/data change fixes that.

2. **Inference coordinate conversion — DONE (2026-06-01).** Shared pair in
   `src/vla_memory/qwen_subgoal/coords.py`: `to_qwen_xy` (pixels `<y,x>` 0–256 →
   Qwen `<x,y>` 0–1000, now also the builder's `_to_qwen_xy`) + `from_qwen_xy` (the
   inverse). `rollout.rollout()` applies `from_qwen_xy(sampled_subgoal)` at the
   SINGLE point VLM output reaches the executor — the oracle warm-up / `rollout_oracle`
   / freeze probes use native coords and are NOT touched. Both trainer executor paths
   (direct `worker.rollout` and `_score_rollout`) route through there. Round-trip +
   axis-order pinned in `tests/test_coords.py` (passes; identity to ±1px).

3. **Held-out VLM rollout probe — PASSED (2026-06-01).** `modal run
   modal_train.py::probe_vlm_rollout` (ckpt-150, 12 unseen seeds): **VLM 0.750 / 67%
   vs ORACLE 0.833 / 75%**, SHIFTED 0.396 (gap +0.354, 10/1). All three gates green
   (bootstrap, generalizes, coord load-bearing). The hierarchy works end-to-end on
   held-out seeds; the conversion is verified in-sim. Driver: `grpo/probe_vlm_rollout.py`.
   **Next: one-shot GRPO as a LOOP SMOKE TEST, not a results-driver.** Run
   `--no-joint-selection` (default) from `checkpoint-150` on ButtonUnmask:
   ```bash
   modal run modal_train.py::grpo --debug-subgoals \
     --sft-adapter-path <…>/permanence_grounded/v0-20260601-042401/checkpoint-150
   ```
   The coordinate head is ~saturated by SFT (only ~8% headroom, partly hard-seed-bound),
   so the GOAL here is to validate the GRPO LOOP on a known-good policy — reward
   RESPONDS to the subgoal (non-degenerate groups), generations terminate
   (`--debug-subgoals`, the v1 no-EOS check), no collapse — before the harder joint
   task. Every prior GRPO run flatlined (deaf executor / degenerate SFT / no-EOS /
   set_adapter grad no-op), all now fixed; this is the controlled confirmation. Don't
   expect a big jump; "holds ~67% / nudges up, signal healthy" = pass. THEN the
   contribution: joint selection on Swap (#4–6).

4. **Joint selection v1 — SCHEMA FIX DONE (2026-06-01); run on ButtonUnmask (broad
   candidate window).** The SELECT/USE collision is FIXED: SELECT now uses a distinct
   `SELECT_SYSTEM_PROMPT` + `mode="select"` user prompt + a target carrying ONLY
   `keyframe_positions`; USE is byte-identical (ckpt-150 unaffected — pinned by
   `tests/test_select_use_prompts.py`). Threaded `mode` through `prompts.py`,
   `model.py` (sample/greedy/`_prepare_inputs`), `trainer._joint_group` (SELECT call
   `mode="select"`), `build_memory_sft_dataset.py` (SELECT row). One-shot path
   untouched (default `mode="use"`).
   **v1 task = ButtonUnmask/VideoUnmask with the BROAD candidate window** the builder
   already emits (reveal + post-occlusion *covered* frames): selection = "keep the
   cube-visible reveal frames, reject the covered ones" — a real (if modest) selection
   task (`JOINT_MEMORY_DESIGN §5`); the earlier "any selection works" worry only holds
   for a reveal-only window. Run:
   ```bash
   modal run modal_train.py::build_memory_dataset        # joint=True (default) → USE + SELECT rows
   modal run modal_train.py::sft_warmstart               # trains both heads → permanence_grounded/<new v*>
   modal run modal_train.py::validate_memory_checkpoints # checks BOTH heads (USE coord + SELECT)
   modal run modal_train.py::grpo --joint-selection --debug-subgoals \
     --sft-adapter-path <…>/permanence_grounded/<new v*>/checkpoint-<best>
   ```
   `validate_memory_checkpoints` now scores the SELECT head too (DONE 2026-06-01):
   decodes SELECT rows with `mode="select"` → `select_mean_jaccard`/`recall` vs the
   labeled reveal frames, and `select_n_empty` (≈ all = the head collapsed to `[]`,
   the bug the schema fix kills).
   **JOINT SFT RESULT (2026-06-01): schema fix CONFIRMED — `select_n_empty=0/12` on
   every checkpoint** (no collapse). SELECT jaccard 0.37(ckpt50)→0.68(ckpt100)→0.61
   (150/200); USE coord px-dist 467→14.8→6.7→5.0 (0–1000 space, so ≤15 ≈ ≤4px@256 =
   all within-container). Coordinate is NOT the differentiator (all sub-4px); SELECT
   PEAKED at ckpt-100 (0.681) then slipped — so favor ckpt-100 as the GRPO warm-start
   (best selection prior, coord already within-container). Launch:
   `grpo --joint-selection --debug-subgoals --sft-adapter-path
   <…>/permanence_grounded/<joint-v*>/checkpoint-100` (PIN the full path — bare dir →
   resolver grabs MAX=200). First goal: reward responds, SELECT `kf=` VARIES across
   candidates, non-degenerate groups.
9. **ButtonUnmask joint GRPO = DEGENERATE, as predicted (2026-06-01). KILL it.** Step 1:
   `n_groups_used=1, n_groups_dropped=11/12, grad_norm 0.068, ~1.8h/step`. The dynamic
   sampler threw away 11 of 12 groups — every candidate kept reveal frames and succeeded,
   so zero reward variance → no selection gradient. This is the EMPIRICAL proof that
   ButtonUnmask can't train selection (any reveal subset works) and that the contribution
   needs **Swap** (#5). The run also confirmed the joint loop machinery is sound (SELECT
   varies, apply_selection works, USE coords within-container, generations terminate). Don't
   run it to completion; the coordinate head is solved and selection has no signal here.

5. **Joint selection v2 — Swap variants (IN PROGRESS).** This is the run that
   actually demonstrates learned selection (ButtonUnmask v1 proved degenerate — see
   "Where we are" #9). `ButtonUnmaskSwap` swaps containers at steps **64/114/164**
   (`_refresh_swap_schedule`, per `swap_times`), AFTER the t=0–64 reveal and aligned
   with its TWO button presses (`press first/second button → pick`).
   - **DONE (2026-06-01): decision-point fix.** The builder detected the pick by
     "first subgoal change", which on Swap fires at press1→press2 (~step 64, pre-pick).
     Now it detects the pick subgoal directly (`"pick up the container" in simple_online`)
     — correct for Swap AND identical-frame for single-button ButtonUnmask (parity safe).
   - **DONE: `inspect_swap_demos` probe** — confirmed the structure: pick starts at
     rel ~165–180, `grounded_subgoal_online` at the pick IS the post-swap final position
     (e.g. `<93,163>`), completed-before-pick = both presses. **Also revealed
     ButtonUnmaskSwap is MULTI-PICK** (1–2 picks, randomized; e.g. blue then green).
   - **DONE: multi-history** — builder USE row + `rollout._warmup`/`peek_at_decision_point`
     now collect ALL completed non-pick subgoals (both presses on Swap), identical logic
     in both → prompt parity. ButtonUnmask unchanged (single press → same as `[phase0]`).
   - **DONE: memer-style SELECT label** — `_reveal_label` replaced by `_memer_important`
     (subgoal transitions + `find_local_minima` action-velocity minima, reused from the
     submodule) + `_memer_label` (keep candidates NEAREST those important timesteps). On
     Swap this spans reveal→swaps (verified mapping in a unit test); falls back to even
     spacing if none found (never empty).
   - **DONE: joint USE-keyframe alignment** — in the JOINT path the USE row keyframes are
     now the SELECT-selected candidate frames (`cand_paths[sel_label]`), NOT a reveal-window
     slice. At GRPO the USE call receives `apply_selection`'s output, so SFT-USE ≡ GRPO-USE;
     on Swap this trains the USE head on the swap-tracking frames it will actually get
     (else it'd be trained reveal-only → couldn't ground the post-swap position → the run
     would go degenerate-toward-FAILURE). One-shot path keeps reveal-window (ckpt-150 parity).
   - **DONE: multi-pick per-pick reward (2026-06-01).** Each pick is its OWN GRPO state:
     the oracle drives picks 0..i-1 (executing them), the VLM owns pick i, scored by
     absolute progress. Both picks earn reward; GRPO's within-group advantage is relative
     so the later picks' progress floor (the oracle's earlier picks) cancels — each pick's
     selection gets a clean gradient. Chose per-pick STATES over a resumable
     multi-decision rollout: simpler, reuses the existing single-decision machinery, and
     CLEANER credit (each selection scored by its own pick, not entangled in one
     accumulated reward). Implemented across: `state_dataset` (`pick_index` +
     `picks_per_episode`), `rollout._warmup(pick_index)` (warm to the i-th pick, cap ×(i+1),
     history = completed subtasks incl. earlier picks), `peek`/`rollout` plumbing, `trainer`
     (passes `state.pick_index`), `main.py`/`modal grpo` (`--picks-per-episode`,
     `decision-warm-cap` 150→250), builder (`_emit_pick` loops over ALL picks; history +
     keyframes + grounded per pick). Builder↔rollout history parity unit-verified.
   - **Run it:**
     ```bash
     modal run modal_train.py::build_memory_dataset --only-tasks ButtonUnmaskSwap   # per-pick USE+SELECT rows
     modal run modal_train.py::sft_warmstart
     modal run modal_train.py::validate_memory_checkpoints   # both heads; pick high SELECT jaccard
     modal run modal_train.py::grpo --joint-selection --only-tasks ButtonUnmaskSwap \
       --picks-per-episode 2 --debug-subgoals --num-steps 40 \
       --sft-adapter-path <…>/permanence_grounded/<v*>/checkpoint-<best>
     ```
     **Core question:** are Swap groups NON-degenerate (reward varies with selection),
     unlike ButtonUnmask's 11/12-dropped? Note: pick-2 states only reach their decision
     when the oracle completes pick-1 (~83%), so expect some pick-2 groups dropped — that's
     warm-up yield, not a selection failure.

6. **Eval vs baselines** (Phase 5) — ours (GRPO-VLM + GroundSG) vs MemER-IL vs
   memoryless `pi05_baseline` on the **Swap** Permanence val/test (the task where the
   selection claim is meaningful).

### Deferred (don't block the above)
- **GRPO speed — IMPLEMENTED (opt-in, gated; 2026-06-01).** `_ensure_env` rebuilt the
  env per rollout (K+1 warm-ups/group). New `--snapshot-branching` (default OFF) warms
  up ONCE, snapshots the env at the decision point (`grpo/env_snapshot.py`: physics
  `get_state_dict` + the WHOLE wrapper-chain bookkeeping incl. `_elapsed_steps` + the
  `DemonstrationWrapper` layer — a bare `get_state_dict` is insufficient), and restores
  it per candidate (`rollout.peek_and_snapshot`/`rollout_from_snapshot`). Also removes
  warm-up sampling noise between candidates. **GATE before trusting it:**
  `modal run modal_train.py::probe_snapshot_parity` (record-then-replay parity; must
  exit 0 on ButtonUnmask + ButtonUnmaskSwap pick 0/1) — keeps the rebuild path as the
  verified default. Copy-policy unit test: `tests/test_env_snapshot.py`.
- **Learned keyframe selection refinement** beyond the heuristic SELECT labels.

### Named fallback (if joint-RL selection is too weak)
RL-for-keyframe-selection is unprecedented — neither RoboMME nor MemER used RL at
all, and under GRPO selection is rewarded only indirectly through task success
(hard credit assignment). If the joint GRPO selection signal proves too weak, keep
selection **SFT-supervised** (MemER style, direct index labels) and RL only the
grounding head. Still a novel RL memory hierarchy — minus the selection risk. Hold
this as a named fallback, don't discover it mid-run.

## Run-stage map (`modal_train.py`)

| Stage | Entry | Purpose |
|---|---|---|
| download GroundSG | `::download_groundsg` | subgoal-conditioned π0.5 from `Yinpei/mme_vla_suite/symbolic-grounded-subgoal/79999` |
| causality probe | `::causality_probe` | BLOCKER check (passed): does π0.5 read the subgoal? |
| cube-visibility | `::probe_cube_visibility` | can the base VLM see/localize the cubes? (yes — in 0–1000) |
| build memory SFT | `::build_memory_dataset [--no-joint]` | multi-task ButtonUnmask+VideoUnmask, augment 2; coordinate `<x,y>` 0–1000 |
| inspect swap demos | `::inspect_swap_demos` | dump Swap subgoal sequence + grounded targets + timing (probe before Swap build) |
| SFT | `--stage sft` | LoRA warmstart (lr 2e-5, label_smoothing 0.0, save/eval every 50) → `permanence_grounded` |
| validate (faithful) | `::validate_memory_checkpoints` | output vs target on DISTINCT-episode inputs + `mean_coord_px_dist` |
| held-out probe | `::probe_vlm_rollout` | **pre-GRPO gate**: VLM greedy subgoal → GroundSG on unseen seeds vs oracle/shifted |
| GRPO | `::grpo [--joint-selection]` | one-shot or joint trajectory GRPO |
| download demos | `--stage download_data` | `Yinpei/robomme_data_h5` (~30GB, done) |

## File map (current)

```
modal_train.py            # all stages above
MEMORY_PIPELINE.md         # Phase 0–5 plan + the pi05→GroundSG reframe
JOINT_MEMORY_DESIGN.md     # joint select-then-use architecture (the contribution)
src/vla_memory/
  qwen_subgoal/
    prompts.py             # MemER prompt; mode="use" (subtask) | "select" (keyframes) — distinct system+user+target
    model.py               # QwenSubgoalPolicy: multi-frame input, sample/greedy (mode-aware), logprob recompute
    coords.py              # to_qwen_xy / from_qwen_xy — SFT-target ↔ GroundSG coordinate conversion
  grpo/
    selection.py           # apply_selection (keyframe_positions → kept frames)
    rollout.py             # oracle warm-up → keyframe capture → candidate window; from_qwen_xy on VLM subgoal
    probe_vlm_rollout.py   # held-out gate: VLM greedy subgoal → GroundSG vs oracle/shifted (pre-GRPO)
    trainer.py             # trajectory GRPO: _oneshot_group / _joint_group / _score_generation
    reward.py              # task-completion fraction
    causality_probe.py     # Phase 0 diagnostics (oracle ceiling, freeze-at-transition)
    main.py                # GRPO entry (micromamba env); --joint-selection flag
    env_runner.py          # EnvRunner subclass (dataset=train, seeded reset)
  data/build_memory_sft_dataset.py   # USE + SELECT rows; coordinate-focused, <x,y> 0–1000
robomme_policy_learning/   # submodule — DO NOT EDIT
```

## Key gotchas (this session)

1. **`pi05_baseline` ignores the subgoal** — use GroundSG (`config=mme_vla_suite`,
   the ckpt's `history_config.txt` activates grounded-subgoal memory). `project_pi05_baseline_ignores_subgoal`.
2. **Cubes spawn in `make_env`, not `reset`** → `_ensure_env` must REBUILD per
   rollout, else the cube layout is corrupted (the long "fixed subgoal can't finish" red herring).
3. **The task goal gives the cube colour** — so colour is the *spec*, not memory;
   the *coordinate* is the memory. Keep colour in the prompt, force the coordinate.
4. **Qwen-VL coordinates are `<x,y>` 0–1000**, not raw `<y,x>` 0–256 — align SFT
   targets to the VLM's native space (`_to_qwen_xy`) or it can't learn and emits a constant.
5. **SFT collapses above lr ~8e-6** on this narrow data; lr 4e-6 + `label_smoothing 0.0`
   (smoothing blurs the exact coordinate). `project_sft_v4_adapter_degenerate`.
6. **`attn_implementation="sdpa"`** everywhere (no flash-attn: cudnn-runtime has no nvcc).

## Memories to read (in `/memory/`)
- `project_memory_pipeline_direction.md` — the pivot + current direction
- `project_pi05_baseline_ignores_subgoal.md` — the executor swap
- `project_sft_v4_adapter_degenerate.md` — the SFT-collapse saga
- `project_qwen_grpo_layout.md`, `project_robomme_setup.md` — stack + project shape
