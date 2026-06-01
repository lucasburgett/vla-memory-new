# JOINT_MEMORY_DESIGN.md — joint keyframe-selection + subtask via GRPO

> Status: DESIGN (2026-05-31). The project's actual contribution: train the
> VLM's **keyframe selection** and **subtask** *jointly*, in one GRPO run, like
> MemER (which does it jointly via IL) — but with RL reward instead of imitation.

---

## 1. Goal & why joint

MemER's high-level VLM emits, every call, **both** a subtask and **keyframe
nominations** (which frames to remember), trained under one objective so they
**co-adapt**: the selector keeps frames the subtask-predictor will need; the
predictor learns to exploit what's kept. Training them separately loses that
coupling. We train both jointly with GRPO, one run.

## 2. The load-bearing requirement: selection must be CAUSAL

Today `keyframe_positions` is a **dead** output — the keyframes fed to the VLM
are heuristic (reveal window), so the selection changes nothing and GRPO can't
learn it. For selection to be trained, **the selected frames must become the
memory the VLM uses for its decision, and the non-selected frames must be
discarded** — otherwise the VLM "saw everything anyway" and selection is moot.

## 3. Architecture — SELECT-THEN-USE at the decision point (v1)

Two VLM calls per candidate, using the SAME MemER prompt (`past keyframes` +
`recent frames` → `{current_subtask, keyframe_positions}`), interpreting a
different field each call:

```
peek/warm-up  → capture a BROAD candidate window C (reveal + some post-occlusion
                frames, e.g. 12) + the current occluded frame f_now.

SELECT call   prompt(past_keyframes=[], recent_frames=C)        ← sees candidates
              → output.keyframe_positions = indices of C to KEEP (≤ K, e.g. ≤4)
              selected = [C[i-1] for i in keyframe_positions]   (1-indexed, clamped)

USE call      prompt(past_keyframes=selected, recent_frames=[f_now])   ← sees ONLY kept
              → output.current_subtask = "pick container at <y,x> ..."

rollout       execute the pick subtask on GroundSG π0.5 → task-success reward
```

The USE call sees **only the selected keyframes** (+ current frame), never the
full candidate set — that's what makes "kept the cube frame → right pick →
reward / kept a useless frame → had to guess → fail" a learnable signal.

`v2 (later): streaming` — VLM selects incrementally through the episode (MemER's
form), needed to scale to long horizons where C can't hold everything. v1 (select
from a bounded captured window) trains the same selection mechanism without the
streaming machinery; we start there.

## 4. Joint GRPO objective (trajectory-level)

Each candidate is a **trajectory** of two VLM generations `o_sel, o_use`. One
episode reward `R`. For a group of K trajectories:

```
A_k   = R_k − mean(R)                      (Dr.GRPO: no /std)
loss  = − Σ_k A_k · ( logπ(o_sel^k) + logπ(o_use^k) ) / C
```

Both generations get gradient from the **same** reward → selection and subtask
trained jointly, one run. Selection is rewarded *indirectly* (a good keep makes
the pick succeed). KL-to-SFT and per-candidate backward as in the current loss.

## 5. Why ButtonUnmask still needs care (task choice)

ButtonUnmask's reveal is fixed-time and cubes are visible the whole reveal, so if
C = only reveal frames, *any* selection works → nothing to learn. Fix for v1:
make **C a broad window (reveal + post-occlusion covered frames)** so the VLM must
learn to keep the **cube-visible** frames over the covered ones — a real (if
modest) selection task. The stronger demonstration is a **Swap** variant
(`ButtonUnmaskSwap`/`VideoUnmaskSwap`): the cube moves after reveal, so *when* to
grab a keyframe matters — that's where joint selection earns its keep. Plan:
ButtonUnmask (broad C) first, Swap second.

## 6. SFT warm-start (selection-aware)

GRPO from scratch on a 2-call trajectory is unstable; warm-start both heads:
- **USE rows** (already built): `past=selected/reveal keyframes`, `recent=[f_now]`
  → subtask. (Our `build_memory_sft_dataset`.)
- **SELECT rows** (new): `past=[]`, `recent=C` → `keyframe_positions` = the
  cube-visible (reveal) frame indices within C. The submodule's
  `build_vlm_subgoal_dataset_memer.py` already labels "important" frames
  (action-velocity minima + subgoal transitions) — reuse that labeling for the
  selection target; for ButtonUnmask the important frames are the reveal frames.

## 7. Implementation plan (ADDITIVE — keep the one-shot path working)

A `joint_selection: bool` flag on `GRPOConfig` gates the new path so the current
subtask-only SFT/GRPO keeps working and we can A/B.

| File | Change |
|---|---|
| `qwen_subgoal/model.py` | none — `sample_subgoals` already returns both `subtask` and `keyframe_positions`; the SELECT call uses the latter, the USE call the former |
| `grpo/selection.py` (new) | `apply_selection(keyframe_positions, candidates, max_k)` → selected frames (pure, tested) |
| `grpo/rollout.py` | `peek` also returns a broad `candidate_frames` window + `current_frame` (new `DecisionPoint` fields); env execution unchanged |
| `grpo/trainer.py` | joint branch: SELECT call (K samples) → `apply_selection` → USE call (per selection) → rollout; trajectory loss over both generations |
| `data/build_memory_sft_dataset.py` | also emit SELECT rows (candidates → keyframe-index labels) |
| `grpo/main.py`, `modal_train.py` | `--joint-selection` flag, joint SFT dataset path |

## 8. Open decisions

- **|C| (candidate count)** and **K (max kept keyframes)** — start C≈12, K≤4.
- **Candidate window span** — reveal + how many post-occlusion frames (to create
  "bad" candidates the selector must reject). Start: reveal[0:64] + a few covered.
- **Selection reward shaping** — pure task reward (indirect) first; add an
  auxiliary "kept-a-cube-frame" bonus only if credit assignment is too weak.
- **Empty/over-long selection** — clamp to ≤K; empty selection → USE call sees
  only `f_now` (no memory) → should fail → negative signal (that's correct).
