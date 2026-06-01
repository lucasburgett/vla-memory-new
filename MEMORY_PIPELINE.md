# MEMORY_PIPELINE.md — Hierarchical Memory VLA via GRPO (MemER-structured)

> Status: PLAN (2026-05-31). Supersedes the PickXtimes/t=0 GRPO direction.
> Goal: a memory-augmented manipulation hierarchy where a high-level VLM
> *remembers* task-relevant visual facts (e.g. which container hides the cube)
> and issues grounded subgoals to a frozen low-level π0.5 — trained with **GRPO**,
> not imitation learning.

---

## TL;DR

- **Structure = MemER** (Pan et al., *Scaling Up Memory for Robot Control via
  Experience Retrieval*, arXiv:2510.20328): a high-level video-VLM (Qwen) that
  conditions on **recent frames + selected keyframes** and emits a **subtask +
  keyframe nominations**, driving a **frozen low-level π0.5** action expert.
- **Our one difference from MemER:** MemER trains the high-level VLM with
  **imitation learning** on ~50 demos/task. **We train it with GRPO** — reward =
  "did the low-level policy, executing the VLM's grounded subgoal, complete the
  memory task in sim."
- **Decisive finding (changes everything):** the current frozen low-level policy
  `pi05_baseline` **discards the subgoal entirely** — it is conditioned only on
  the task goal. The VLM's output has had **zero causal effect** on actions. No
  subgoal hierarchy (memory or otherwise) can work until we swap it for a
  **subgoal-conditioned (GroundSG) π0.5**.

---

## 1. The decisive finding: the hierarchy was deaf

`mme_vla_suite/training/config.py` → `ModelTransformFactory` (PI05 case):

```python
symbolic_memory_type = None
if model_config.use_history and model_config.history_config is not None:
    loaded = get_history_config(model_config.history_config)
    if loaded.representation_type == "symbolic":
        symbolic_memory_type = loaded.symbolic_memory.type   # simple/grounded
```

`pi05_baseline` has `use_history=False`, `history_config=None`, so
`symbolic_memory_type` stays `None`. Then in `TokenizePromptWithSymbolicMemory`:

```python
if self.symbolic_memory_type is None:
    data.pop("simple_subgoal")      # <-- subgoal thrown away
    data.pop("grounded_subgoal")    # <-- subgoal thrown away
    return {... "tokenized_prompt": tokens ...}   # prompt = task goal only
```

So every GRPO rollout we ran fed the subgoal into `element["simple_subgoal"]`
(`rollout.py:_infer_actions`) and the policy server **silently dropped it**.
Within-group reward variance came purely from π0.5's stochastic flow sampling,
not from the subgoal. This is a **second, independent root cause** of the
flatline, orthogonal to the degenerate SFT adapter
(`project_sft_v4_adapter_degenerate`): even a perfect VLM subgoal would have
changed nothing.

This is also conceptually right: `pi05_baseline` is RoboMME's *memoryless
baseline*, designed to **fail** memory tasks. Using it as the executor of a
memory hierarchy is backwards.

---

## 2. MemER architecture, and how it maps onto this repo

MemER high-level policy, per timestep `t`:

```
INPUT   R_t = last N recent frames (N=8)         ─┐
        K_t = selected keyframes (|K|≤8)          ├─► VLM ─► { subtask l'_t ,
        l_t = high-level task instruction        ─┘            keyframe noms J_t }
OUTPUT  l'_t  → low-level π0.5 (executes ~1s)
        J_t   → clustered (1D single-linkage, d=5) → updates K_{t+1}
```

Low-level = π0.5 (frozen). High-level = Qwen2.5-VL-7B (MemER) / Qwen3-VL-4B (us).

**The submodule already ships every MemER piece** — RoboMME's authors built
MemER (IL-trained). We reuse them:

| MemER piece                  | Where it already lives                                                  |
| ---------------------------- | ----------------------------------------------------------------------- |
| Subgoal-conditioned π0.5     | `config="mme_vla_suite"` + history cfg `symbolic-grounded-subgoal.yaml` ("GroundSG") |
| Keyframe + multi-image data  | `dataset_builder/build_vlm_subgoal_dataset_memer.py`                    |
| MemER prompt format          | system + `Past Keyframe i: <image>` / `Executed Frame i: <image>`, JSON `{current_subtask, keyframe_positions}` |
| Memory tasks                 | Permanence suite: `ButtonUnmask(Swap)`, `VideoUnmask(Swap)`             |
| Per-step oracle subgoal      | `info["grounded_subgoal_online"]` (mid-episode supervision)             |

**GroundSG = a subgoal-conditioned π0.5 + a subgoal predictor.** MemER's
predictor is a QwenVL fine-tuned with IL. **Ours is the same QwenVL trained with
GRPO.** That's the whole project, and the code path exists.

---

## 3. Target task: `ButtonUnmask` (Permanence suite → spatial memory)

`ButtonUnmask.step()` calls
`lift_and_drop_objects_back_to_original(bin, start_step=0, end_step=64, cur_step=t)`:

```
 t=0 ........... reveal window ........... t≈64 .................... end
 bins LIFT, colored cubes (R/G/B) visible │ bins DROP, cubes COVERED
 VLM must SEE this                         │ VLM must REMEMBER which bin
                                           │ → "pick up the container at <bbox>
                                           │    that hides the red cube"
```

- Color→bin mapping is **seed-shuffled** → the answer can't be memorized; it must
  be *remembered from the reveal*.
- Oracle subgoal `"pick up the container at <> that hides the {color} cube"`:
  the memory lives in the **grounding** (`<>` = which physical container). So the
  task is only solvable in **`grounded_subgoal`** mode — which is why MemER /
  GroundSG use grounded subgoals.
- `easy` (3 bins, 1 pick) is the cleanest first target; `medium`/`hard` add bins.

---

## 4. Memory mechanism (what we build)

1. **VLM input = recent frames + reveal-window keyframes** (MemER multi-image).
   The pre-occlusion frames (bins up) are the memory that disambiguates the bin.
2. **Mid-episode prediction.** Query the VLM *after* occlusion (`warm_steps`
   replays past the reveal) so it must use memory, not luck. `rollout.py` already
   supports `warm_steps`; `state_dataset` must emit `frame_idx>0` decision points.
3. **Grounded subgoal → GroundSG π0.5.** The VLM emits a grounded subgoal whose
   bbox names the remembered container; the (now subgoal-conditioned) π0.5
   executes it.
4. **Reward = task-completion fraction** (already in `reward.py`): picking the
   correct container advances the sequential-task tracker; the wrong one trips
   `failure_func`.

**Keyframe selection — phased:**
- **Phase A (first): heuristic keyframes.** Feed the reveal-window frames
  directly (no learned selection). Clean GRPO signal; isolates "can the VLM *use*
  memory." This is the one deliberate simplification vs. full MemER.
- **Phase B (later): learned selection.** Add MemER's `keyframe_positions` output
  and reward/clustering. Harder credit assignment under GRPO — deferred until A
  works.

---

## 5. Phased build

```
Phase 0  UNBLOCK: subgoal-conditioned low-level policy        [BLOCKER]
  - Source a frozen GroundSG π0.5 (symbolic-grounded-subgoal).
  - serve_policy: config mme_vla_suite + history_config=symbolic-grounded-subgoal.
  - CAUSALITY PROBE: same scene, two different grounded subgoals → different
    actions/outcome. If the subgoal doesn't move behavior, STOP — nothing else
    matters. (Cheapest, highest-value experiment in the whole plan.)

Phase 1  VLM memory input (fork-independent, safe to build now)
  - prompts.py: MemER multi-image prompt (Past Keyframe i / Executed Frame i),
    JSON output, optional <video>.
  - model.py _prepare_inputs: accept a LIST of frames (keyframes+recent), expand
    one <image> per frame; keep grounded <box> markers.

Phase 2  Env memory capture + mid-episode
  - rollout/env_runner: capture reveal-window frames; expose a keyframe set.
  - state_dataset: emit post-occlusion decision points (frame_idx>0 via warm_steps).
  - trainer._rollout_group: sample subgoals with (keyframes, recent_frames),
    not history_subgoals=[].

Phase 3  SFT warmstart on memory data
  - Build ButtonUnmask grounded+keyframe SFT data via the memer builder
    (or our build_sft_dataset adapted to its multi-image format).
  - SFT (lessons from project_sft_hyperparameters / v4 collapse: low peak LR,
    early-stop on token_acc, save_total_limit high).

Phase 4  GRPO on ButtonUnmask
  - GRPO from the SFT warmstart; reward = pick-correct-container fraction;
    seed_match_group pins the shell-game layout across a group.
  - Held-out probe (greedy subgoal on unseen seeds) + oracle ceiling.

Phase 5  Eval
  - Compare: ours (GRPO-VLM) vs MemER-IL vs memoryless pi05_baseline,
    on Permanence val/test. The thesis claim lives here.
```

---

## 6. Open decision (needs user)

**How to source the subgoal-conditioned low-level policy (Phase 0)?** The whole
plan hinges on a frozen π0.5 that *reads* grounded subgoals. Options: download a
pretrained GroundSG checkpoint from `Yinpei/mme_vla_suite` (if published),
vs. train our own (JAX, expensive), vs. verify availability first. See chat.

---

## 7. Risks / unknowns

- **GroundSG checkpoint availability.** `Yinpei/mme_vla_suite` is referenced but
  the specific grounded-subgoal checkpoint isn't confirmed downloadable. Verify
  first; if absent, training a π0.5 variant is a large (JAX) detour.
- **Grounded-format plumbing — RESOLVED.** GroundSG consumes the literal oracle
  text `"...at <y, x>..."` (verified: the probe fed exactly that). The submodule's
  MemER builder instead rewrites `<y,x>` → `<bbox>` placeholder + `objects.bbox`
  (Qwen grounding tokens), which would need a conversion step. **Decision:** train
  our VLM to emit the LITERAL `<y, x>` text (Phase 3 builds SFT targets from
  `grounded_subgoal_online` without the `<bbox>` conversion). No conversion at
  inference → `rollout.py` passing `c.subtask` straight to GroundSG is correct.
- **Causality probe v1 was color-only (false negative).** Swapping the colour word
  while keeping the coordinate tests a redundant signal — a correct grounded policy
  ignores it. Probe v2 shifts the GROUNDING POINT ~100px (`_shift_coord`); that is
  the real test. Re-run before trusting the verdict.
- **GRPO credit assignment** stays hard even unblocked: reward is sparse-ish per
  episode and π0.5 is stochastic. Keep dense progress fraction + rollouts_per_subgoal.
- **Cost.** Permanence rollouts are longer than PickXtimes (reveal + pick); budget
  accordingly.
