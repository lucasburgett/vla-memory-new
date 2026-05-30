# Training Pipeline — How the Data, SFT, and GRPO All Fit Together

A walkthrough of what each training stage does, what data it consumes, what
signal it optimizes against, and how to tell if it's working. Companion doc to
`HANDOFF.md` (which is more about *what's running where* on Modal).

---

## The stack we're building

```
        ┌─────────────────────────────────────────┐
        │  Qwen3-VL-4B + LoRA (we train this)     │   ← src/vla_memory/*
        │  Subgoal predictor / "memory module"    │
        └────────────────┬────────────────────────┘
                         │ subgoal text (e.g. "pick up the first red cube")
                         ▼
        ┌─────────────────────────────────────────┐
        │  pi0.5 baseline, frozen (JAX)           │   ← modal_server.py serves
        │  scripts/serve_policy.py, WebSocket     │
        └────────────────┬────────────────────────┘
                         │ action chunks (joint targets)
                         ▼
        ┌─────────────────────────────────────────┐
        │  ManiSkill / RoboMME simulator          │
        │  16 tasks × 50 val episodes             │
        └─────────────────────────────────────────┘
```

The only weights that change during training are the LoRA adapters on the Qwen
language tower. Vision encoder, vision-language aligner, π0.5 expert, and
simulator are all frozen.

---

## Pipeline stages at a glance

| Stage | Entry | Signal | Data source | What it optimizes |
|---|---|---|---|---|
| 3. Build SFT JSONL | `modal_train.py --stage build_dataset` | — | RoboMME H5 demos | Materializes (image, subgoal) pairs |
| 4. SFT warmstart | `./scripts/train_sft.sh` | Cross-entropy loss | SFT JSONL | P(oracle subgoal \| image, task) |
| 5. GRPO | `modal_train.py --stage grpo` | Task-completion reward | Live ManiSkill rollouts | E[reward(rollout(sampled subgoal))] |
| 6. Eval | `modal_server.py` w/ adapter | Pass-rate | Held-out test episodes | (verification only) |

SFT and GRPO are **two completely different optimization problems** that
happen to update the same LoRA adapter. SFT gives the model the right
*output format and vocabulary*; GRPO refines it to maximize what actually
makes π0.5 succeed downstream.

---

## The SFT data

Each row of `simple_subgoal_train.jsonl` is one Qwen3-VL chat example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of language subgoals"
    },
    {
      "role": "user",
      "content": "The task goal is: put two red cubes into the bin, then press the button to stop\nThis is the initial turn for prediction\n<image>What's the next language subgoal based on current observation?"
    },
    {
      "role": "assistant",
      "content": "pick up the first red cube"
    }
  ],
  "images": ["/mnt/robomme/.../images/BinFill_ep0_step0.png"]
}
```

### Anatomy of a row

- **system** — fixed prompt defined at `build_vlm_subgoal_dataset_qwenvl.py:23`.
- **user** — task goal + optional history of prior subgoals + `<image>` token that
  Qwen replaces with the vision embedding of `images[0]`.
  - *Initial turn*: `"This is the initial turn for prediction"`.
  - *Subsequent turns*: `"The history of previous predicted language subgoals are: <prior>"`.
- **assistant** — the **label** the model is trained to produce. One short
  imperative ("pick up the first red cube", "put it into the bin", etc.).
- **images** — single PNG. Filename pattern: `{TaskName}_ep{N}_step{M}.png`.

### Where the assistant labels come from — the oracle

The labels are **not human-annotated**. They come from the RoboMME *scripted task
controller* — a hand-written program that knows how to solve each task. As the
controller plays an episode to record a demonstration, it writes its current
intent at each timestep as plain text into the env's `info` dict under the key
`simple_subgoal_online`:

```python
# robomme_policy_learning/.../build_robomme_dataset.py:268
simple_subgoal_online = ts["info"]["simple_subgoal_online"][()].decode()
```

That oracle text stream is what the SFT JSONL builder reads, keyframes, and
emits as the assistant label. So in practice: **the SFT label at any frame is
"what the oracle solver was thinking about at that frame."**

### Row count math

- 16 RoboMME tasks × 100 demonstrations each = **1,600 episodes**.
- Each episode emits ~4 subgoal-transition keyframes → **~6,400–7,000 rows total**.
- The builder also produces `grounded_subgoal_train.jsonl` (same image, label with
  bbox reference). Project currently trains only the simple head.
- Three artifacts must stay in sync: simple JSONL rows == grounded JSONL rows ==
  image file count. `scripts/check_sft_dataset.sh` verifies this *and*
  task coverage (since a build can be perfectly consistent but task-incomplete
  if it times out mid-loop — see `memory/project_sft_dataset_row_counts.md`).

---

## Stage 4: SFT warmstart

### What gets optimized

Standard chat-SFT: **next-token cross-entropy on the assistant tokens only**.

```
loss = -Σ_i log p_model(assistant_token_i | system + user + image + assistant_<i)
```

The system prompt, user prompt, and image tokens are masked out of the loss —
the model isn't trying to predict its inputs.

### The command

`./scripts/train_sft.sh` → `modal run modal_train.py --stage sft` →
`modal_train.py:253` invokes ms-swift:

```bash
swift sft \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --dataset .../simple_subgoal_train.jsonl \
  --train_type lora --lora_rank 16 --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true --freeze_aligner true \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --attn_impl sdpa \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true --deepspeed zero2 \
  --max_length 3200
```

### Why these specific choices

- **`freeze_vit` + `freeze_aligner`**: we're only teaching output distribution.
  Don't waste compute (or risk overfitting) on the vision tower.
- **LoRA rank 16 on `all-linear`**: full FT of a 4B model + Adam state would blow
  the 80 GB VRAM budget. LoRA gives ~95% of the quality at ~5% of the cost.
- **`attn_impl sdpa`**: the base image has no nvcc, so flash-attn won't compile.
  See `memory/feedback_no_flash_attn.md`.
- **`max_length 3200`**: covers image tokens (~256 each, capped by
  `IMAGE_MAX_TOKEN_NUM`) + system + user (with history) + assistant, with headroom
  for late-episode rows where the history string is long.
- **`learning_rate 1e-4`** with 5% warmup: standard LoRA range; LoRA tolerates
  higher LR than full FT because the rank-16 update is small.

### Effective batch size

4 GPUs × 4 per-device × 4 grad-accum = **64 examples per optimizer step**.

### Outputs

Every 100 steps a `checkpoint-{N}/` lands under
`/mnt/robomme/ckpts/qwen_sft/simple_subgoal/`, keeping the last 2. Each
checkpoint contains:

- `adapter_config.json` + `adapter_model.safetensors` — the LoRA weights
  (the only thing GRPO consumes).
- `trainer_state.json` — training metadata, loss curve.

### How you know SFT is succeeding

Watch the swift log's `loss` field — it should drop sharply in the first
few hundred steps then plateau around some task-dependent floor (typically
0.3–0.8 for short-label tasks like these). Per-epoch loss should be lower than
the previous epoch's. No reward, no rollouts — pure supervised learning.

---

## Stage 5: GRPO

### What "the data" is — and isn't

GRPO **does not consume the JSONL.** The "dataset" is just a list of starting
points (`state_dataset.py:19`):

```python
@dataclass
class StateSample:
    task_name: str      # e.g. "PickXtimes"
    episode_id: int     # 0..N within that task
    frame_idx: int = 0  # always 0 today; mid-episode is TODO
```

Built from a flat enumeration: `task × episode_id` (`state_dataset.py:43`).

Defaults in `modal_train.py:310`:

- `only_tasks="PickXtimes"`, `episodes_per_task=20` → **20 starting states** for
  the first smoke-test run.
- Scale-up: 16 tasks × 50 train episodes = **800 starting states** max.

Per training step:

- Sample `batch_states=4` of those.
- For each state, sample `group_size=4` candidate subgoals from the policy.
- Run all 4 × 4 = **16 ManiSkill rollouts**.
- Use the rewards to compute advantages and update the LoRA.

The "training data" is generated **on the fly** by the simulator. The image at
rollout time comes live from `env_runner.get_init_obs()` — never from disk.

### The reward formula

From `reward.py:32`:

```
reward = outcome + shaping

outcome  = 1.0               if success
         = 0.5 × progress    otherwise
         = -1.0              on error

shaping  = 0.1 × difflib_ratio(sampled_subgoal, oracle_subgoal)
```

where `progress` is the rollout worker's coarse self-report (`rollout.py:148`):

```
success → 1.0
timeout → 0.5
fail    → 0.0
error   → 0.0
```

Realized reward per candidate:

| Outcome | outcome | + max shaping | total |
|---|---|---|---|
| success | 1.0 | 0.1 | **1.0–1.1** |
| timeout | 0.25 (0.5 × 0.5) | 0.1 | **0.25–0.35** |
| fail | 0.0 | 0.1 | **0.0–0.1** |
| error | −1.0 | — | **−1.0** |

The shaping is a cheap `difflib.SequenceMatcher` ratio — the file header notes
it's a placeholder ("Could swap for a sentence-embedding similarity if shaping
turns out to matter"). The ×0.1 weight keeps it dominated by the outcome term.

### Important: the oracle shaping is currently inactive

`trainer.py:74` defaults the oracle lookup to a no-op:

```python
self.oracle_subgoal_fn = oracle_subgoal_fn or (lambda _s: None)
```

`modal_train.py`'s GRPO stage doesn't pass one in. So although
`use_oracle_shaping=True` by default, the oracle is always `None` and the
shaping term is silently skipped (`reward.py:63`). **Today's GRPO runs on
outcome reward only.** To activate shaping, wire a callback that reads
`env_runner` info → `simple_subgoal_online` at the rollout start and pass it as
`oracle_subgoal_fn` when constructing `GRPOTrainer`.

### Group-normalized advantage — the "GR" in GRPO

From `trainer.py:213`:

```python
r = np.asarray(group_rewards, dtype=np.float32)
adv = r - r.mean()
if r.std() > 1e-8:
    adv = adv / (r.std() + 1e-8)
```

Within each group of K=4 candidates **for the same starting state**, the
advantage is `(r − mean) / std`. So a candidate is rewarded *relative to its
peers from the same state*, not in absolute terms. Implications:

| Scenario | Effect |
|---|---|
| All 4 succeed | All advantages = 0 → no learning signal that step |
| 1 succeeds, 3 fail | Winner gets large positive Â; losers small negative |
| 1 errors, 3 finish | Error gets strongly negative Â — policy learns to avoid that output |
| Std ≈ 0 (all same reward) | No update — `mean_reward_std` is the diagnostic |

### The loss

From `trainer.py:229`:

```
L_per_candidate = -Â · sum log p_policy(subgoal_tokens) + β · (log p_policy - log p_ref)²
                   └────── policy gradient ──────┘     └────── KL anchor ─────┘
```

- **Policy gradient**: pushes log-prob of high-advantage candidates up,
  low-advantage candidates down.
- **KL anchor**: quadratic surrogate for KL divergence against the *frozen SFT
  reference adapter*. Prevents the policy from drifting too far from what SFT
  already learned. β = `kl_beta` = 0.04 by default.

The backward pass is done **one candidate at a time** (`trainer.py:218`)
because holding 16 Qwen3-VL-4B activation graphs in memory simultaneously
would OOM the A100-80GB. PyTorch accumulates gradients in `param.grad` across
candidates; a single `optimizer.step()` fires after the full minibatch.

### What gets logged

Per step → `runs/grpo/.../train_log.jsonl` (`trainer.py:129`):

| Field | Meaning |
|---|---|
| `mean_reward` | Avg reward across this step's 16 rollouts |
| `mean_reward_std` | Avg within-group std — diagnostic for advantage signal |
| `pg_loss` | Policy-gradient loss |
| `kl_loss` | KL penalty against SFT reference |
| `loss` | Combined: `pg_loss + β · kl_loss` |
| `n_generated_tokens` | Total subgoal tokens this step |
| `mean_per_group_kl` | Per-candidate KL average |
| `wall_seconds` | Wall-clock per step |
| `step` | Step index |

LoRA checkpoint saved every `save_every=25` steps to
`runs/grpo/.../step{N}/`.

### How you know GRPO is succeeding

1. **`mean_reward` trends up over the first 30–50 steps.** This is the
   headline signal. If it stays flat, the handoff says (`HANDOFF.md:147`):
   bump sample temperature 0.9 → 1.0, or shrink `kl_beta` to let the policy
   move further from SFT.

2. **`mean_reward_std` stays > 0 most steps.** If it's ~0, all candidates in
   each group are getting the same reward → no advantage → no gradient. Bump
   temperature for more diversity.

3. **`kl_loss` stays bounded.** Runaway KL = policy drifting too fast.
   Counterintuitive fix: `HANDOFF.md` says drop `kl_beta` 0.04 → 0.01 (the
   quadratic surrogate amplifies drift; a smaller weight actually stabilizes).

4. **Eval pass-rate beats SFT-only baseline.** Train-time reward rising but
   eval rate flat = the policy is overfitting to the `episodes_per_task=20`
   rollout pool. The honest success metric is loading the new adapter into
   `modal_server.py` and running RoboMME eval (stage 6).

### What's missing from current logging (worth adding)

The trainer logs aggregate `mean_reward` but **not**:

- Per-outcome counts (n_success / n_timeout / n_fail / n_error per step) —
  what you actually want to plot.
- Sample of the candidate subgoal strings — to eyeball whether the policy is
  producing reasonable text or degenerating into repetition.
- Eval pass-rate on a held-out subset — without this you can't separate
  training reward from generalization.

---

## SFT vs GRPO at a glance

|  | SFT (stage 4) | GRPO (stage 5) |
|---|---|---|
| Signal | Cross-entropy loss | Task-completion reward |
| Per-step input | (image, prompt, **label**) from JSONL | (image, prompt) only |
| "Label" source | Oracle subgoal from H5 demos | None — model invents candidates |
| Optimizes | P(oracle subgoal \| state) | E[reward(rollout(sampled subgoal))] |
| Failure mode | Imitates oracle even when oracle is suboptimal | Reward hacking — KL-anchored to SFT to prevent |
| Simulator? | No — pure offline | Yes — 16 rollouts per step |
| Logged metric | `train_loss` (lower better) | `mean_reward` (higher better) |
| Wall clock | ~3–6 h on 4× A100-80GB | ~10–30 min/step (sim-bound) |

---

## Quick command reference

```bash
# Stage 3 — build the SFT JSONL (timeout bumped to 4h in modal_train.py)
modal run modal_train.py --stage build_dataset

# Verify the build (consistency + task coverage)
./scripts/check_sft_dataset.sh

# Stage 4 — SFT warmstart
./scripts/train_sft.sh

# Stage 5 — GRPO smoke test on one task
modal run modal_train.py::grpo --only-tasks PickXtimes --num-steps 30 --episodes-per-task 5

# Stage 5 — GRPO full run
modal run modal_train.py --stage grpo

# Stage 6 — eval with new adapter (load into modal_server.py)
modal run modal_server.py --only-tasks PickXtimes
```

---

## Related docs

- `HANDOFF.md` — current pipeline status, Modal volume layout, known gotchas.
- `CLAUDE.md` — project conventions and what NOT to touch.
- `memory/project_sft_dataset_row_counts.md` — why row count alone doesn't
  prove the SFT build is complete.
- `memory/feedback_no_flash_attn.md` — why we use sdpa attention.
