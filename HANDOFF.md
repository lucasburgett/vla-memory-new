# Handoff — Qwen3-VL GRPO Subgoal Predictor

Last updated: 2026-05-27

## What this is

Hierarchical-memory VLA on RoboMME: train a Qwen3-VL-4B subgoal predictor
(LoRA) with SFT → GRPO so it sits on top of a frozen pi0.5 expert.

```
        ┌─────────────────────────────────────────┐
        │  Qwen3-VL-4B + LoRA (we train this)     │   ← src/vla_memory/*
        │  Subgoal predictor / "memory module"    │
        └────────────────┬────────────────────────┘
                         │ subgoal text (+ optional bbox)
                         ▼
        ┌─────────────────────────────────────────┐
        │  pi0.5 baseline, frozen (JAX)           │   ← modal_server.py serves this
        │  scripts/serve_policy.py, WebSocket     │
        └────────────────┬────────────────────────┘
                         │ action chunks
                         ▼
        ┌─────────────────────────────────────────┐
        │  ManiSkill / RoboMME simulator (CPU)    │
        │  16 tasks × 50 val episodes             │
        └─────────────────────────────────────────┘
```

## Where we are in the pipeline

| Stage | Modal entry | Status |
|---|---|---|
| 1. Download pi0.5 ckpt | `modal_server.py --download-only` | ✅ done earlier |
| 2. Download RoboMME H5 demos | `modal_train.py --stage download_data` | ✅ done |
| 3. Build SFT JSONL | `modal_train.py --stage build_dataset` | 🔄 re-running after cv2/h5py image fix |
| 4. SFT warmstart | `./scripts/train_sft.sh` | ⏳ next |
| 5. GRPO fine-tune | `./scripts/train_grpo.sh` or `modal_train.py::grpo` | ⏳ after SFT |
| 6. Eval | `modal_server.py` with new adapter | ⏳ after GRPO |

## File map

```
modal_server.py                              # Frozen pi0.5 eval app (don't touch)
modal_train.py                               # Training app: download_data / build_dataset / sft / grpo
scripts/
  train_sft.sh                               # Wraps `modal run modal_train.py --stage sft`
  train_grpo.sh                              # Wraps `modal run modal_train.py --stage grpo`
src/vla_memory/
  qwen_subgoal/
    prompts.py                               # MUST stay byte-identical to submodule's SFT builder
    model.py                                 # QwenSubgoalPolicy: PEFT + sample + logprob recompute
  grpo/
    trainer.py                               # GRPOTrainer + GRPOConfig (per-candidate backward)
    rollout.py                               # RolloutWorker: subgoal → pi0.5 → ManiSkill → reward
    reward.py                                # Outcome + small subgoal-oracle shaping
    state_dataset.py                         # Enumerates rollout starting states
  data/build_sft_dataset.py                  # Thin wrapper over submodule DatasetBuilder
robomme_policy_learning/                     # Submodule — DO NOT EDIT
```

## Modal volume layout (`robomme-vla-data`)

```
/mnt/robomme/
  ckpts/
    pi05_baseline/pi05_baseline/79999/       # frozen pi0.5 weights
    qwen_sft/simple_subgoal/checkpoint-*     # SFT'd LoRA adapter (populated by stage 4)
  data/
    robomme_data_h5/                         # raw H5 demonstrations from Yinpei/robomme_data_h5
    preprocessed/qwenvl/vlm_subgoal/         # SFT JSONL + images, populated by stage 3
  runs/grpo/                                 # GRPO checkpoints + train_log.jsonl
```

## Commands to run next

After the in-flight `build_dataset` finishes:

```bash
# Confirm the JSONL was actually written
modal volume ls robomme-vla-data /data/preprocessed/qwenvl/vlm_subgoal

# SFT warmstart on 4× A100-80GB (~3-6 hours)
./scripts/train_sft.sh

# Smoke-test GRPO on one task before scaling up
modal run modal_train.py::grpo --only-tasks PickXtimes --num-steps 30 --episodes-per-task 5
```

## Known gotchas (captured in /memory and inline)

1. **No PyTorch flash-attn.** Base image is `nvidia/cuda:12.8.0-cudnn-runtime` —
   no nvcc, can't compile flash-attn. Use `attn_implementation="sdpa"` everywhere.
   The submodule's own `finetune_vlm_subgoal_predictor.sh:29` does the same.
   Saved as `feedback_no_flash_attn.md` in /memory.

2. **Submodule dataset builder needs cv2 + h5py at the system Python level.** They
   live in the submodule's uv venv at `/app/.venv` but Modal's function runtime
   uses the `add_python=3.11` system Python. We pip-install them into our layer.

3. **Path naming is inconsistent in the submodule.** `build_dataset.py` defaults
   to `data/robomme_data_h5` (matches the HF repo name) but
   `vlm_subgoal_dataset_base.py` defaults to `data/robomme_h5_data` (transposed).
   We always pass `raw_data_path` explicitly. JSONL output lands one level deeper
   in a `vlm_subgoal/` subdir because of `BaseVLMSubgoalDatasetBuilder.vlm_dir_name`.

4. **Prompts must match between train and inference.** `src/vla_memory/qwen_subgoal/prompts.py`
   duplicates the submodule's prompt strings. CPU smoke test verifies the exact
   byte-for-byte match. Change both at once or the SFT prior drifts.

5. **PYTHONPATH gotcha in rollout.py.** `from utils import pack_buffer` (not
   `from examples.robomme.utils import ...`) — because modal_train.py inserts
   `/app/examples/robomme` (not `/app`) on sys.path before constructing the worker.

6. **GRPO does per-candidate backward.** `_accumulate_gradients()` in trainer.py
   processes one candidate at a time and calls `.backward()` per candidate to
   release the activation graph. Otherwise 16 retained Qwen3-VL-4B graphs OOM
   the A100-80GB. PyTorch accumulates gradients in `param.grad` across
   candidates; single `optimizer.step()` per minibatch.

## Open issues / TODOs

| Item | Where | Impact | When |
|---|---|---|---|
| Mid-episode rollouts (env.set_state) | `rollout.py`, `state_dataset.py` | Currently we only sample subgoals at t=0 | After first reward curve |
| `num_return_sequences=k` with multimodal | `model.py:sample_subgoals` | Untested on real Modal; may need K parallel forwards instead | First GRPO run will reveal |
| `selected_adapters` save path | `model.py:save_policy_adapter` | PEFT writes the adapter into a subdirectory; verify the load path matches | First checkpoint save |
| sdpa is ~25% slower than flash_attention_2 | everywhere | Adds ~10s/step. Acceptable for now | Revisit once loop produces reward curve. Use prebuilt flash-attn wheel matching torch +cuXXX, do NOT compile from source |
| Reward shaping uses difflib | `reward.py` | Crude; could swap for a sentence-embedding similarity if shaping turns out to matter | If GRPO converges slowly |
| Single-container architecture | `modal_train.py::grpo` | Qwen training + frozen pi0.5 + ManiSkill all on one A100-80GB | Bump to 2-GPU split if VRAM is tight |

## Memory entries

Three project memories already exist for future sessions:
- `project_robomme_setup.md` — overall project shape
- `project_qwen_grpo_layout.md` — this stack's architecture + design choices
- `feedback_no_flash_attn.md` — don't reintroduce flash-attn

## If something breaks

- Modal image build fails: re-read `feedback_no_flash_attn.md` first. If a new
  ModuleNotFoundError, check `robomme_policy_learning/uv.lock` for the missing
  package and add to `modal_train.py`'s `pip_install` list.
- `from mme_vla_suite...` fails: PYTHONPATH must include `/app/src`. The image
  env sets this; if it ever changes, GRPO and build_dataset both break.
- GRPO loss explodes: drop `kl_beta` from 0.04 to 0.01 or 0.005. SFT prior is
  strong, so smaller KL is fine.
- Reward stays flat for 30+ steps: bump sample temperature from 0.9 → 1.0, or
  shrink `kl_beta` to let the policy move further from SFT.
