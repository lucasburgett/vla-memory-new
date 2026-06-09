# Memory-Augmented VLM Planners for Long-Horizon VLA Control via RL

**CS 224R Final Project — Stanford University**
*Krish Sharma, Lucas Burgett — Advisor: Marcel Torne*

Hierarchical memory VLA for long-horizon robotic manipulation. A Qwen3-VL-4B planner with a persistent keyframe buffer, trained via streaming GRPO on task-completion reward from simulation (zero human demonstrations), drives a frozen GroundSG π₀.₅ action expert on [RoboMME](https://robomme.github.io/).

**Result:** Streaming GRPO achieves **31.4% ± 3.1% success rate** on ButtonUnmaskSwap, exceeding MemER-IL (21.3%, ~50 human demos/task) by +10.1 pp with zero demonstrations.

## Architecture

```
Qwen3-VL-4B + LoRA  (trained with streaming GRPO)
    ↓  grounded subgoal  <x,y> 0–1000 → <y,x> 0–256
GroundSG π₀.₅  (frozen)
    ↓  action chunks
ManiSkill / RoboMME simulator
```

## Setup

### Prerequisites
- Python ≥ 3.10
- [Modal](https://modal.com) account + CLI: `pip install modal && modal setup`
- [git-lfs](https://git-lfs.github.com): `brew install git-lfs` (macOS)

### Clone and initialize
```bash
git clone https://github.com/lucasburgett/vla-memory-new
cd vla-memory-new
./scripts/setup.sh
cp .env.example .env  # add HF_TOKEN and WANDB_API_KEY
```

## Training Pipeline

All stages run on Modal A100-80GB GPUs. State persists on the `robomme-vla-data` volume.

```bash
# 1. Download data and checkpoints (one-time)
modal run modal_train.py --stage download_data        # ~30 GB RoboMME demos
modal run modal_train.py --stage download_groundsg    # GroundSG π₀.₅ checkpoint

# 2. Build coordinate-aligned SFT dataset (Permanence suite)
modal run modal_train.py::build_memory_dataset \
  --only-tasks "ButtonUnmask,ButtonUnmaskSwap,VideoUnmask,VideoUnmaskSwap"

# 3. SFT warm-start (5 epochs, ~2 hrs on A100)
modal run modal_train.py --stage sft

# 4. Streaming GRPO (our primary contribution)
modal run --detach modal_train.py::grpo \
  --streaming-memory --only-tasks ButtonUnmaskSwap \
  --num-steps 200 --group-size 8

# 5. Evaluate (faithful streaming evaluator)
modal run --detach modal_train.py::eval_streaming \
  --adapter-path /mnt/robomme/runs/grpo/... \
  --task ButtonUnmaskSwap --episodes 50
```

### Alternative RL algorithms (Krish's contribution)
```bash
modal run --detach modal_train.py::ppo   # bandit-PPO with value head
modal run --detach modal_train.py::rloo  # REINFORCE leave-one-out
```

### Buffer design ablations
```bash
# streaming_buffer_reset: clear buffer before each pick
modal run --detach modal_train.py::grpo --streaming-memory --streaming-buffer-reset ...

# streaming_no_fifo: keyframe buffer only, no sliding window
modal run --detach modal_train.py::grpo --streaming-memory --streaming-no-fifo ...
```

## Key Results (ButtonUnmaskSwap, streaming eval, 5 seeds)

| Method | Success Rate | Demos |
|--------|-------------|-------|
| Frozen π₀.₅ (no memory) | 6.7% | 0 |
| MemER-IL | 21.3% | ~50/task |
| SFT only (ours) | 26.2% | 0 |
| **GRPO streaming (ours)** | **31.4% ± 3.1%** | **0** |
| Buffer-reset ablation | 9.5% ± 2.8% | 0 |
| No-FIFO ablation | 30.7% ± 2.9% | 0 |

## Repository Structure

```
modal_server.py              # Frozen π₀.₅ baseline eval on all 16 RoboMME tasks
modal_train.py               # All training stages: build_memory, sft, grpo, ppo, rloo, eval_streaming
scripts/
  setup.sh                   # One-time init (submodules, prerequisites)
  train_ppo.sh / train_rloo.sh
src/vla_memory/
  grpo/                      # Streaming GRPO trainer + rollout (Lucas)
    trainer.py               # GRPOTrainer with streaming_buffer_reset / streaming_no_fifo flags
    rollout.py               # rollout_streaming generator + KeyframeBuffer
    keyframe_buffer.py       # MemER-style temporal clustering
    eval_streaming.py        # Faithful streaming evaluator
  ppo/                       # Bandit-PPO with value head (Krish)
  rloo/                      # REINFORCE leave-one-out (Krish)
  qwen_subgoal/
    model.py                 # QwenSubgoalPolicy + ValueHead
    coords.py                # to_qwen_xy / from_qwen_xy coordinate alignment
    prompts.py               # MemER prompt format
  data/
    build_memory_sft_dataset.py  # Coord-aligned SFT data builder
robomme_policy_learning/     # Submodule: RoboMME training + eval + ManiSkill simulator
figures/
  reward_curves.png          # Training curves for buffer ablation
report.md                    # Final paper (CS 224R)
poster.tex                   # Conference poster
```

## Critical Engineering Note: Coordinate Space Alignment

Qwen3-VL grounds in `<x,y>` 0–1000 (normalized). ManiSkill oracle subgoals use `<y,x>` 0–256 (pixels). Without alignment, SFT converges to a constant coordinate (~`<101,101>`) and all group-based RL (GRPO, RLOO) produces zero gradient. See `src/vla_memory/qwen_subgoal/coords.py`.

## Citation

```bibtex
@article{dai2026robomme,
  title={RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies},
  author={Dai, Yinpei and Fu, Hongze and Lee, Jayjun and Liu, Yuejiang and Zhang, Haoran
          and Yang, Jianing and Finn, Chelsea and Fazeli, Nima and Chai, Joyce},
  journal={arXiv preprint arXiv:2603.04639},
  year={2026}
}
```
