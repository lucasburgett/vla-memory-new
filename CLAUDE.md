# vla-memory-new — CLAUDE.md

## Project Purpose

Class project: hierarchical memory VLA for long-horizon robotic manipulation.

**Current phase:** Run RoboMME evaluations with frozen pi0.5 baseline VLAs on Modal.  
**Next phase:** Implement Qwen VLM fine-tuning with GRPO to produce a hierarchical memory
module that sits above the frozen pi0.5 VLAs.

## Repository Layout

```
.
├── modal_server.py              # Main Modal app (this phase's entry point)
├── robomme_policy_learning/     # Git submodule — RoboMME training + eval code
│   ├── src/mme_vla_suite/       # 14 MME-VLA variants built on π0.5
│   ├── src/openpi/              # OpenPI fork (JAX policy framework)
│   ├── scripts/serve_policy.py  # WebSocket policy server
│   ├── scripts/train.py         # Training entry point
│   ├── examples/robomme/eval.py # RoboMME eval client
│   └── third_party/
│       └── robomme_benchmark/   # Nested submodule — ManiSkill simulator
├── scripts/
│   └── setup.sh                 # One-time setup (run before first modal run)
├── pyproject.toml               # Project deps (torch/transformers for Qwen phase)
└── .env.example                 # Env var template
```

## Key Facts

### RoboMME
- Benchmark: 16 tasks across 4 suites (Counting, Permanence, Reference, Imitation)
- 1,600 demonstrations (100 per task), 50 val + 50 test episodes per task
- Evaluation uses a WebSocket server (policy) + ManiSkill simulator (client)

### pi0.5 Frozen Baseline
- Config name: `pi05_baseline`
- No history (`--args.no-use-history`)
- Checkpoint: `Yinpei/pi05_baseline` on HuggingFace, default step = 79999
- Backbone: Physical Intelligence π0.5 (JAX/Flax, SigLIP vision encoder frozen)
- Policy server: `scripts/serve_policy.py`, WebSocket
- Depends on JAX 0.5.3 + CUDA 12.8 (managed by uv lockfile)

### Modal Setup
- App name: `robomme-pi05-frozen-vla`
- Volume: `robomme-vla-data` (persists across runs; holds ckpts + results)
- GPU: A100-40GB
- Image: built from `robomme_policy_learning/Dockerfile` + libosmesa6
- Both policy server and simulator run in the same container (localhost WebSocket)
- Simulator uses CPU rendering (SAPIEN_RENDER_DEVICE=cpu, MUJOCO_GL=osmesa)

### Run Commands
```bash
# One-time setup
./scripts/setup.sh
modal setup           # authenticate with Modal
modal run modal_server.py --download-only   # download checkpoint (~4 GB)

# Evaluation
modal run modal_server.py                                      # all 16 tasks
modal run modal_server.py --only-tasks "PickXtimes,BinFill"   # subset
modal run modal_server.py --seed 42 --ckpt-id 59999           # custom params
```

## What NOT to Do

- Do not modify files inside `robomme_policy_learning/` — it's a submodule; changes
  won't be tracked here and will cause confusion during rebases.
- Do not add `robomme_policy_learning/data/` or `robomme_policy_learning/runs/` to git.
- Do not commit `.env`.

## Next Phase (Qwen VLM)

The planned hierarchical structure:
```
[Qwen VLM — fine-tuned with GRPO]
        ↓  subgoal / language instruction
[frozen π0.5 VLA — action expert]
        ↓  joint actions
[ManiSkill simulator]
```

The VLM code will live in `src/` with dependencies already in `pyproject.toml`
(`transformers`, `qwen-vl-utils`, `torch`).  Training on Modal using the same
`robomme-vla-data` volume for data access.
