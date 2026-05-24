# vla-memory-new

Hierarchical memory VLA for long-horizon robotic manipulation.

**Current phase:** RoboMME evaluation with frozen π0.5 baseline VLAs on Modal.  
**Next phase:** Qwen VLM fine-tuning (GRPO) as a hierarchical memory module.

## Background

[RoboMME](https://robomme.github.io/) is a benchmark (ICML 2026 Spotlight) for evaluating
memory-augmented Vision-Language-Action (VLA) models on 16 robotic tasks across four cognitive
suites: Counting, Permanence, Reference, and Imitation.

This repo runs the frozen **π0.5 baseline** — π0.5 fine-tuned on RoboMME data with the vision
encoder frozen — as a performance floor before adding the Qwen memory module.

## Setup

### Prerequisites

- Python ≥ 3.10
- [Modal](https://modal.com) account + CLI: `pip install modal && modal setup`
- [git-lfs](https://git-lfs.github.com): `brew install git-lfs` (macOS) or `apt install git-lfs`
- ~4 GB free disk on the Modal volume (checkpoint storage)

### Clone and initialize

```bash
git clone https://github.com/<your-org>/vla-memory-new
cd vla-memory-new
./scripts/setup.sh
```

`setup.sh` initializes the `robomme_policy_learning` submodule and its nested
`third_party/robomme_benchmark` submodule.

### Environment variables

```bash
cp .env.example .env
# Fill in HF_TOKEN (optional — pi05_baseline is a public repo)
# Fill in WANDB_API_KEY if you want training metrics
```

## Running a RoboMME Trial

### Step 1 — Download the checkpoint (one-time, ~4 GB)

```bash
modal run modal_server.py --download-only
```

This clones `Yinpei/pi05_baseline` from HuggingFace into a Modal Volume
(`robomme-vla-data`) that persists across runs.

### Step 2 — Run evaluation

```bash
# All 16 tasks (takes ~1–1.5 h on A100):
modal run modal_server.py

# Smoke-test with one task:
modal run modal_server.py --only-tasks "PickXtimes" --overwrite

# Different seed or checkpoint step:
modal run modal_server.py --seed 42 --ckpt-id 59999
```

On the first run, Modal builds the container image from
`robomme_policy_learning/Dockerfile` — this takes ~15–25 min the first time
(JAX CUDA packages are large). Subsequent runs use the cached image.

### What happens inside Modal

```
A100-40GB container
├── serve_policy.py  (JAX, GPU)  — WebSocket server on a random localhost port
└── eval.py  (ManiSkill, CPU rendering via OSMesa)  — connects to the server
```

Results are written to the Modal Volume at:
```
robomme-vla-data/results/pi05_baseline/ckpt<N>/seed<S>/
```

The `modal run` command prints a JSON summary when evaluation finishes.

## Tasks Reference

| Suite      | Tasks                                                         |
|------------|---------------------------------------------------------------|
| Counting   | BinFill, PickXtimes, SwingXtimes, StopCube                   |
| Permanence | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap  |
| Reference  | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder |
| Imitation  | MoveCube, InsertPeg, PatternLock, RouteStick                  |

## Repository Structure

```
.
├── modal_server.py              # Modal app entry point
├── robomme_policy_learning/     # Submodule: training + eval code
│   ├── src/mme_vla_suite/       # MME-VLA model variants
│   ├── src/openpi/              # JAX policy framework (openpi fork)
│   ├── scripts/serve_policy.py  # Policy WebSocket server
│   └── third_party/
│       └── robomme_benchmark/   # ManiSkill simulator
├── scripts/
│   └── setup.sh
└── pyproject.toml
```

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
