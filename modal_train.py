"""Modal app: Qwen3-VL subgoal predictor training (SFT warmstart + GRPO).

Pipeline (each step is a separate Modal function — runnable independently):

  modal run modal_train.py::build_dataset   # HDF5 → QwenVL JSONL on the volume
  modal run modal_train.py::sft_warmstart   # ms-swift SFT → LoRA adapter
  modal run modal_train.py::grpo            # GRPO fine-tune on top of SFT

State lives on the shared ``robomme-vla-data`` volume (same one used by
``modal_server.py``). The pi0.5 frozen checkpoint already lives there too,
so GRPO can boot the policy server in-container without re-downloading.

Layout under the volume:
  /mnt/robomme/ckpts/pi05_baseline/...        ← from modal_server.py
  /mnt/robomme/data/robomme_h5_data/          ← RoboMME demonstrations
  /mnt/robomme/data/preprocessed/qwenvl/      ← SFT JSONL (built here)
  /mnt/robomme/ckpts/qwen_sft/                ← SFT LoRA adapter
  /mnt/robomme/runs/grpo/                     ← GRPO checkpoints + logs
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App + shared resources
# ---------------------------------------------------------------------------

app = modal.App("vla-memory-qwen-train")

volume = modal.Volume.from_name("robomme-vla-data", create_if_missing=True)
MOUNT = "/mnt/robomme"

# ---------------------------------------------------------------------------
# Image
#
# We reuse the submodule's Dockerfile (same one as modal_server.py) — it ships
# ManiSkill + JAX + uv venv + the robomme micromamba env. On top of that we add
# the Qwen/PEFT/swift stack used for SFT + GRPO. Keeping a single image avoids
# the cost of two container builds when GRPO needs both stacks at once.
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_dockerfile(
        "robomme_policy_learning/Dockerfile",
        context_dir="robomme_policy_learning",
        ignore=["data/", "runs/", ".git/"],
        add_python="3.11",
    )
    .apt_install(["libosmesa6", "libosmesa6-dev"])
    .pip_install(
        "wandb>=0.19",
        # Don't bump above ~4.57: ms-swift 3.x's setup.py declares
        # `transformers<4.58`. A `>=5.0` pin makes pip's resolver explode
        # (ResolutionImpossible) trying every ms-swift 3.x release in turn.
        # Qwen3VLForConditionalGeneration is present across this range.
        "transformers>=4.49",
        "peft>=0.13",
        "accelerate>=1.2",
        "torch>=2.4",
        # transformers 5.x's AutoVideoProcessor hard-requires torchvision;
        # Qwen3-VL's processor chain triggers it even for image-only training.
        # Without this we got `ImportError: AutoVideoProcessor requires the
        # Torchvision library` at swift's _prepare_model_tokenizer step.
        "torchvision>=0.19",
        # `decord` is the video-decode backend swift prefers for video inputs.
        # We're image-only today, but installing it silences the
        # `[WARNING:swift] Please install the package: pip install "decord"`
        # line on every run and future-proofs for video-conditioned subgoals.
        "decord>=0.6",
        # ms-swift 4.x dropped/renamed `--train_type lora` (and other flags
        # upstream's recipe uses). Pin to 3.x so the upstream
        # finetune_vlm_subgoal_predictor.sh:24 flag set still parses.
        # Without this we got: `ValueError: remaining_argv: ['--train_type', 'lora']`.
        "ms-swift>=3.0,<4.0",
        "qwen-vl-utils>=0.0.8",
        "Pillow>=10",
        "imageio>=2.34",
        "imageio-ffmpeg>=0.5",
        # Needed by the submodule's dataset_builder (build_vlm_subgoal_dataset_qwenvl.py
        # imports cv2 + h5py). They live in robomme_policy_learning's uv venv at
        # /app/.venv but Modal's function runtime uses the system Python that
        # add_python=3.11 injects, so we install them here too. opencv-python-headless
        # avoids pulling X11 / GUI deps that would just sit unused in a headless
        # container.
        "opencv-python-headless>=4.10",
        "h5py>=3.10",
        # NOTE: NOT installing deepspeed. Upstream's recipe uses
        # `--deepspeed zero2` but deepspeed 0.19's
        # `fp_quantizer.is_compatible()` shells out to nvcc at *import* time
        # (deepspeed/ops/op_builder/builder.py:53) with no fallback. Our base
        # image is cudnn-RUNTIME, which has no nvcc. For LoRA on 4× A100-80GB
        # ZeRO-2 is a no-op anyway (LoRA Adam state is ~400MB; fits per GPU
        # trivially), so we drop the flag and use plain DDP. To re-enable
        # later: switch base image to cudnn-devel or apt-install cuda-nvcc,
        # then add `"deepspeed>=0.14"` here and `--deepspeed zero2` to the
        # swift cmd.
    )
    # GRPO main loop runs in the submodule's micromamba `robomme` env
    # (sapien/mani_skill/torch all consistent there). Modal Python is just
    # the launcher. We add the Qwen-training deps the micromamba env doesn't
    # already pin in examples/robomme/requirements.txt.
    .run_commands(
        "micromamba run -n robomme pip install "
        "'peft>=0.13' 'accelerate>=1.2' 'wandb>=0.19'"
    )
    # Note: flash-attn is NOT installed here. The base image
    # ``nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`` ships CUDA runtime libs
    # but not the nvcc toolchain, so building flash-attn from source fails
    # ("No such file or directory: '/usr/local/cuda/bin/nvcc'"). The submodule's
    # own SFT recipe (finetune_vlm_subgoal_predictor.sh:29) uses --attn_impl sdpa
    # too, so we follow suit. Slower than flash_attention_2 by ~20-30% but
    # numerically identical and works on any GPU without a toolchain.
    # Pin the project's src/ into the container so we can import vla_memory.*
    # Ignore both `.git` dirs: skipping them avoids copying ~287MB of git
    # history into the image, AND prevents "FETCH_HEAD was modified during
    # build" errors when an IDE git plugin auto-fetches mid-copy.
    .add_local_dir(
        ".",
        "/workspace",
        copy=True,
        ignore=[".git", "robomme_policy_learning/.git", "__pycache__"],
    )
    .env(
        {
            "OPENPI_DATA_HOME": f"{MOUNT}/openpi",
            "PYTHONPATH": "/workspace/src:/app/src",
            "SAPIEN_RENDER_DEVICE": "cpu",
            "MUJOCO_GL": "osmesa",
            "DISPLAY": "",
            "XDG_RUNTIME_DIR": "/tmp/runtime-root",
            "IMAGE_MAX_TOKEN_NUM": "256",
            "VIDEO_MAX_TOKEN_NUM": "64",
            "FPS_MAX_FRAMES": "10",
        }
    )
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# Stage A.0 — Download the RoboMME demonstration HDF5s
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=8.0,
    memory=16 * 1024,
    timeout=4 * 3600,           # ~30+ GB of LFS objects + .tar.xz decompression
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def download_demonstrations(
    out_dir: str = f"{MOUNT}/data/robomme_data_h5",
    jobs: int = 16,
) -> str:
    """Clone Yinpei/robomme_data_h5 from HuggingFace and decompress the .tar.xz shards.

    Safe to call multiple times — skips both the clone and the decompression
    when the completion marker is present. The marker is written only after
    decompression finishes successfully.
    """
    import shutil

    out_path = Path(out_dir)
    done_marker = out_path / ".download_complete"
    if done_marker.exists():
        print(f"Demonstrations already present at {out_path}")
        return str(out_path)

    # Clean any partial state from an interrupted previous attempt — git clone
    # refuses a non-empty destination and partial .tar.xz files would confuse
    # the decompressor.
    if out_path.exists():
        print(f"Removing partial download at {out_path} …")
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", "")

    print("Cloning Yinpei/robomme_data_h5 (LFS skip) …")
    subprocess.run(
        [
            "git", "clone",
            "https://huggingface.co/datasets/Yinpei/robomme_data_h5",
            str(out_path),
        ],
        check=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
    )

    lfs_env = {**os.environ}
    if hf_token:
        lfs_env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("Pulling LFS objects (~30+ GB of .tar.xz shards) …")
    subprocess.run(["git", "lfs", "pull"], check=True, cwd=str(out_path), env=lfs_env)

    print("Decompressing .h5.tar.xz shards in place …")
    subprocess.run(
        [
            "uv", "run", "scripts/tarxz_h5.py", "decompress",
            "--input_dir", str(out_path),
            "--jobs", str(jobs),
            "--remove_archive",
        ],
        check=True,
        cwd="/app",
    )

    done_marker.touch()
    volume.commit()
    print(f"Demonstrations ready at {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Stage A.1 — Build the SFT dataset
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=8.0,
    memory=32 * 1024,
    timeout=4 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def build_dataset(
    raw_data_path: str = f"{MOUNT}/data/robomme_data_h5",
    out_dir: str = f"{MOUNT}/data/preprocessed/qwenvl",
    max_episodes: int = 0,
    visualize: bool = False,
) -> dict:
    """Run the QwenVL SFT dataset builder against the demonstration HDF5s."""
    from vla_memory.data.build_sft_dataset import build_sft_dataset

    paths = build_sft_dataset(
        raw_data_path=raw_data_path,
        preprocessed_data_path=out_dir,
        max_episodes=max_episodes or None,
        visualize=visualize,
    )
    volume.commit()
    return paths


# ---------------------------------------------------------------------------
# Stage A.2 — SFT warmstart
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB:4",
    timeout=12 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def sft_warmstart(
    dataset_path: str = f"{MOUNT}/data/preprocessed/qwenvl/vlm_subgoal/simple_subgoal_train.jsonl",
    output_dir: str = f"{MOUNT}/ckpts/qwen_sft/simple_subgoal",
    # Hyperparameter rationale (2026-05-28 research synthesis):
    # - Our 6343-row narrow-vocab dataset hit loss 0.25 at step 80 of ~209
    #   with the upstream-recipe values, clearly overfitting.
    # - Unsloth's "loss < 0.2 = overfitting" threshold + ms-swift's official
    #   Qwen3-VL recipe both point at LOWER capacity than upstream defaults.
    # - Coordinated reduction: rank 16→8, alpha 32→16 (preserves 2× ratio),
    #   lr 1e-4→5e-5, plus dropout 0.1 + label_smoothing 0.1 in the cmd.
    # - 1 epoch (~104 steps with effective batch 64) is plenty for narrow
    #   imperative labels — official ms-swift Qwen3-VL recipe also uses 1.
    num_train_epochs: int = 1,
    per_device_batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 4e-5,
    lora_rank: int = 8,
    lora_alpha: int = 16,
) -> dict:
    """Run ms-swift SFT to produce the LoRA prior for GRPO.

    Mirrors ``robomme_policy_learning/scripts/finetune_vlm_subgoal_predictor.sh``
    — keep the hyperparameters aligned with the upstream recipe so we can reuse
    their reported numbers as a sanity check.
    """
    run_name = f"qwen-sft-simple-{int(time.time())}"
    # Cache the 8GB Qwen weights on the persistent volume so subsequent runs
    # don't re-download. Ephemeral container storage flushes between runs;
    # the volume mount survives. First run: ~5 min download from HF. Later
    # runs: instant.
    hf_cache_dir = f"{MOUNT}/.cache/huggingface"
    Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "IMAGE_MAX_TOKEN_NUM": "256",
        "VIDEO_MAX_TOKEN_NUM": "64",
        "FPS_MAX_FRAMES": "10",
        "NPROC_PER_NODE": "4",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        # WANDB_API_KEY arrives via the .env Modal secret; project + run name
        # are set explicitly so the run lands in a predictable place.
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "vla-memory-sft"),
        "WANDB_RUN_NAME": run_name,
        # Force HuggingFace instead of ModelScope. ModelScope (Chinese hub)
        # is heavily throttled from Modal's US infra — we saw downloads drop
        # from 40 MB/s to 1 MB/s within the same session. HF is consistently
        # fast (50-100 MB/s) from US datacenters.
        "USE_HF": "1",
        # Persist HF model cache on the Modal volume.
        "HF_HOME": hf_cache_dir,
        "HF_HUB_CACHE": f"{hf_cache_dir}/hub",
        "TRANSFORMERS_CACHE": f"{hf_cache_dir}/hub",
    }

    cmd = [
        "swift",
        "sft",
        "--model", "Qwen/Qwen3-VL-4B-Instruct",
        # Explicit hub override — matches USE_HF=1 in the env above. Without
        # this swift's args parser may still try ModelScope first.
        "--use_hf", "true",
        "--dataset", dataset_path,
        # 5% held-out validation. Lets wandb show eval_loss alongside
        # train_loss so we can detect overfitting *during* training
        # instead of guessing from train-loss shape alone.
        "--split_dataset_ratio", "0.05",
        "--load_from_cache_file", "true",
        "--packing", "false",
        "--train_type", "lora",
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", str(num_train_epochs),
        "--per_device_train_batch_size", str(per_device_batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--attn_impl", "sdpa",
        "--padding_free", "false",
        "--learning_rate", str(learning_rate),
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        # 0.05 (default) → 0.1 — Unsloth's small-dataset recommendation.
        # Helps LoRA layers generalize on 6.3k rows of repetitive labels.
        "--lora_dropout", "0.1",
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--gradient_checkpointing", "true",
        "--vit_gradient_checkpointing", "false",
        # save_steps lowered from 100 → 25. The previous run was killed
        # at ~step 90 (mid-overfitting) and produced ZERO checkpoints
        # because save_steps=100 never fired. With 25, the latest ~4
        # checkpoints are always within killing distance.
        "--save_steps", "25",
        "--save_total_limit", "4",
        # Also evaluate every 25 steps so eval_loss is plotted at
        # checkpoint granularity.
        "--eval_steps", "25",
        "--eval_strategy", "steps",
        "--logging_steps", "10",
        "--max_length", "3200",
        "--output_dir", output_dir,
        "--warmup_ratio", "0.05",
        # Label smoothing 0.1: distributes 10% of probability mass uniformly
        # across the vocab instead of putting 100% on the target token.
        # On a narrow-vocab task this prevents the model from collapsing to
        # overconfident memorization of the ~10-50 distinct subgoal phrases.
        "--label_smoothing_factor", "0.1",
        # NOTE: `--deepspeed zero2` removed — see image build comment above.
        # torchrun + DDP is the fallback and is sufficient for LoRA.
        "--dataset_num_proc", "4",
        "--dataloader_num_workers", "4",
        "--report_to", "wandb",
        "--run_name", run_name,
    ]

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env, cwd="/workspace")
    finally:
        # Commit the volume even on partial failure so any checkpoints written
        # mid-training are persisted. Without this, a crash at hour 3 of a
        # 4-hour run could lose every save_steps interval.
        volume.commit()

    # Swift transforms `--output_dir X` into `X/v{N}-{timestamp}/` internally
    # and writes checkpoints under that subdirectory, NOT under X directly.
    # Pick the most recent v* run, then the largest checkpoint within it.
    versioned_runs = sorted(Path(output_dir).glob("v*-*"))
    if not versioned_runs:
        raise RuntimeError(
            f"swift produced no versioned run directory under {output_dir} — "
            "training likely failed before saving any checkpoint"
        )
    latest_run = versioned_runs[-1]
    checkpoints = list(latest_run.glob("checkpoint-*"))
    if not checkpoints:
        raise RuntimeError(
            f"swift produced no checkpoint under {latest_run} — "
            f"training may have failed before save_steps=100 was reached"
        )
    latest = max(checkpoints, key=lambda p: int(p.name.split("-", 1)[1]))
    volume.commit()
    return {"output_dir": output_dir, "latest_checkpoint": str(latest)}


# ---------------------------------------------------------------------------
# Stage B — GRPO
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def grpo(
    # Pinned to the 2026-05-28 best-eval-loss checkpoint from the v4 SFT
    # run. The resolver below recognizes paths ending in `checkpoint-*` and
    # uses them as-is; to fall back to "latest in latest v*" auto-resolution,
    # pass `--sft-adapter-path {MOUNT}/ckpts/qwen_sft/simple_subgoal` instead.
    # To pin a different checkpoint, override at the CLI:
    #   modal run modal_train.py::grpo --sft-adapter-path /mnt/.../checkpoint-N
    sft_adapter_path: str = f"{MOUNT}/ckpts/qwen_sft/simple_subgoal/v4-20260528-025027/checkpoint-400",
    pi05_ckpt_dir: str = f"{MOUNT}/ckpts/pi05_baseline/pi05_baseline/79999",
    output_dir: str = f"{MOUNT}/runs/grpo/simple_subgoal_v1",
    num_steps: int = 200,
    batch_states: int = 4,
    group_size: int = 8,
    kl_beta: float = 0.0,             # KL off (DAPO/TRL default); reference only used if >0
    learning_rate: float = 1e-4,      # ~10x v0; LoRA RL wants a higher lr than SFT
    sample_temperature: float = 1.0,
    rollouts_per_subgoal: int = 1,    # >1 averages reward to cut pi0.5 flow-sampling noise
    rollout_max_steps: int = 200,
    only_tasks: str = "PickXtimes",   # restrict early experiments to one task
    episodes_per_task: int = 20,
    subgoal_type: str = "simple_subgoal",
    seed: int = 0,
) -> dict:
    """Launch the GRPO loop in the submodule's micromamba env.

    Architecture (matches modal_server.py:307-329 for the simulator side):
      - This Modal function (Modal Python) is a thin launcher.
      - π0.5 policy server runs in /app/.venv via `uv run`.
      - GRPO main loop (Qwen + simulator) runs in the micromamba `robomme`
        env via `micromamba run -n robomme python main.py`, where sapien
        and mani_skill see the torch/sapien versions they were built against.

    Putting Qwen + the simulator into one Modal-Python process segfaults
    during sapien init (torch ABI mismatch + CUDA-context interactions),
    so we keep them strictly separated.
    """
    # ------------------------------------------------------------------
    # 1. Boot the frozen π0.5 policy server in a background process.
    # ------------------------------------------------------------------
    port = _free_port()
    server_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "OPENPI_DATA_HOME": f"{MOUNT}/openpi",
        # Don't let JAX preallocate ~60GB on the 80GB A100 — the GRPO
        # subprocess needs ~15GB for Qwen + activations and a few more for
        # sapien/mani_skill. JAX allocates on demand instead; π0.5 only
        # actually uses ~3GB in bf16.
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    print(f"Starting frozen π0.5 server on localhost:{port} …")
    server_proc = subprocess.Popen(
        [
            "uv", "run", "scripts/serve_policy.py",
            f"--port={port}",
            f"--seed={seed}",
            "policy:checkpoint",
            f"--policy.dir={pi05_ckpt_dir}",
            "--policy.config=pi05_baseline",
        ],
        cwd="/app",
        env=server_env,
    )
    # Wait for JAX compilation (~60–90 s on A100). We poll a TCP connect to be
    # more reliable than a fixed sleep.
    for _ in range(180):
        if server_proc.poll() is not None:
            raise RuntimeError("π0.5 policy server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                print("π0.5 server is up.")
                break
        except OSError:
            time.sleep(1.0)
    else:
        raise TimeoutError("π0.5 policy server did not become reachable")

    # ------------------------------------------------------------------
    # 2. Launch the GRPO main loop inside the micromamba `robomme` env.
    #    The script handles SFT adapter resolution, policy build, dataset
    #    construction, and the GRPO loop. We just pipe args through.
    # ------------------------------------------------------------------
    hf_cache_dir = f"{MOUNT}/.cache/huggingface"
    Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)
    grpo_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Same HF cache strategy as SFT — persists Qwen weights across runs.
        "USE_HF": "1",
        "HF_HOME": hf_cache_dir,
        "HF_HUB_CACHE": f"{hf_cache_dir}/hub",
        "TRANSFORMERS_CACHE": f"{hf_cache_dir}/hub",
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "vla-memory-grpo"),
        "WANDB_RUN_NAME": f"qwen-grpo-{int(time.time())}",
    }
    grpo_cmd = [
        "micromamba", "run", "-n", "robomme",
        "python", "-u", "/workspace/src/vla_memory/grpo/main.py",
        f"--port={port}",
        f"--sft-adapter-path={sft_adapter_path}",
        f"--output-dir={output_dir}",
        f"--num-steps={num_steps}",
        f"--batch-states={batch_states}",
        f"--group-size={group_size}",
        f"--kl-beta={kl_beta}",
        f"--learning-rate={learning_rate}",
        f"--sample-temperature={sample_temperature}",
        f"--rollouts-per-subgoal={rollouts_per_subgoal}",
        f"--rollout-max-steps={rollout_max_steps}",
        f"--only-tasks={only_tasks}",
        f"--episodes-per-task={episodes_per_task}",
        f"--subgoal-type={subgoal_type}",
        f"--seed={seed}",
    ]
    import shlex
    print("Launching GRPO main loop:", shlex.join(grpo_cmd), flush=True)
    try:
        subprocess.run(grpo_cmd, check=True, env=grpo_env, cwd="/workspace")
    finally:
        # Commit volume even on failure so any checkpoints written mid-run
        # are persisted. Guard the commit so a Modal-side hiccup can't
        # orphan the π0.5 server process — server termination MUST run.
        try:
            volume.commit()
        except Exception as commit_exc:
            print(f"[grpo] volume.commit() failed: {commit_exc!r}", flush=True)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    return {"output_dir": output_dir, "log_path": str(Path(output_dir) / "train_log.jsonl")}


# ---------------------------------------------------------------------------
# Local entrypoint convenience
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(stage: str = "grpo"):
    """Run a single stage end-to-end. Example::

        modal run modal_train.py --stage download_data    # clone + decompress H5 demos (~30 GB)
        modal run modal_train.py --stage build_dataset    # H5 -> QwenVL JSONL
        modal run modal_train.py --stage sft              # SFT warmstart
        modal run modal_train.py --stage grpo             # GRPO fine-tune
    """
    if stage == "download_data":
        print(download_demonstrations.remote())
    elif stage == "build_dataset":
        print(build_dataset.remote())
    elif stage == "sft":
        print(sft_warmstart.remote())
    elif stage == "grpo":
        print(grpo.remote())
    else:
        raise SystemExit(f"unknown stage: {stage}")
