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
# Stage A.0b — Download the GroundSG (subgoal-conditioned) π0.5 checkpoint
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=8.0,
    memory=16 * 1024,
    timeout=2 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def download_groundsg(
    out_dir: str = f"{MOUNT}/ckpts/mme_vla_suite",
    ckpt_step: int = 79999,
) -> str:
    """Download the GroundSG (grounded-subgoal) π0.5 from ``Yinpei/mme_vla_suite``.

    This is the SUBGOAL-CONDITIONED low-level policy the memory hierarchy needs;
    ``pi05_baseline`` discards the subgoal (``project_pi05_baseline_ignores_subgoal``).
    We fetch only the ``symbolic-grounded-subgoal/`` variant (the checkpoint zip +
    its ``history_config.txt``) and unzip so ``create_trained_policy`` finds
    ``<ckpt>/params``, ``<ckpt>/assets`` and ``<parent>/history_config.txt``.

    Idempotent — skips when the completion marker is present.
    """
    import zipfile

    from huggingface_hub import snapshot_download

    variant = "symbolic-grounded-subgoal"
    out_path = Path(out_dir)
    ckpt_dir = out_path / variant / str(ckpt_step)
    done_marker = ckpt_dir / ".download_complete"
    if done_marker.exists():
        print(f"GroundSG checkpoint already present at {ckpt_dir}")
        return str(ckpt_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    print(f"Downloading {variant}/* from Yinpei/mme_vla_suite …")
    snapshot_download(
        repo_id="Yinpei/mme_vla_suite",
        allow_patterns=[f"{variant}/*"],
        local_dir=str(out_path),
        token=token,
    )

    zip_path = out_path / variant / f"{ckpt_step}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Expected {zip_path} after download; found "
            f"{list((out_path / variant).glob('*'))}"
        )
    print(f"Unzipping {zip_path} → {ckpt_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_path / variant)

    # The zip may extract straight to <variant>/<step>/ or nest one level deeper;
    # normalize so checkpoint_dir/params exists where create_trained_policy looks.
    if not (ckpt_dir / "params").exists():
        found = next(iter((out_path / variant).rglob("params")), None)
        if found is not None and found.parent != ckpt_dir:
            print(f"Normalizing checkpoint layout: {found.parent} → {ckpt_dir}")
            found.parent.rename(ckpt_dir)
    if not (ckpt_dir / "params").exists():
        raise FileNotFoundError(
            f"No params/ under {ckpt_dir} after unzip — inspect "
            f"{[str(p) for p in (out_path / variant).rglob('*')][:20]}"
        )

    hist = out_path / variant / "history_config.txt"
    print(f"history_config.txt present: {hist.exists()} ({hist})")
    if not hist.exists():
        print(
            "[warn] history_config.txt missing — create_trained_policy will NOT "
            "activate grounded-subgoal memory without it (subgoal would be ignored)."
        )
    done_marker.touch()
    volume.commit()
    print(f"GroundSG ready at {ckpt_dir}")
    return str(ckpt_dir)


# ---------------------------------------------------------------------------
# Stage A.0c — Causality probe (BLOCKER check: does π0.5 read the subgoal?)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=2 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def causality_probe(
    low_level_ckpt_dir: str = f"{MOUNT}/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999",
    policy_config: str = "mme_vla_suite",
    task: str = "ButtonUnmask",
    episodes: int = 6,        # 3 rollout conditions/episode now (oracle + correct + wrong)
    seed: int = 0,
    rollout_max_steps: int = 400,   # generous so the oracle has room to finish the pick
) -> dict:
    """Boot the GroundSG server and probe whether the subgoal changes π0.5's actions.

    The cheapest, highest-value check in the whole pipeline. Compares the action
    chunk at one fixed post-occlusion state under correct vs colour-swapped vs
    repeated-correct subgoals (see ``causality_probe.py``). Exit 0 / ``causal:True``
    means the hierarchy is viable; nonzero means the subgoal is being ignored and
    you must NOT proceed to SFT/GRPO.
    """
    port = _free_port()
    server_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "OPENPI_DATA_HOME": f"{MOUNT}/openpi",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    print(f"Starting GroundSG π0.5 server on localhost:{port} …")
    server_proc = subprocess.Popen(
        [
            "uv", "run", "scripts/serve_policy.py",
            f"--port={port}",
            f"--seed={seed}",
            "policy:checkpoint",
            f"--policy.dir={low_level_ckpt_dir}",
            f"--policy.config={policy_config}",
        ],
        cwd="/app",
        env=server_env,
    )
    for _ in range(180):
        if server_proc.poll() is not None:
            raise RuntimeError("GroundSG policy server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                print("GroundSG server is up.")
                break
        except OSError:
            time.sleep(1.0)
    else:
        raise TimeoutError("GroundSG policy server did not become reachable")

    probe_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SAPIEN_RENDER_DEVICE": "cpu",
        "MUJOCO_GL": "osmesa",
    }
    cmd = [
        "micromamba", "run", "-n", "robomme",
        "python", "-u", "/workspace/src/vla_memory/grpo/causality_probe.py",
        f"--port={port}",
        f"--task={task}",
        f"--episodes={episodes}",
        f"--seed={seed}",
        f"--rollout-max-steps={rollout_max_steps}",
    ]
    import shlex
    print("Launching causality probe:", shlex.join(cmd), flush=True)
    rc = 1
    try:
        proc = subprocess.run(cmd, env=probe_env, cwd="/workspace")
        rc = proc.returncode
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
    return {"task": task, "episodes": episodes, "probe_exit_code": rc, "causal": rc == 0}


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
# Stage A.1b — Build the MEMORY SFT dataset (MemER-style, reveal keyframes)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=8.0,
    memory=32 * 1024,
    timeout=4 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def build_memory_dataset(
    raw_data_path: str = f"{MOUNT}/data/robomme_data_h5",
    out_dir: str = f"{MOUNT}/data/preprocessed",
    # Multi-task Permanence (NON-Swap) warm-up. Single-task narrow data forced
    # lr≤4e-6 (collapse above ~8e-6) which UNDERFITS the coordinate
    # (project_qwen_coordinate_space RESULT: constant ~<315,305>). Adding VideoUnmask
    # lifts the collapse threshold so the SFT can run a higher lr and actually fit
    # the grounding. VideoUnmask is builder-compatible: phase0 is "static" (vs
    # "press the button") but the decision-point + "pick up the container at <> that
    # hides the {color} cube" target + reveal_window=64 are identical. Swap variants
    # are EXCLUDED — the cube moves mid-episode, so reveal-window keyframes go stale
    # (those belong to the later keyframe-selection work; see JOINT_MEMORY_DESIGN).
    only_tasks: str = "ButtonUnmask,VideoUnmask",
    n_key_frames: int = 4,
    n_recent_frames: int = 2,
    reveal_window: int = 64,
    # 5→2: augment_factor emits N rows/episode with the SAME target but different
    # reveal keyframes. =5 over-teaches keyframe-INVARIANCE ("frames don't matter")
    # which feeds input-blindness; 2 keeps a little subset-robustness. Multi-task
    # already 2×'s the distinct-episode count, so volume isn't the concern.
    augment_factor: int = 2,   # rows/episode via varied reveal keyframe subsets
    n_candidate_frames: int = 12,
    max_keyframes: int = 4,
    joint: bool = True,        # also emit SELECT rows (candidate window → keyframe labels)
                              # for the joint pipeline. False = USE rows only (one-shot SFT).
    seed: int = 0,
    max_episodes: int = 0,
) -> dict:
    """Build the MemER-style memory SFT dataset: USE rows (reveal keyframes + recent
    + LITERAL ``<y,x>`` grounded pick subgoal) and, when ``joint``, SELECT rows
    (candidate window → reveal-frame keyframe labels). Warm-start data for GRPO.
    Writes ``<out_dir>/memory/grounded_subgoal_train.jsonl`` + images.
    """
    from vla_memory.data.build_memory_sft_dataset import build_memory_sft_dataset

    paths = build_memory_sft_dataset(
        raw_data_path=raw_data_path,
        preprocessed_data_path=out_dir,
        only_tasks=tuple(t for t in only_tasks.split(",") if t),
        n_key_frames=n_key_frames,
        n_recent_frames=n_recent_frames,
        reveal_window=reveal_window,
        augment_factor=augment_factor,
        n_candidate_frames=n_candidate_frames,
        max_keyframes=max_keyframes,
        also_select=joint,
        seed=seed,
        max_episodes=max_episodes or None,
    )
    volume.commit()
    print(f"[build_memory_dataset] {paths}", flush=True)
    return paths


# ---------------------------------------------------------------------------
# Stage A.1b — Swap-demo inspection (probe before building the Swap SFT data)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=4.0,
    memory=16 * 1024,
    timeout=1800,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def inspect_swap_demos(
    raw_data_path: str = f"{MOUNT}/data/robomme_data_h5",
    only_task: str = "ButtonUnmaskSwap",
    n_episodes: int = 3,
    every: int = 8,
) -> dict:
    """Dump the per-timestep online subgoal structure for a few Swap demo episodes.

    The Swap builder changes (decision point, SELECT labels, history) depend on facts
    we can only see in the demo data: where the PICK subgoal starts (vs the two button
    presses), whether ``grounded_subgoal_online`` at the pick gives the POST-swap final
    container position, and how the press/pick phases line up with the swap windows
    (``ButtonUnmaskSwap`` swaps at steps 64/114/164 per ``swap_times``). This probe
    prints the subgoal sequence + transitions + grounded targets so we design the
    swap-aware label against REAL structure instead of guessing. Read-only; no GPU.
    """
    import os as _os

    import h5py

    from mme_vla_suite.dataset_builder.robomme_h5_utils import (  # type: ignore
        first_execution_step,
        get_env_id_from_filename,
        get_episode_indices,
        get_task_goal,
        get_timestep_indices,
    )

    files = sorted(f for f in _os.listdir(raw_data_path) if f.endswith(".h5"))
    matched = [f for f in files if get_env_id_from_filename(f) == only_task]
    if not matched:
        raise RuntimeError(
            f"No H5 for task {only_task!r} under {raw_data_path}. Available env_ids: "
            f"{sorted({get_env_id_from_filename(f) for f in files})}"
        )

    out: list = []
    for file in matched:
        with h5py.File(_os.path.join(raw_data_path, file), "r") as data:
            for ep in list(get_episode_indices(data, n_episodes)):
                ed = data[f"episode_{ep}"]
                goal = get_task_goal(ed, lower=True)
                exec_start = int(first_execution_step(ed))
                tsx = [ts for ts in get_timestep_indices(ed) if ts >= exec_start]
                print(
                    f"\n=== {only_task} ep{ep} | goal={goal!r} | exec_start={exec_start} "
                    f"| n_exec_ts={len(tsx)} ===",
                    flush=True,
                )
                prev = None
                transitions: list = []
                for ts in tsx:
                    info = ed[f"timestep_{ts}"]["info"]
                    s = info["simple_subgoal_online"][()].decode().strip()
                    g = info["grounded_subgoal_online"][()].decode().strip()
                    rel = ts - exec_start
                    changed = s != prev
                    if changed:
                        transitions.append((rel, s, g))
                        print(f"  t={ts:>4} (rel {rel:>4}) <-- CHANGE simple={s!r}", flush=True)
                        print(f"             grounded={g!r}", flush=True)
                    elif rel % every == 0:
                        print(f"  t={ts:>4} (rel {rel:>4})          simple={s!r}", flush=True)
                    prev = s
                pick_rel = next((r for r, s, _ in transitions if "pick up the container" in s.lower()), None)
                print(
                    f"  -> {len(transitions)} phase(s); PICK starts at rel={pick_rel} "
                    f"(swap windows on ButtonUnmaskSwap: 64/114/164). "
                    f"completed-before-pick={[s for r, s, _ in transitions if pick_rel is not None and r < pick_rel]}",
                    flush=True,
                )
                out.append({"episode": int(ep), "n_transitions": len(transitions), "pick_rel": pick_rel})
    return {"task": only_task, "episodes": out}


# ---------------------------------------------------------------------------
# Stage A.2 — SFT warmstart
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",          # single GPU — a 4B+LoRA fits in <40GB; multi-GPU only
                              # bought the num_items_in_batch loss bug that collapsed v4.
    timeout=12 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def sft_warmstart(
    dataset_path: str = f"{MOUNT}/data/preprocessed/memory/grounded_subgoal_train.jsonl",
    output_dir: str = f"{MOUNT}/ckpts/qwen_sft/permanence_grounded",
    # Hyperparameter rationale (2026-05-28 research synthesis):
    # - Our 6343-row narrow-vocab dataset hit loss 0.25 at step 80 of ~209
    #   with the upstream-recipe values, clearly overfitting.
    # - Unsloth's "loss < 0.2 = overfitting" threshold + ms-swift's official
    #   Qwen3-VL recipe both point at LOWER capacity than upstream defaults.
    # - Coordinated reduction: rank 16→8, alpha 32→16 (preserves 2× ratio),
    #   lr 1e-4→5e-5, plus dropout 0.1 + label_smoothing 0.1 in the cmd.
    # The 500-row memory dataset is small (~60 steps/epoch at batch 8). A 1-epoch
    # run left loss still STEEPLY decreasing (28→21) with token_acc flat at 0.65 —
    # undertrained, NOT collapsed (grad_norm a sane ~25). More epochs keep the loss
    # dropping and let token_acc climb once the argmax flips on the coordinate/colour
    # tokens. Watch token_acc for the v4-style collapse; pick the best checkpoint via
    # validate_subgoal_checkpoints. Drop back if eval_token_acc diverges (overfit).
    num_train_epochs: int = 5,
    # batch 8, NO gradient accumulation, single GPU → num_items_in_batch cannot be
    # mis-normalized on either the device or accumulation axis (the v4 collapse).
    # v4 log showed ~18GB at per-device 4, so 8 is safe on an 80GB A100.
    per_device_batch_size: int = 8,
    grad_accum: int = 1,
    # Peak LR — RAISED to 2e-5 for the MULTI-TASK run (was 4e-6). The vise: on
    # SINGLE-task narrow data, peak 1e-5 collapsed (~step 80, token_acc 0.65→0.50) as
    # lr decayed through ~8e-6, so 4e-6 was the only safe single-task value — but 4e-6
    # UNDERFITS (ckpt-300 emitted a constant ~<315,305> regardless of input,
    # project_qwen_coordinate_space RESULT). Single-task can't go fast enough to fit
    # the coordinate without collapsing. The escape is DATA DIVERSITY: multi-task
    # (ButtonUnmask+VideoUnmask) lifts the collapse threshold (RoboMME ran lr 1e-4
    # across 1,600 demos, no collapse). 2e-5 is a first step into that regime (5× the
    # old peak, 5× below RoboMME). MONITOR coord-px-dist per ckpt via
    # validate_memory_checkpoints, NOT token_acc (template-inflated): back off toward
    # 1e-5 if it collapses, raise toward 4e-5 if it still underfits.
    # See project_sft_v4_adapter_degenerate, project_qwen_coordinate_space.
    learning_rate: float = 2e-5,
    lora_rank: int = 8,
    lora_alpha: int = 16,
) -> dict:
    """Run ms-swift SFT to produce the LoRA prior for GRPO.

    Mirrors ``robomme_policy_learning/scripts/finetune_vlm_subgoal_predictor.sh``
    — keep the hyperparameters aligned with the upstream recipe so we can reuse
    their reported numbers as a sanity check.
    """
    run_name = f"qwen-sft-memory-{int(time.time())}"
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
        # Single GPU (see the gpu= decorator + per_device comments): sidesteps the
        # multi-GPU num_items_in_batch loss-normalization bug that collapsed v4.
        "NPROC_PER_NODE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
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
        # save/eval every 50 steps (~6 checkpoints over the 5-epoch ~300-step run).
        "--save_steps", "50",
        # Keep ~all checkpoints (~16 saves over a 1-epoch batch-8 run). v4 collapsed
        # early but --save_total_limit 4 kept only the dead late checkpoints and
        # deleted the good early one; 20 retains the full run so
        # validate_subgoal_checkpoints can pick a coherent one. See
        # project_sft_v4_adapter_degenerate.
        "--save_total_limit", "20",
        # Evaluate every 50 steps. WATCH token_acc, NOT eval_loss — in the v4
        # collapse eval_loss kept dropping while token_acc crashed to 0.
        "--eval_steps", "50",
        "--eval_strategy", "steps",
        "--logging_steps", "10",
        # 4096 (was 3200): the joint SELECT rows show up to 12 candidate frames
        # (~256 vision tokens each ≈ 3072) + prompt + output; 3200 would truncate.
        "--max_length", "4096",
        "--output_dir", output_dir,
        "--warmup_ratio", "0.05",
        # --- loss normalization: root-cause fix for the v4 collapse ---
        # v4 logged loss 476 (a per-token CE mean maxes at ~ln(vocab)≈12) and
        # grad_norm 530 at lr≈0 → the loss was mis-normalized. With 4-GPU DDP,
        # ms-swift's average_tokens_across_devices defaults to FALSE, so the
        # non-masked token count isn't all_reduce'd across devices: num_items_in_batch
        # is wrong and the loss/gradient inflate ~T×, blowing up the EFFECTIVE LR →
        # collapse (token_acc 0.85→0). Sync the token count so loss is a true
        # per-token mean. (transformers multi-GPU + grad-accum bug; HF #37474/#37766.)
        "--average_tokens_across_devices", "true",
        # ms-swift's recommended clip for multimodal stability — bounds residual spikes.
        "--max_grad_norm", "0.5",
        # Label smoothing OFF for the memory task. 0.1 was tuned to stop
        # overconfident memorization of the narrow ~10-50 subgoal PHRASE vocab —
        # but here the load-bearing output is the exact pixel COORDINATE
        # (`<y, x>`), which varies per episode and must be read from the keyframes.
        # Smoothing spreads 10% of mass off every target token, which blurs the
        # coordinate digits — exactly the precision we need. The model got the
        # phrase + colour but emitted NO coordinate at ckpt-100; let it commit.
        "--label_smoothing_factor", "0.0",
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
# Stage A.3 — Validate SFT checkpoints (catch the v4 'Wait'-collapse)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=2 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def validate_subgoal_checkpoints(
    run_dir: str = f"{MOUNT}/ckpts/qwen_sft/permanence_grounded",
    task_goal: str = "press the button then pick up the container that hides the red cube",
    max_new_tokens: int = 128,   # MemER JSON output is longer than a bare phrase
) -> dict:
    """Greedy-decode one subgoal from each SFT checkpoint to catch collapse.

    ``eval_loss`` alone hid the v4 adapter collapsing into 'Wait...' repetition
    (see ``project_sft_v4_adapter_degenerate``). This loads each checkpoint,
    greedy-decodes on a neutral synthetic frame, and flags degenerate ones, so you
    can pick the last COHERENT checkpoint to warmstart GRPO from. Coarse collapse
    gate (synthetic image), NOT a quality eval — for real-frame checks run
    ``grpo ... --debug-subgoals``.

    Run on the in-flight v4 checkpoints, or after a fresh sft_warmstart:
      modal run modal_train.py::validate_subgoal_checkpoints
    """
    import gc

    import numpy as np
    import torch

    from vla_memory.qwen_subgoal.model import QwenSubgoalPolicy, _looks_degenerate

    base = Path(run_dir)
    # Accept a swift output_dir (resolve latest v*-*) or a specific run dir.
    versioned = sorted(base.glob("v*-*"))
    run = versioned[-1] if versioned else base
    checkpoints = sorted(
        run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-", 1)[1])
    )
    if not checkpoints:
        raise RuntimeError(f"No checkpoint-* directories under {run}")

    # Mid-gray frame. Collapse is image-independent (it happened on real frames),
    # so a synthetic image is enough to detect a degenerate adapter.
    image = np.full((256, 256, 3), 127, dtype=np.uint8)

    results = []
    print(f"[validate] checking {len(checkpoints)} checkpoints under {run}", flush=True)
    for ckpt in checkpoints:
        row = {"checkpoint": ckpt.name}
        policy = None
        try:
            policy = QwenSubgoalPolicy(adapter_init_path=str(ckpt), device="cuda")
            # MemER prompt: no keyframes, one synthetic "recent" frame. Collapse is
            # image-independent, so the synthetic frame is enough to detect it.
            text, gen_ids = policy.greedy_subgoal(
                key_frames=[], recent_frames=[image], task_goal=task_goal,
                max_new_tokens=max_new_tokens,
            )
            degenerate = _looks_degenerate(gen_ids, max_new_tokens)
            row.update(
                ntok=int(gen_ids.numel()),
                degenerate=bool(degenerate),
                subgoal=text[:120],
            )
            print(
                f"[validate] {ckpt.name}: ntok={int(gen_ids.numel())} "
                f"{'DEGENERATE' if degenerate else 'ok'} subgoal={text[:80]!r}",
                flush=True,
            )
        except Exception as exc:
            row["error"] = repr(exc)[:200]
            print(f"[validate] {ckpt.name}: ERROR {exc!r}", flush=True)
        finally:
            del policy
            gc.collect()
            torch.cuda.empty_cache()
        results.append(row)

    coherent = [r["checkpoint"] for r in results if r.get("degenerate") is False]
    print(
        f"\n[validate] coherent checkpoints (usable GRPO warmstart): {coherent or 'NONE'}",
        flush=True,
    )
    if not coherent:
        print(
            "[validate] ALL checkpoints degenerate — the SFT run collapsed. Reduce "
            "training (fewer epochs/steps), re-run, and do NOT warmstart GRPO from these.",
            flush=True,
        )
    return {"run": str(run), "results": results, "coherent": coherent}


# ---------------------------------------------------------------------------
# Stage A.3b — FAITHFUL memory validation (real keyframes, output vs target)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=2 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def validate_memory_checkpoints(
    run_dir: str = f"{MOUNT}/ckpts/qwen_sft/permanence_grounded",
    dataset_path: str = f"{MOUNT}/data/preprocessed/memory/grounded_subgoal_train.jsonl",
    n_samples: int = 12,
    n_key_frames: int = 4,
    max_new_tokens: int = 128,
) -> dict:
    """Greedy-decode each checkpoint on REAL dataset inputs and score BOTH heads.

    USE head (coordinate): decode the 4-key+2-recent rows and report
    ``mean_coord_px_dist`` — small (≲30px ≈ within a container) means the model points
    at roughly the right container; large/random means it learned the template but not
    the memory. (``validate_subgoal_checkpoints`` uses a synthetic gray frame with 0
    keyframes — a prompt-structure MISMATCH — so it only detects collapse; this uses
    the exact training inputs.)

    SELECT head (joint pipeline only): decode the SELECT rows with ``mode="select"``
    and compare predicted ``keyframe_positions`` to the labeled reveal frames —
    ``select_mean_jaccard``/``recall`` (higher = picked the right frames) and
    ``select_n_empty`` (≈ all = the head collapsed to ``[]``, the contradictory-
    supervision bug the SELECT/USE schema separation fixes). Empty for a --no-joint
    dataset (no SELECT rows).
    """
    import gc
    import json
    import os
    import re

    import numpy as np
    import torch
    from PIL import Image

    from vla_memory.qwen_subgoal.model import QwenSubgoalPolicy
    from vla_memory.qwen_subgoal.prompts import parse_subgoal_output

    _COORD = re.compile(r"<\s*(-?\d+)\s*,\s*(-?\d+)\s*>")

    def coord(s: str):
        m = _COORD.search(s or "")
        return (int(m.group(1)), int(m.group(2))) if m else None

    # One row per DISTINCT episode, restricted to coordinate (USE) rows. The old
    # "first n_samples rows" took augmentations of the SAME episode (the reported
    # ckpt-300 read had 5/6 rows sharing target <605,332>), so it could NOT reveal
    # whether the output VARIES per episode — the whole point of the metric. We key
    # on the episode id in the image filename ("{env}_ep{idx}_{ts}.png"), keep the
    # FIRST row per episode (the a==0 canonical even-spacing layout the builder emits
    # first, which also matches inference), and skip SELECT rows (no coordinate
    # target). Even-subsample across all episodes so a multi-task dataset
    # (ButtonUnmask then VideoUnmask in file order) is covered across both tasks.
    _EP = re.compile(r"^(.*_ep\d+)_\d+\.(?:png|jpg|jpeg)$", re.IGNORECASE)

    def episode_key(row) -> str:
        for p in row.get("images", []):
            mm = _EP.search(os.path.basename(p))
            if mm:
                return mm.group(1)
        return ""

    # Classify each row by target: USE rows carry a coordinate (current_subtask),
    # SELECT rows carry non-empty keyframe_positions and no coordinate. Collect ONE
    # row per distinct episode for each (the a==0 canonical layout the builder emits
    # first). SELECT rows are absent in a --no-joint dataset → that section is skipped.
    seen_use: set = set()
    seen_sel: set = set()
    use_rows: list = []
    sel_rows: list = []
    with open(dataset_path) as f:
        for line in f:
            r = json.loads(line)
            sub, pos = parse_subgoal_output(r["messages"][2]["content"])
            ek = episode_key(r)
            if coord(sub) is not None:           # USE row — coordinate target
                if ek not in seen_use:
                    seen_use.add(ek)
                    use_rows.append(r)
            elif pos:                            # SELECT row — keyframe_positions, no coord
                if ek not in seen_sel:
                    seen_sel.add(ek)
                    sel_rows.append(r)
    if not use_rows:
        raise RuntimeError(f"No coordinate (USE) rows found in {dataset_path}")

    def _even(rows: list) -> list:
        if len(rows) > n_samples:
            idx = np.linspace(0, len(rows) - 1, num=n_samples, dtype=int)
            return [rows[i] for i in idx]
        return rows

    use_rows = _even(use_rows)
    sel_rows = _even(sel_rows)

    def _load(paths):
        return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]

    samples = []
    for r in use_rows:
        imgs = _load(r["images"])
        user = r["messages"][1]["content"]
        m = re.search(r"The task goal is:\s*(.+)", user)
        hm = re.search(r"The subtasks already completed are:\s*(.+)", user)
        history = (
            [re.sub(r"^\s*\d+\.\s*", "", p.strip()) for p in hm.group(1).split(";") if p.strip()]
            if hm else []
        )
        target_sub, _ = parse_subgoal_output(r["messages"][2]["content"])
        samples.append({
            "key": imgs[:n_key_frames],
            "recent": imgs[n_key_frames:],
            "task_goal": m.group(1).strip() if m else "",
            "history": history,
            "target": target_sub,
        })

    # SELECT samples: the row's images ARE the candidate window (n_key_frames=0).
    # We decode mode="select" and compare predicted keyframe_positions to the labeled
    # reveal frames — the check the coordinate metric can't see.
    sel_samples = []
    for r in sel_rows:
        user = r["messages"][1]["content"]
        m = re.search(r"The task goal is:\s*(.+)", user)
        _, tgt_pos = parse_subgoal_output(r["messages"][2]["content"])
        sel_samples.append({
            "candidates": _load(r["images"]),
            "task_goal": m.group(1).strip() if m else "",
            "target_pos": sorted({int(p) for p in tgt_pos}),
        })

    print(
        f"[mem-validate] {len(samples)} USE + {len(sel_samples)} SELECT distinct-episode "
        f"samples (USE of {len(seen_use)} eps, SELECT of {len(seen_sel)} eps); "
        f"USE target example: {samples[0]['target']!r}",
        flush=True,
    )

    base = Path(run_dir)
    versioned = sorted(base.glob("v*-*"))
    run = versioned[-1] if versioned else base
    checkpoints = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-", 1)[1]))
    if not checkpoints:
        raise RuntimeError(f"No checkpoint-* directories under {run}")

    results = []
    for ckpt in checkpoints:
        policy = None
        n_grounded = n_empty = 0
        dists = []
        sel_jacc: list = []
        sel_rec: list = []
        sel_empty = 0
        try:
            policy = QwenSubgoalPolicy(adapter_init_path=str(ckpt), device="cuda")
            print(f"\n[mem-validate] === {ckpt.name} ===", flush=True)
            for i, s in enumerate(samples):
                text, _ = policy.greedy_subgoal(
                    key_frames=s["key"], recent_frames=s["recent"],
                    task_goal=s["task_goal"], history_subgoals=s["history"],
                    max_new_tokens=max_new_tokens,
                )
                out_sub, _ = parse_subgoal_output(text)
                oc, tc = coord(out_sub), coord(s["target"])
                n_empty += 0 if out_sub.strip() else 1
                if oc is not None:
                    n_grounded += 1
                    if tc is not None:
                        dists.append(((oc[0] - tc[0]) ** 2 + (oc[1] - tc[1]) ** 2) ** 0.5)
                print(f"  s{i}: OUT={out_sub!r}\n      TGT={s['target']!r}", flush=True)
            # SELECT head: greedy mode="select" → predicted keyframe_positions vs the
            # labeled reveal frames. The headline is select_n_empty: empty≈all means the
            # head collapsed to [] (the contradictory-supervision bug the schema fix kills).
            for j, s in enumerate(sel_samples):
                text, _ = policy.greedy_subgoal(
                    key_frames=[], recent_frames=s["candidates"],
                    task_goal=s["task_goal"], history_subgoals=[],
                    max_new_tokens=max_new_tokens, mode="select",
                )
                _, pred = parse_subgoal_output(text)
                ncand = len(s["candidates"])
                pred_set = {p for p in pred if 1 <= p <= ncand}   # clamp to valid indices
                tgt_set = set(s["target_pos"])
                inter = len(pred_set & tgt_set)
                union = len(pred_set | tgt_set)
                sel_jacc.append(inter / union if union else 1.0)
                sel_rec.append(inter / len(tgt_set) if tgt_set else 0.0)
                sel_empty += 1 if not pred_set else 0
                print(f"  sel{j}: PRED={sorted(pred_set)} TGT={sorted(tgt_set)}", flush=True)
        except Exception as exc:
            print(f"[mem-validate] {ckpt.name}: ERROR {exc!r}", flush=True)
            results.append({"checkpoint": ckpt.name, "error": repr(exc)[:200]})
            continue
        finally:
            del policy
            gc.collect()
            torch.cuda.empty_cache()
        mean_dist = float(np.mean(dists)) if dists else None
        result = {
            "checkpoint": ckpt.name, "n_grounded": n_grounded, "n_empty": n_empty,
            "n_samples": len(samples), "mean_coord_px_dist": mean_dist,
        }
        print(
            f"[mem-validate] {ckpt.name}: grounded {n_grounded}/{len(samples)}, "
            f"empty {n_empty}, mean coord px-dist={mean_dist}",
            flush=True,
        )
        if sel_samples:
            mean_jacc = float(np.mean(sel_jacc))
            mean_rec = float(np.mean(sel_rec))
            result.update({
                "select_mean_jaccard": mean_jacc,
                "select_mean_recall": mean_rec,
                "select_n_empty": sel_empty,
                "n_select_samples": len(sel_samples),
            })
            print(
                f"[mem-validate] {ckpt.name}: SELECT jaccard={mean_jacc:.3f} "
                f"recall={mean_rec:.3f} empty={sel_empty}/{len(sel_samples)} "
                "(empty≈all → SELECT head collapsed to []; the schema fix should prevent it)",
                flush=True,
            )
        results.append(result)
    return {"run": str(run), "results": results}


# ---------------------------------------------------------------------------
# Stage A.3c — Cube-visibility diagnostic (can the VLM SEE the cubes at all?)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def probe_cube_visibility(
    dataset_path: str = f"{MOUNT}/data/preprocessed/memory/grounded_subgoal_train.jsonl",
    n_samples: int = 4,
    adapter_path: str = "",   # "" = BASE Qwen3-VL (no SFT); else a checkpoint dir
) -> dict:
    """Ask the VLM directly what cubes it sees + where, on a real reveal keyframe.

    The constant-coordinate SFT failure (``<100,100>`` for every episode) means the
    model ignores the keyframes. This isolates WHY: if the model can name the cube
    colours + rough pixel locations here, the keyframes ARE usable and the fix is to
    stop forcing pixel REGRESSION (give candidate container coords → select). If it
    can't, the keyframes are unusable (resolution/clarity) and we fix the inputs.
    Defaults to the BASE model so we test the VLM's raw perception, not the adapter.
    """
    import json

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda", trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Take USE rows (grounded coordinate targets); their first image is a reveal keyframe.
    rows = []
    with open(dataset_path) as f:
        for line in f:
            r = json.loads(line)
            if "container at <" in r["messages"][2]["content"]:
                rows.append(r)
            if len(rows) >= n_samples:
                break

    question = (
        "This is a 256x256 robot camera image. A few containers each briefly reveal a "
        "colored cube (red, green, and/or blue) underneath. For EACH cube you can see, "
        "give its colour and its approximate pixel location as (x, y). If you cannot make "
        "out any cubes, say 'no cubes visible'."
    )
    results = []
    for i, r in enumerate(rows):
        kf = r["images"][0]
        img = Image.fromarray(__import__("numpy").asarray(Image.open(kf).convert("RGB")))
        messages = [
            {"role": "system", "content": "You are a precise visual grounding assistant."},
            {"role": "user", "content": f"<image>{question}"},
        ]
        chat_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_text = chat_text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        inputs = proc(text=[chat_text], images=[img], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        ans = proc.tokenizer.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        tgt = r["messages"][2]["content"]
        print(f"[cube-probe] sample{i} kf={kf}\n  TARGET (true container coord): {tgt}\n  VLM SEES: {ans}\n", flush=True)
        results.append({"keyframe": kf, "target": tgt, "answer": ans})
    return {"adapter": adapter_path or "BASE", "results": results}


# ---------------------------------------------------------------------------
# Stage A.4 — Held-out VLM rollout probe (the gate before GRPO)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=4 * 3600,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def probe_vlm_rollout(
    adapter_path: str = f"{MOUNT}/ckpts/qwen_sft/permanence_grounded",
    low_level_ckpt_dir: str = f"{MOUNT}/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999",
    policy_config: str = "mme_vla_suite",
    task: str = "ButtonUnmask",
    episodes: int = 12,
    seed: int = 20260601,        # held-out base, distinct from training demos / GRPO seed=0
    rollout_max_steps: int = 400,
) -> dict:
    """Boot GroundSG and roll out the SFT'd VLM's greedy subgoal on HELD-OUT seeds.

    The gate before GRPO. ``validate_memory_checkpoints`` proved the VLM conditions
    on the keyframes on TRAIN rows; this runs the full inference path (peek →
    greedy subgoal → ``coords.from_qwen_xy`` → GroundSG) on unseen seeds and checks
    the three things GRPO needs: VLM success > 0 (bootstrap), VLM ≈ oracle ceiling
    (generalizes), VLM > coordinate-shifted contrast (the grounding + conversion are
    load-bearing). Exit 0 / ``proceed:True`` means launch GRPO; nonzero means fix the
    SFT ckpt / conversion first. Pin a checkpoint with ``--adapter-path .../checkpoint-N``.
    See ``probe_vlm_rollout.py``.
    """
    port = _free_port()
    server_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "OPENPI_DATA_HOME": f"{MOUNT}/openpi",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    print(f"Starting GroundSG π0.5 server on localhost:{port} …")
    server_proc = subprocess.Popen(
        [
            "uv", "run", "scripts/serve_policy.py",
            f"--port={port}",
            f"--seed={seed}",
            "policy:checkpoint",
            f"--policy.dir={low_level_ckpt_dir}",
            f"--policy.config={policy_config}",
        ],
        cwd="/app",
        env=server_env,
    )
    for _ in range(180):
        if server_proc.poll() is not None:
            raise RuntimeError("GroundSG policy server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                print("GroundSG server is up.")
                break
        except OSError:
            time.sleep(1.0)
    else:
        raise TimeoutError("GroundSG policy server did not become reachable")

    hf_cache_dir = f"{MOUNT}/.cache/huggingface"
    Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)
    probe_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # CPU rendering for the simulator (same as the GRPO/eval container).
        "SAPIEN_RENDER_DEVICE": "cpu",
        "MUJOCO_GL": "osmesa",
        # Reuse the volume-cached Qwen weights (same strategy as SFT/GRPO).
        "USE_HF": "1",
        "HF_HOME": hf_cache_dir,
        "HF_HUB_CACHE": f"{hf_cache_dir}/hub",
        "TRANSFORMERS_CACHE": f"{hf_cache_dir}/hub",
    }
    cmd = [
        "micromamba", "run", "-n", "robomme",
        "python", "-u", "/workspace/src/vla_memory/grpo/probe_vlm_rollout.py",
        f"--port={port}",
        f"--adapter-path={adapter_path}",
        f"--task={task}",
        f"--episodes={episodes}",
        f"--seed={seed}",
        f"--rollout-max-steps={rollout_max_steps}",
    ]
    import shlex
    print("Launching held-out VLM rollout probe:", shlex.join(cmd), flush=True)
    rc = 1
    try:
        proc = subprocess.run(cmd, env=probe_env, cwd="/workspace")
        rc = proc.returncode
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
    return {"task": task, "episodes": episodes, "adapter": adapter_path,
            "probe_exit_code": rc, "proceed": rc == 0}


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
    # GRPO warmstart = the grounded ButtonUnmask SFT adapter (Phase 3 output). The
    # resolver accepts a swift output_dir (auto-picks the latest v*/checkpoint-*) or
    # a specific checkpoint-N dir. Override at the CLI once SFT completes:
    #   modal run modal_train.py::grpo --sft-adapter-path /mnt/.../checkpoint-N
    sft_adapter_path: str = f"{MOUNT}/ckpts/qwen_sft/permanence_grounded",
    # GroundSG (subgoal-conditioned) π0.5 — NOT pi05_baseline, which discards the
    # subgoal (project_pi05_baseline_ignores_subgoal). create_trained_policy reads
    # symbolic-grounded-subgoal/history_config.txt (next to the ckpt) and activates
    # grounded-subgoal symbolic memory automatically. Download via
    # `modal run modal_train.py::download_groundsg`.
    low_level_ckpt_dir: str = f"{MOUNT}/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999",
    policy_config: str = "mme_vla_suite",
    output_dir: str = f"{MOUNT}/runs/grpo/buttonunmask_groundsg_v0",
    num_steps: int = 200,
    batch_states: int = 4,
    group_size: int = 8,
    kl_beta: float = 0.0,             # KL off (DAPO/TRL default); reference only used if >0
    learning_rate: float = 1e-4,      # ~10x v0; LoRA RL wants a higher lr than SFT
    sample_temperature: float = 1.0,
    rollouts_per_subgoal: int = 1,    # >1 averages reward to cut pi0.5 flow-sampling noise
    rollout_max_steps: int = 200,
    only_tasks: str = "ButtonUnmask",   # Permanence/spatial-memory task (was PickXtimes)
    episodes_per_task: int = 20,
    subgoal_type: str = "grounded_subgoal",   # memory lives in the grounding (which container)
    joint_selection: bool = False,    # joint select-then-use: train keyframe selection + subtask
    n_candidate_frames: int = 12,     # SELECT-call candidate window (joint)
    max_keyframes: int = 4,           # cap on kept keyframes (joint)
    seed: int = 0,
    debug_subgoals: bool = False,     # print each sampled subgoal's text + token count
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
            f"--policy.dir={low_level_ckpt_dir}",
            f"--policy.config={policy_config}",
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
        f"--n-candidate-frames={n_candidate_frames}",
        f"--max-keyframes={max_keyframes}",
        f"--seed={seed}",
    ]
    grpo_cmd.append("--joint-selection" if joint_selection else "--no-joint-selection")
    if debug_subgoals:
        grpo_cmd.append("--debug-subgoals")
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
        modal run modal_train.py --stage download_groundsg # subgoal-conditioned π0.5 (~GB)
        modal run modal_train.py --stage causality_probe   # BLOCKER check: does π0.5 read the subgoal?
        modal run modal_train.py --stage build_dataset    # H5 -> QwenVL JSONL (simple subgoal)
        modal run modal_train.py --stage build_memory     # H5 -> memory SFT JSONL (Permanence: ButtonUnmask+VideoUnmask grounded)
        modal run modal_train.py --stage sft              # SFT warmstart (memory data)
        modal run modal_train.py --stage probe_vlm        # held-out VLM rollout gate (pre-GRPO)
        modal run modal_train.py --stage grpo             # GRPO fine-tune
    """
    if stage == "download_data":
        print(download_demonstrations.remote())
    elif stage == "download_groundsg":
        print(download_groundsg.remote())
    elif stage == "causality_probe":
        print(causality_probe.remote())
    elif stage == "build_dataset":
        print(build_dataset.remote())
    elif stage == "build_memory":
        print(build_memory_dataset.remote())
    elif stage == "inspect_swap":
        print(inspect_swap_demos.remote())
    elif stage == "sft":
        print(sft_warmstart.remote())
    elif stage == "probe_vlm":
        print(probe_vlm_rollout.remote())
    elif stage == "grpo":
        print(grpo.remote())
    else:
        raise SystemExit(f"unknown stage: {stage}")
