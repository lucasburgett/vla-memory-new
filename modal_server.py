"""
Modal app for running RoboMME experiments with the frozen pi0.5 baseline VLA.

Architecture (all runs on a single Modal A100):
  1. serve_policy.py  — JAX policy server, WebSocket on a random localhost port
  2. eval.py          — RoboMME simulator (CPU rendering via OSMesa) connects to it
  Results are written to the Modal Volume and returned as a JSON dict.

Usage:
  # First run — download the checkpoint (~4 GB):
  modal run modal_server.py --download-only

  # Full evaluation (all 16 tasks):
  modal run modal_server.py

  # Quick smoke-test (one task, overwrite previous results):
  modal run modal_server.py --only-tasks "PickXtimes" --overwrite

  # Different seed / checkpoint step:
  modal run modal_server.py --seed 42 --ckpt-id 59999
"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App + persistent storage
# ---------------------------------------------------------------------------

app = modal.App("robomme-pi05-frozen-vla")

# Persists across runs: checkpoint files + evaluation results
volume = modal.Volume.from_name("robomme-vla-data", create_if_missing=True)
MOUNT = "/mnt/robomme"

# ---------------------------------------------------------------------------
# Container image
#
# Built from robomme_policy_learning/Dockerfile, which gives us:
#   - JAX 0.5.3 + CUDA 12.8 (via uv sync --frozen)
#   - micromamba "robomme" env with ManiSkill + RoboMME simulator
#
# PREREQUISITE: run `scripts/setup.sh` first to initialize the submodule and
# its nested third_party/robomme_benchmark before deploying.
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_dockerfile(
        # Modal 1.x: context_dir sets the Docker build context directory.
        # The Dockerfile's COPY instructions resolve relative to that dir.
        "robomme_policy_learning/Dockerfile",
        context_dir="robomme_policy_learning",
        # Skip large generated dirs that shouldn't be in the build context
        ignore=["data/", "runs/", ".git/"],
        # The Dockerfile installs Python 3.11 only via uv into /app/.venv —
        # no system Python exists for Modal's runtime to detect.
        # add_python injects a system Python 3.11 at /usr/local after the
        # Docker build completes; uv's venv is unaffected.
        add_python="3.11",
    )
    # Add OSMesa for headless CPU rendering in the ManiSkill simulator
    .apt_install(["libosmesa6", "libosmesa6-dev"])
    # wandb is already in the uv venv, but Modal's function runtime uses the
    # system Python injected by add_python — install it there too.
    .pip_install("wandb>=0.19")
    .env(
        {
            # Policy-server side (JAX / openpi)
            "OPENPI_DATA_HOME": f"{MOUNT}/openpi",
            "PYTHONPATH": "/app/src",
            # Simulator side — use CPU rasterization; no display required
            "SAPIEN_RENDER_DEVICE": "cpu",
            "MUJOCO_GL": "osmesa",
            "DISPLAY": "",
            # Misc
            "XDG_RUNTIME_DIR": "/tmp/runtime-root",
        }
    )
)

# ---------------------------------------------------------------------------
# Helper: free port
# ---------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# Function 1 — download checkpoint
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    # CPU-only is fine for downloading; saves GPU cost.
    # 1 h ceiling: Yinpei/pi05_baseline is ~11.6 GB of LFS objects — clone +
    # `git lfs pull` + unzip + volume.commit() of the extracted data runs well
    # past 15 min, which is what killed the earlier 900 s attempt.
    timeout=3600,
    volumes={MOUNT: volume},
    # Ship the local .env into the container (provides HF_TOKEN)
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def download_pi05_baseline() -> str:
    """Download Yinpei/pi05_baseline from HuggingFace into the volume.

    Safe to call multiple times — skips if already present.
    Returns the path to the checkpoint root on the volume.
    """
    ckpt_root = Path(f"{MOUNT}/ckpts/pi05_baseline")
    done_marker = ckpt_root / ".download_complete"

    if done_marker.exists():
        print(f"Checkpoint already present at {ckpt_root}")
        return str(ckpt_root)

    ckpt_root.parent.mkdir(parents=True, exist_ok=True)

    # A previous run may have been interrupted (e.g. timeout) after cloning but
    # before the marker was written. `git clone` refuses a non-empty
    # destination, so clear any partial state for a clean retry.
    if ckpt_root.exists():
        import shutil

        print(f"Removing incomplete previous download at {ckpt_root} …")
        shutil.rmtree(ckpt_root)

    # Clone without LFS objects first (fast metadata clone)
    print("Cloning Yinpei/pi05_baseline (LFS skip) …")
    subprocess.run(
        [
            "git",
            "clone",
            "https://huggingface.co/Yinpei/pi05_baseline",
            str(ckpt_root),
        ],
        check=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
    )

    # Pull the actual weight files via LFS
    hf_token = os.environ.get("HF_TOKEN", "")
    lfs_env = {**os.environ}
    if hf_token:
        lfs_env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("Pulling LFS objects (model weights) …")
    subprocess.run(["git", "lfs", "pull"], check=True, cwd=str(ckpt_root), env=lfs_env)

    # Unzip checkpoint archives produced by robomme_policy_learning's training
    print("Unzipping checkpoint archives …")
    subprocess.run(
        ["uv", "run", "scripts/unzip_ckpt.py", str(ckpt_root)],
        check=True,
        cwd="/app",
    )

    done_marker.touch()
    volume.commit()
    print(f"Checkpoint ready at {ckpt_root}")
    return str(ckpt_root)


# ---------------------------------------------------------------------------
# Function 2 — run trial
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,          # 2 h ceiling; full eval is ~1–1.5 h
    volumes={MOUNT: volume},
    # Ship the local .env into the container (provides WANDB_API_KEY / WANDB_PROJECT)
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def run_trial(
    ckpt_id: int = 79999,
    seed: int = 7,
    only_tasks: str = "",
    overwrite: bool = False,
) -> dict:
    """Run a complete RoboMME evaluation with the frozen pi0.5 baseline VLA.

    - Policy server (JAX, GPU) and simulator (CPU rendering) run on the same A100.
    - Results are written to the volume under MOUNT/results/ and returned here.

    Args:
        ckpt_id:    Checkpoint step to evaluate (default: 79999 = final ckpt).
        seed:       Random seed for the policy server.
        only_tasks: Comma-separated task names to restrict evaluation (empty = all 16).
        overwrite:  Re-evaluate even if results already exist.

    Returns:
        dict with keys: exit_code, results_dir, and any parsed result JSON files.
    """
    # The HF repo Yinpei/pi05_baseline nests its checkpoints inside a top-level
    # pi05_baseline/ folder, so after `git clone` into {MOUNT}/ckpts/pi05_baseline
    # the unzipped checkpoint steps land one directory deeper.
    ckpt_dir = Path(f"{MOUNT}/ckpts/pi05_baseline/pi05_baseline/{ckpt_id}")
    results_root = Path(f"{MOUNT}/results")

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_dir}\n"
            "Run first:  modal run modal_server.py --download-only"
        )

    # ------------------------------------------------------------------
    # Drop stale log.json on a full-eval resume.
    #
    # examples/robomme/eval.py uses `save_dir/log.json` as its "evaluation
    # complete" marker:  `while not os.path.exists(save_dir / "log.json"):`.
    # But it writes log.json at the end of any pass, including a partial run
    # with --only-tasks, so a smoke test leaves a log.json containing only
    # the smoke-tested task. A follow-up full run (only_tasks="", no
    # --overwrite) then short-circuits and returns the stale partial result.
    # progress.json (per-episode resume) is still trustworthy, so we keep it.
    # ------------------------------------------------------------------
    if not only_tasks and not overwrite:
        stale_log = results_root / f"pi05_baseline/ckpt{ckpt_id}/seed{seed}/log.json"
        if stale_log.exists():
            print(f"Removing stale {stale_log} so the full eval can resume.")
            stale_log.unlink()

    # ------------------------------------------------------------------
    # Copy reference norm_stats if the checkpoint doesn't already include
    # them (pi05_baseline checkpoints from HF may or may not ship these).
    # ------------------------------------------------------------------
    repo_norm_stats = Path("/app/assets/norm_stats.json")
    for config_name in ("pi05_baseline", "mme_vla_suite"):
        dst = Path(f"{MOUNT}/assets/{config_name}/robomme/norm_stats.json")
        if repo_norm_stats.exists() and not dst.exists():
            import shutil
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(repo_norm_stats, dst)

    # ------------------------------------------------------------------
    # W&B run — init before the eval so wall-clock time is accurate
    # ------------------------------------------------------------------
    import wandb

    _wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    if _wandb_enabled:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "vla-memory-new"),
            name=f"pi05_baseline_ckpt{ckpt_id}_seed{seed}",
            job_type="eval",
            config={
                "policy": "pi05_baseline",
                "ckpt_id": ckpt_id,
                "seed": seed,
                "tasks": only_tasks or "all",
            },
        )
    else:
        print("WANDB_API_KEY not set — skipping W&B logging")

    # ------------------------------------------------------------------
    # Start the policy server (background process)
    # ------------------------------------------------------------------
    port = _free_port()
    server_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "OPENPI_DATA_HOME": f"{MOUNT}/openpi",
    }

    print(f"Starting policy server on localhost:{port} …")
    server_proc = subprocess.Popen(
        [
            "uv",
            "run",
            "scripts/serve_policy.py",
            f"--port={port}",
            f"--seed={seed}",
            "policy:checkpoint",
            f"--policy.dir={ckpt_dir}",
            "--policy.config=pi05_baseline",
        ],
        cwd="/app",
        env=server_env,
    )

    # Wait for JAX XLA compilation + server init (~60–90 s on A100)
    print("Waiting 90 s for JAX compilation and server startup …")
    for tick in range(18):
        time.sleep(5)
        if server_proc.poll() is not None:
            raise RuntimeError(
                f"Policy server exited unexpectedly during startup "
                f"(returncode={server_proc.returncode})."
            )
        print(f"  {(tick + 1) * 5}s …")

    # ------------------------------------------------------------------
    # Run the RoboMME evaluation client (micromamba 'robomme' env)
    # ------------------------------------------------------------------
    eval_cmd = [
        "micromamba",
        "run",
        "-n",
        "robomme",
        "python",
        "examples/robomme/eval.py",
        "--args.host=127.0.0.1",
        f"--args.port={port}",
        f"--args.model_seed={seed}",
        "--args.policy_name=pi05_baseline",
        f"--args.model_ckpt_id={ckpt_id}",
        "--args.no-use-history",      # frozen baseline: no history
        f"--args.save_dir={results_root}",
    ]
    if only_tasks:
        eval_cmd += [f"--args.only_tasks={only_tasks}"]
    if overwrite:
        eval_cmd += ["--args.overwrite"]

    print("Starting evaluation …")
    print(" ".join(eval_cmd))
    eval_result = subprocess.run(eval_cmd, cwd="/app")

    # Shutdown server gracefully; force-kill if it hangs
    server_proc.terminate()
    try:
        server_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        server_proc.wait()

    volume.commit()

    # ------------------------------------------------------------------
    # Collect and return results
    # ------------------------------------------------------------------
    results_dir = results_root / f"pi05_baseline/ckpt{ckpt_id}/seed{seed}"
    summary: dict = {
        "exit_code": eval_result.returncode,
        "results_dir": str(results_dir),
    }

    if results_dir.exists():
        for json_file in sorted(results_dir.glob("*.json")):
            try:
                summary[json_file.stem] = json.loads(json_file.read_text())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Log results to W&B
    # ------------------------------------------------------------------
    if _wandb_enabled:
        flat: dict = {"eval/exit_code": eval_result.returncode}
        for key, val in summary.items():
            if key in ("exit_code", "results_dir"):
                continue
            if isinstance(val, dict):
                # Flatten one level: e.g. {"PickXtimes": {"success_rate": 0.7}}
                for metric, mv in val.items():
                    if isinstance(mv, (int, float)):
                        flat[f"eval/{key}/{metric}"] = mv
            elif isinstance(val, (int, float)):
                flat[f"eval/{key}"] = val
        wandb.log(flat)
        wandb.finish()

    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    ckpt_id: int = 79999,
    seed: int = 7,
    only_tasks: str = "",
    download_only: bool = False,
    overwrite: bool = False,
):
    """Orchestrate a RoboMME trial on Modal.

    Steps:
      1. Ensure the pi05_baseline checkpoint is on the volume (download if needed).
      2. Launch run_trial on an A100 — runs policy server + eval end-to-end.
      3. Print the result summary.

    Examples::

        # Full run (downloads checkpoint on first call, ~4 GB):
        modal run modal_server.py

        # Specific tasks only:
        modal run modal_server.py --only-tasks "PickXtimes,BinFill"

        # Download checkpoint without running eval:
        modal run modal_server.py --download-only

        # Re-run with a different random seed:
        modal run modal_server.py --seed 42
    """
    print(f"[robomme-pi05]  ckpt={ckpt_id}  seed={seed}  tasks='{only_tasks or 'all'}'")

    print("\n--- Step 1: ensure checkpoint is downloaded ---")
    download_pi05_baseline.remote()

    if download_only:
        print("Download complete. Exiting (--download-only).")
        return

    print("\n--- Step 2: running evaluation on Modal A100 ---")
    result = run_trial.remote(
        ckpt_id=ckpt_id,
        seed=seed,
        only_tasks=only_tasks,
        overwrite=overwrite,
    )

    print("\n=== Result ===")
    print(json.dumps(result, indent=2))
