"""GRPO entry point — runs entirely inside the micromamba `robomme` env.

``modal_train.py:grpo`` launches this script with::

    micromamba run -n robomme python -u src/vla_memory/grpo/main.py --port N ...

We run here (not in Modal's add_python="3.11" runtime) so that sapien /
mani_skill / torch live in one consistent environment — the same one the
submodule's Dockerfile sets up via `pip install -e third_party/...`. Mixing
them with Modal Python's torch 2.12 segfaults during sapien init.

This mirrors ``modal_server.py``'s pattern of shelling into the micromamba
env for the simulator side (modal_server.py:307-329).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Path setup: we're invoked from /workspace by micromamba run, but our imports
# come from both /workspace/src (vla_memory package) and /app/examples/robomme
# (the submodule's utils.py + env_runner.py that our subclass extends).
sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "/app/examples/robomme")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO main loop (micromamba env)")
    parser.add_argument("--port", type=int, required=True,
                        help="Port the π0.5 policy server is listening on (localhost).")
    parser.add_argument("--sft-adapter-path", default=None,
                        help="Either a swift output_dir or a specific checkpoint-N dir. "
                             "Omit (or pass empty) to COLD-START from base Qwen + a fresh "
                             "LoRA — the no-SFT-warmup baseline ablation. Cold-start has no "
                             "reference adapter, so keep --kl-beta 0.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=25,
                        help="Save a LoRA-adapter checkpoint every N steps. The CPU-sim "
                             "bottleneck caps a 24h streaming run at ~15 steps, so the "
                             "default 25 would bank nothing before timeout — set low "
                             "(e.g. 3) for long streaming runs.")
    parser.add_argument("--batch-states", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--rollouts-per-subgoal", type=int, default=1,
                        help="Rollouts per subgoal, averaged — cuts frozen-pi0.5 flow-sampling noise.")
    parser.add_argument("--dynamic-sampling", action=argparse.BooleanOptionalAction, default=True,
                        help="Resample states until the batch has B non-degenerate (signal) groups.")
    parser.add_argument("--dynamic-sampling-max-multiplier", type=int, default=3,
                        help="Cap group attempts at batch_states * this per step (bounds rollout cost).")
    parser.add_argument("--normalize-advantage-std", action=argparse.BooleanOptionalAction, default=False,
                        help="Divide advantage by group std (default off — Dr.GRPO unbiased baseline).")
    parser.add_argument("--seed-match-group", action=argparse.BooleanOptionalAction, default=True,
                        help="Pin the env reset seed across a group's rollouts (env-side CRN).")
    parser.add_argument("--rollout-max-steps", type=int, default=700,
                        help="Bounds warm-up + forward execution. Must clear the latest pick's "
                             "warm-up (Swap pick 2 ~rel 410) + the pick, or pick_index=1 states "
                             "truncate before their decision point. ButtonUnmask finishes well under.")
    parser.add_argument("--only-tasks", default="ButtonUnmask",
                        help="Permanence/spatial-memory task by default (was PickXtimes).")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--subgoal-type", default="grounded_subgoal",
                        help="Memory lives in the grounding (which container) — GroundSG uses grounded.")
    parser.add_argument("--n-key-frames", type=int, default=4,
                        help="Reveal-window keyframes fed to the VLM as memory.")
    parser.add_argument("--n-recent-frames", type=int, default=2,
                        help="Recent execution frames (current context) fed to the VLM.")
    parser.add_argument("--reveal-window", type=int, default=64,
                        help="Steps over which the scene reveals (ButtonUnmask lifts/drops bins 0-64).")
    parser.add_argument("--decision-warm-cap", type=int, default=250,
                        help="Per-pick cap on oracle warm-up steps (scaled ×(pick_index+1) in "
                             "rollout). 250 covers ButtonUnmaskSwap pick 1 (~rel 180); pick 2 "
                             "gets 500 (~rel 410). ButtonUnmask stops at its pick well before this.")
    parser.add_argument("--picks-per-episode", type=int, default=1,
                        help="Multi-pick: emit one GRPO state per pick (oracle drives earlier "
                             "picks, VLM owns pick i). >1 for ButtonUnmaskSwap so each pick earns "
                             "reward. The seed-dependent true pick count caps it; extra states drop.")
    # --- joint keyframe-selection (JOINT_MEMORY_DESIGN.md) ---
    parser.add_argument("--joint-selection", action=argparse.BooleanOptionalAction, default=False,
                        help="Select-then-use: VLM picks keyframes from a candidate window, then "
                             "acts on ONLY those. Trains selection + subtask jointly. Off = one-shot.")
    parser.add_argument("--n-candidate-frames", type=int, default=12,
                        help="SELECT-call candidate-window breadth (joint).")
    parser.add_argument("--max-keyframes", type=int, default=4,
                        help="Cap on kept keyframes the USE call sees (joint).")
    parser.add_argument("--snapshot-branching", action=argparse.BooleanOptionalAction, default=False,
                        help="Speed path: warm up once per group, snapshot the env at the "
                             "decision point, restore it per candidate (no K re-warms). Gate with "
                             "modal_train.py::probe_snapshot_parity before enabling. Off = rebuild path.")
    parser.add_argument("--streaming-memory", action=argparse.BooleanOptionalAction, default=False,
                        help="MemER-faithful single-call output: the VLM owns the whole episode and a "
                             "keyframe buffer accumulates across pick decision points, so a selection at "
                             "pick 0 causally drives pick 1's grounding. ONE state per episode. Use with "
                             "--only-tasks ButtonUnmaskSwap. Mutually exclusive with --joint-selection.")
    parser.add_argument("--streaming-max-picks", type=int, default=4,
                        help="Cap on decision points (picks) per streaming episode.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-subgoals", action="store_true",
                        help="Print each sampled subgoal's text + token count — "
                             "use to confirm generations terminate (v1 no-EOS check).")
    args = parser.parse_args()

    if args.streaming_memory and args.joint_selection:
        parser.error(
            "--streaming-memory and --joint-selection are mutually exclusive: streaming is "
            "the single-call MemER path; joint-selection is the two-call select-then-use path."
        )

    # Echo the resolved config so the first run's log makes it obvious which
    # hyperparameters and paths landed in this process (vs. defaults in the
    # launcher).
    print(
        "[grpo] config: " + " ".join(
            f"{k}={v}" for k, v in sorted(vars(args).items())
        ),
        flush=True,
    )

    import torch

    from vla_memory.grpo.env_runner import EnvRunner
    from vla_memory.grpo.reward import RewardConfig
    from vla_memory.grpo.rollout import RolloutWorker
    from vla_memory.grpo.state_dataset import StateDataset
    from vla_memory.grpo.trainer import GRPOConfig, GRPOTrainer
    from vla_memory.qwen_subgoal.model import QwenSubgoalPolicy

    from openpi_client import websocket_client_policy as _wp  # type: ignore
    from utils import TASK_NAME_LIST, TASK_WITH_VIDEO_DEMO  # type: ignore

    # 1. Resolve the SFT adapter path. Accept either a swift output_dir
    #    (resolve to latest v*/checkpoint-*) or a specific checkpoint dir.
    #    Empty/None => cold-start baseline (no SFT warmup): pass None so
    #    QwenSubgoalPolicy builds a fresh LoRA on base Qwen (has_reference=False).
    if not args.sft_adapter_path:
        resolved_adapter = None
        print("GRPO COLD-START: no SFT adapter — fresh LoRA on base Qwen "
              "(no reference; keep --kl-beta 0).", flush=True)
    else:
        adapter_dir = Path(args.sft_adapter_path)
        if adapter_dir.name.startswith("checkpoint-"):
            resolved_adapter = str(adapter_dir)
        else:
            versioned_runs = sorted(adapter_dir.glob("v*-*"))
            if not versioned_runs:
                raise FileNotFoundError(
                    f"No versioned swift run dir under {adapter_dir} — "
                    "did SFT complete? Pass --sft-adapter-path to override."
                )
            checkpoints = list(versioned_runs[-1].glob("checkpoint-*"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint under {versioned_runs[-1]}")
            resolved_adapter = str(
                max(checkpoints, key=lambda p: int(p.name.split("-", 1)[1]))
            )
        print(f"GRPO loading SFT adapter from {resolved_adapter}", flush=True)

    # 2. Build the policy (SFT'd Qwen + LoRA, reference adapter from same dir).
    policy = QwenSubgoalPolicy(
        adapter_init_path=resolved_adapter,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

    # 3. Build rollout factory. RolloutWorker now lives entirely in this
    #    process (micromamba env), no bridging needed.
    task_list = args.only_tasks.split(",") if args.only_tasks else list(TASK_NAME_LIST)

    def rollout_factory(state):
        env_runner = EnvRunner(
            env_id=state.task_name,
            video_save_dir=str(Path(args.output_dir) / "rollout_videos"),
            max_steps=args.rollout_max_steps,
            dataset="train",
        )
        client = _wp.MMEVLAWebsocketClientPolicy("127.0.0.1", args.port)
        return RolloutWorker(
            env_runner=env_runner,
            policy_client=client,
            obs_horizon=16,
            max_steps=args.rollout_max_steps,
            use_history=False,
            subgoal_type=args.subgoal_type,
            decision_warm_cap=args.decision_warm_cap,
            n_key_frames=args.n_key_frames,
            n_recent_frames=args.n_recent_frames,
            reveal_window=args.reveal_window,
            n_candidate_frames=args.n_candidate_frames,
            max_keyframes=args.max_keyframes,
        )

    dataset = StateDataset.from_task_list(
        task_names=task_list,
        episodes_per_task=args.episodes_per_task,
        tasks_with_video_demo=list(TASK_WITH_VIDEO_DEMO),
        picks_per_episode=args.picks_per_episode,
        streaming=args.streaming_memory,
    )

    cfg = GRPOConfig(
        num_steps=args.num_steps,
        save_every=args.save_every,
        batch_states=args.batch_states,
        group_size=args.group_size,
        kl_beta=args.kl_beta,
        learning_rate=args.learning_rate,
        sample_temperature=args.sample_temperature,
        rollouts_per_subgoal=args.rollouts_per_subgoal,
        dynamic_sampling=args.dynamic_sampling,
        dynamic_sampling_max_multiplier=args.dynamic_sampling_max_multiplier,
        normalize_advantage_std=args.normalize_advantage_std,
        seed_match_group=args.seed_match_group,
        rollout_max_steps=args.rollout_max_steps,
        subgoal_type=args.subgoal_type,
        joint_selection=args.joint_selection,
        streaming_memory=args.streaming_memory,
        streaming_max_picks=args.streaming_max_picks,
        n_candidate_frames=args.n_candidate_frames,
        max_keyframes=args.max_keyframes,
        snapshot_branching=args.snapshot_branching,
        output_dir=args.output_dir,
        seed=args.seed,
        debug_subgoals=args.debug_subgoals,
    )

    trainer = GRPOTrainer(
        policy=policy,
        state_dataset=dataset,
        rollout_factory=rollout_factory,
        reward_cfg=RewardConfig(),
        cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
