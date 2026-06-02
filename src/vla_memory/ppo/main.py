"""PPO entry point — runs inside the micromamba `robomme` env.

Launched by modal_train.py::ppo via:
    micromamba run -n robomme python -u src/vla_memory/ppo/main.py --port N ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "/app/examples/robomme")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO main loop (micromamba env)")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sft-adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-states", type=int, default=4)
    parser.add_argument("--rollouts-per-state", type=int, default=8)
    parser.add_argument("--coeff-vf", type=float, default=0.5)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--rollouts-per-subgoal", type=int, default=1)
    parser.add_argument("--seed-match-group", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollout-max-steps", type=int, default=200)
    parser.add_argument("--only-tasks", default="ButtonUnmask")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--subgoal-type", default="grounded_subgoal")
    parser.add_argument("--n-key-frames", type=int, default=4)
    parser.add_argument("--n-recent-frames", type=int, default=2)
    parser.add_argument("--reveal-window", type=int, default=64)
    parser.add_argument("--decision-warm-cap", type=int, default=150)
    parser.add_argument("--n-candidate-frames", type=int, default=12)
    parser.add_argument("--max-keyframes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-subgoals", action="store_true")
    args = parser.parse_args()

    print(
        "[ppo] config: " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())),
        flush=True,
    )

    import torch

    from vla_memory.grpo.env_runner import EnvRunner
    from vla_memory.grpo.reward import RewardConfig
    from vla_memory.grpo.rollout import RolloutWorker
    from vla_memory.grpo.state_dataset import StateDataset
    from vla_memory.ppo.trainer import PPOConfig, PPOTrainer
    from vla_memory.qwen_subgoal.model import QwenSubgoalPolicy

    from openpi_client import websocket_client_policy as _wp  # type: ignore
    from utils import TASK_NAME_LIST, TASK_WITH_VIDEO_DEMO  # type: ignore

    # Resolve SFT adapter path (same logic as grpo/main.py).
    adapter_dir = Path(args.sft_adapter_path)
    if adapter_dir.name.startswith("checkpoint-"):
        resolved_adapter = str(adapter_dir)
    else:
        versioned_runs = sorted(adapter_dir.glob("v*-*"))
        if not versioned_runs:
            raise FileNotFoundError(f"No versioned swift run dir under {adapter_dir}")
        checkpoints = list(versioned_runs[-1].glob("checkpoint-*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint under {versioned_runs[-1]}")
        resolved_adapter = str(
            max(checkpoints, key=lambda p: int(p.name.split("-", 1)[1]))
        )
    print(f"PPO loading SFT adapter from {resolved_adapter}", flush=True)

    policy = QwenSubgoalPolicy(
        adapter_init_path=resolved_adapter,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

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
        )

    dataset = StateDataset.from_task_list(
        task_names=task_list,
        episodes_per_task=args.episodes_per_task,
        tasks_with_video_demo=list(TASK_WITH_VIDEO_DEMO),
    )

    cfg = PPOConfig(
        num_steps=args.num_steps,
        batch_states=args.batch_states,
        rollouts_per_state=args.rollouts_per_state,
        coeff_vf=args.coeff_vf,
        kl_beta=args.kl_beta,
        learning_rate=args.learning_rate,
        sample_temperature=args.sample_temperature,
        rollouts_per_subgoal=args.rollouts_per_subgoal,
        seed_match_group=args.seed_match_group,
        rollout_max_steps=args.rollout_max_steps,
        subgoal_type=args.subgoal_type,
        n_key_frames=args.n_key_frames,
        n_recent_frames=args.n_recent_frames,
        reveal_window=args.reveal_window,
        decision_warm_cap=args.decision_warm_cap,
        n_candidate_frames=args.n_candidate_frames,
        max_keyframes=args.max_keyframes,
        output_dir=args.output_dir,
        seed=args.seed,
        debug_subgoals=args.debug_subgoals,
    )

    trainer = PPOTrainer(
        policy=policy,
        state_dataset=dataset,
        rollout_factory=rollout_factory,
        reward_cfg=RewardConfig(),
        cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
