"""RLOO (REINFORCE Leave-One-Out) trainer for the Qwen3-VL subgoal predictor.

RLOO vs GRPO:
  - GRPO: A_k = r_k - mean(r)             (subtract full group mean)
  - RLOO: A_k = r_k - mean(r_{j≠k})       (leave-one-out mean as baseline)
         = r_k - (sum(r) - r_k) / (K - 1)

With K=8 the difference is small, but RLOO is theoretically unbiased: each
sample's baseline is formed from the other K-1 independent rollouts rather
than including its own reward. This matters when K is small or when rewards
are highly variable across a batch.

RLOO also avoids GRPO's group-degenerate filtering logic — every state has a
valid leave-one-out baseline as long as K >= 2, so dynamic sampling is not
needed (though it's still supported for compatibility).

All other infrastructure (rollout worker, reward, dynamic sampling, KL) is
shared directly with GRPO via import, keeping the delta to the advantage
computation only.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..qwen_subgoal.model import QwenSubgoalPolicy, SampleResult
from ..grpo.reward import RewardConfig, compute_reward
from ..grpo.rollout import RolloutWorker
from ..grpo.state_dataset import StateDataset, StateSample
from ..grpo.trainer import GRPOTrainer, GRPOConfig

Trajectory = List[SampleResult]


@dataclasses.dataclass
class RLOOConfig:
    """Mirrors GRPOConfig; only the advantage estimator changes."""
    num_steps: int = 200
    batch_states: int = 4
    group_size: int = 8             # K — must be >= 2 for leave-one-out
    kl_beta: float = 0.0
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    sample_temperature: float = 1.0
    sample_top_p: float = 0.95
    max_new_tokens: int = 128
    rollout_max_steps: int = 200
    subgoal_type: str = "grounded_subgoal"
    use_history: bool = False
    normalize_advantage_std: bool = False
    loss_token_normalizer: int = 0
    rollouts_per_subgoal: int = 1
    seed_match_group: bool = True
    dynamic_sampling: bool = False  # RLOO has a valid baseline for every state,
                                    # so dynamic sampling is off by default.
    dynamic_sampling_max_multiplier: int = 3
    advantage_std_floor: float = 1e-6
    log_every: int = 1
    save_every: int = 25
    output_dir: str = "runs/rloo"
    seed: int = 0
    debug_subgoals: bool = False
    n_key_frames: int = 4
    n_recent_frames: int = 2
    reveal_window: int = 64
    decision_warm_cap: int = 150
    n_candidate_frames: int = 12
    max_keyframes: int = 4


def _rloo_advantages(rewards: np.ndarray) -> np.ndarray:
    """Leave-one-out advantage: A_k = r_k - mean(r_{j≠k})."""
    K = len(rewards)
    if K < 2:
        return rewards - rewards.mean()
    total = rewards.sum()
    baselines = (total - rewards) / (K - 1)
    return rewards - baselines


class RLOOTrainer(GRPOTrainer):
    """GRPO with leave-one-out advantage instead of group mean.

    Subclasses GRPOTrainer and overrides only _accumulate_gradients to swap
    the advantage estimator. Everything else — rollout collection, logging,
    checkpointing, KL regularization — is inherited unchanged.
    """

    def __init__(
        self,
        policy: QwenSubgoalPolicy,
        state_dataset: StateDataset,
        rollout_factory: Callable[[StateSample], RolloutWorker],
        reward_cfg: RewardConfig = RewardConfig(),
        cfg: RLOOConfig = RLOOConfig(),
    ) -> None:
        # Convert RLOOConfig → GRPOConfig so the parent __init__ works unchanged.
        grpo_cfg = GRPOConfig(
            num_steps=cfg.num_steps,
            batch_states=cfg.batch_states,
            group_size=cfg.group_size,
            kl_beta=cfg.kl_beta,
            learning_rate=cfg.learning_rate,
            grad_clip=cfg.grad_clip,
            sample_temperature=cfg.sample_temperature,
            sample_top_p=cfg.sample_top_p,
            max_new_tokens=cfg.max_new_tokens,
            rollout_max_steps=cfg.rollout_max_steps,
            subgoal_type=cfg.subgoal_type,
            use_history=cfg.use_history,
            normalize_advantage_std=cfg.normalize_advantage_std,
            loss_token_normalizer=cfg.loss_token_normalizer,
            rollouts_per_subgoal=cfg.rollouts_per_subgoal,
            seed_match_group=cfg.seed_match_group,
            dynamic_sampling=cfg.dynamic_sampling,
            dynamic_sampling_max_multiplier=cfg.dynamic_sampling_max_multiplier,
            advantage_std_floor=cfg.advantage_std_floor,
            log_every=cfg.log_every,
            save_every=cfg.save_every,
            output_dir=cfg.output_dir,
            seed=cfg.seed,
            debug_subgoals=cfg.debug_subgoals,
            n_candidate_frames=cfg.n_candidate_frames,
            max_keyframes=cfg.max_keyframes,
        )

        # Patch wandb project name before parent __init__ uses it.
        os.environ.setdefault("WANDB_PROJECT", "vla-memory-rloo")
        super().__init__(
            policy=policy,
            state_dataset=state_dataset,
            rollout_factory=rollout_factory,
            reward_cfg=reward_cfg,
            cfg=grpo_cfg,
        )
        self._rloo_cfg = cfg

        # Override the log path so RLOO runs don't clobber GRPO logs.
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.out_dir / "train_log.jsonl"

    def _accumulate_gradients(
        self,
        groups: List[List[Trajectory]],
        rewards: List[List[float]],
    ) -> dict:
        """Identical to GRPOTrainer._accumulate_gradients but uses RLOO advantage."""
        n_cands = float(sum(len(g) for g in groups))
        cand_scale = 1.0 / max(n_cands, 1.0)
        token_norm = float(self.cfg.loss_token_normalizer or self.cfg.max_new_tokens)
        use_kl = self.policy.has_reference and self.cfg.kl_beta > 0.0

        pg_sum = 0.0
        kl_sum = 0.0
        n_tokens = 0
        per_group_kls: List[float] = []

        for group, group_rewards in zip(groups, rewards):
            r = np.asarray(group_rewards, dtype=np.float32)
            # --- RLOO advantage (the only line that differs from GRPOTrainer) ---
            adv = _rloo_advantages(r)
            if self._rloo_cfg.normalize_advantage_std and r.std() > 1e-8:
                adv = adv / (r.std() + 1e-8)

            for traj_idx, traj in enumerate(group):
                a = float(adv[traj_idx])
                if a == 0.0 and not use_kl:
                    continue
                for cand in traj:
                    scored = self._score_generation(cand, a, use_kl, token_norm, cand_scale)
                    if scored is None:
                        print(f"[rloo] WARN: skipping empty generation (traj={traj_idx})")
                        continue
                    pg_val, kl_val, n_tok = scored
                    pg_sum += pg_val
                    n_tokens += n_tok
                    if use_kl:
                        kl_sum += kl_val
                        per_group_kls.append(kl_val)

        adv_abs: List[float] = []
        for grp_rewards in rewards:
            arr = np.asarray(grp_rewards, dtype=np.float32)
            adv_abs.extend(np.abs(_rloo_advantages(arr)).tolist())

        return {
            "pg_loss": pg_sum,
            "kl_loss": kl_sum,
            "loss": pg_sum + self.cfg.kl_beta * kl_sum if use_kl else pg_sum,
            "n_generated_tokens": n_tokens,
            "mean_per_group_kl": float(np.mean(per_group_kls)) if per_group_kls else 0.0,
            "mean_abs_advantage": float(np.mean(adv_abs)) if adv_abs else 0.0,
        }


__all__ = ["RLOOTrainer", "RLOOConfig"]
