"""Reward function for GRPO subgoal training.

The reward is the env's **task-completion fraction** — a dense signal in [0, 1]
derived from RoboMME's sequential-task tracker (``current_task_index /
len(task_list)``; see ``rollout.py:_progress_estimate``). A full success is
fraction 1.0; a rollout that finishes 2 of 3 subtasks scores 0.667.

This replaces the original sparse 0/1 success reward. Pure-binary reward made
most GRPO groups degenerate (all-fail or all-success → zero group advantage →
zero gradient); a graded fraction gives within-group variance far more often,
which is what the policy-gradient signal needs (DAPO arXiv:2503.14476;
"Gradient Starvation in Binary-Reward GRPO").
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class RewardConfig:
    success_bonus: float = 0.0   # extra reward added on full success, on top of fraction 1.0
    error_penalty: float = 0.0   # reward when the sim raises (0.0 = neutral; negative to discourage)


def compute_reward(
    status: str,
    progress: float,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """Map a rollout outcome to a scalar reward.

    Args:
        status: terminal status from ``env.info`` — one of
            ``"success" | "fail" | "timeout" | "error" | "unknown"``.
        progress: task-completion fraction in [0, 1] at rollout end
            (``current_task_index / len(task_list)``); 1.0 iff all subtasks done.
        cfg: reward weights.

    Returns:
        Scalar reward in [error_penalty, 1.0 + success_bonus].
    """
    if status == "error":
        return cfg.error_penalty

    reward = max(0.0, min(1.0, float(progress)))
    if status == "success":
        # Guard against a progress signal that didn't reach 1.0 on the success
        # step (e.g. the tracker latching a frame late), and add the bonus.
        reward = 1.0 + cfg.success_bonus
    return reward


__all__ = ["RewardConfig", "compute_reward"]
