"""GRPO trainer for the Qwen3-VL subgoal predictor.

Standard GRPO loop:

    for step in range(num_steps):
        states = sample_minibatch(state_dataset, B)
        for s in states:
            cands = policy.sample_subgoals(s, k=K)      # K candidates per state
            rewards = [rollout(s, c) for c in cands]    # K rewards
            advantages = group_normalize(rewards)
        loss = -E[Â · log p_pol(sg|s)] + β · KL(pol ‖ ref)
        loss.backward(); opt.step()

We keep this single-process and synchronous so the math is auditable. Going
multi-GPU / distributed is a follow-up — first we want to see the reward curve
move on at least one task.
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
from .reward import RewardConfig, compute_reward
from .rollout import RolloutResult, RolloutWorker
from .state_dataset import StateDataset, StateSample


@dataclasses.dataclass
class GRPOConfig:
    num_steps: int = 200
    batch_states: int = 4              # B — non-degenerate groups per gradient step
    group_size: int = 8                # K — candidate subgoals per state
    kl_beta: float = 0.0               # KL vs SFT reference; 0 = no KL (DAPO / TRL default)
    learning_rate: float = 1e-4        # LoRA RL wants ~10x SFT lr ("LoRA Without Regret")
    grad_clip: float = 1.0
    sample_temperature: float = 1.0
    sample_top_p: float = 0.95
    max_new_tokens: int = 64
    rollout_max_steps: int = 200
    rollout_obs_horizon: int = 16
    subgoal_type: str = "simple_subgoal"
    use_history: bool = False
    # --- advantage / loss shaping ---
    normalize_advantage_std: bool = False  # Dr.GRPO: don't divide by group std (it amplifies
                                           # near-degenerate groups). Use Â = r - mean(r).
    loss_token_normalizer: int = 0         # 0 -> max_new_tokens. Dr.GRPO constant normalizer:
                                           # per-token loss WITHOUT the per-sequence-length mean.
    # --- variance reduction ---
    rollouts_per_subgoal: int = 1          # >1 averages reward per subgoal to cut the frozen
                                           # pi0.5 flow-sampling noise (server seed isn't client-set).
    seed_match_group: bool = True          # pin the env reset seed across a group's rollouts.
    # --- dynamic sampling (DAPO) ---
    dynamic_sampling: bool = True          # resample states until the batch has B non-degenerate groups.
    dynamic_sampling_max_multiplier: int = 3  # cap group attempts at B * this per step (bounds cost).
    advantage_std_floor: float = 1e-6      # group with reward std <= floor is degenerate (no signal).
    log_every: int = 1
    save_every: int = 25
    output_dir: str = "runs/grpo"
    seed: int = 0


class GRPOTrainer:
    """Group-Relative Policy Optimization for a multimodal subgoal predictor."""

    def __init__(
        self,
        policy: QwenSubgoalPolicy,
        state_dataset: StateDataset,
        rollout_factory: Callable[[StateSample], RolloutWorker],
        reward_cfg: RewardConfig = RewardConfig(),
        cfg: GRPOConfig = GRPOConfig(),
    ) -> None:
        self.policy = policy
        self.dataset = state_dataset
        self.rollout_factory = rollout_factory
        self.reward_cfg = reward_cfg
        self.cfg = cfg

        if cfg.group_size < 2:
            raise ValueError(
                "GRPO needs a group-relative baseline: group_size must be >= 2, "
                f"got {cfg.group_size}. With K=1 every group has zero reward "
                "variance and dynamic sampling drops it (no gradient)."
            )

        self.optimizer = torch.optim.AdamW(
            policy.trainable_parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

        self._rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.out_dir / "train_log.jsonl"

        # wandb logging is optional — if `wandb.init()` fails for any reason
        # (no API key, network blip, package missing), we keep going with
        # JSONL-only logging. Training reliability shouldn't depend on telemetry.
        self._wandb_run = None
        try:
            import wandb  # type: ignore

            self._wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "vla-memory-grpo"),
                name=os.environ.get("WANDB_RUN_NAME"),
                config={
                    **dataclasses.asdict(cfg),
                    "reward_config": dataclasses.asdict(reward_cfg),
                    "dataset_size": len(state_dataset),
                },
                dir=str(self.out_dir),
                reinit=False,
            )
            self._wandb = wandb
            print(f"[grpo] wandb run: {self._wandb_run.url}", flush=True)
        except Exception as exc:
            print(f"[grpo] wandb disabled ({exc!r}); JSONL logging only", flush=True)
            self._wandb = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        try:
            self._train_loop()
        finally:
            if self._wandb_run is not None:
                try:
                    self._wandb.finish()
                except Exception as exc:
                    print(f"[grpo] wandb.finish() failed: {exc!r}", flush=True)

    def _train_loop(self) -> None:
        for step in range(self.cfg.num_steps):
            t0 = time.time()
            # DAPO dynamic sampling: roll out groups until B of them carry a
            # learning signal (non-zero reward variance).
            all_groups, all_rewards, n_attempts, n_dropped = self._collect_groups(
                self.cfg.batch_states, step
            )

            # Per-candidate backward — one forward+backward at a time so we
            # never hold more than one Qwen3-VL-4B activation graph in memory.
            # Gradients accumulate in ``param.grad`` across candidates; we
            # call ``optimizer.step()`` once after the whole minibatch (one step
            # per fresh batch → ratio == 1, no PPO clipping needed).
            self.optimizer.zero_grad()
            metrics = self._accumulate_gradients(all_groups, all_rewards)
            # Only step if some candidate actually produced gradient. When every
            # group is degenerate (all-equal rewards), dynamic sampling falls back
            # to a single zero-advantage group that contributes nothing; stepping
            # then is a wasted, misleading "trained" update.
            stepped = metrics["n_generated_tokens"] > 0
            if stepped:
                # Re-activate the trainable 'policy' adapter before clip/step: a
                # KL reference forward leaves 'reference' active, and PEFT would
                # then have grad-clip / the optimizer touch the frozen adapter.
                self.policy.activate_policy()
                torch.nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), self.cfg.grad_clip)
                self.optimizer.step()
            else:
                print(
                    f"[grpo] step {step}: all groups degenerate (no gradient) — "
                    "skipping optimizer step",
                    flush=True,
                )
            metrics["optimizer_stepped"] = int(stepped)

            dt = time.time() - t0
            flat = [r for grp in all_rewards for r in grp]
            metrics.update(
                step=step,
                wall_seconds=dt,
                mean_reward=float(np.mean(flat)) if flat else 0.0,
                mean_reward_std=float(
                    np.mean([np.std(r) if len(r) > 1 else 0.0 for r in all_rewards])
                ) if all_rewards else 0.0,
                n_groups_used=len(all_groups),
                n_groups_dropped=n_dropped,
                n_group_attempts=n_attempts,
            )

            if step % self.cfg.log_every == 0:
                print(json.dumps(metrics))
            with open(self._log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            if self._wandb is not None:
                try:
                    # Use the trainer's own step counter so wandb's x-axis
                    # aligns with the GRPO step (not wandb's internal counter).
                    self._wandb.log(metrics, step=step)
                except Exception as exc:
                    print(f"[grpo] wandb.log failed at step {step}: {exc!r}", flush=True)

            if step > 0 and step % self.cfg.save_every == 0:
                self._save_checkpoint(step)

        self._save_checkpoint(self.cfg.num_steps)

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def _collect_groups(
        self, n_target: int, step: int
    ) -> tuple[List[List[SampleResult]], List[List[float]], int, int]:
        """Roll out groups until ``n_target`` of them are non-degenerate.

        DAPO dynamic sampling: a group whose rewards are all equal has zero group
        advantage and contributes no gradient, so we drop it and sample another
        state. Attempts are capped at ``n_target * dynamic_sampling_max_multiplier``
        to bound rollout cost; on hitting the cap we proceed with whatever
        non-degenerate groups we have (or the last group if every attempt was
        degenerate — the caller then skips the optimizer step).
        """
        groups: List[List[SampleResult]] = []
        rewards: List[List[float]] = []
        n_dropped = 0
        attempts = 0
        max_attempts = n_target * max(1, self.cfg.dynamic_sampling_max_multiplier)
        last: Optional[tuple] = None
        while len(groups) < n_target and attempts < max_attempts:
            state = self._sample_states(1)[0]
            attempts += 1
            cands, grp_rewards, _ = self._rollout_group(state, step)
            last = (cands, grp_rewards)
            degenerate = float(np.std(grp_rewards)) <= self.cfg.advantage_std_floor
            if self.cfg.dynamic_sampling and degenerate:
                n_dropped += 1
                continue
            groups.append(cands)
            rewards.append(grp_rewards)
        if not groups and last is not None:
            groups.append(last[0])
            rewards.append(last[1])
        return groups, rewards, attempts, n_dropped

    def _sample_states(self, n: int) -> List[StateSample]:
        idxs = self._rng.sample(range(len(self.dataset)), k=min(n, len(self.dataset)))
        return [self.dataset[i] for i in idxs]

    def _rollout_group(
        self, state: StateSample, step: int
    ) -> tuple[List[SampleResult], List[float], None]:
        """Sample K subgoals for ``state`` and roll each out to a (dense) reward.

        Each subgoal is rolled out ``rollouts_per_subgoal`` times and the reward
        averaged — the frozen pi0.5 is a stochastic flow policy, so a single
        rollout is a noisy estimate of a subgoal's quality. All rollouts in the
        group share one env seed (``seed_match_group``) so reward differences
        track the subgoal rather than the scene.
        """
        worker = self.rollout_factory(state)
        # One seed per (run seed, episode, step): all K candidates + their repeats
        # in this group share a scene (within-group common-random-numbers), while
        # the same episode drawn on a later step gets a fresh scene — avoids
        # overfitting to one frozen layout per episode. None → upstream
        # re-randomizes on every reset (no CRN).
        group_seed = None
        if self.cfg.seed_match_group:
            group_seed = ((self.cfg.seed * 1_000_003 + state.episode_id) * 9_973 + step) % 2_147_483_647

        try:
            # peek_init constructs the env once and keeps it open across the K
            # rollouts of this group — avoids paying the asset-loading cost
            # K+1 times per state.
            init_image, task_goal = worker.peek_init(state.episode_id, seed=group_seed)

            cands = self.policy.sample_subgoals(
                image=init_image,
                task_goal=task_goal,
                history_subgoals=[],
                k=self.cfg.group_size,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.sample_temperature,
                top_p=self.cfg.sample_top_p,
                subgoal_type=self.cfg.subgoal_type,
                has_video_demo=state.has_video_demo,
            )

            rewards: List[float] = []
            for c in cands:
                trials: List[float] = []
                for _ in range(max(1, self.cfg.rollouts_per_subgoal)):
                    result = worker.rollout(
                        episode_id=state.episode_id,
                        sampled_subgoal=c.text,
                        seed=group_seed,
                    )
                    trials.append(
                        compute_reward(
                            status=result.success_flag,
                            progress=result.progress,
                            cfg=self.reward_cfg,
                        )
                    )
                rewards.append(float(np.mean(trials)))
        finally:
            worker.close()
        return cands, rewards, None

    # ------------------------------------------------------------------
    # Loss — per-candidate backward to keep memory bounded.
    # ------------------------------------------------------------------

    def _accumulate_gradients(
        self,
        groups: List[List[SampleResult]],
        rewards: List[List[float]],
    ) -> dict:
        """GRPO objective applied one candidate at a time (bounded memory).

        Per candidate: ``L = −Â · (Σ_t log π_t)/C  [+ β · KL/C]``, ``backward()``
        to free the graph, accumulate ``param.grad`` across candidates, one
        ``optimizer.step()`` in the caller. ``C`` is a constant token normalizer
        (Dr.GRPO) — per-token loss WITHOUT the per-sequence-length mean, so long
        subgoals aren't down-weighted.

        CRITICAL ORDERING: when a KL term is used, the reference forward runs
        FIRST (under no_grad) and the policy forward LAST, so the trainable
        'policy' adapter is the active one at ``backward()``. PEFT's
        ``set_adapter`` flips ``requires_grad=False`` on the non-active adapter,
        and a leaf that is ``requires_grad=False`` at backward() time receives NO
        gradient. The original order (policy forward, then reference forward,
        then backward) silently zeroed the policy gradient — the v0 no-op. See
        ``feedback_peft_set_adapter_zeroes_grad``.
        """
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
            adv = r - r.mean()
            if self.cfg.normalize_advantage_std and r.std() > 1e-8:
                adv = adv / (r.std() + 1e-8)

            for cand_idx, cand in enumerate(group):
                if cand.token_ids.numel() == 0:
                    # Empty generation (EOS at position 0): logp over a (0,)
                    # tensor is NaN and would poison every parameter. Skip it.
                    print(f"[grpo] WARN: skipping empty subgoal candidate (idx={cand_idx})")
                    continue

                a = float(adv[cand_idx])
                if a == 0.0 and not use_kl:
                    # Zero advantage and no KL → zero gradient; skip the
                    # forward/backward entirely to save compute.
                    continue

                ref_logp = None
                if use_kl:
                    # Reference FIRST and under no_grad, so 'policy' is the
                    # active (trainable) adapter at backward() below.
                    with torch.no_grad():
                        ref_logp = self.policy.policy_logprobs(
                            prompt_input_ids=cand.prompt_input_ids,
                            prompt_attention_mask=cand.prompt_attention_mask,
                            gen_token_ids=cand.token_ids,
                            pixel_values=cand.pixel_values,
                            image_grid_thw=cand.image_grid_thw,
                            adapter="reference",
                            gradient_enabled=False,
                        )

                policy_logp = self.policy.policy_logprobs(
                    prompt_input_ids=cand.prompt_input_ids,
                    prompt_attention_mask=cand.prompt_attention_mask,
                    gen_token_ids=cand.token_ids,
                    pixel_values=cand.pixel_values,
                    image_grid_thw=cand.image_grid_thw,
                    adapter="policy",
                    gradient_enabled=True,
                )

                pg_term = -a * policy_logp.sum() / token_norm * cand_scale
                cand_loss = pg_term

                if use_kl:
                    # Quadratic per-token KL surrogate (leading-order of TRL's k3
                    # estimator), constant-normalized like the PG term. policy_logp
                    # carries gradient; ref_logp does not.
                    kl_term = (policy_logp - ref_logp).pow(2).sum() / token_norm * cand_scale
                    cand_loss = cand_loss + self.cfg.kl_beta * kl_term
                    kl_val = float(kl_term.detach().cpu().item())
                    kl_sum += kl_val
                    per_group_kls.append(kl_val)

                cand_loss.backward()
                pg_sum += float(pg_term.detach().cpu().item())
                n_tokens += int(policy_logp.shape[0])

        adv_abs: List[float] = []
        for grp_rewards in rewards:
            arr = np.asarray(grp_rewards, dtype=np.float32)
            adv_abs.extend(np.abs(arr - arr.mean()).tolist())

        return {
            "pg_loss": pg_sum,
            "kl_loss": kl_sum,
            "loss": pg_sum + self.cfg.kl_beta * kl_sum if use_kl else pg_sum,
            "n_generated_tokens": n_tokens,
            "mean_per_group_kl": float(np.mean(per_group_kls)) if per_group_kls else 0.0,
            "mean_abs_advantage": float(np.mean(adv_abs)) if adv_abs else 0.0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = self.out_dir / f"step{step}"
        self.policy.save_policy_adapter(str(ckpt_dir))
        print(f"[grpo] saved checkpoint to {ckpt_dir}")


__all__ = ["GRPOTrainer", "GRPOConfig"]
