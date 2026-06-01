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

# A trajectory is the list of VLM generations scored by ONE episode reward:
# one-shot = [subtask]; joint select-then-use = [selection, use]. GRPO sums logp
# over a trajectory's generations with the trajectory's shared advantage.
Trajectory = List[SampleResult]


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
    max_new_tokens: int = 128          # MemER target is a JSON object, not a bare phrase
    rollout_max_steps: int = 200
    rollout_obs_horizon: int = 16
    subgoal_type: str = "simple_subgoal"
    use_history: bool = False
    # --- joint keyframe-selection (JOINT_MEMORY_DESIGN.md) ---
    joint_selection: bool = False          # select-then-use: SELECT call picks keyframes from the
                                           # candidate window, USE call acts on ONLY those. Trains
                                           # selection + subtask jointly. False = one-shot path.
    n_candidate_frames: int = 12           # SELECT-call candidate window breadth
    max_keyframes: int = 4                 # cap on kept keyframes the USE call sees
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
    debug_subgoals: bool = False       # print each sampled subgoal's text + token
                                       # count — confirms generations terminate.


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
            grad_norm = 0.0
            if stepped:
                # Re-activate the trainable 'policy' adapter before clip/step: a
                # KL reference forward leaves 'reference' active, and PEFT would
                # then have grad-clip / the optimizer touch the frozen adapter.
                self.policy.activate_policy()
                # clip_grad_norm_ returns the total grad norm BEFORE clipping —
                # capture it. This is THE diagnostic for "is the gradient
                # vanishing": pg_loss is ~0 by construction (Σ advantage = 0
                # within every group), so the loss value is NOT a gradient
                # proxy. A grad_norm collapsing toward 0 confirms a dead/uninformative
                # gradient; a healthy nonzero norm means the signal is the issue.
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.trainable_parameters(), self.cfg.grad_clip
                    )
                )
                self.optimizer.step()
            else:
                print(
                    f"[grpo] step {step}: all groups degenerate (no gradient) — "
                    "skipping optimizer step",
                    flush=True,
                )
            metrics["optimizer_stepped"] = int(stepped)
            metrics["grad_norm"] = grad_norm

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
    ) -> tuple[List[List[Trajectory]], List[List[float]], int, int]:
        """Roll out groups until ``n_target`` of them are non-degenerate.

        DAPO dynamic sampling: a group whose rewards are all equal has zero group
        advantage and contributes no gradient, so we drop it and sample another
        state. Attempts are capped at ``n_target * dynamic_sampling_max_multiplier``
        to bound rollout cost; on hitting the cap we proceed with whatever
        non-degenerate groups we have (or the last group if every attempt was
        degenerate — the caller then skips the optimizer step).
        """
        groups: List[List[Trajectory]] = []
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
            degenerate = (
                not grp_rewards
                or float(np.std(grp_rewards)) <= self.cfg.advantage_std_floor
            )
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
    ) -> tuple[List[Trajectory], List[float], None]:
        """Roll out a group for ``state`` → K TRAJECTORIES + their episode rewards.

        Dispatches on ``cfg.joint_selection``: one-shot (one VLM call → subtask) or
        joint select-then-use (SELECT call picks keyframes → USE call acts on only
        those). All rollouts in the group share one env seed (``seed_match_group``),
        so reward differences track the VLM output, not the scene.
        """
        worker = self.rollout_factory(state)
        # One seed per (run seed, episode, step): within-group common-random-numbers
        # so the cube→container layout is identical across candidates; a later step
        # on the same episode gets a fresh scene (avoids overfitting one layout).
        group_seed = None
        if self.cfg.seed_match_group:
            group_seed = ((self.cfg.seed * 1_000_003 + state.episode_id) * 9_973 + step) % 2_147_483_647

        try:
            # peek warms up with the oracle to the post-occlusion decision point and
            # returns the reveal keyframes + recent context (+ the broad candidate
            # window for joint selection). Each rollout below rebuilds the env from
            # the same seed (cube layout matches) — see _ensure_env on why we rebuild.
            dp = worker.peek_at_decision_point(state.episode_id, seed=group_seed)
            if dp.terminated_early and self.cfg.debug_subgoals:
                print(
                    f"[grpo][debug] step={step} ep={state.episode_id} warm-up "
                    f"terminated early ({dp.success_flag}) after {dp.warm_steps} "
                    "steps — no memory decision; group should be degenerate.",
                    flush=True,
                )
            if self.cfg.joint_selection:
                trajectories, rewards = self._joint_group(dp, state, worker, group_seed, step)
            else:
                trajectories, rewards = self._oneshot_group(dp, state, worker, group_seed, step)
        finally:
            worker.close()
        return trajectories, rewards, None

    def _score_rollout(self, subtask: str, state: StateSample, worker, group_seed) -> float:
        """Execute ``subtask`` on the low-level policy → dense reward.

        Blank subtask → 0 without a rollout (π0.5 with no subgoal can't do better;
        the negative advantage still trains the policy away from it). Averages
        ``rollouts_per_subgoal`` trials to cut π0.5 flow-sampling noise.
        """
        if not subtask.strip():
            return 0.0
        trials: List[float] = []
        for _ in range(max(1, self.cfg.rollouts_per_subgoal)):
            result = worker.rollout(
                episode_id=state.episode_id, sampled_subgoal=subtask, seed=group_seed,
            )
            trials.append(compute_reward(result.success_flag, result.progress, self.reward_cfg))
        return float(np.mean(trials))

    def _oneshot_group(self, dp, state, worker, group_seed, step) -> tuple[List[Trajectory], List[float]]:
        """One VLM call → subtask; each candidate is its own length-1 trajectory."""
        cands = self.policy.sample_subgoals(
            key_frames=dp.key_frames,
            recent_frames=dp.recent_frames,
            task_goal=dp.task_goal,
            history_subgoals=dp.history_subgoals,
            k=self.cfg.group_size,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.sample_temperature,
            top_p=self.cfg.sample_top_p,
            has_video_demo=state.has_video_demo,
            debug=self.cfg.debug_subgoals,
        )
        if self.cfg.debug_subgoals:
            for ci, c in enumerate(cands):
                print(
                    f"[grpo][debug] step={step} ep={state.episode_id} cand={ci} "
                    f"ntok={c.token_ids.numel()} subtask={c.subtask!r} kf={c.keyframe_positions}",
                    flush=True,
                )
        rewards = [self._score_rollout(c.subtask, state, worker, group_seed) for c in cands]
        return [[c] for c in cands], rewards

    def _joint_group(self, dp, state, worker, group_seed, step) -> tuple[List[Trajectory], List[float]]:
        """SELECT call (pick keyframes from the candidate window) → USE call (act on
        ONLY the kept frames) → rollout. Trajectory = [selection, use]; both
        generations are trained by the episode reward, so selection + subtask are
        learned jointly (JOINT_MEMORY_DESIGN.md §3-4)."""
        from .selection import apply_selection

        sel_cands = self.policy.sample_subgoals(
            key_frames=[],
            recent_frames=dp.candidate_frames,
            task_goal=dp.task_goal,
            history_subgoals=[],   # SELECT call is "observe & select" — no completed-subtask
                                   # line; matches the SELECT SFT rows (build_memory_sft_dataset).
            k=self.cfg.group_size,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.sample_temperature,
            top_p=self.cfg.sample_top_p,
            has_video_demo=state.has_video_demo,
            debug=self.cfg.debug_subgoals,
            mode="select",   # distinct SELECT prompt/schema → emits keyframe_positions only
        )
        recent_base = [dp.current_frame] if dp.current_frame is not None else []
        trajectories: List[Trajectory] = []
        rewards: List[float] = []
        for si, sel in enumerate(sel_cands):
            kept = apply_selection(sel.keyframe_positions, dp.candidate_frames, self.cfg.max_keyframes)
            if not kept and not recent_base:
                # No frames to condition the USE call on (degenerate warm-up).
                trajectories.append([sel])
                rewards.append(0.0)
                continue
            use = self.policy.sample_subgoals(
                key_frames=kept,
                recent_frames=recent_base,
                task_goal=dp.task_goal,
                history_subgoals=dp.history_subgoals,
                k=1,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.sample_temperature,
                top_p=self.cfg.sample_top_p,
                has_video_demo=state.has_video_demo,
                debug=self.cfg.debug_subgoals,
            )[0]
            if self.cfg.debug_subgoals:
                print(
                    f"[grpo][debug] step={step} ep={state.episode_id} sel={si} "
                    f"kf={sel.keyframe_positions} kept={len(kept)}/{len(dp.candidate_frames)} "
                    f"subtask={use.subtask!r}",
                    flush=True,
                )
            rewards.append(self._score_rollout(use.subtask, state, worker, group_seed))
            trajectories.append([sel, use])
        return trajectories, rewards

    # ------------------------------------------------------------------
    # Loss — per-candidate backward to keep memory bounded.
    # ------------------------------------------------------------------

    def _accumulate_gradients(
        self,
        groups: List[List[Trajectory]],
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

            # One advantage per TRAJECTORY; every generation in it (one-shot:
            # [subtask]; joint: [selection, use]) gets that advantage, so selection
            # and subtask are trained jointly from one episode reward.
            for traj_idx, traj in enumerate(group):
                a = float(adv[traj_idx])
                if a == 0.0 and not use_kl:
                    # Zero advantage and no KL → zero gradient for the whole
                    # trajectory; skip the forward/backward to save compute.
                    continue
                for cand in traj:
                    scored = self._score_generation(cand, a, use_kl, token_norm, cand_scale)
                    if scored is None:
                        # Empty generation (EOS at position 0): logp over a (0,)
                        # tensor is NaN — skip it rather than poison every param.
                        print(f"[grpo] WARN: skipping empty generation (traj={traj_idx})")
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
            adv_abs.extend(np.abs(arr - arr.mean()).tolist())

        return {
            "pg_loss": pg_sum,
            "kl_loss": kl_sum,
            "loss": pg_sum + self.cfg.kl_beta * kl_sum if use_kl else pg_sum,
            "n_generated_tokens": n_tokens,
            "mean_per_group_kl": float(np.mean(per_group_kls)) if per_group_kls else 0.0,
            "mean_abs_advantage": float(np.mean(adv_abs)) if adv_abs else 0.0,
        }

    def _score_generation(self, cand, a, use_kl, token_norm, cand_scale):
        """Forward+backward ONE generation under advantage ``a`` (memory-bounded).

        Returns ``(pg_val, kl_val, n_tokens)`` or ``None`` if the generation is
        empty. KL ordering matters: the reference forward runs FIRST under no_grad
        so the trainable 'policy' adapter is active at ``backward()`` — PEFT zeroes
        grad on the non-active adapter (``feedback_peft_set_adapter_zeroes_grad``).
        """
        if cand.token_ids.numel() == 0:
            return None
        ref_logp = None
        if use_kl:
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
        loss = pg_term
        kl_val = 0.0
        if use_kl:
            # Quadratic per-token KL surrogate (leading-order of TRL's k3
            # estimator), constant-normalized like the PG term.
            kl_term = (policy_logp - ref_logp).pow(2).sum() / token_norm * cand_scale
            loss = loss + self.cfg.kl_beta * kl_term
            kl_val = float(kl_term.detach().cpu().item())
        loss.backward()
        return float(pg_term.detach().cpu().item()), kl_val, int(policy_logp.shape[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = self.out_dir / f"step{step}"
        self.policy.save_policy_adapter(str(ckpt_dir))
        print(f"[grpo] saved checkpoint to {ckpt_dir}")


__all__ = ["GRPOTrainer", "GRPOConfig"]
