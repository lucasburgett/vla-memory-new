"""PPO trainer (contextual-bandit variant) for the Qwen3-VL subgoal predictor.

Architecture vs GRPO:
  - GRPO baseline: group mean of K rewards → advantage = r_k - mean(r)
  - PPO baseline:  learned value function V(state) → advantage = r_k - V(state)

Since the VLM makes one decision per episode (contextual bandit), we don't need
GAE or multi-step bootstrapping. The key improvement over GRPO is that V(state)
is trained over thousands of steps and learns a personalized, lower-variance
baseline that captures per-state difficulty — the group mean from K=8 samples
can't do that.

Loss per step:
    L_pg  = -A_k · Σ_t log π(t|state) / C       (policy gradient per candidate)
    L_vf  = coeff_vf · (V(state) - mean_reward)²  (value MSE per state)
    A_k   = r_k - V(state).detach()               (advantage; stops grad to value)
    C     = max_new_tokens                         (Dr.GRPO constant token norm)

No ratio clipping: we do one on-policy update per batch (ratio ≡ 1 at update
time), so clipping adds no benefit on the first (and only) epoch. Can be added
if replay / multiple epochs are introduced later.
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

from ..qwen_subgoal.model import QwenSubgoalPolicy, SampleResult
from ..grpo.reward import RewardConfig, compute_reward
from ..grpo.rollout import RolloutWorker
from ..grpo.state_dataset import StateDataset, StateSample


@dataclasses.dataclass
class PPOConfig:
    num_steps: int = 200
    batch_states: int = 4           # states per gradient step
    rollouts_per_state: int = 8     # K rollouts per state (replaces group_size)
    coeff_vf: float = 0.5           # value loss weight
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    sample_temperature: float = 1.0
    sample_top_p: float = 0.95
    max_new_tokens: int = 128
    rollout_max_steps: int = 200
    subgoal_type: str = "grounded_subgoal"
    use_history: bool = False
    loss_token_normalizer: int = 0  # 0 → max_new_tokens (Dr.GRPO convention)
    rollouts_per_subgoal: int = 1   # >1 averages reward to cut π0.5 noise
    seed_match_group: bool = True   # pin env seed across rollouts per state
    log_every: int = 1
    save_every: int = 25
    output_dir: str = "runs/ppo"
    seed: int = 0
    debug_subgoals: bool = False
    n_key_frames: int = 4
    n_recent_frames: int = 2
    reveal_window: int = 64
    decision_warm_cap: int = 150
    n_candidate_frames: int = 12
    max_keyframes: int = 4


class PPOTrainer:
    """Actor-critic (bandit PPO) for a multimodal subgoal predictor.

    The value head in QwenSubgoalPolicy is trained jointly with the LoRA policy
    adapter. Advantage A_k = r_k - V(state).detach() provides a lower-variance
    baseline than GRPO's group mean while V is learned over many steps.
    """

    def __init__(
        self,
        policy: QwenSubgoalPolicy,
        state_dataset: StateDataset,
        rollout_factory: Callable[[StateSample], RolloutWorker],
        reward_cfg: RewardConfig = RewardConfig(),
        cfg: PPOConfig = PPOConfig(),
    ) -> None:
        self.policy = policy
        self.dataset = state_dataset
        self.rollout_factory = rollout_factory
        self.reward_cfg = reward_cfg
        self.cfg = cfg

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

        self._wandb_run = None
        try:
            import wandb  # type: ignore

            self._wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "vla-memory-ppo"),
                name=os.environ.get("WANDB_RUN_NAME"),
                config={**dataclasses.asdict(cfg), "reward_config": dataclasses.asdict(reward_cfg)},
                dir=str(self.out_dir),
                reinit=False,
            )
            self._wandb = wandb
            print(f"[ppo] wandb run: {self._wandb_run.url}", flush=True)
        except Exception as exc:
            print(f"[ppo] wandb disabled ({exc!r}); JSONL logging only", flush=True)
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
                    print(f"[ppo] wandb.finish() failed: {exc!r}", flush=True)

    def _train_loop(self) -> None:
        for step in range(self.cfg.num_steps):
            t0 = time.time()
            states = self._sample_states(self.cfg.batch_states)

            self.optimizer.zero_grad()
            metrics = self._update_step(states, step)

            stepped = metrics["n_generated_tokens"] > 0
            grad_norm = 0.0
            if stepped:
                self.policy.activate_policy()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.trainable_parameters(), self.cfg.grad_clip
                    )
                )
                self.optimizer.step()
            else:
                print(f"[ppo] step {step}: no tokens generated — skipping optimizer step", flush=True)

            metrics.update(
                step=step,
                optimizer_stepped=int(stepped),
                grad_norm=grad_norm,
                wall_seconds=time.time() - t0,
            )

            if step % self.cfg.log_every == 0:
                print(json.dumps(metrics))
            with open(self._log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            if self._wandb is not None:
                try:
                    self._wandb.log(metrics, step=step)
                except Exception as exc:
                    print(f"[ppo] wandb.log failed at step {step}: {exc!r}", flush=True)

            if step > 0 and step % self.cfg.save_every == 0:
                self._save_checkpoint(step)

        self._save_checkpoint(self.cfg.num_steps)

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def _update_step(self, states: List[StateSample], step: int) -> dict:
        token_norm = float(self.cfg.loss_token_normalizer or self.cfg.max_new_tokens)
        n_states = float(len(states))
        state_scale = 1.0 / max(n_states, 1.0)

        pg_sum = 0.0
        vf_sum = 0.0
        n_tokens = 0
        all_rewards: List[float] = []
        all_values: List[float] = []

        for state in states:
            group_seed = None
            if self.cfg.seed_match_group:
                group_seed = (
                    (self.cfg.seed * 1_000_003 + state.episode_id) * 9_973 + step
                ) % 2_147_483_647

            worker = self.rollout_factory(state)
            try:
                dp = worker.peek_at_decision_point(state.episode_id, seed=group_seed)
                cands = self.policy.sample_subgoals(
                    key_frames=dp.key_frames,
                    recent_frames=dp.recent_frames,
                    task_goal=dp.task_goal,
                    history_subgoals=dp.history_subgoals,
                    k=self.cfg.rollouts_per_state,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.sample_temperature,
                    top_p=self.cfg.sample_top_p,
                    has_video_demo=state.has_video_demo,
                    debug=self.cfg.debug_subgoals,
                )

                rewards = []
                for c in cands:
                    if not c.subtask.strip():
                        rewards.append(0.0)
                        continue
                    trials = []
                    for _ in range(max(1, self.cfg.rollouts_per_subgoal)):
                        result = worker.rollout(
                            episode_id=state.episode_id,
                            sampled_subgoal=c.subtask,
                            seed=group_seed,
                        )
                        trials.append(
                            compute_reward(result.success_flag, result.progress, self.reward_cfg)
                        )
                    rewards.append(float(np.mean(trials)))

                if self.cfg.debug_subgoals:
                    for ci, (c, r) in enumerate(zip(cands, rewards)):
                        print(
                            f"[ppo][debug] step={step} ep={state.episode_id} "
                            f"cand={ci} ntok={c.token_ids.numel()} r={r:.3f} "
                            f"subtask={c.subtask!r}",
                            flush=True,
                        )

                # Representative prompt inputs for the value forward pass
                # (all candidates share the same prompt for this state).
                first = cands[0]
                v = self.policy.get_value(
                    prompt_input_ids=first.prompt_input_ids,
                    prompt_attention_mask=first.prompt_attention_mask,
                    pixel_values=first.pixel_values,
                    image_grid_thw=first.image_grid_thw,
                )  # scalar with grad flowing to value_head

                r_mean = float(np.mean(rewards))
                all_rewards.extend(rewards)
                all_values.append(float(v.detach().item()))

                # Value loss (grad → value_head only, no LoRA gradient)
                vf_loss = self.cfg.coeff_vf * (v - r_mean) ** 2 * state_scale
                vf_loss.backward()
                vf_sum += float(vf_loss.detach().item())

                # Policy gradient per candidate (advantage uses detached V)
                v_detached = float(v.detach().item())
                cand_scale = 1.0 / max(float(len(cands)), 1.0) * state_scale
                for c, r in zip(cands, rewards):
                    if c.token_ids.numel() == 0:
                        continue
                    adv = r - v_detached
                    policy_logp = self.policy.policy_logprobs(
                        prompt_input_ids=c.prompt_input_ids,
                        prompt_attention_mask=c.prompt_attention_mask,
                        gen_token_ids=c.token_ids,
                        pixel_values=c.pixel_values,
                        image_grid_thw=c.image_grid_thw,
                        adapter="policy",
                        gradient_enabled=True,
                    )
                    pg_term = -adv * policy_logp.sum() / token_norm * cand_scale
                    pg_term.backward()
                    pg_sum += float(pg_term.detach().item())
                    n_tokens += int(policy_logp.shape[0])

            finally:
                worker.close()

        return {
            "pg_loss": pg_sum,
            "vf_loss": vf_sum,
            "loss": pg_sum + vf_sum,
            "n_generated_tokens": n_tokens,
            "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "mean_value": float(np.mean(all_values)) if all_values else 0.0,
            "value_error": float(
                np.mean([abs(v - np.mean(all_rewards)) for v in all_values])
            ) if all_values and all_rewards else 0.0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_states(self, n: int) -> List[StateSample]:
        idxs = self._rng.sample(range(len(self.dataset)), k=min(n, len(self.dataset)))
        return [self.dataset[i] for i in idxs]

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = self.out_dir / f"step{step}"
        self.policy.save_policy_adapter(str(ckpt_dir))
        self.policy.save_value_head(str(ckpt_dir))
        print(f"[ppo] saved checkpoint to {ckpt_dir}")


__all__ = ["PPOTrainer", "PPOConfig"]
