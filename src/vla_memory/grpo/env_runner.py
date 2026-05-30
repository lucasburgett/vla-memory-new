"""EnvRunner subclass that picks the dataset split.

Upstream ``robomme_policy_learning/examples/robomme/env_runner.py:34`` hardcodes
``dataset="test"`` in the ``BenchmarkEnvBuilder`` it constructs. For RL training
we need ``dataset="train"`` so we don't overfit to the held-out eval episodes.

We subclass instead of editing the submodule (which CLAUDE.md forbids).
"""

from __future__ import annotations

import sys
from typing import Any

# The submodule's example modules live at /app/examples/robomme inside the
# Modal container. We insert here so this file can be imported from anywhere
# in vla_memory without callers needing to remember the path mutation.
if "/app/examples/robomme" not in sys.path:
    sys.path.insert(0, "/app/examples/robomme")

from env_runner import EnvRunner as _UpstreamEnvRunner, pack_state  # type: ignore
from robomme.env_record_wrapper import BenchmarkEnvBuilder  # type: ignore
from utils import TASK_NAME_LIST  # type: ignore


class EnvRunner(_UpstreamEnvRunner):
    """``examples/robomme/env_runner.py:EnvRunner`` with overridable dataset.

    Defaults to ``dataset="train"`` for GRPO so we never accidentally roll out
    from test seeds during RL.
    """

    def __init__(
        self,
        env_id: str,
        video_save_dir: str,
        max_steps: int = 1300,
        dataset: str = "train",
    ) -> None:
        if env_id not in TASK_NAME_LIST:
            raise ValueError(f"Environment ID {env_id} not in {TASK_NAME_LIST}")
        self.env_id = env_id
        self.video_save_dir = video_save_dir
        self.env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset=dataset,
            action_space="joint_angle",
            gui_render=False,
            max_steps=max_steps,
        )
        # Same defaults as upstream — set after make_env().
        self.env: Any = None
        self.episode_id: int | None = None
        self.task_goal: str = ""

    def get_init_obs(self, seed: int | None = None) -> dict[str, Any]:
        """Reset and return the initial observation, optionally with a fixed seed.

        Upstream ``get_init_obs`` resets with no seed, so successive resets on the
        same env instance advance the episode RNG — the K rollouts of a GRPO
        group then start from DIFFERENT scenes. Passing a fixed ``seed`` pins
        every rollout in a group to the same initial state (env-side
        common-random-numbers), so reward differences reflect the subgoal rather
        than scene luck. Falls back to a plain reset if the env rejects ``seed``.

        (Duplicates upstream's obs packing because we can't edit the submodule
        and ``super().get_init_obs()`` would reset a second time without a seed.)
        """
        if seed is None:
            return super().get_init_obs()
        try:
            obs, self.info = self.env.reset(seed=int(seed))
        except Exception as exc:
            # Seeded reset is a best-effort variance-reduction nicety; if the
            # env's episode machinery dislikes an explicit seed, fall back to the
            # upstream plain reset rather than crashing the (costly) run.
            if not getattr(self, "_warned_seed_reset", False):
                print(f"[env_runner] seeded reset failed ({exc!r}); using plain reset", flush=True)
                self._warned_seed_reset = True
            return super().get_init_obs()
        if isinstance(self.info["task_goal"], list):
            self.task_goal = self.info["task_goal"][0]
        else:
            self.task_goal = self.info["task_goal"]
        images = obs["front_rgb_list"]
        wrist_images = obs["wrist_rgb_list"]
        states = [
            pack_state(joint_state, gripper_state)
            for joint_state, gripper_state in zip(obs["joint_state_list"], obs["gripper_state_list"])
        ]
        return {
            "images": images,
            "wrist_images": wrist_images,
            "states": states,
            "task_goal": self.task_goal,
        }


__all__ = ["EnvRunner"]
