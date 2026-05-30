"""Rollout worker: subgoal → frozen pi0.5 → ManiSkill → reward.

Each call to ``rollout()`` executes one candidate subgoal for ``horizon`` steps,
returning the success flag, progress fraction, and the final frame. We re-use
the existing WebSocket policy client from openpi-client, so the policy server
keeps the same JAX/CUDA process that ``modal_server.py`` already spins up.

The rollout worker is *task-agnostic* — task name and env_runner come in from
above. This lets us keep the worker stateless across episodes and pick states
to rollout from in the trainer's outer loop.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np


@dataclasses.dataclass
class RolloutResult:
    success_flag: str        # "success" | "fail" | "timeout" | "error" | "unknown"
    progress: float          # [0, 1]
    final_image: np.ndarray  # last frame
    n_steps: int


class RolloutWorker:
    """Wraps an openpi WebSocket client + an ``EnvRunner`` from the submodule.

    The submodule's ``examples/robomme/env_runner.py`` is a thin wrapper around
    a single task's env. We accept an instance of it and a fresh client per
    rollout — the client itself is cheap (it's just a socket connection),
    while the env is expensive to construct (it loads scene assets).
    """

    def __init__(
        self,
        env_runner,                       # examples/robomme/env_runner.py:EnvRunner
        policy_client,                    # openpi_client.websocket_client_policy.MMEVLAWebsocketClientPolicy
        obs_horizon: int = 16,
        max_steps: int = 200,
        use_history: bool = False,        # frozen pi0.5 baseline runs without history
    ) -> None:
        self.env_runner = env_runner
        self.client = policy_client
        self.obs_horizon = obs_horizon
        self.max_steps = max_steps
        self.use_history = use_history
        # The env is constructed once per episode_id and reused across the K
        # candidates of a GRPO group — make_env is the expensive op (loads
        # scene assets), env.reset() between candidates is cheap.
        self._env_open = False
        self._cached_episode_id: Optional[int] = None

    def _ensure_env(self, episode_id: int, seed: Optional[int] = None) -> dict:
        """Build env if not built (or episode changed) and return a fresh init obs.

        ``seed`` pins the reset so every rollout in a group starts from the same
        scene (env-side common-random-numbers); without it the upstream reset
        advances the RNG and each rollout gets a different layout. The env stays
        open across calls with the same ``episode_id``.
        """
        if not self._env_open or self._cached_episode_id != episode_id:
            if self._env_open:
                try:
                    self.env_runner.close_env()
                except Exception:
                    pass
                self._env_open = False
            self.env_runner.make_env(episode_id)
            self._env_open = True
            self._cached_episode_id = episode_id
        return self.env_runner.get_init_obs(seed=seed)

    def peek_init(self, episode_id: int, seed: Optional[int] = None) -> Tuple[np.ndarray, str]:
        """Return ``(init_image, task_goal)`` for sampling subgoals.

        Opens the env if needed and caches it for the upcoming K rollouts of
        this episode. The trainer should call this once per state before
        sampling, then loop ``rollout()`` for each candidate, then ``close()``.
        Pass the same ``seed`` here and to each ``rollout()`` so the subgoals are
        sampled from the same scene the rollouts start in.
        """
        init = self._ensure_env(episode_id, seed=seed)
        return init["images"][-1], init["task_goal"]

    def close(self) -> None:
        """Close the underlying env. Safe to call multiple times."""
        if self._env_open:
            try:
                self.env_runner.close_env()
            except Exception:
                pass
            self._env_open = False
        self._cached_episode_id = None

    def rollout(
        self,
        episode_id: int,
        sampled_subgoal: str,
        warm_steps: int = 0,
        seed: Optional[int] = None,
    ) -> RolloutResult:
        """Drive one rollout under a single sampled subgoal.

        ``warm_steps``: if > 0, we first replay the env from reset for this many
        steps before evaluating the subgoal. Useful when we want the policy to
        see a mid-episode state instead of just t=0.

        TODO: for GRPO at scale we should serialize / restore env state so we
        don't have to replay from scratch each rollout. ManiSkill exposes
        ``env.get_state()`` / ``env.set_state()`` — wiring that here is a clear
        next optimization once the loop is working end-to-end.
        """
        # Reset both the policy server and the env. ``_ensure_env`` keeps the
        # env open across candidates of the same episode_id, so we only pay
        # the asset-loading cost once per state.
        resp = self.client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

        init = self._ensure_env(episode_id, seed=seed)

        image_buf = deque(init["images"], maxlen=64)
        wrist_buf = deque(init["wrist_images"], maxlen=64)
        state_buf = deque(init["states"], maxlen=64)
        exec_start_idx = len(image_buf) - 1
        task_goal = init["task_goal"]

        img = image_buf[-1]
        wrist = wrist_buf[-1]
        robot_state = state_buf[-1]

        n_steps = 0
        success_flag = "unknown"

        # If we want a mid-episode state, advance with policy-only (no subgoal
        # changes) before scoring.
        for _ in range(warm_steps):
            actions = self._infer_actions(img, wrist, robot_state, task_goal, subgoal=None,
                                          image_buf=image_buf, state_buf=state_buf,
                                          exec_start_idx=exec_start_idx)
            for action in actions:
                (img, wrist, robot_state), stop, success_flag = self.env_runner.step(action)
                if stop:
                    return self._finalize(success_flag, img, n_steps)
                image_buf.append(img)
                wrist_buf.append(wrist)
                state_buf.append(robot_state)
                n_steps += 1
                if n_steps >= self.max_steps:
                    break
            if n_steps >= self.max_steps:
                break

        # Now run with the sampled subgoal until success/fail/timeout.
        while n_steps < self.max_steps:
            actions = self._infer_actions(img, wrist, robot_state, task_goal,
                                          subgoal=sampled_subgoal,
                                          image_buf=image_buf, state_buf=state_buf,
                                          exec_start_idx=exec_start_idx)
            for action in actions:
                (img, wrist, robot_state), stop, success_flag = self.env_runner.step(action)
                n_steps += 1
                if img is None:
                    success_flag = "error"
                    return self._finalize(success_flag, image_buf[-1], n_steps)
                image_buf.append(img)
                wrist_buf.append(wrist)
                state_buf.append(robot_state)
                if stop or n_steps >= self.max_steps:
                    break
            if stop or n_steps >= self.max_steps:
                break

        if n_steps >= self.max_steps and success_flag in ("unknown", ""):
            success_flag = "timeout"
        return self._finalize(success_flag, img, n_steps)

    def _finalize(self, success_flag: str, image: np.ndarray, n_steps: int) -> RolloutResult:
        # NOTE: env is intentionally NOT closed here. The trainer reuses the
        # env across the K candidates of a GRPO group; closing happens in
        # ``RolloutWorker.close()`` after the full group is done.
        progress = self._progress_estimate(success_flag)
        return RolloutResult(
            success_flag=success_flag,
            progress=progress,
            final_image=image,
            n_steps=n_steps,
        )

    def _progress_estimate(self, success_flag: str) -> float:
        """Dense task-completion fraction from the env's sequential-task tracker.

        RoboMME envs expose ``current_task_index`` (0..N, set to N once all N
        subtasks are done) and ``task_list`` on ``env.unwrapped`` (set by
        ``subgoal_evaluate_func.sequential_task_check``). The fraction
        ``current_task_index / len(task_list)`` is a graded progress signal —
        1.0 iff success — available for every RoboMME task (PickXtimes:
        pick/place×N + button → N completed-of-total). Falls back to a 0/1
        success estimate if the attributes aren't present (e.g. the env errored
        before any evaluate()).
        """
        if success_flag == "success":
            return 1.0
        try:
            env = self.env_runner.env.unwrapped
            total = len(getattr(env, "task_list", []) or [])
            idx = float(getattr(env, "current_task_index", 0) or 0)
            if total > 0:
                return max(0.0, min(1.0, idx / total))
        except Exception:
            pass
        return 0.0

    def _infer_actions(
        self,
        img: np.ndarray,
        wrist: np.ndarray,
        robot_state: np.ndarray,
        task_goal: str,
        subgoal: Optional[str],
        image_buf,
        state_buf,
        exec_start_idx: int,
    ):
        from openpi_client import websocket_client_policy as _wp  # noqa: F401
        # ``utils`` resolves because modal_train.py inserts
        # ``/app/examples/robomme`` on sys.path before constructing the worker.
        # We deliberately don't use the ``examples.robomme.utils`` form because
        # that would require ``/app`` (not ``/app/examples/robomme``) on path.
        from utils import pack_buffer  # type: ignore

        if self.use_history:
            resp = self.client.add_buffer(
                pack_buffer(list(image_buf), list(state_buf), exec_start_idx)
            )
            while not resp.get("add_buffer_finished", False):
                time.sleep(0.05)

        element = {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": robot_state,
            "prompt": task_goal,
        }
        if subgoal is not None:
            element["simple_subgoal"] = subgoal
            element["grounded_subgoal"] = subgoal
        return self.client.infer(element)["actions"][: self.obs_horizon]


__all__ = ["RolloutWorker", "RolloutResult"]
