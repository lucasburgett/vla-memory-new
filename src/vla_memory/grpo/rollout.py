"""Rollout worker: memory-conditioned subgoal → GroundSG π0.5 → ManiSkill → reward.

The memory pipeline (ButtonUnmask and the rest of the Permanence suite) needs the
VLM to make a subgoal decision *mid-episode, after an occlusion*, using frames it
saw *before* the occlusion. So a rollout is no longer "run one subgoal from t=0";
it is:

    1. WARM UP with the ORACLE subgoal — the robot does the pre-decision part of
       the task (e.g. press the button) while the scene reveals then hides the
       cubes (ButtonUnmask lifts/drops the bins over steps 0–64, driven by the
       timestep, not the robot). We capture the reveal frames here.
    2. STOP at the oracle's subgoal TRANSITION — the moment the task switches to
       the memory-dependent step ("pick the container that hides the red cube").
       This is the decision point the VLM is scored on.
    3. EXECUTE the candidate subgoal (the VLM's prediction) to the episode end.
    4. Reward = task-completion fraction (``reward.py``).

``peek_at_decision_point`` runs steps 1–2 and returns ``(key_frames,
recent_frames, task_goal)`` for the VLM; ``rollout`` re-runs steps 1–2 (the
warm-up is oracle-driven and the reveal is timestep-deterministic, so it
reproduces the same decision state up to π0.5 sampling noise) and then runs
step 3 with the candidate.

TODO(optimization): snapshot the env at the decision point with
``env.get_state()`` once per group and ``set_state()`` for each candidate, instead
of re-warming K times. Correct but slower today.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

from vla_memory.qwen_subgoal.coords import from_qwen_xy


@dataclasses.dataclass
class RolloutResult:
    success_flag: str        # "success" | "fail" | "timeout" | "error" | "unknown"
    progress: float          # [0, 1]
    final_image: np.ndarray  # last frame
    n_steps: int


@dataclasses.dataclass
class DecisionPoint:
    """What the VLM conditions on at a mid-episode memory decision."""

    key_frames: List[np.ndarray]     # remembered past (reveal-window keyframes) — one-shot path
    recent_frames: List[np.ndarray]  # current execution context
    task_goal: str
    history_subgoals: List[str]      # completed subtasks (timing signal, e.g. ["press the button"])
    warm_steps: int                  # steps taken to reach the decision point
    terminated_early: bool           # episode ended during warm-up (no decision to make)
    success_flag: str                # status if it terminated during warm-up
    # Joint select-then-use path (JOINT_MEMORY_DESIGN.md): the SELECT call shows the
    # VLM this BROAD candidate window (reveal cube-visible + post-occlusion covered
    # frames) and it picks which to keep; the USE call sees only the kept ones +
    # current_frame. Empty in one-shot mode.
    candidate_frames: List[np.ndarray] = dataclasses.field(default_factory=list)
    current_frame: Optional[np.ndarray] = None


class RolloutWorker:
    """Wraps an openpi WebSocket client + an ``EnvRunner`` from the submodule.

    The env is constructed once per ``episode_id`` (``make_env`` loads scene
    assets) and reused across the K candidates of a GRPO group; ``reset`` between
    candidates is cheap. The worker is otherwise stateless across episodes.
    """

    def __init__(
        self,
        env_runner,                       # vla_memory.grpo.env_runner.EnvRunner
        policy_client,                    # openpi_client websocket client
        obs_horizon: int = 16,
        max_steps: int = 200,
        use_history: bool = False,        # frozen π0.5 baseline runs without history
        subgoal_type: str = "grounded_subgoal",  # which oracle to warm up with
        decision_warm_cap: int = 150,     # safety cap on warm-up if no transition is seen
        n_key_frames: int = 4,            # reveal keyframes fed to the VLM as memory
        n_recent_frames: int = 2,         # recent execution frames (current context)
        reveal_window: int = 64,          # steps over which the scene reveals (ButtonUnmask: 0–64)
        n_candidate_frames: int = 12,     # joint SELECT call: breadth of the candidate window
    ) -> None:
        self.env_runner = env_runner
        self.client = policy_client
        self.obs_horizon = obs_horizon
        self.max_steps = max_steps
        self.use_history = use_history
        if subgoal_type not in ("simple_subgoal", "grounded_subgoal"):
            raise ValueError(f"subgoal_type must be simple/grounded, got {subgoal_type!r}")
        self.subgoal_type = subgoal_type
        self.decision_warm_cap = decision_warm_cap
        self.n_key_frames = n_key_frames
        self.n_recent_frames = n_recent_frames
        self.reveal_window = reveal_window
        self.n_candidate_frames = n_candidate_frames
        self._env_open = False
        self._cached_episode_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Env lifecycle
    # ------------------------------------------------------------------

    def _ensure_env(self, episode_id: int, seed: Optional[int] = None) -> dict:
        """REBUILD the env and return a fresh init obs.

        We rebuild (``make_env``) on every call rather than caching + resetting.
        ButtonUnmask and the Permanence suite spawn the cubes/bins in
        ``_load_scene`` (run by ``make_env``), NOT in ``reset`` /
        ``_initialize_episode`` (which only re-poses the robot). So a cached env
        that is merely reset keeps the cubes wherever a prior warm-up left them —
        the second warm-up of the ``peek``→``rollout`` path then starts from a
        corrupted layout and the pick fails. The causality probe proved a freshly
        built env completes the pick (frozen-correct == oracle) while the
        cached-reset path does not. ``seed`` pins the scene so a group's rollouts
        share the same cube→container layout (seed-shuffled).

        TODO(perf): rebuilding per rollout pays ``make_env`` every time. The right
        optimization is ``env.get_state``/``set_state`` to snapshot the decision
        point once per group and branch K candidates from it — which also removes
        the warm-up sampling noise between candidates. Deferred until the loop is
        validated end-to-end.
        """
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

    def close(self) -> None:
        """Close the underlying env. Safe to call multiple times."""
        if self._env_open:
            try:
                self.env_runner.close_env()
            except Exception:
                pass
            self._env_open = False
        self._cached_episode_id = None

    # ------------------------------------------------------------------
    # Oracle helper
    # ------------------------------------------------------------------

    def _oracle_subgoal(self) -> str:
        """Current oracle subgoal of the configured type, from the last env step.

        ``env_runner`` exposes ``info["simple_subgoal_online"]`` /
        ``["grounded_subgoal_online"]`` as properties. We warm up with the
        grounded oracle so the GroundSG π0.5 gets the format it was trained on.
        """
        if self.subgoal_type == "grounded_subgoal":
            return self.env_runner.grounded_subgoal_oracle
        return self.env_runner.simple_subgoal_oracle

    @staticmethod
    def _simple_phase(env_runner) -> str:
        """The simple (text-only) oracle subgoal — used for transition detection.

        We detect the decision point off the *simple* subgoal because the grounded
        one carries a bbox that can jitter frame-to-frame even within one phase.
        """
        try:
            return str(env_runner.simple_subgoal_oracle).strip().lower()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Warm-up to the mid-episode decision point
    # ------------------------------------------------------------------

    def _warmup(
        self, episode_id: int, seed: Optional[int]
    ) -> Tuple[List[np.ndarray], str, int, bool, str, Tuple]:
        """Reset and drive the env with the ORACLE subgoal until the subgoal phase
        changes (the memory decision point) or a cap/termination is hit.

        Returns ``(captured_frames, task_goal, n_steps, terminated, success_flag,
        carry)`` where ``carry`` is the live ``(img, wrist, state, image_buf,
        wrist_buf, state_buf, exec_start_idx)`` needed to keep stepping in
        ``rollout`` without re-resetting.
        """
        resp = self.client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

        init = self._ensure_env(episode_id, seed=seed)
        image_buf = deque(init["images"], maxlen=64)
        wrist_buf = deque(init["wrist_images"], maxlen=64)
        state_buf = deque(init["states"], maxlen=64)
        exec_start_idx = len(image_buf) - 1
        task_goal = init["task_goal"]

        img, wrist, robot_state = image_buf[-1], wrist_buf[-1], state_buf[-1]
        captured: List[np.ndarray] = [img]
        # Track COMPLETED subtasks (distinct non-pick simple phases) up to the pick —
        # the timing signal carried in the prompt. MUST match the builder's history
        # (prompt parity). ButtonUnmask → ["press the button"]; ButtonUnmaskSwap →
        # ["press the first button", "press the second button"].
        history: List[str] = []

        def _record(phase: str) -> None:
            if phase and "pick up the container" not in phase and phase not in history:
                history.append(phase)

        _record(self._simple_phase(self.env_runner))

        n_steps = 0
        success_flag = "unknown"
        terminated = False
        reached_transition = False

        while n_steps < self.decision_warm_cap:
            subgoal = self._oracle_subgoal()
            actions = self._infer_actions(
                img, wrist, robot_state, task_goal, subgoal=subgoal,
                image_buf=image_buf, state_buf=state_buf, exec_start_idx=exec_start_idx,
            )
            for action in actions:
                (img, wrist, robot_state), stop, success_flag = self.env_runner.step(action)
                if img is None:
                    return captured, task_goal, n_steps, True, "error", (
                        None, None, None, image_buf, wrist_buf, state_buf, exec_start_idx
                    ), history
                image_buf.append(img)
                wrist_buf.append(wrist)
                state_buf.append(robot_state)
                captured.append(img)
                n_steps += 1
                if stop:
                    terminated = True
                    break
                # Decision point = the PICK subgoal (not the first phase change): a
                # multi-phase task (ButtonUnmaskSwap presses TWO buttons) changes phase
                # at press1→press2, BEFORE the swaps finish. Record each completed
                # non-pick phase as we pass it; stop at the pick so the VLM owns it.
                # Single-button ButtonUnmask resolves to the SAME frame/history as
                # before (parity preserved).
                phase = self._simple_phase(self.env_runner)
                if "pick up the container" in phase:
                    reached_transition = True
                    break
                _record(phase)
                if n_steps >= self.decision_warm_cap:
                    break
            if terminated or reached_transition or n_steps >= self.decision_warm_cap:
                break

        carry = (img, wrist, robot_state, image_buf, wrist_buf, state_buf, exec_start_idx)
        return captured, task_goal, n_steps, terminated, success_flag, carry, history

    def _select_memory_frames(
        self, captured: Sequence[np.ndarray], warm_steps: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split captured warm-up frames into (key_frames, recent_frames).

        Keyframes are sampled evenly from the **reveal window** (the early steps
        where the cubes were visible) — that is the memory. Recent frames are the
        last few captured frames (the current, post-occlusion context). ``captured``
        includes the initial frame at index 0, so it has ``warm_steps + 1`` frames.
        """
        n = len(captured)
        if n == 0:
            return [], []
        # Reveal window in capture-index space (clamped to what we actually saw).
        reveal_end = min(self.reveal_window, n - 1)
        recent = [captured[i] for i in range(max(0, n - self.n_recent_frames), n)]
        if reveal_end <= 0 or self.n_key_frames <= 0:
            return [], recent
        # Evenly spaced indices across [0, reveal_end]; dedupe preserves order.
        idxs = np.linspace(0, reveal_end, num=self.n_key_frames, dtype=int).tolist()
        seen, key_idxs = set(), []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                key_idxs.append(i)
        key_frames = [captured[i] for i in key_idxs]
        return key_frames, recent

    def _select_candidate_frames(self, captured: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Broad, evenly-spaced sample across the WHOLE captured window for the
        joint SELECT call. Spans the reveal (cube-visible) AND the post-occlusion
        (covered, bins-dropped-at-64) frames, so the VLM must learn to keep the
        informative ones rather than any frame working (the one-shot heuristic's
        triviality). Empty in degenerate cases.
        """
        n = len(captured)
        if n == 0:
            return []
        if n <= self.n_candidate_frames:
            return list(captured)
        idxs = np.linspace(0, n - 1, num=self.n_candidate_frames, dtype=int).tolist()
        seen, out = set(), []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                out.append(captured[i])
        return out

    def peek_at_decision_point(
        self, episode_id: int, seed: Optional[int] = None
    ) -> DecisionPoint:
        """Warm up to the memory decision point and return what the VLM conditions on.

        Call once per state before sampling K candidates; then call ``rollout``
        for each candidate with the same ``episode_id``/``seed``. Each call
        rebuilds the env from the seed (so the cube→container layout matches what
        the VLM saw here) and warms up afresh to the decision point.
        """
        captured, task_goal, warm_steps, terminated, success_flag, _, history = self._warmup(
            episode_id, seed
        )
        key_frames, recent_frames = self._select_memory_frames(captured, warm_steps)
        candidate_frames = self._select_candidate_frames(captured)
        return DecisionPoint(
            key_frames=key_frames,
            recent_frames=recent_frames,
            task_goal=task_goal,
            # The completed-subtask timing signal (e.g. ["press the button"], or both
            # presses on ButtonUnmaskSwap); the SFT prompt carries the same line so
            # train/inference agree (prompt parity).
            history_subgoals=history,
            candidate_frames=candidate_frames,
            current_frame=captured[-1] if captured else None,
            warm_steps=warm_steps,
            terminated_early=terminated,
            success_flag=success_flag,
        )

    # ------------------------------------------------------------------
    # Rollout under a candidate subgoal
    # ------------------------------------------------------------------

    def rollout(
        self,
        episode_id: int,
        sampled_subgoal: str,
        seed: Optional[int] = None,
    ) -> RolloutResult:
        """Warm up to the decision point (oracle-driven), then execute
        ``sampled_subgoal`` to the episode end and score the outcome."""
        # ``sampled_subgoal`` is the VLM's prediction in Qwen-native <x,y> 0–1000
        # (the SFT target space). GroundSG was trained on the oracle's <y,x> 0–256
        # PIXEL format, so convert here — the SINGLE point where VLM output reaches
        # the executor (the oracle warm-up below and rollout_oracle use native
        # coords and must NOT be converted). Skipping this feeds GroundSG a wrong
        # coordinate → silently wrong reward → fake flatline. See coords.from_qwen_xy.
        sampled_subgoal = from_qwen_xy(sampled_subgoal)
        captured, task_goal, n_steps, terminated, success_flag, carry, _history = self._warmup(
            episode_id, seed
        )
        img, wrist, robot_state, image_buf, wrist_buf, state_buf, exec_start_idx = carry

        # If the episode already ended during warm-up there's no decision to make;
        # return the warm-up outcome so the trainer's degeneracy filter handles it.
        if terminated or img is None:
            return self._finalize(success_flag, image_buf[-1], n_steps)

        while n_steps < self.max_steps:
            actions = self._infer_actions(
                img, wrist, robot_state, task_goal, subgoal=sampled_subgoal,
                image_buf=image_buf, state_buf=state_buf, exec_start_idx=exec_start_idx,
            )
            for action in actions:
                (img, wrist, robot_state), stop, success_flag = self.env_runner.step(action)
                n_steps += 1
                if img is None:
                    return self._finalize("error", image_buf[-1], n_steps)
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

    def rollout_oracle(
        self, episode_id: int, seed: Optional[int] = None, log_every: int = 0
    ) -> RolloutResult:
        """GroundSG + ONLINE-ORACLE upper bound — drive the WHOLE episode with the
        per-chunk oracle subgoal (re-queried each inference, as MemER's high-level
        runs at ~1 Hz).

        Diagnostic only: separates "the low-level policy + env can complete this
        task with perfect subgoals" from "our fixed-subgoal harness is too rigid."
        ``log_every>0`` prints the oracle subgoal every ~that many steps so we can
        see whether it EVOLVES through the pick (approach→grasp→lift) or stays a
        single persistent target. Not used in training.
        """
        resp = self.client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)
        init = self._ensure_env(episode_id, seed=seed)
        image_buf = deque(init["images"], maxlen=64)
        wrist_buf = deque(init["wrist_images"], maxlen=64)
        state_buf = deque(init["states"], maxlen=64)
        exec_start_idx = len(image_buf) - 1
        task_goal = init["task_goal"]
        img, wrist, robot_state = image_buf[-1], wrist_buf[-1], state_buf[-1]

        n_steps = 0
        success_flag = "unknown"
        last_logged = -10**9
        while n_steps < self.max_steps:
            subgoal = self._oracle_subgoal()   # re-queried each chunk (updates online)
            if log_every and n_steps - last_logged >= log_every:
                print(f"        [oracle t={n_steps:>3}] {subgoal!r}", flush=True)
                last_logged = n_steps
            actions = self._infer_actions(
                img, wrist, robot_state, task_goal, subgoal=subgoal,
                image_buf=image_buf, state_buf=state_buf, exec_start_idx=exec_start_idx,
            )
            for action in actions:
                (img, wrist, robot_state), stop, success_flag = self.env_runner.step(action)
                n_steps += 1
                if img is None:
                    return self._finalize("error", image_buf[-1], n_steps)
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

    def rollout_freeze_at_transition(
        self,
        episode_id: int,
        seed: Optional[int] = None,
        corrupt=None,
        log_every: int = 0,
    ) -> Tuple[RolloutResult, Optional[str]]:
        """Continuous (one-reset) rollout: drive the PRESS phase with the oracle,
        then at the subgoal transition FREEZE the oracle's pick subgoal and HOLD it.

        Structurally identical to ``rollout_oracle`` EXCEPT it holds the pick
        subgoal instead of re-reading it. So comparing the two isolates "does the
        pick need an EVOLVING subgoal" from the double-warm-up reset artifact in
        the ``peek``→``rollout`` path (which resets a mid-episode env a second
        time). ``corrupt`` optionally rewrites the frozen subgoal (e.g. shift the
        grounding point, for the wrong-subgoal contrast). Returns
        ``(result, frozen_subgoal)``.
        """
        resp = self.client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)
        init = self._ensure_env(episode_id, seed=seed)
        image_buf = deque(init["images"], maxlen=64)
        wrist_buf = deque(init["wrist_images"], maxlen=64)
        state_buf = deque(init["states"], maxlen=64)
        exec_start_idx = len(image_buf) - 1
        task_goal = init["task_goal"]
        img, wrist, robot_state = image_buf[-1], wrist_buf[-1], state_buf[-1]
        phase0 = self._simple_phase(self.env_runner)

        frozen: Optional[str] = None
        n_steps = 0
        success_flag = "unknown"
        last_logged = -10**9
        while n_steps < self.max_steps:
            if frozen is not None:
                subgoal = frozen
            elif self._simple_phase(self.env_runner) != phase0:
                frozen = self._oracle_subgoal()           # capture the pick subgoal ONCE
                if corrupt is not None:
                    frozen = corrupt(frozen)
                subgoal = frozen
            else:
                subgoal = self._oracle_subgoal()           # press phase
            if log_every and n_steps - last_logged >= log_every:
                print(f"        [freeze t={n_steps:>3}] frozen={frozen is not None} {subgoal!r}", flush=True)
                last_logged = n_steps
            actions = self._infer_actions(
                img, wrist, robot_state, task_goal, subgoal=subgoal,
                image_buf=image_buf, state_buf=state_buf, exec_start_idx=exec_start_idx,
            )
            for action in actions:
                (img, wrist, robot_state), stop, success_flag = self.env_runner.step(action)
                n_steps += 1
                if img is None:
                    return self._finalize("error", image_buf[-1], n_steps), frozen
                image_buf.append(img)
                wrist_buf.append(wrist)
                state_buf.append(robot_state)
                if stop or n_steps >= self.max_steps:
                    break
            if stop or n_steps >= self.max_steps:
                break
        if n_steps >= self.max_steps and success_flag in ("unknown", ""):
            success_flag = "timeout"
        return self._finalize(success_flag, img, n_steps), frozen

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _finalize(self, success_flag: str, image: np.ndarray, n_steps: int) -> RolloutResult:
        # NOTE: env is intentionally NOT closed here — the trainer reuses it across
        # the K candidates of a group; closing happens in ``close()``.
        progress = self._progress_estimate(success_flag)
        return RolloutResult(
            success_flag=success_flag,
            progress=progress,
            final_image=image,
            n_steps=n_steps,
        )

    def _progress_estimate(self, success_flag: str) -> float:
        """Dense task-completion fraction from the env's sequential-task tracker.

        RoboMME envs expose ``current_task_index`` (0..N) and ``task_list`` on
        ``env.unwrapped``. ``current_task_index / len(task_list)`` is a graded
        progress signal — 1.0 iff success. For ButtonUnmask the subtasks are
        [press button, pick correct container, …], so picking the *wrong*
        container trips ``failure_func`` and does not advance the index.
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
        # ``utils`` resolves because main.py inserts ``/app/examples/robomme`` on
        # sys.path before constructing the worker.
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


__all__ = ["RolloutWorker", "RolloutResult", "DecisionPoint"]
