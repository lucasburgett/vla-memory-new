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

SNAPSHOT BRANCHING (the speed path, opt-in): re-warming K times pays
``make_env`` + ~150–410 oracle inference steps per candidate, which dominates
wall-clock. ``peek_and_snapshot`` warms up ONCE and captures the env at the
decision point (``env_snapshot.snapshot_env`` — physics + the whole wrapper
chain's bookkeeping); ``rollout_from_snapshot`` restores that snapshot and runs
only step 3. The trainer enables this via ``GRPOConfig.snapshot_branching``;
``env_snapshot`` documents why a bare ``get_state_dict`` is insufficient, and
``snapshot_parity_probe.py`` certifies that a restored env reproduces an
identical reward + task-index trajectory bit-for-bit. The rebuild path
(``peek_at_decision_point`` + ``rollout``) is preserved unchanged as the default.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from vla_memory.grpo.env_snapshot import EnvSnapshot, _copy_value, restore_env, snapshot_env
from vla_memory.grpo.keyframe_buffer import KeyframeBuffer, TaggedFrame
from vla_memory.grpo.selection import valid_keyframe_positions
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


@dataclasses.dataclass
class Snapshot:
    """Everything needed to branch a candidate rollout from the decision point
    WITHOUT re-warming: the captured env state plus the live stepping carry.

    ``env`` is the physics + full-wrapper-chain bookkeeping snapshot. The carry
    fields are the warm-up's exit state — copied (np arrays, deque contents as
    lists) so the snapshot is immutable across the K candidates restored from it.
    """

    env: EnvSnapshot
    task_goal: str
    n_steps: int
    exec_start_idx: int
    terminated: bool
    success_flag: str
    img: Optional[np.ndarray]
    wrist: Optional[np.ndarray]
    robot_state: Optional[np.ndarray]
    image_buf: List[np.ndarray]
    wrist_buf: List[np.ndarray]
    state_buf: List[np.ndarray]
    # ``EnvRunner.info`` lives on the runner, NOT the wrapper chain — so
    # ``restore_env`` (which only touches ``env_runner.env``) leaves it stale. It
    # carries the oracle subgoals + ``status``; we restore it for a consistent
    # post-restore runner state (candidate execution re-reads its own subgoal and
    # ``step`` refreshes ``info``, so this matters mainly for correctness hygiene
    # and the parity probe's invariant check).
    info: Optional[dict] = None


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
        max_keyframes: int = 4,           # streaming: cap on positions nominated per call
        keyframe_buffer_cap: int = 8,     # streaming: cap on the accumulated keyframe buffer (MemER ≤8)
        keyframe_cluster_dist: int = 8,   # streaming: merge nominations within this many steps (MemER d=8)
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
        self.max_keyframes = max_keyframes
        self.keyframe_buffer_cap = keyframe_buffer_cap
        self.keyframe_cluster_dist = keyframe_cluster_dist
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

        PERF: rebuilding per rollout pays ``make_env`` every time. The snapshot
        path (``peek_and_snapshot`` → ``rollout_from_snapshot``, enabled by
        ``GRPOConfig.snapshot_branching``) bypasses this entirely: it warms up
        ONCE, calls ``env_snapshot.snapshot_env`` here-equivalent state, and
        ``restore_env``s it per candidate — also removing the warm-up sampling
        noise between candidates. This rebuild path stays the default and the
        verified fallback (see ``snapshot_parity_probe.py``).
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

    @staticmethod
    def _history_before_pick(phase_seq: List[str]) -> List[str]:
        """Distinct completed subtasks before the target pick. If we stopped ON a pick
        (last phase is a pick), drop it; otherwise return the non-pick phases seen.
        On Swap pick 2 this yields [press1, press2, pick-1, put-down] → the prompt tells
        the model which colour it's now on."""
        if phase_seq and "pick up the container" in phase_seq[-1]:
            return list(phase_seq[:-1])
        return [p for p in phase_seq if "pick up the container" not in p]

    def _warmup(
        self, episode_id: int, seed: Optional[int], pick_index: int = 0
    ) -> Tuple[List[np.ndarray], str, int, bool, str, Tuple, List[str]]:
        """Reset and drive the env with the ORACLE subgoal up to the ``pick_index``-th
        PICK (the memory decision point) or a cap/termination.

        For multi-pick tasks the oracle EXECUTES picks 0..pick_index-1 (and the
        put-downs between them) on the way, so the VLM owns pick ``pick_index`` with the
        earlier picks correctly done — that's how each pick earns its own reward.
        Returns ``(captured, task_goal, n_steps, terminated, success_flag, carry,
        history)``; ``carry`` is the live stepping state and ``history`` the distinct
        completed subtasks before the target pick (prompt parity with the builder).
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

        # Distinct simple phases in order of first appearance (dedup consecutive). The
        # decision point is the (pick_index+1)-th PICK phase; the oracle drives through
        # earlier picks/put-downs to get there. ButtonUnmask (pick_index 0) stops at the
        # one press→pick transition exactly as before (parity).
        phase_seq: List[str] = []

        def _push(phase: str) -> None:
            if phase and (not phase_seq or phase_seq[-1] != phase):
                phase_seq.append(phase)

        def _pick_count() -> int:
            return sum(1 for p in phase_seq if "pick up the container" in p)

        _push(self._simple_phase(self.env_runner))

        # Later picks need a proportionally longer warm-up (ButtonUnmaskSwap: pick 1
        # ~rel 180, pick 2 ~rel 410), so scale the cap by the pick index.
        cap = self.decision_warm_cap * (pick_index + 1)
        n_steps = 0
        success_flag = "unknown"
        terminated = False
        reached_transition = _pick_count() >= pick_index + 1

        while n_steps < cap and not reached_transition:
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
                    ), self._history_before_pick(phase_seq)
                image_buf.append(img)
                wrist_buf.append(wrist)
                state_buf.append(robot_state)
                captured.append(img)
                n_steps += 1
                if stop:
                    terminated = True
                    break
                _push(self._simple_phase(self.env_runner))
                if _pick_count() >= pick_index + 1:
                    reached_transition = True
                    break
                if n_steps >= cap:
                    break
            if terminated or reached_transition or n_steps >= cap:
                break

        history = self._history_before_pick(phase_seq)
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

    def _build_decision_point(
        self,
        captured: List[np.ndarray],
        warm_steps: int,
        task_goal: str,
        history: List[str],
        terminated: bool,
        success_flag: str,
    ) -> DecisionPoint:
        """Assemble the ``DecisionPoint`` (VLM-conditioning view) from warm-up
        output. Shared by ``peek_at_decision_point`` (rebuild path) and
        ``peek_and_snapshot`` (snapshot path) so both expose the same frames."""
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

    def peek_at_decision_point(
        self, episode_id: int, seed: Optional[int] = None, pick_index: int = 0
    ) -> DecisionPoint:
        """Warm up to the ``pick_index``-th memory decision point and return what the
        VLM conditions on.

        Call once per state before sampling K candidates; then call ``rollout``
        for each candidate with the same ``episode_id``/``seed``/``pick_index``. Each
        call rebuilds the env from the seed (so the layout matches what the VLM saw
        here) and warms up afresh to the same pick.
        """
        captured, task_goal, warm_steps, terminated, success_flag, _, history = self._warmup(
            episode_id, seed, pick_index=pick_index
        )
        return self._build_decision_point(
            captured, warm_steps, task_goal, history, terminated, success_flag
        )

    def peek_and_snapshot(
        self, episode_id: int, seed: Optional[int] = None, pick_index: int = 0
    ) -> Tuple[DecisionPoint, Snapshot]:
        """Warm up ONCE and return the decision point AND a restorable env snapshot.

        The speed path: instead of re-warming for each of the K candidates, the
        trainer calls this once, then ``rollout_from_snapshot`` per candidate. The
        env is left OPEN (no ``close_env``/``make_env``) so ``restore_env`` can
        rewind it in place. Captures the env state (physics + full wrapper chain)
        plus the warm-up's exit carry (copied so the snapshot survives K restores).
        """
        captured, task_goal, warm_steps, terminated, success_flag, carry, history = self._warmup(
            episode_id, seed, pick_index=pick_index
        )
        dp = self._build_decision_point(
            captured, warm_steps, task_goal, history, terminated, success_flag
        )
        img, wrist, robot_state, image_buf, wrist_buf, state_buf, exec_start_idx = carry
        snapshot = Snapshot(
            env=snapshot_env(self.env_runner.env),
            task_goal=task_goal,
            n_steps=warm_steps,
            exec_start_idx=exec_start_idx,
            terminated=terminated,
            success_flag=success_flag,
            img=None if img is None else img.copy(),
            wrist=None if wrist is None else wrist.copy(),
            robot_state=None if robot_state is None else robot_state.copy(),
            image_buf=[f.copy() for f in image_buf],
            wrist_buf=[f.copy() for f in wrist_buf],
            state_buf=[s.copy() for s in state_buf],
            info=_copy_value(getattr(self.env_runner, "info", None)),
        )
        return dp, snapshot

    # ------------------------------------------------------------------
    # Rollout under a candidate subgoal
    # ------------------------------------------------------------------

    def rollout(
        self,
        episode_id: int,
        sampled_subgoal: str,
        seed: Optional[int] = None,
        pick_index: int = 0,
    ) -> RolloutResult:
        """Warm up to the ``pick_index``-th decision point (oracle-driven, executing
        earlier picks), then execute ``sampled_subgoal`` to the episode end and score
        the outcome. Reward = absolute progress: later picks have a higher floor (the
        oracle's earlier picks) but GRPO's within-group advantage is relative, so the
        floor cancels and each pick's selection still gets a clean gradient."""
        # ``sampled_subgoal`` is the VLM's prediction in Qwen-native <x,y> 0–1000
        # (the SFT target space). GroundSG was trained on the oracle's <y,x> 0–256
        # PIXEL format, so convert here — the SINGLE point where VLM output reaches
        # the executor (the oracle warm-up below and rollout_oracle use native
        # coords and must NOT be converted). Skipping this feeds GroundSG a wrong
        # coordinate → silently wrong reward → fake flatline. See coords.from_qwen_xy.
        sampled_subgoal = from_qwen_xy(sampled_subgoal)
        captured, task_goal, n_steps, terminated, success_flag, carry, _history = self._warmup(
            episode_id, seed, pick_index=pick_index
        )
        img, wrist, robot_state, image_buf, wrist_buf, state_buf, exec_start_idx = carry

        # If the episode already ended during warm-up there's no decision to make;
        # return the warm-up outcome so the trainer's degeneracy filter handles it.
        if terminated or img is None:
            return self._finalize(success_flag, image_buf[-1], n_steps)

        return self._execute_forward(
            sampled_subgoal, task_goal, img, wrist, robot_state,
            image_buf, wrist_buf, state_buf, exec_start_idx, n_steps, success_flag,
        )

    def rollout_from_snapshot(
        self, snapshot: Snapshot, sampled_subgoal: str
    ) -> RolloutResult:
        """Branch a candidate from a ``peek_and_snapshot`` snapshot: restore the
        decision-point env in place, then execute ``sampled_subgoal`` to the end.

        Replaces the per-candidate warm-up (env rebuild + oracle re-drive) with a
        ``restore_env`` — the whole point of the optimization. Forward execution is
        byte-identical to ``rollout`` (shared ``_execute_forward``). ``from_qwen_xy``
        still converts the VLM coordinate at this single point (same as ``rollout``).
        """
        sampled_subgoal = from_qwen_xy(sampled_subgoal)
        # The policy server accumulates a history buffer only when use_history is on.
        # We skip the re-warm here, so reset the server per candidate to avoid leaking
        # the prior candidate's frames. GRPO runs use_history=False → infer is
        # stateless → no reset needed (and we avoid the reset round-trip).
        if self.use_history:
            resp = self.client.reset()
            while not resp.get("reset_finished", False):
                time.sleep(0.1)

        restore_env(self.env_runner.env, snapshot.env)
        if snapshot.info is not None:
            # info lives on the runner, outside the wrapper chain restore_env touches.
            self.env_runner.info = _copy_value(snapshot.info)
        image_buf = deque(snapshot.image_buf, maxlen=64)
        wrist_buf = deque(snapshot.wrist_buf, maxlen=64)
        state_buf = deque(snapshot.state_buf, maxlen=64)
        img = None if snapshot.img is None else snapshot.img.copy()
        wrist = None if snapshot.wrist is None else snapshot.wrist.copy()
        robot_state = None if snapshot.robot_state is None else snapshot.robot_state.copy()

        if snapshot.terminated or img is None:
            last = image_buf[-1] if image_buf else img
            return self._finalize(snapshot.success_flag, last, snapshot.n_steps)

        return self._execute_forward(
            sampled_subgoal, snapshot.task_goal, img, wrist, robot_state,
            image_buf, wrist_buf, state_buf, snapshot.exec_start_idx,
            snapshot.n_steps, snapshot.success_flag,
        )

    def _execute_forward(
        self,
        sampled_subgoal: str,
        task_goal: str,
        img: np.ndarray,
        wrist: np.ndarray,
        robot_state: np.ndarray,
        image_buf,
        wrist_buf,
        state_buf,
        exec_start_idx: int,
        n_steps: int,
        success_flag: str,
    ) -> RolloutResult:
        """Step ``sampled_subgoal`` from the (warmed-or-restored) decision state to
        the episode end and score the outcome. Identical body for both the rebuild
        (``rollout``) and snapshot (``rollout_from_snapshot``) paths."""
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

    # ------------------------------------------------------------------
    # Decision-point STREAMING (MemER-faithful single-call output + GRPO)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_pick_phase(phase: str) -> bool:
        """True iff the simple oracle phase is a memory-dependent PICK (the
        decision the VLM owns). The put-downs / presses / approaches around it are
        deterministic scaffolding the oracle drives. Same substring the per-pick
        path keys on (``_warmup`` / ``_history_before_pick``) → consistent."""
        return "pick up the container" in phase

    def _stream_apply(
        self, actions, image_buf, wrist_buf, state_buf, captured, captured_steps, n_steps,
        recorder=None, subgoal=None,
    ):
        """Apply ONE inferred action chunk, updating the rolling buffers and the
        unbounded (frame, abs_step) capture. Returns
        ``(img, wrist, robot_state, n_steps, stop, success_flag, dead)`` — ``dead``
        is True iff the sim returned a None frame (error). Shared by the oracle-scaffold
        and VLM-pick loops of ``rollout_streaming`` so the stepping/capture is in one place.

        ``recorder`` (eval-only, default None) is the submodule's ``RolloutRecorder``;
        when provided, each stepped frame is annotated with ``subgoal`` and appended to
        the video. None in training → this is a no-op and the path is byte-identical.
        """
        img = wrist = robot_state = None
        stop = False
        success_flag = "unknown"
        for action in actions:
            (img, wrist, robot_state), stop, success_flag = self.env_runner.step(action)
            n_steps += 1
            if img is None:
                return None, None, None, n_steps, True, "error", True
            image_buf.append(img)
            wrist_buf.append(wrist)
            state_buf.append(robot_state)
            captured.append(img)
            captured_steps.append(n_steps)
            if recorder is not None:
                recorder.record(img, wrist, robot_state, subgoal=subgoal)
            if stop or n_steps >= self.max_steps:
                break
        return img, wrist, robot_state, n_steps, stop, success_flag, False

    def _stream_window(
        self, captured: List[np.ndarray], captured_steps: List[int], start: int
    ) -> Tuple[List[np.ndarray], List[int]]:
        """The broad recent window for a pick: ``n_candidate_frames`` evenly-spaced
        frames from ``captured[start:]`` with their absolute steps. For pick 0
        (``start=0``) this spans reveal→swaps→approach, so the VLM can both ground the
        current pick AND nominate the post-swap frames it will need for a LATER pick."""
        sub = captured[start:]
        sub_steps = captured_steps[start:]
        n = len(sub)
        if n == 0:
            return [], []
        if n <= self.n_candidate_frames:
            return list(sub), list(sub_steps)
        idxs = np.linspace(0, n - 1, num=self.n_candidate_frames, dtype=int).tolist()
        seen, fr, st = set(), [], []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                fr.append(sub[i])
                st.append(sub_steps[i])
        return fr, st

    def rollout_streaming(
        self,
        episode_id: int,
        seed: Optional[int] = None,
        max_picks: int = 4,
        per_pick_max_steps: Optional[int] = None,
        recorder=None,
    ) -> Iterator[object]:
        """GENERATOR: the VLM owns the whole episode, accumulating a keyframe buffer
        across pick decision points — MemER's mechanism, single-call output, trained
        with GRPO (JOINT/STREAMING design). Drive it from the trainer:

            gen = worker.rollout_streaming(ep, seed)
            dp = next(gen)
            while isinstance(dp, DecisionPoint):
                cand = policy.sample_subgoals(key_frames=dp.key_frames,
                            recent_frames=dp.recent_frames, ..., k=1, mode="use")[0]
                dp = gen.send((cand.subtask, cand.keyframe_positions))
            reward = compute_reward(dp.success_flag, dp.progress, cfg)   # dp is RolloutResult

        At each pick the VLM sees ``key_frames`` = the accumulated buffer (empty at
        pick 0) + ``recent_frames`` = a broad window, and emits ONE
        ``{current_subtask, keyframe_positions}``. The nominated positions merge into
        the buffer (MemER clustering); what's kept at pick 0 (whose window spans the
        reveal+swaps) is the ONLY memory of pick 1's now-occluded target → selection
        causally moves reward. The oracle drives the presses / put-downs between
        picks; the VLM owns each grounded pick. ``from_qwen_xy`` converts the VLM
        coordinate at the SINGLE point it reaches the executor (oracle subgoals are
        NOT converted). The worker owns env lifecycle (the env is NOT closed here —
        the trainer closes the worker after the group, as ``rollout`` does)."""
        per_pick_cap = per_pick_max_steps or self.decision_warm_cap

        resp = self.client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

        init = self._ensure_env(episode_id, seed=seed)
        image_buf = deque(init["images"], maxlen=64)
        wrist_buf = deque(init["wrist_images"], maxlen=64)
        state_buf = deque(init["states"], maxlen=64)
        exec_start_idx = len(image_buf) - 1
        task_goal = init["task_goal"]
        if recorder is not None:
            recorder.task_goal = task_goal   # overlay the real per-episode instruction
        img, wrist, robot_state = image_buf[-1], wrist_buf[-1], state_buf[-1]

        n_steps = 0
        success_flag = "unknown"
        # Unbounded capture (an episode is < rollout_max_steps, cheap) so the early
        # reveal frames survive for the buffer; the rolling buffers above stay maxlen-64.
        captured: List[np.ndarray] = [img]
        captured_steps: List[int] = [n_steps]
        phase_seq: List[str] = []

        def _push(phase: str) -> None:
            if phase and (not phase_seq or phase_seq[-1] != phase):
                phase_seq.append(phase)

        _push(self._simple_phase(self.env_runner))
        buffer = KeyframeBuffer(dist=self.keyframe_cluster_dist, cap=self.keyframe_buffer_cap)
        picks_done = 0
        window_start = 0   # index into `captured` where THIS pick's broad window begins

        while picks_done < max_picks:
            # 1. Oracle-drive the deterministic scaffolding until we ENTER a pick phase.
            while (
                not self._is_pick_phase(self._simple_phase(self.env_runner))
                and n_steps < self.max_steps
            ):
                subgoal = self._oracle_subgoal()
                actions = self._infer_actions(
                    img, wrist, robot_state, task_goal, subgoal=subgoal,
                    image_buf=image_buf, state_buf=state_buf, exec_start_idx=exec_start_idx,
                )
                img, wrist, robot_state, n_steps, stop, success_flag, dead = self._stream_apply(
                    actions, image_buf, wrist_buf, state_buf, captured, captured_steps, n_steps,
                    recorder=recorder, subgoal=subgoal,
                )
                if dead:
                    yield self._finalize("error", image_buf[-1], n_steps)
                    return
                _push(self._simple_phase(self.env_runner))
                if stop:
                    break
            if not self._is_pick_phase(self._simple_phase(self.env_runner)):
                # Episode ended (stop) or ran out of budget before another pick — no
                # more decisions. Yield the outcome so the trainer scores the episode.
                if n_steps >= self.max_steps and success_flag in ("unknown", ""):
                    success_flag = "timeout"
                yield self._finalize(success_flag, img, n_steps)
                return

            # 2. At a pick decision point — hand the VLM the buffer + broad window.
            recent_frames, recent_steps = self._stream_window(captured, captured_steps, window_start)
            dp = DecisionPoint(
                key_frames=buffer.frames(),       # accumulated memory (empty at pick 0)
                recent_frames=recent_frames,      # broad window: ground current pick + nominate for later
                task_goal=task_goal,
                history_subgoals=self._history_before_pick(phase_seq),
                candidate_frames=recent_frames,
                current_frame=img,
                warm_steps=n_steps,
                terminated_early=False,
                success_flag=success_flag,
            )
            sent = yield dp
            subtask_qwen, kf_positions = sent if sent is not None else ("", [])

            # 3. Merge the nominated keyframes into the buffer (MemER clustering). Map
            #    the validated 1-indexed positions back to their (frame, abs_step).
            kept_pos = valid_keyframe_positions(kf_positions, len(recent_frames), self.max_keyframes)
            buffer.merge([TaggedFrame(recent_frames[p - 1], recent_steps[p - 1]) for p in kept_pos])

            # 4. Execute the VLM subtask until this pick completes / fails / budget.
            subtask = from_qwen_xy(subtask_qwen)   # SINGLE VLM→executor conversion point
            if not subtask.strip():
                # Blank pick — π0.5 with no subgoal can't advance; score where we are.
                yield self._finalize(success_flag, img, n_steps)
                return
            pick_deadline = n_steps + per_pick_cap
            stop = False
            while (
                self._is_pick_phase(self._simple_phase(self.env_runner))
                and n_steps < pick_deadline
                and n_steps < self.max_steps
            ):
                actions = self._infer_actions(
                    img, wrist, robot_state, task_goal, subgoal=subtask,
                    image_buf=image_buf, state_buf=state_buf, exec_start_idx=exec_start_idx,
                )
                img, wrist, robot_state, n_steps, stop, success_flag, dead = self._stream_apply(
                    actions, image_buf, wrist_buf, state_buf, captured, captured_steps, n_steps,
                    recorder=recorder, subgoal=subtask,
                )
                if dead:
                    yield self._finalize("error", image_buf[-1], n_steps)
                    return
                _push(self._simple_phase(self.env_runner))
                if stop:
                    break
            picks_done += 1
            window_start = len(captured) - 1   # next pick grounds only on post-this-pick frames + buffer
            if stop or n_steps >= self.max_steps:
                if n_steps >= self.max_steps and success_flag in ("unknown", ""):
                    success_flag = "timeout"
                yield self._finalize(success_flag, img, n_steps)
                return

        yield self._finalize(success_flag, img, n_steps)

    def rollout_oracle(
        self, episode_id: int, seed: Optional[int] = None, log_every: int = 0, recorder=None
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
                if recorder is not None:
                    recorder.record(img, wrist, robot_state, subgoal=subgoal)
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


__all__ = ["RolloutWorker", "RolloutResult", "DecisionPoint", "Snapshot"]
