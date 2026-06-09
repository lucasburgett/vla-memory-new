"""Build the memory SFT dataset for the MemER-style Qwen subgoal predictor.

This is the warm-start data for GRPO on the Permanence suite (ButtonUnmask first).
It deliberately diverges from the submodule's MemER builder
(``build_vlm_subgoal_dataset_memer.py``) in two ways the causality probes forced:

1. **Keyframes come from the REVEAL window**, not subgoal-transition frames. The
   bins drop at ~t=64 (cubes covered), so transition-time frames (~t=80) show
   covered cubes and carry no memory. We sample keyframes from the first
   ``reveal_window`` execution frames — exactly what ``rollout._select_memory_frames``
   feeds the VLM at inference (the prompt-parity invariant).

2. **Literal ``<y, x>`` grounding**, not the builder's ``<bbox>`` placeholder +
   ``objects.bbox`` (Qwen grounding tokens). GroundSG consumes the literal oracle
   text ``"...at <y, x>..."`` directly (verified by the probe), so the VLM is
   trained to emit that — no conversion at inference.

Each episode's first post-occlusion decision (the press→pick subgoal transition)
is the memory choice the VLM makes once and holds (the probe showed a single held
subgoal == the online oracle). ButtonUnmask train has only ~100 episodes, which is
too thin for SFT, so we emit ``augment_factor`` rows per episode with DIFFERENT
reveal-window keyframe subsets but the SAME target — any subset of the reveal
window shows the cube layout, so this teaches robustness to keyframe selection
without changing the answer. Row 0 is the exact even-spacing inference uses.

Row shape is ms-swift's: ``{"messages": [...], "images": [keyframes..., recent...]}``
with the user turn built by ``prompts.build_user_prompt`` (same function as
inference) and the assistant turn the JSON ``{"current_subtask": "<grounded pick
subgoal>", "keyframe_positions": []}``.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from typing import List, Optional, Tuple

import numpy as np

# Strips the copyable "that hides the {colour} cube" suffix off the grounded pick
# subgoal so the COORDINATE is the only variable token span in the SFT target.
_HIDES_SUFFIX = re.compile(r"\s+that hides the .*$", re.IGNORECASE)

# Pixel <y,x> 0–256 → Qwen-native <x,y> 0–1000. The pair (with from_qwen_xy, used
# at inference in rollout) lives in qwen_subgoal.coords so the build-time target
# and the inference round-trip share one source of truth.
from vla_memory.qwen_subgoal.coords import to_qwen_xy as _to_qwen_xy
from vla_memory.qwen_subgoal.prompts import (
    SELECT_SYSTEM_PROMPT,
    SUBGOAL_SYSTEM_PROMPT,
    build_user_prompt,
)


class MemorySFTBuilder:
    """Build memory SFT rows from RoboMME demonstration HDF5s for chosen tasks."""

    def __init__(
        self,
        raw_data_path: str,
        preprocessed_data_path: str,
        vlm_dir_name: str = "memory",
        only_tasks: Tuple[str, ...] = ("ButtonUnmask",),
        n_key_frames: int = 4,
        n_recent_frames: int = 2,
        reveal_window: int = 64,
        augment_factor: int = 5,
        n_candidate_frames: int = 12,
        max_keyframes: int = 4,
        also_select: bool = True,
        streaming: bool = False,
        keyframe_buffer_cap: int = 8,
        seed: int = 0,
        max_episodes: Optional[int] = None,
    ) -> None:
        self.raw_data_path = raw_data_path
        self.only_tasks = set(only_tasks) if only_tasks else set()
        self.n_key_frames = n_key_frames
        self.n_recent_frames = n_recent_frames
        self.reveal_window = reveal_window
        self.augment_factor = max(1, augment_factor)
        self.n_candidate_frames = n_candidate_frames
        self.max_keyframes = max_keyframes
        # streaming = MemER-faithful SINGLE-call rows ({current_subtask, keyframe_positions}
        # BOTH populated), with the keyframe buffer carried across picks. Supersedes the
        # two-call SELECT/USE rows; when set, also_select is ignored (no separate SELECT row).
        self.streaming = streaming
        self.keyframe_buffer_cap = keyframe_buffer_cap
        self.also_select = also_select and not streaming
        self.max_episodes = max_episodes
        self._rng = random.Random(seed)

        self.data_dir = os.path.join(preprocessed_data_path, vlm_dir_name)
        self.images_dir = os.path.join(self.data_dir, "images")
        self.grounded_train_path = os.path.join(self.data_dir, "grounded_subgoal_train.jsonl")
        self._setup_output_dirs()

    def _setup_output_dirs(self) -> None:
        if os.path.exists(self.images_dir):
            shutil.rmtree(self.images_dir)
        os.makedirs(self.images_dir, exist_ok=True)
        if os.path.exists(self.grounded_train_path):
            os.remove(self.grounded_train_path)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Process the selected tasks' H5 files. Returns the number of rows written."""
        import h5py  # lazy: only present in the Modal/submodule env

        from mme_vla_suite.dataset_builder.robomme_h5_utils import (  # type: ignore
            get_env_id_from_filename,
            get_episode_indices,
        )

        n_rows = 0
        n_eps = 0
        files = sorted(f for f in os.listdir(self.raw_data_path) if f.endswith(".h5"))
        for file in files:
            env_id = get_env_id_from_filename(file)
            if self.only_tasks and env_id not in self.only_tasks:
                continue
            print(f"[memory-sft] processing {file} (env={env_id})", flush=True)
            with h5py.File(os.path.join(self.raw_data_path, file), "r") as data:
                for episode_idx in get_episode_indices(data, self.max_episodes):
                    written = self._process_episode(data, env_id, episode_idx)
                    n_rows += written
                    n_eps += 1 if written else 0
        print(
            f"[memory-sft] wrote {n_rows} rows from {n_eps} episodes "
            f"(augment_factor={self.augment_factor}) → {self.grounded_train_path}",
            flush=True,
        )
        return n_rows

    def _process_episode(self, env_dataset, env_id: str, episode_idx: int) -> int:
        from mme_vla_suite.dataset_builder.robomme_h5_utils import (  # type: ignore
            first_execution_step,
            get_task_goal,
            get_timestep_indices,
        )

        episode_data = env_dataset[f"episode_{episode_idx}"]
        task_goal = get_task_goal(episode_data, lower=True)
        timestep_idxs = list(get_timestep_indices(episode_data))
        if not timestep_idxs:
            return 0
        exec_start = int(first_execution_step(episode_data))

        def simple_online(ts: int) -> str:
            return (
                episode_data[f"timestep_{ts}"]["info"]["simple_subgoal_online"][()]
                .decode()
                .strip()
                .lower()
            )

        captured = [ts for ts in timestep_idxs if ts >= exec_start]
        if len(captured) < 2:
            return 0

        # All PICK decision points. A multi-pick task (ButtonUnmaskSwap picks blue THEN
        # green) has several; each is its own memory decision + SFT row-set so joint GRPO
        # earns reward on every pick. Detect the START of each distinct "pick up the
        # container ..." phase (single-pick tasks yield exactly one).
        pick_starts: List[int] = []
        prev = None
        for ts in captured:
            s = simple_online(ts)
            if "pick up the container" in s and s != prev:
                pick_starts.append(ts)
            prev = s
        if not pick_starts:
            return 0

        # Frame cache shared across picks + augmentations (each frame written once).
        saved: dict = {}

        def save(ts: int) -> str:
            if ts not in saved:
                saved[ts] = self._save_frame(episode_data, env_id, episode_idx, ts)
            return saved[ts]

        if self.streaming:
            return self._emit_episode_streaming(
                episode_data, task_goal, captured, pick_starts, simple_online, save
            )

        n_rows = 0
        for decision_ts in pick_starts:
            n_rows += self._emit_pick(
                episode_data, task_goal, captured, decision_ts, simple_online, save
            )
        return n_rows

    def _emit_episode_streaming(
        self,
        episode_data,
        task_goal: str,
        captured: List[int],
        pick_starts: List[int],
        simple_online,
        save,
    ) -> int:
        """Emit the MemER single-call STREAMING rows for one episode.

        One row per (pick, augment): a single MemER call with key_frames = the keyframe
        buffer the streaming rollout would hold at that pick (empty at pick 0; the prior
        picks' nominations after, accumulated + capped) and recent_frames = this pick's
        broad window. Target = ``{current_subtask, keyframe_positions}`` with BOTH fields
        populated. The buffer is threaded across picks so SFT mirrors the streaming
        rollout (rollout.rollout_streaming): pick 0's window spans reveal→swaps so its
        nominations capture the memory; pick i>0's window is occluded so grounding must
        come from the buffer. Mirrors trainer._streaming_group's prompt structure (parity).
        """
        # Split captured into per-pick windows at the pick boundaries: pick 0 spans
        # reveal→pick0 (cube-visible); pick i>0 spans (pick(i-1)_start, pick_i] (occluded),
        # matching the streaming rollout's window_start advancing past each completed pick.
        windows: List[List[int]] = []
        prev = None
        for ds in pick_starts:
            if prev is None:
                windows.append([ts for ts in captured if ts <= ds])
            else:
                windows.append([ts for ts in captured if prev < ts <= ds])
            prev = ds

        n_rows = 0
        buffer_paths: List[str] = []   # the keyframe buffer the rollout holds at this pick
        for decision_ts, window in zip(pick_starts, windows):
            if not window:
                continue
            # History = distinct completed subtasks before THIS pick (parity with
            # rollout._history_before_pick); on Swap pick 1 it includes the completed pick 0.
            history: List[str] = []
            for ts in captured:
                if ts >= decision_ts:
                    break
                sg = simple_online(ts)
                if sg and sg not in history:
                    history.append(sg)
            written, next_buffer = self._emit_pick_streaming(
                episode_data, task_goal, window, history, decision_ts, simple_online, save, buffer_paths
            )
            n_rows += written
            # Accumulate the buffer across picks (MemER), capped — most-recent picks' frames.
            buffer_paths = (buffer_paths + next_buffer)[-self.keyframe_buffer_cap:]
        return n_rows

    def _emit_pick_streaming(
        self,
        episode_data,
        task_goal: str,
        window: List[int],
        history: List[str],
        decision_ts: int,
        simple_online,
        save,
        buffer_paths: List[str],
    ) -> Tuple[int, List[str]]:
        """Emit the streaming rows for ONE pick. Returns (n_written, next_buffer_paths)
        where next_buffer_paths = the frames this pick nominates (a=0 even-spacing,
        inference-matching), to carry into the next pick's buffer."""
        grounded = (
            episode_data[f"timestep_{decision_ts}"]["info"]["grounded_subgoal_online"][()]
            .decode()
            .strip()
        )
        if not grounded:
            return 0, []
        grounded = _HIDES_SUFFIX.sub("", grounded).strip()
        grounded = _to_qwen_xy(grounded)   # <y,x> 0–255 px → Qwen-native <x,y> 0–1000

        n = len(window)
        important_ts = self._memer_important(episode_data, window, simple_online)

        next_buffer: List[str] = []
        n_written = 0
        for a in range(self.augment_factor):
            cand_pos = self._candidate_window(n, a)
            cand_ts = [window[p] for p in cand_pos]
            cand_paths = [save(ts) for ts in cand_ts]
            sel_label = self._memer_label(cand_ts, important_ts)
            user = build_user_prompt(
                task_goal=task_goal,
                n_key_frames=len(buffer_paths),
                n_recent_frames=len(cand_paths),
                history_subgoals=history,
                has_video_demo=False,
            )
            # The MemER single-call target: BOTH current_subtask AND keyframe_positions.
            assistant = json.dumps({"current_subtask": grounded, "keyframe_positions": sel_label})
            self._write_row(user, assistant, buffer_paths + cand_paths)
            n_written += 1
            if a == 0:
                next_buffer = [cand_paths[p - 1] for p in sel_label]
        return n_written, next_buffer

    def _emit_pick(
        self,
        episode_data,
        task_goal: str,
        captured: List[int],
        decision_ts: int,
        simple_online,
        save,
    ) -> int:
        """Emit the USE (+ SELECT, if joint) rows for ONE pick at ``decision_ts``.

        The oracle drives earlier picks at rollout time, so each pick is an independent
        memory decision — its own keyframes, grounded target, and completed history.
        """
        # History = distinct completed subtasks before THIS pick (presses + earlier
        # picks + put-downs). On Swap pick 2 this INCLUDES the completed pick 1 → the
        # prompt tells the model which colour it's now on. MUST match
        # rollout._history_before_pick for prompt parity.
        history: List[str] = []
        for ts in captured:
            if ts >= decision_ts:
                break
            sg = simple_online(ts)
            if sg and sg not in history:
                history.append(sg)

        grounded = (
            episode_data[f"timestep_{decision_ts}"]["info"]["grounded_subgoal_online"][()]
            .decode()
            .strip()
        )
        if not grounded:
            return 0
        # Coordinate-focused target: drop "that hides the {colour} cube" so the
        # coordinate (the memory output) is the dominant, only-variable span; colour
        # stays in the PROMPT. GroundSG is coordinate-driven (probe: colour-insensitive).
        grounded = _HIDES_SUFFIX.sub("", grounded).strip()
        grounded = _to_qwen_xy(grounded)   # <y,x> 0–255 px → Qwen-native <x,y> 0–1000

        window = [ts for ts in captured if ts <= decision_ts]
        n = len(window)
        reveal_end = min(self.reveal_window, n - 1)
        recent_ts = window[max(0, n - self.n_recent_frames):]
        if not recent_ts:
            return 0
        recent_paths = [save(ts) for ts in recent_ts]

        # Memer-style "important" timesteps over THIS pick's pre-decision window:
        # subgoal transitions + action-velocity minima. On Swap the transitions align
        # with the container swaps, so the kept frames span reveal→swaps (what the USE
        # call needs to deduce the post-swap container position).
        important_ts = self._memer_important(episode_data, window, simple_online)

        n_written = 0
        for a in range(self.augment_factor):
            if self.also_select:
                # JOINT: the USE call must train on the SAME frames a correct SELECT
                # keeps — at GRPO the USE call receives apply_selection's output, NOT a
                # reveal-window slice. So build the candidate window + memer SELECT label
                # first, then point the USE keyframes at the selected candidates. This
                # keeps SFT-USE ≡ GRPO-USE (the kept frames) and, on Swap, trains the
                # USE head on the swap-tracking frames it will actually receive.
                cand_pos = self._candidate_window(n, a)
                cand_ts = [window[p] for p in cand_pos]
                cand_paths = [save(ts) for ts in cand_ts]
                sel_label = self._memer_label(cand_ts, important_ts)
                key_paths = [cand_paths[p - 1] for p in sel_label]
            else:
                # ONE-SHOT: reveal-window even-spacing — matches rollout._select_memory_frames
                # and preserves the validated coordinate checkpoint's prompt (parity).
                key_pos = self._key_positions(reveal_end, a)
                key_paths = [save(window[p]) for p in key_pos]

            user = build_user_prompt(
                task_goal=task_goal,
                n_key_frames=len(key_paths),
                n_recent_frames=len(recent_paths),
                # The completed-subtask timing signal (e.g. ["press the button"], or
                # both presses on ButtonUnmaskSwap). Without it the model can't tell
                # the press phase is done and collapses to a constant "press the
                # button" output (the keyframes-only prompt's failure mode). Matches
                # rollout.peek_at_decision_point's history (prompt parity).
                history_subgoals=history,
                has_video_demo=False,
            )
            assistant = json.dumps({"current_subtask": grounded, "keyframe_positions": []})
            self._write_row(user, assistant, key_paths + recent_paths)
            n_written += 1

            # SELECT row (joint pipeline): from the broad candidate window, KEEP the
            # memer-important frames (transitions + action-velocity minima → spans
            # reveal and, on Swap, the container swaps). DISTINCT schema from the USE
            # row — SELECT_SYSTEM_PROMPT + mode="select" prompt + a target carrying
            # ONLY keyframe_positions (no current_subtask). The two calls share near-
            # identical images, so without distinct prompts the model collapsed
            # keyframe_positions to the USE majority `[]` and the SELECT head learned
            # nothing (project_sft_plan_adjustments). history=[] (observe & select);
            # parity with trainer._joint_group's SELECT call (mode="select").
            if self.also_select:
                sel_user = build_user_prompt(
                    task_goal=task_goal,
                    n_key_frames=0,
                    n_recent_frames=len(cand_paths),
                    history_subgoals=[],
                    has_video_demo=False,
                    mode="select",
                )
                sel_assistant = json.dumps({"keyframe_positions": sel_label})
                self._write_row(
                    sel_user, sel_assistant, cand_paths, system=SELECT_SYSTEM_PROMPT
                )
                n_written += 1
        return n_written

    def _write_row(
        self,
        user: str,
        assistant: str,
        image_paths: List[str],
        system: str = SUBGOAL_SYSTEM_PROMPT,
    ) -> None:
        row = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "images": image_paths,
        }
        with open(self.grounded_train_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _candidate_window(self, n_window: int, aug_index: int) -> List[int]:
        """Positions into ``window`` for the SELECT candidate frames. a=0 even
        spacing (matches ``rollout._select_candidate_frames``); a>0 jittered for
        robustness to the rollout's variable warm-up length."""
        nc = min(self.n_candidate_frames, n_window)
        if nc <= 0:
            return []
        if aug_index == 0 or nc >= n_window:
            pos = np.linspace(0, n_window - 1, num=nc, dtype=int).tolist()
        else:
            pos = sorted(self._rng.sample(range(n_window), nc))
        seen, out = set(), []
        for p in pos:
            if p not in seen:
                seen.add(p)
                out.append(int(p))
        return out

    def _memer_important(self, episode_data, window: List[int], simple_online) -> List[int]:
        """Memer-style 'important' timesteps over the pre-pick ``window``: subgoal
        transitions + action-velocity minima (reuses the submodule's
        ``find_local_minima``). RoboMME's own keyframe definition — task-agnostic, so
        it generalizes from ButtonUnmask to Swap (the transitions line up with the
        container swaps). Returns absolute timesteps; the start frame (reveal) is
        always included."""
        if not window:
            return []
        from mme_vla_suite.dataset_builder.build_vlm_subgoal_dataset_memer import (  # type: ignore
            find_local_minima,
        )

        trans, prev = [], None
        for ts in window:
            s = simple_online(ts)
            if prev is not None and s != prev:
                trans.append(ts)
            prev = s
        try:
            minima_pos = find_local_minima(episode_data, window)
        except Exception as exc:  # missing joint_state/action keys, etc. — degrade gracefully
            print(f"[memory-sft] find_local_minima failed ({exc!r}); transitions only", flush=True)
            minima_pos = []
        minima_ts = [window[i] for i in minima_pos if 0 <= i < len(window)]
        return sorted(set([window[0]] + trans + minima_ts))

    def _memer_label(self, cand_ts: List[int], important_ts: List[int]) -> List[int]:
        """1-indexed candidate positions NEAREST the memer important timesteps — the
        SELECT target, capped at ``max_keyframes``. Falls back to even spacing if no
        important frames were found, so the SELECT head still gets a non-empty signal."""
        if not cand_ts:
            return []
        if not important_ts:
            k = min(self.max_keyframes, len(cand_ts))
            return sorted({int(p) + 1 for p in np.linspace(0, len(cand_ts) - 1, num=k, dtype=int)})
        labels = set()
        for imp in important_ts:
            j = min(range(len(cand_ts)), key=lambda k: abs(cand_ts[k] - imp))
            labels.add(j + 1)
        out = sorted(labels)
        if len(out) > self.max_keyframes:
            idx = np.linspace(0, len(out) - 1, num=self.max_keyframes, dtype=int)
            out = [out[i] for i in idx]
        return out

    # ------------------------------------------------------------------
    # Memory-frame selection
    # ------------------------------------------------------------------

    def _key_positions(self, reveal_end: int, aug_index: int) -> List[int]:
        """Keyframe positions within the reveal window [0, reveal_end].

        ``aug_index == 0`` is even spacing — EXACTLY what ``rollout._select_memory_frames``
        uses at inference (parity). ``aug_index > 0`` draws a random distinct subset
        from the reveal window (same cube layout, different frames) so the VLM
        learns the answer depends on the layout, not the specific frames.
        """
        if reveal_end <= 0 or self.n_key_frames <= 0:
            return []
        if aug_index == 0:
            pos = np.linspace(0, reveal_end, num=self.n_key_frames, dtype=int).tolist()
        else:
            k = min(self.n_key_frames, reveal_end + 1)
            pos = sorted(self._rng.sample(range(0, reveal_end + 1), k))
        seen, out = set(), []
        for p in pos:
            if p not in seen:
                seen.add(p)
                out.append(int(p))
        return out

    def _select_memory_idxs(self, window: List[int]) -> Tuple[List[int], List[int]]:
        """Inference-matching split of a window into (even-spaced keyframes, recent).

        Kept for the parity test; ``_process_episode`` uses ``_key_positions`` so it
        can also emit augmented subsets. Index space matches
        ``rollout._select_memory_frames``.
        """
        n = len(window)
        if n == 0:
            return [], []
        reveal_end = min(self.reveal_window, n - 1)
        recent = window[max(0, n - self.n_recent_frames):]
        key = [window[p] for p in self._key_positions(reveal_end, 0)]
        return key, recent

    def _save_frame(self, episode_data, env_id: str, episode_idx: int, ts: int) -> str:
        import imageio  # lazy

        img = np.asarray(episode_data[f"timestep_{ts}"]["obs"]["front_rgb"][()], dtype=np.uint8)
        fname = f"{env_id}_ep{episode_idx}_{ts}.png"
        path = os.path.join(self.images_dir, fname)
        imageio.imwrite(path, img)
        return path


def build_memory_sft_dataset(
    raw_data_path: str,
    preprocessed_data_path: str,
    only_tasks: Tuple[str, ...] = ("ButtonUnmask",),
    n_key_frames: int = 4,
    n_recent_frames: int = 2,
    reveal_window: int = 64,
    augment_factor: int = 5,
    n_candidate_frames: int = 12,
    max_keyframes: int = 4,
    also_select: bool = True,
    streaming: bool = False,
    keyframe_buffer_cap: int = 8,
    seed: int = 0,
    max_episodes: Optional[int] = None,
) -> dict:
    """Build the memory SFT dataset and return its output paths + row count."""
    builder = MemorySFTBuilder(
        raw_data_path=raw_data_path,
        preprocessed_data_path=preprocessed_data_path,
        only_tasks=only_tasks,
        n_key_frames=n_key_frames,
        n_recent_frames=n_recent_frames,
        reveal_window=reveal_window,
        augment_factor=augment_factor,
        n_candidate_frames=n_candidate_frames,
        max_keyframes=max_keyframes,
        also_select=also_select,
        streaming=streaming,
        keyframe_buffer_cap=keyframe_buffer_cap,
        seed=seed,
        max_episodes=max_episodes,
    )
    n_rows = builder.run()
    return {
        "grounded_subgoal_train": builder.grounded_train_path,
        "images_dir": builder.images_dir,
        "n_rows": n_rows,
    }


__all__ = ["MemorySFTBuilder", "build_memory_sft_dataset"]
