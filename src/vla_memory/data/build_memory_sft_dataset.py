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
import shutil
from typing import List, Optional, Tuple

import numpy as np

from vla_memory.qwen_subgoal.prompts import SUBGOAL_SYSTEM_PROMPT, build_user_prompt


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
        self.also_select = also_select   # also emit SELECT rows for the joint pipeline
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

        # Decision point: first frame where the online subgoal leaves the initial
        # (press) phase — the post-occlusion memory choice. Mirrors rollout._warmup.
        phase0 = simple_online(captured[0])
        decision_ts = None
        for ts in captured:
            if simple_online(ts) != phase0:
                decision_ts = ts
                break
        if decision_ts is None:
            return 0

        from vla_memory.qwen_subgoal.coords import to_qwen_xy
        grounded = to_qwen_xy(
            episode_data[f"timestep_{decision_ts}"]["info"]["grounded_subgoal_online"][()]
            .decode()
            .strip()
        )
        if not grounded:
            return 0

        window = [ts for ts in captured if ts <= decision_ts]
        n = len(window)
        reveal_end = min(self.reveal_window, n - 1)
        recent_ts = window[max(0, n - self.n_recent_frames):]
        if not recent_ts:
            return 0

        # Cache so a frame reused across augmentations is written once.
        saved: dict = {}

        def save(ts: int) -> str:
            if ts not in saved:
                saved[ts] = self._save_frame(episode_data, env_id, episode_idx, ts)
            return saved[ts]

        recent_paths = [save(ts) for ts in recent_ts]

        n_written = 0
        for a in range(self.augment_factor):
            key_pos = self._key_positions(reveal_end, a)
            key_ts = [window[p] for p in key_pos]
            key_paths = [save(ts) for ts in key_ts]

            user = build_user_prompt(
                task_goal=task_goal,
                n_key_frames=len(key_paths),
                n_recent_frames=len(recent_paths),
                # The completed-subtask timing signal: phase0 is the pre-transition
                # simple subgoal ("press the button"). Without it the model can't
                # tell the press is done and collapses to a constant "press the
                # button" output (the keyframes-only prompt's failure mode).
                history_subgoals=[phase0],
                has_video_demo=False,
            )
            assistant = json.dumps({"current_subtask": grounded, "keyframe_positions": []})
            self._write_row(user, assistant, key_paths + recent_paths)
            n_written += 1

            # SELECT row (joint pipeline): given the broad candidate window, KEEP
            # the reveal (cube-visible) frames. history=[] (the SELECT call is
            # "observe & select"); current_subtask = the action shown (phase0).
            # Parity with rollout._joint_group / _select_candidate_frames.
            if self.also_select:
                cand_pos = self._candidate_window(n, a)
                cand_ts = [window[p] for p in cand_pos]
                cand_paths = [save(ts) for ts in cand_ts]
                reveal_label = self._reveal_label(cand_ts, exec_start)
                sel_user = build_user_prompt(
                    task_goal=task_goal,
                    n_key_frames=0,
                    n_recent_frames=len(cand_paths),
                    history_subgoals=[],
                    has_video_demo=False,
                )
                sel_assistant = json.dumps(
                    {"current_subtask": phase0, "keyframe_positions": reveal_label}
                )
                self._write_row(sel_user, sel_assistant, cand_paths)
                n_written += 1
        return n_written

    def _write_row(self, user: str, assistant: str, image_paths: List[str]) -> None:
        row = {
            "messages": [
                {"role": "system", "content": SUBGOAL_SYSTEM_PROMPT},
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

    def _reveal_label(self, cand_ts: List[int], exec_start: int) -> List[int]:
        """1-indexed candidate positions whose timestep is in the reveal window
        (cube-visible) — the SELECT target — capped at ``max_keyframes``."""
        reveal = [j + 1 for j, ts in enumerate(cand_ts) if (ts - exec_start) < self.reveal_window]
        if len(reveal) > self.max_keyframes:
            idx = np.linspace(0, len(reveal) - 1, num=self.max_keyframes, dtype=int)
            reveal = [reveal[i] for i in idx]
        return reveal

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
