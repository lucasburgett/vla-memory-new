"""State dataset for GRPO: enumerates ``(task, episode_id, frame_idx)`` tuples.

Each sample is a *decision point* — a state at which the subgoal predictor is
asked to emit a subgoal during a real rollout. We don't precompute the actual
image arrays here because that would require running ManiSkill in a separate
process and storing tens of GB of frames. Instead the trainer loads the image
on demand by replaying the env to the requested frame_idx.

For early-stage training, ``frame_idx`` is simply 0 (initial frame). Mid-episode
states are a TODO once env state checkpointing is wired up in ``rollout.py``.
"""

from __future__ import annotations

import dataclasses
from typing import List


@dataclasses.dataclass
class StateSample:
    task_name: str
    episode_id: int
    frame_idx: int = 0           # 0 = initial frame; >0 = mid-episode (TODO)
    has_video_demo: bool = False


class StateDataset:
    """Iterable over rollout starting points.

    Built from the RoboMME task list. The dataset is small enough (16 tasks ×
    ~50 train episodes = 800 samples) that we just hold it in memory.
    """

    def __init__(self, samples: List[StateSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> StateSample:
        return self._samples[idx]

    @classmethod
    def from_task_list(
        cls,
        task_names: List[str],
        episodes_per_task: int,
        tasks_with_video_demo: List[str],
    ) -> "StateDataset":
        samples: List[StateSample] = []
        for task in task_names:
            for ep in range(episodes_per_task):
                samples.append(
                    StateSample(
                        task_name=task,
                        episode_id=ep,
                        frame_idx=0,
                        has_video_demo=task in tasks_with_video_demo,
                    )
                )
        return cls(samples)


__all__ = ["StateDataset", "StateSample"]
