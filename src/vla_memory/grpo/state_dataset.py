"""State dataset for GRPO: enumerates ``(task, episode_id, frame_idx)`` tuples.

Each sample is a *decision point* — a state at which the subgoal predictor is
asked to emit a subgoal during a real rollout. We don't precompute the actual
image arrays here because that would require running ManiSkill in a separate
process and storing tens of GB of frames. Instead the trainer loads the image
on demand by replaying the env to the requested frame_idx.

The mid-episode decision point is no longer encoded here: ``rollout.py`` warms
up with the oracle subgoal and stops at the subgoal *transition* (the moment the
task switches to the memory-dependent step), so each ``(task, episode_id)`` maps
to one post-occlusion decision automatically. ``frame_idx`` is retained for an
optional future "explicit decision step" override but is unused by the current
trainer.
"""

from __future__ import annotations

import dataclasses
from typing import List


@dataclasses.dataclass
class StateSample:
    task_name: str
    episode_id: int
    frame_idx: int = 0           # reserved; decision point is auto-detected in rollout.py
    has_video_demo: bool = False
    # Which PICK in a multi-pick task is this state's decision point (0-based).
    # The rollout oracle-drives picks 0..pick_index-1, then the VLM owns pick_index,
    # scored by progress. Lets each pick of a multi-pick episode (e.g. ButtonUnmaskSwap
    # = pick blue THEN green) be its own GRPO state → both picks earn reward.
    # IGNORED when ``streaming`` is set (the streaming rollout discovers all picks).
    pick_index: int = 0
    # Decision-point STREAMING (MemER-faithful, single-call output): the VLM owns the
    # WHOLE episode, accumulating a keyframe buffer across pick decision points, so a
    # selection at pick 0 causally drives pick 1's grounding. One state per episode;
    # ``rollout.rollout_streaming`` enumerates the picks internally (vs the per-pick
    # ``pick_index`` states above). An explicit boolean — NOT a ``pick_index`` sentinel
    # — so a missed trainer dispatch can't silently flow into ``peek_at_decision_point``.
    streaming: bool = False


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
        picks_per_episode: int = 1,
        streaming: bool = False,
    ) -> "StateDataset":
        """Enumerate states. ``picks_per_episode > 1`` emits one state per pick so each
        pick of a multi-pick task is its own GRPO group (the actual pick count is
        seed-dependent; a pick_index beyond what an episode has just yields a
        degenerate warm-up, dropped by dynamic sampling).

        ``streaming=True`` instead emits ONE state per episode (``picks_per_episode``
        ignored): the streaming rollout owns the whole episode and discovers its picks
        internally, accumulating the keyframe buffer across them."""
        n_picks = 1 if streaming else max(1, picks_per_episode)
        samples: List[StateSample] = []
        for task in task_names:
            for ep in range(episodes_per_task):
                for pick in range(n_picks):
                    samples.append(
                        StateSample(
                            task_name=task,
                            episode_id=ep,
                            frame_idx=0,
                            has_video_demo=task in tasks_with_video_demo,
                            pick_index=pick,
                            streaming=streaming,
                        )
                    )
        return cls(samples)


__all__ = ["StateDataset", "StateSample"]
