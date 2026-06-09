"""Streaming keyframe memory buffer — MemER's across-the-episode memory.

This is the piece that makes keyframe selection *causal* under GRPO. In MemER
(Pan et al., arXiv:2510.20328) the high-level VLM emits, every call, a single
JSON ``{current_subtask, keyframe_positions}``; the nominated ``keyframe_positions``
(frames from the current window) are accumulated into a persistent keyframe buffer
that becomes the ``past keyframes`` memory the VLM conditions on at *later* calls.
A redundancy-reducing clustering keeps the buffer small.

In our decision-point-streaming GRPO rollout (``rollout.rollout_streaming``) the VLM
is queried once per pick. What it nominates at pick 0 (whose broad window spans the
reveal + container-swaps) becomes the memory it must ground pick 1 on — pick 1's
target is occluded, so it can only come from the buffer. That causal link is what
puts a reward gradient on ``keyframe_positions``.

Frames carry their absolute episode step so clustering can merge temporally-nearby
nominations (MemER's ``merge_key_frame_paths(dist=8)``: group consecutive keyframes
within ``dist`` and keep the median). The one-shot / joint paths in ``rollout.py``
never need step tags and are untouched by this module.
"""

from __future__ import annotations

import dataclasses
from typing import List, Sequence

import numpy as np


@dataclasses.dataclass
class TaggedFrame:
    """A captured frame tagged with the absolute episode step it was seen at."""

    frame: np.ndarray
    abs_step: int


def _cluster_median(tagged: Sequence[TaggedFrame], dist: int) -> List[TaggedFrame]:
    """Single-linkage cluster ``tagged`` on ``abs_step`` (gap ``> dist`` splits a
    cluster) and keep each cluster's median element. Mirrors MemER's
    ``merge_key_frame_paths``: it collapses near-duplicate nominations (the same
    physical moment nominated twice across calls) without dropping distinct ones.
    Input need not be sorted; output is sorted by ``abs_step``.
    """
    if not tagged:
        return []
    ordered = sorted(tagged, key=lambda t: t.abs_step)
    clusters: List[List[TaggedFrame]] = [[ordered[0]]]
    for t in ordered[1:]:
        if t.abs_step - clusters[-1][-1].abs_step <= dist:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    # Median element of each cluster (by position — the cluster is sorted).
    return [c[len(c) // 2] for c in clusters]


class KeyframeBuffer:
    """Persistent, redundancy-reduced keyframe memory across an episode's calls.

    ``merge`` adds newly-nominated frames and re-clusters (``dist``); if the buffer
    still exceeds ``cap`` it is evenly subsampled across ``abs_step`` so temporal
    coverage (early reveal frames *and* recent ones) is preserved rather than
    biasing toward the most recent — the early reveal/swap frames are precisely the
    memory a Permanence task needs. Deduplication is by ``abs_step``: the same step
    nominated twice never double-counts.
    """

    def __init__(self, dist: int = 8, cap: int = 8) -> None:
        if cap <= 0:
            raise ValueError(f"KeyframeBuffer cap must be >= 1, got {cap}")
        self.dist = dist
        self.cap = cap
        self._frames: List[TaggedFrame] = []

    def merge(self, new: Sequence[TaggedFrame]) -> None:
        """Add ``new`` tagged frames, cluster to drop near-duplicates, cap by even
        temporal subsampling. Idempotent on repeated identical ``abs_step``s."""
        if not new:
            return
        by_step: dict[int, TaggedFrame] = {t.abs_step: t for t in self._frames}
        for t in new:
            by_step.setdefault(t.abs_step, t)  # first-seen wins; same step never duplicates
        clustered = _cluster_median(list(by_step.values()), self.dist)
        if len(clustered) > self.cap:
            idxs = np.linspace(0, len(clustered) - 1, num=self.cap, dtype=int)
            clustered = [clustered[i] for i in idxs]
        self._frames = clustered

    def frames(self) -> List[np.ndarray]:
        """The buffered frames as raw arrays, in ``abs_step`` order — what the
        next VLM call receives as its ``key_frames`` (past keyframes) memory."""
        return [t.frame for t in self._frames]

    def tagged(self) -> List[TaggedFrame]:
        return list(self._frames)

    def __len__(self) -> int:
        return len(self._frames)


__all__ = ["KeyframeBuffer", "TaggedFrame", "_cluster_median"]
