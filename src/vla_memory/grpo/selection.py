"""Keyframe-selection application — the causal link that makes selection trainable.

The VLM's ``keyframe_positions`` output (1-indexed into the candidate frames it
was shown) is turned here into the actual kept memory frames. In the joint
select-then-use loop (``JOINT_MEMORY_DESIGN.md``), the USE call sees ONLY these
kept frames, so the selection causally determines what the VLM can remember —
which is what lets GRPO put a reward gradient on the selection.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def valid_keyframe_positions(
    keyframe_positions: Sequence[int],
    n_candidates: int,
    max_keyframes: int,
) -> List[int]:
    """The 1-indexed positions ``apply_selection`` keeps: in-range, deduped, in
    original order, capped at ``max_keyframes``.

    Pure index logic, so callers can map the kept positions to BOTH frames and
    their step tags (the streaming rollout needs the abs-step of each kept frame to
    cluster the keyframe buffer). Defensive against free-form generation: non-ints
    and out-of-range / duplicate indices are dropped.
    """
    if n_candidates <= 0 or max_keyframes <= 0:
        return []
    seen: set[int] = set()
    out: List[int] = []
    for p in keyframe_positions:
        try:
            idx = int(p)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= n_candidates and idx not in seen:
            seen.add(idx)
            out.append(idx)
            if len(out) >= max_keyframes:
                break
    return out


def apply_selection(
    keyframe_positions: Sequence[int],
    candidates: Sequence[np.ndarray],
    max_keyframes: int,
) -> List[np.ndarray]:
    """Map 1-indexed ``keyframe_positions`` to the selected candidate frames.

    MemER convention: positions are 1-indexed into ``candidates`` (the frames the
    VLM was shown). An empty / all-invalid selection returns ``[]`` — the call then
    has no memory and should fail, which is the correct (negative) training signal,
    not an error. Delegates the index validation to ``valid_keyframe_positions`` so
    the streaming path (which also needs the kept positions for step-tagging) and
    this frame mapping share one source of truth.
    """
    if not candidates:
        return []
    positions = valid_keyframe_positions(keyframe_positions, len(candidates), max_keyframes)
    return [candidates[p - 1] for p in positions]


__all__ = ["apply_selection", "valid_keyframe_positions"]
