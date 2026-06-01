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


def apply_selection(
    keyframe_positions: Sequence[int],
    candidates: Sequence[np.ndarray],
    max_keyframes: int,
) -> List[np.ndarray]:
    """Map 1-indexed ``keyframe_positions`` to the selected candidate frames.

    MemER convention: positions are 1-indexed into ``candidates`` (the frames the
    VLM was shown). We are defensive because the positions come from free-form
    generation: non-integers and out-of-range / duplicate indices are dropped,
    original order is preserved, and the result is capped at ``max_keyframes``
    (keep the first valid ones). An empty / all-invalid selection returns ``[]``
    — the USE call then has no memory and should fail, which is the correct
    (negative) training signal, not an error.
    """
    if not candidates or max_keyframes <= 0:
        return []
    n = len(candidates)
    seen: set[int] = set()
    out: List[np.ndarray] = []
    for p in keyframe_positions:
        try:
            idx = int(p)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= n and idx not in seen:
            seen.add(idx)
            out.append(candidates[idx - 1])
            if len(out) >= max_keyframes:
                break
    return out


__all__ = ["apply_selection"]
