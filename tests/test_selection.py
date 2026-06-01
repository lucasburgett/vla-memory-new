"""Tests for keyframe-selection application (the causal select→use link).

``apply_selection`` turns the VLM's free-form ``keyframe_positions`` (1-indexed)
into the kept memory frames. It must be defensive: positions come from generation,
so out-of-range / duplicate / non-integer values can't crash a rollout. See
``vla_memory.grpo.selection`` and ``JOINT_MEMORY_DESIGN.md``.
"""

import numpy as np

from vla_memory.grpo.selection import apply_selection


def _frames(n):
    # frame i is a constant image of value i, so we can identify which was kept.
    return [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]


def test_basic_1indexed_order_preserved():
    c = _frames(5)
    sel = apply_selection([1, 3], c, max_keyframes=4)
    assert len(sel) == 2
    assert sel[0][0, 0, 0] == 0  # position 1 → candidates[0]
    assert sel[1][0, 0, 0] == 2  # position 3 → candidates[2]


def test_caps_at_max_keyframes():
    sel = apply_selection([1, 2, 3, 4, 5], _frames(10), max_keyframes=3)
    assert len(sel) == 3


def test_drops_out_of_range_and_duplicates():
    c = _frames(3)
    sel = apply_selection([0, 1, 1, 9, 2, -1], c, max_keyframes=4)
    # valid & unique: 1→idx0, 2→idx1; 0/9/-1 out of range; second 1 is a dup
    assert [f[0, 0, 0] for f in sel] == [0, 1]


def test_empty_and_all_invalid_selection():
    assert apply_selection([], _frames(3), 4) == []
    assert apply_selection([5, 6, 99], _frames(3), 4) == []  # nothing in range


def test_non_integer_tolerated():
    c = _frames(3)
    sel = apply_selection(["2", None, 1.0], c, max_keyframes=4)
    # "2"→2 (idx1), None skipped, 1.0→1 (idx0); order preserved
    assert [f[0, 0, 0] for f in sel] == [1, 0]


def test_zero_max_keyframes_returns_empty():
    assert apply_selection([1, 2], _frames(3), 0) == []
