"""KeyframeBuffer — MemER's streaming keyframe memory (clustering + cap).

Pins the behaviour the streaming GRPO rollout relies on:
1. near-duplicate nominations (within ``dist`` steps) collapse to one (median);
2. distinct moments survive;
3. the cap preserves temporal COVERAGE (keeps early reveal frames, not just recent);
4. re-nominating the same ``abs_step`` never double-counts (idempotent merge);
5. ``frames()`` returns arrays in ``abs_step`` order.
"""

import numpy as np

from vla_memory.grpo.keyframe_buffer import KeyframeBuffer, TaggedFrame, _cluster_median


def _tf(step: int) -> TaggedFrame:
    # Encode the step into the frame so we can assert WHICH frame survived.
    return TaggedFrame(frame=np.full((2, 2, 3), step % 256, dtype=np.uint8), abs_step=step)


def _steps(buf: KeyframeBuffer):
    return [t.abs_step for t in buf.tagged()]


def test_clusters_collapse_near_duplicates_to_median():
    # 10,12,14 are within dist=8 of each other -> one cluster, median=12.
    out = _cluster_median([_tf(10), _tf(12), _tf(14)], dist=8)
    assert [t.abs_step for t in out] == [12]


def test_distinct_moments_survive():
    # 5 and 40 are > dist apart -> two clusters.
    out = _cluster_median([_tf(40), _tf(5)], dist=8)
    assert [t.abs_step for t in out] == [5, 40]


def test_merge_dedups_same_step_idempotent():
    buf = KeyframeBuffer(dist=8, cap=8)
    buf.merge([_tf(5), _tf(50)])
    buf.merge([_tf(5), _tf(50)])  # exact repeat — must not grow
    assert _steps(buf) == [5, 50]


def test_merge_accumulates_across_calls():
    buf = KeyframeBuffer(dist=4, cap=8)
    buf.merge([_tf(5)])
    buf.merge([_tf(60)])
    buf.merge([_tf(120)])
    assert _steps(buf) == [5, 60, 120]


def test_cap_preserves_temporal_coverage_not_just_recent():
    # Far-apart steps (no clustering at dist=2); cap=3 must keep span ends,
    # NOT drop the early reveal frame (the memory a Permanence task needs).
    buf = KeyframeBuffer(dist=2, cap=3)
    buf.merge([_tf(s) for s in (0, 16, 32, 48, 64, 200, 400)])
    steps = _steps(buf)
    assert len(steps) == 3
    assert steps[0] == 0 and steps[-1] == 400  # span endpoints retained
    assert steps == sorted(steps)


def test_frames_in_step_order_and_identity():
    buf = KeyframeBuffer(dist=4, cap=8)
    buf.merge([_tf(70), _tf(10)])
    frames = buf.frames()
    assert len(frames) == 2
    # Sorted by abs_step -> first frame encodes step 10, second step 70.
    assert int(frames[0][0, 0, 0]) == 10 and int(frames[1][0, 0, 0]) == 70


def test_empty_merge_is_noop():
    buf = KeyframeBuffer()
    buf.merge([])
    assert len(buf) == 0 and buf.frames() == []


def test_cap_must_be_positive():
    try:
        KeyframeBuffer(cap=0)
    except ValueError:
        return
    raise AssertionError("KeyframeBuffer accepted cap=0")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all keyframe_buffer tests passed")
