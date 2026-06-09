"""Snapshot / restore a ManiSkill env at a mid-episode decision point.

The GRPO loop warms an env to a decision point once per group, then branches K
candidate rollouts from it. Re-warming K times (``rollout._warmup`` →
``_ensure_env`` → ``make_env``) is correct but dominates wall-clock. This module
captures the env state once and restores it for each candidate.

Why this is more than ``env.get_state_dict()``: ManiSkill's
``get_state_dict``/``set_state_dict`` restores only the *physics* (actor /
articulation poses + velocities). It does NOT restore:

  * ``_elapsed_steps`` — the time counter ButtonUnmask/Swap key their scripted
    scene dynamics on (bin lift/drop over 0–64; container swaps in [64,114),
    [114,164),[164,214); cube-gluing until ~214). A decision point on Swap sits
    *inside* those windows, so a wrong ``elapsed_steps`` re-fires swaps → garbage.
  * The sequential task tracker (``utils/subgoal_evaluate_func.sequential_task_check``):
    ``timestep`` (persistent counter), ``current_task_index`` (read by the GRPO
    dense reward), ``_timelimit_deadlines``, ``first_timestep``, ``button_list``
    (mutated by ``.remove()``), ``_lift_drop_cache`` / ``_two_lane_swaps`` (inner
    dicts holding mutable np arrays), swap-pair actor refs, ``currentpickup`` …
  * ``DemonstrationWrapper`` state ONE LAYER above the base env:
    ``episode_success`` / ``steps_without_demonstration`` → ``info["status"]``
    (→ the rollout's ``success_flag``), plus the grounded-subgoal fill cache.

So we snapshot the FULL wrapper chain's ``__dict__`` (not just ``.unwrapped``)
with a recursive *copy-plain / share-complex* policy: deep-copy the mutable
bookkeeping (so a candidate's in-place mutations never leak into the snapshot),
clone tensors (restores ``_elapsed_steps`` & the success flags), reset RNG
generators, and SHARE by reference the heavy sim handles (actors, scene, agent,
sensors, lambdas) whose physical state ``set_state_dict`` already restores.

This is unavoidably coupled to internals we cannot edit (the submodule), so the
correctness contract is verified empirically by ``snapshot_parity_probe.py``: a
record-then-replay parity check that asserts a restored env reproduces an
identical reward + task-index trajectory + final physics, bit-for-bit.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Tuple

import numpy as np
import torch

# Recursion backstop for the copy walk. Env bookkeeping nests at most ~3–4 deep
# (e.g. ``task_list`` list→dict→lambda, ``_two_lane_swaps`` dict→dict→ndarray);
# beyond this we share by reference rather than risk a pathological cycle.
_MAX_DEPTH = 8


@dataclasses.dataclass
class _GeneratorState:
    """A captured ``torch.Generator`` — the live generator plus a clone of its
    byte-state. Restored in place (the env keeps the same Generator object; we
    only rewind its RNG state) so seeded draws during ``step`` reproduce across
    candidates."""

    gen: torch.Generator
    state: torch.Tensor


def _copy_value(v: Any, depth: int = 0) -> Any:
    """Recursive copy-plain / share-complex.

    Plain/mutable bookkeeping is copied (so the snapshot is immutable across K
    restores); engine handles are shared by reference (``set_state_dict`` owns
    their physical state). See the module docstring for the rationale.
    """
    # Immutable scalars — safe to share.
    if v is None or isinstance(v, (bool, int, float, complex, str, bytes)):
        return v
    if isinstance(v, np.generic):  # np scalar (np.int64, np.float32, …) — immutable
        return v
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    if isinstance(v, torch.Generator):
        return _GeneratorState(v, v.get_state().clone())
    if isinstance(v, _GeneratorState):
        # Re-copying a previously captured generator (restore path): keep the same
        # live generator, re-clone the saved state so it is never consumed.
        return _GeneratorState(v.gen, v.state.clone())
    if isinstance(v, bytearray):
        return bytearray(v)
    if depth >= _MAX_DEPTH:
        # Too deep to keep recursing safely — share by reference.
        return v
    if isinstance(v, dict):
        return {k: _copy_value(val, depth + 1) for k, val in v.items()}
    if isinstance(v, list):
        return [_copy_value(x, depth + 1) for x in v]
    if isinstance(v, tuple):
        return tuple(_copy_value(x, depth + 1) for x in v)
    if isinstance(v, set):
        return {_copy_value(x, depth + 1) for x in v}
    # Everything else (sapien actors/scene/agent/sensors, lambdas, modules,
    # builders, np.random.Generator, …) — share by reference. Physics is
    # restored separately via set_state_dict; callables are stable.
    return v


def snapshot_layer(obj: Any) -> dict:
    """Capture ``obj.__dict__`` (one wrapper/env layer). Records the key set so
    restore can delete attributes created *after* the snapshot (e.g.
    ``first_timestep``, ``_swap_rng``)."""
    d = obj.__dict__
    return {"keys": set(d.keys()), "values": {k: _copy_value(v) for k, v in d.items()}}


def restore_layer(obj: Any, snap: dict) -> None:
    """Restore one layer: delete post-snapshot keys, then reassign each captured
    value (re-copied from the snapshot so it survives K restores intact)."""
    for k in list(obj.__dict__.keys()):
        if k not in snap["keys"]:
            try:
                delattr(obj, k)
            except AttributeError:
                pass
    for k, stored in snap["values"].items():
        if isinstance(stored, _GeneratorState):
            setattr(obj, k, stored.gen)            # keep the same Generator object
            stored.gen.set_state(stored.state.clone())   # rewind its RNG state
        else:
            setattr(obj, k, _copy_value(stored))


@dataclasses.dataclass
class EnvSnapshot:
    """A restorable env state: physics (``get_state_dict``) + per-layer
    bookkeeping for the whole wrapper chain (outermost → unwrapped)."""

    physics: Any
    layers: List[Tuple[Any, dict]]


def _walk_chain(env: Any) -> List[Any]:
    """Wrapper chain outermost → unwrapped, via gymnasium ``Wrapper.env``."""
    nodes: List[Any] = []
    node = env
    for _ in range(17):  # 16-layer cap; real chain is FailAware→Demonstration→Base
        nodes.append(node)
        if node is node.unwrapped:
            return nodes
        nxt = getattr(node, "env", None)
        if nxt is None or nxt is node:
            return nodes
        node = nxt
    raise RuntimeError("env wrapper chain too deep or cyclic (>16 layers)")


def snapshot_env(env: Any) -> EnvSnapshot:
    """Snapshot every wrapper layer's bookkeeping + the base env physics."""
    layers = [(node, snapshot_layer(node)) for node in _walk_chain(env)]
    physics = _copy_value(env.unwrapped.get_state_dict())
    return EnvSnapshot(physics=physics, layers=layers)


def restore_env(env: Any, snap: EnvSnapshot) -> None:
    """Restore physics first (poses/velocities), then per-layer bookkeeping.

    The physics dict is re-copied so a later in-place sim step can't alias the
    snapshot — keeping restore idempotent across the K candidates of a group.
    """
    env.unwrapped.set_state_dict(_copy_value(snap.physics))
    for node, layer_snap in snap.layers:
        restore_layer(node, layer_snap)


__all__ = ["EnvSnapshot", "snapshot_env", "restore_env", "snapshot_layer", "restore_layer"]
