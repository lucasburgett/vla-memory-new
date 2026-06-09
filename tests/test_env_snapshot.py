"""Unit tests for the env-snapshot copy policy (``grpo/env_snapshot.py``).

These pin the *copy-plain / share-complex* contract without needing sapien /
ManiSkill: a wrong policy here silently corrupts GRPO rewards (a candidate's
in-place mutation leaking into the snapshot → the next candidate branches from a
drifted state). The full in-sim fidelity check is ``snapshot_parity_probe.py``;
this is the fast local guard on the pure logic.
"""

import numpy as np
import torch

from vla_memory.grpo.env_snapshot import (
    _copy_value,
    _GeneratorState,
    restore_layer,
    snapshot_layer,
)


class _FakeActor:
    """Stands in for a sapien actor: an opaque handle that must be SHARED, not
    copied (its physical state is owned by set_state_dict, not the snapshot)."""

    def __init__(self, name):
        self.name = name


class _FakeEnv:
    """A bag of attributes mimicking the env bookkeeping we snapshot."""


def _make_env():
    env = _FakeEnv()
    a, b = _FakeActor("bin_0"), _FakeActor("bin_1")
    env.timestep = 1
    env.current_task_index = 1
    env._elapsed_steps = torch.tensor([180])
    env.successflag = torch.tensor([False])
    env._timelimit_deadlines = {2: 400}
    env._lift_drop_cache = {(id(a), 0, 64): {"origin": np.array([1.0, 2.0, 3.0])}}
    env.button_list = [a, b]            # mutated by .remove() in the real env
    env.swap_schedule = [(a, b, 64, 114)]
    env.spawned_bins = [a, b]           # shared actor refs
    env.generator = torch.Generator()
    env.generator.manual_seed(42)
    return env, a, b


def test_actor_refs_shared_not_copied():
    env, a, b = _make_env()
    snap = snapshot_layer(env)
    # Containers are fresh objects (so .remove()/pop can be undone)...
    assert snap["values"]["button_list"] is not env.button_list
    # ...but the actor elements inside are the SAME objects (identity preserved).
    assert snap["values"]["button_list"][0] is a
    assert snap["values"]["spawned_bins"][1] is b
    assert snap["values"]["swap_schedule"][0][0] is a


def test_restore_undoes_list_mutation():
    env, a, b = _make_env()
    snap = snapshot_layer(env)
    env.button_list.remove(a)           # simulate is_any_button_pressed_removelist
    assert env.button_list == [b]
    restore_layer(env, snap)
    assert env.button_list == [a, b]
    assert env.button_list[0] is a      # restored ref is the original actor


def test_restore_undoes_nested_dict_mutation():
    env, a, b = _make_env()
    snap = snapshot_layer(env)
    # In-place mutate the inner np array + pop the cache (what statechange.py does).
    env._lift_drop_cache[(id(a), 0, 64)]["origin"][0] = 999.0
    env._timelimit_deadlines.pop(2, None)
    env._timelimit_deadlines[5] = 1
    restore_layer(env, snap)
    key = (id(a), 0, 64)
    assert env._lift_drop_cache[key]["origin"][0] == 1.0   # inner array restored
    assert env._timelimit_deadlines == {2: 400}            # dict restored


def test_tensor_restored_by_value():
    env, a, b = _make_env()
    snap = snapshot_layer(env)
    env._elapsed_steps += 50            # in-place tensor mutation during rollout
    env.successflag = torch.tensor([True])
    restore_layer(env, snap)
    assert int(env._elapsed_steps.item()) == 180
    assert bool(env.successflag.item()) is False


def test_restore_is_idempotent_across_k_restores():
    env, a, b = _make_env()
    snap = snapshot_layer(env)
    for _ in range(3):                  # K candidates branch from one snapshot
        env.button_list.clear()
        env._elapsed_steps += 7
        env._timelimit_deadlines.clear()
        restore_layer(env, snap)
        assert env.button_list == [a, b]
        assert int(env._elapsed_steps.item()) == 180
        assert env._timelimit_deadlines == {2: 400}


def test_keys_added_after_snapshot_are_deleted():
    env, a, b = _make_env()
    snap = snapshot_layer(env)
    env.first_timestep = 170            # created mid-rollout by static_check
    env._swap_rng = torch.Generator()
    restore_layer(env, snap)
    assert not hasattr(env, "first_timestep")
    assert not hasattr(env, "_swap_rng")


def test_generator_state_rewound():
    env, a, b = _make_env()
    snap = snapshot_layer(env)
    before = torch.rand(3, generator=env.generator)   # advance RNG
    restore_layer(env, snap)
    assert env.generator is not None
    after = torch.rand(3, generator=env.generator)     # same draws as `before`
    assert torch.equal(before, after)


def test_copy_value_shares_unknown_objects():
    a = _FakeActor("x")
    assert _copy_value(a) is a                          # opaque handle shared
    lam = lambda: 1                                     # noqa: E731
    assert _copy_value(lam) is lam                      # callable shared


def test_generator_captured_as_state_wrapper():
    g = torch.Generator()
    g.manual_seed(7)
    captured = _copy_value(g)
    assert isinstance(captured, _GeneratorState)
    assert captured.gen is g


if __name__ == "__main__":
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all env_snapshot tests passed")
