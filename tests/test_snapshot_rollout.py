"""Integration test for the snapshot-branching PLUMBING, against a deterministic
fake env — runs locally (no sapien / no policy server).

The copy policy itself is pinned in ``test_env_snapshot.py``; the in-sim fidelity
gate is ``snapshot_parity_probe.py`` (Modal). This fills the gap between them:
that ``peek_and_snapshot`` → ``restore_env`` → ``rollout_from_snapshot`` and the
probe's record/replay loop are wired correctly — caught here in milliseconds
instead of after booting a GPU. A deterministic fake env means record == replay
iff the restore is faithful, so it exercises the exact property the real probe does.
"""

import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

# RolloutWorker._infer_actions imports these lazily (submodule-only on Modal).
_oc = types.ModuleType("openpi_client")
_oc.websocket_client_policy = types.ModuleType("openpi_client.websocket_client_policy")
sys.modules.setdefault("openpi_client", _oc)
sys.modules.setdefault("openpi_client.websocket_client_policy", _oc.websocket_client_policy)
_utils = types.ModuleType("utils")
_utils.pack_buffer = lambda *a, **k: None
sys.modules.setdefault("utils", _utils)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vla_memory.grpo.env_snapshot import restore_env  # noqa: E402
from vla_memory.grpo.rollout import RolloutWorker  # noqa: E402
from vla_memory.grpo.snapshot_parity_probe import (  # noqa: E402
    _prestep_invariants,
    _run_segment,
)

_GOAL_POS = 20          # pos at which the (single) pick completes
_PICK_AT_STEP = 3       # elapsed step where the oracle phase flips press→pick


class _FakeBase:
    """Unwrapped env: physics (``pos``/``_elapsed_steps``) in get/set_state_dict,
    plus Python bookkeeping (``current_task_index``, ``_cache``, ``button_list``)
    that ONLY the wrapper-chain ``__dict__`` snapshot restores."""

    def __init__(self):
        self.pos = 0
        self._elapsed_steps = torch.tensor([0])
        self.current_task_index = 0
        self.timestep = 0
        self.task_list = ["press", "pick", "done"]
        self._cache = {"k": [1, 2, 3]}      # mutable nested bookkeeping
        self.button_list = ["b0", "b1"]     # mutated by .remove() analogue

    @property
    def unwrapped(self):
        return self

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def get_state_dict(self):
        return {"pos": torch.tensor([self.pos]), "elapsed": self._elapsed_steps.clone()}

    def set_state_dict(self, d):
        self.pos = int(d["pos"][0].item())
        self._elapsed_steps = d["elapsed"].clone()


class _FakeWrapper:
    """Gymnasium-style layer: holds ``.env`` + its own bookkeeping (mimics
    DemonstrationWrapper.episode_success)."""

    def __init__(self, env, **extra):
        self.env = env
        for k, v in extra.items():
            setattr(self, k, v)

    @property
    def unwrapped(self):
        return self.env.unwrapped


def _make_chain():
    base = _FakeBase()
    demo = _FakeWrapper(base, episode_success=False, steps_without_demo=0)
    outer = _FakeWrapper(demo, _last_obs=None)
    return outer


class _FakeEnvRunner:
    def __init__(self):
        self.env = _make_chain()

    def make_env(self, episode_id):
        pass

    def close_env(self):
        pass

    def get_init_obs(self, seed=None):
        b = self.env.unwrapped
        b.pos = 0
        b._elapsed_steps = torch.tensor([0])
        b.current_task_index = 0
        b.timestep = 0
        b._cache = {"k": [1, 2, 3]}
        b.button_list = ["b0", "b1"]
        img = np.zeros((2, 2), dtype=np.uint8)
        return {
            "images": [img], "wrist_images": [img],
            "states": [np.zeros(8, dtype=np.float32)], "task_goal": "do it",
        }

    def step(self, action):
        b = self.env.unwrapped
        b.pos += int(np.asarray(action).reshape(-1)[0])
        b._elapsed_steps = b._elapsed_steps + 1
        b._cache["k"].append(b.pos)              # in-place bookkeeping mutation
        t = int(b._elapsed_steps.item())
        if t >= _PICK_AT_STEP:
            b.current_task_index = 1
        if b.pos >= _GOAL_POS:
            b.current_task_index = 2
        img = np.full((2, 2), b.pos % 256, dtype=np.uint8)
        stop = b.current_task_index >= 2
        return (img, img, np.zeros(8, dtype=np.float32)), stop, ("success" if stop else "ongoing")

    @property
    def simple_subgoal_oracle(self):
        t = int(self.env.unwrapped._elapsed_steps.item())
        return "pick up the container" if t >= _PICK_AT_STEP else "press the button"

    @property
    def grounded_subgoal_oracle(self):
        return "pick up the container at <500, 500>"


class _FakeClient:
    """Returns a fixed action chunk — π0.5 is stochastic in reality, but the probe
    only relies on REPLAY reproducing RECORD, which a fixed chunk also tests."""

    def reset(self):
        return {"reset_finished": True}

    def infer(self, element):
        return {"actions": [np.array([1.0], dtype=np.float32) for _ in range(16)]}

    def add_buffer(self, *a, **k):
        return {"add_buffer_finished": True}


def _worker():
    return RolloutWorker(
        env_runner=_FakeEnvRunner(), policy_client=_FakeClient(),
        obs_horizon=16, max_steps=200, use_history=False,
        subgoal_type="grounded_subgoal", decision_warm_cap=50,
    )


def test_peek_and_snapshot_then_restore_roundtrip():
    w = _worker()
    dp, snap = w.peek_and_snapshot(episode_id=0, seed=0, pick_index=0)
    assert not dp.terminated_early
    ref = _prestep_invariants(w.env_runner)
    assert ref["task_index"] == 1 and ref["elapsed"] == _PICK_AT_STEP
    cache_at_decision = list(w.env_runner.env.unwrapped._cache["k"])

    # Drive the env far past the decision point, mutating physics + bookkeeping.
    for _ in range(30):
        w.env_runner.step(np.array([1.0]))
    assert w.env_runner.env.unwrapped.pos >= _GOAL_POS
    assert w.env_runner.env.unwrapped._cache["k"] != cache_at_decision

    restore_env(w.env_runner.env, snap.env)
    assert _prestep_invariants(w.env_runner) == ref           # physics + tracker back
    assert w.env_runner.env.unwrapped._cache["k"] == cache_at_decision  # nested bookkeeping back


def test_run_segment_record_replay_parity():
    w = _worker()
    dp, snap = w.peek_and_snapshot(episode_id=0, seed=0, pick_index=0)
    subgoal = str(w.env_runner.grounded_subgoal_oracle)

    trace_rec, actions, status_rec, n_rec, pre_rec, phys_t0_rec = _run_segment(w, snap, subgoal, None)
    reward_rec = w._progress_estimate(status_rec)
    assert len(trace_rec) > 0 and status_rec == "success"     # pick completes

    for _ in range(3):                                        # idempotent across replays
        trace_rep, _, status_rep, n_rep, pre_rep, phys_t0_rep = _run_segment(w, snap, subgoal, actions)
        assert pre_rep == pre_rec
        assert trace_rep == trace_rec
        assert status_rep == status_rec
        assert w._progress_estimate(status_rep) == reward_rec
        # deterministic fake env → t=0 physics restores identically every replay
        assert phys_t0_rep.keys() == phys_t0_rec.keys()
        for k in phys_t0_rec:
            assert torch.equal(phys_t0_rep[k], phys_t0_rec[k])


def test_rollout_from_snapshot_idempotent():
    w = _worker()
    _dp, snap = w.peek_and_snapshot(episode_id=0, seed=0, pick_index=0)
    # The coordinate is run through from_qwen_xy inside rollout_from_snapshot; the
    # fake env ignores subgoal content, so two restores must give identical results.
    r1 = w.rollout_from_snapshot(snap, "pick up the container at <500, 500>")
    r2 = w.rollout_from_snapshot(snap, "pick up the container at <500, 500>")
    assert (r1.success_flag, r1.progress, r1.n_steps) == (r2.success_flag, r2.progress, r2.n_steps)
    assert r1.success_flag == "success"


def test_terminated_warmup_snapshot_short_circuits():
    # A snapshot whose warm-up already ended → rollout_from_snapshot returns the
    # warm-up outcome without stepping (mirrors rollout's early-out).
    w = _worker()
    _dp, snap = w.peek_and_snapshot(episode_id=0, seed=0, pick_index=0)
    snap.terminated = True
    snap.success_flag = "fail"
    res = w.rollout_from_snapshot(snap, "pick up the container at <500, 500>")
    assert res.success_flag == "fail"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all snapshot_rollout tests passed")
