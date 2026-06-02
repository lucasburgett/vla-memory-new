"""RolloutWorker.rollout_streaming — the MemER decision-point-streaming generator.

Drives the generator with a SCRIPTED fake env (no sim/torch/openpi needed) to pin
the core new behaviour the streaming GRPO path relies on:

1. the generator yields a DecisionPoint at EACH pick and a RolloutResult at the end;
2. the keyframe buffer ACCUMULATES — pick 1's ``key_frames`` are exactly the frames
   pick 0 nominated (the causal memory link selection is trained through);
3. ``from_qwen_xy`` converts the VLM subtask at the SINGLE executor point, while the
   oracle scaffolding subgoals are passed through UNCONVERTED;
4. pick 0's window spans the early (reveal) frames; pick 1's window is later frames.

We override ``_infer_actions`` on the instance (its real body imports openpi_client /
utils) to return one action per call and to RECORD the subgoal it was handed.
"""

import numpy as np

from vla_memory.grpo.rollout import DecisionPoint, RolloutResult, RolloutWorker
from vla_memory.qwen_subgoal.coords import from_qwen_xy


class _FakeClient:
    def reset(self):
        return {"reset_finished": True}


class _Unwrapped:
    task_list = ["press1", "press2", "pick0", "putdown", "pick1"]
    current_task_index = 5


class _FakeEnv:
    unwrapped = _Unwrapped()


class _FakeEnvRunner:
    """Scripts a 2-pick ButtonUnmaskSwap-like timeline purely by step count.

    Phase schedule (absolute step): press → pick0 → putdown → pick1 → success.
    Phase transitions are step-driven (deterministic) so we test the GENERATOR
    PROTOCOL + buffer wiring, not physics. The grounded oracle encodes the phase so
    we can assert scaffolding used the (unconverted) oracle subgoal.
    """

    def __init__(self):
        self.env = _FakeEnv()
        self.step_n = 0
        # frame encodes its own step in pixel [0,0,0] so we can identify buffered frames.
        self._frame = lambda s: np.full((2, 2, 3), s % 256, dtype=np.uint8)

    def make_env(self, episode_id):
        self.step_n = 0

    def close_env(self):
        pass

    def get_init_obs(self, seed=None):
        f = self._frame(0)
        return {"images": [f], "wrist_images": [f], "states": [np.zeros(8)], "task_goal": "swap task"}

    def _phase(self):
        s = self.step_n
        if s < 5:
            return "press the first button"
        if s < 12:
            return "pick up the container at <10, 20>"   # pick 0
        if s < 16:
            return "put down the container"
        return "pick up the container at <30, 40>"        # pick 1

    @property
    def simple_subgoal_oracle(self):
        return self._phase()

    @property
    def grounded_subgoal_oracle(self):
        return self._phase()  # scaffolding drives with this; must reach the executor UNCONVERTED

    def step(self, action):
        self.step_n += 1
        f = self._frame(self.step_n)
        stop = self.step_n >= 24
        success_flag = "success" if stop else "unknown"
        return (f, f, np.zeros(8)), stop, success_flag


def _make_worker():
    w = RolloutWorker(
        env_runner=_FakeEnvRunner(),
        policy_client=_FakeClient(),
        use_history=False,
        subgoal_type="grounded_subgoal",
        n_candidate_frames=6,
        max_keyframes=4,
        keyframe_buffer_cap=8,
        keyframe_cluster_dist=2,   # tiny so distinct scripted frames don't collapse
        max_steps=200,
        decision_warm_cap=100,
    )
    calls = []   # (phase_is_pick_exec, subgoal) recorded per _infer_actions

    def fake_infer(img, wrist, robot_state, task_goal, subgoal, image_buf, state_buf, exec_start_idx):
        calls.append(subgoal)
        return [np.zeros(7)]   # one action -> one env step per chunk

    w._infer_actions = fake_infer
    return w, calls


def _drive(worker, subtasks, kf=(1, 5)):
    """Run the generator, feeding ``subtasks[i]`` (qwen <x,y>) at pick i; nominate
    keyframe positions ``kf`` (default well-separated so the cluster-merge in
    KeyframeBuffer — tested separately — doesn't collapse them). Returns
    (decision_points, final_result)."""
    gen = worker.rollout_streaming(episode_id=0, seed=0)
    dps = []
    dp = next(gen)
    i = 0
    while isinstance(dp, DecisionPoint):
        dps.append(dp)
        subtask = subtasks[min(i, len(subtasks) - 1)]
        dp = gen.send((subtask, list(kf)))
        i += 1
    return dps, dp


def test_yields_two_decision_points_then_result():
    worker, _ = _make_worker()
    dps, result = _drive(worker, ["pick up the container at <500, 600>",
                                  "pick up the container at <700, 800>"])
    assert len(dps) == 2, f"expected 2 picks, got {len(dps)}"
    assert isinstance(result, RolloutResult)
    assert result.success_flag == "success"


def test_buffer_accumulates_across_picks():
    worker, _ = _make_worker()
    dps, _ = _drive(worker, ["pick up the container at <500, 600>",
                             "pick up the container at <700, 800>"])
    dp0, dp1 = dps
    # Pick 0: buffer empty (nothing nominated yet).
    assert dp0.key_frames == []
    # Pick 1: buffer = the frames pick 0 nominated (positions [1,5] of dp0.recent_frames).
    assert len(dp1.key_frames) == 2
    expected = [dp0.recent_frames[0], dp0.recent_frames[4]]
    for got, exp in zip(dp1.key_frames, expected):
        assert int(got[0, 0, 0]) == int(exp[0, 0, 0]), "pick-1 buffer != pick-0 nominations"


def test_from_qwen_xy_applied_to_vlm_subtask_not_oracle():
    worker, calls = _make_worker()
    vlm0 = "pick up the container at <500, 600>"
    _drive(worker, [vlm0, "pick up the container at <700, 800>"])
    converted = from_qwen_xy(vlm0)
    assert converted != vlm0, "test precondition: conversion should change the string"
    # The converted VLM subtask must appear among the executor subgoals...
    assert converted in calls, "VLM subtask was not from_qwen_xy-converted at the executor"
    # ...and the raw qwen-space VLM string must NEVER reach the executor.
    assert vlm0 not in calls, "unconverted VLM coordinate leaked to the executor"
    # ...while the oracle scaffolding subgoal (press phase) passed through unconverted.
    assert "press the first button" in calls


def test_pick0_window_is_early_pick1_window_is_later():
    worker, _ = _make_worker()
    dps, _ = _drive(worker, ["pick up the container at <500, 600>",
                             "pick up the container at <700, 800>"])
    dp0, dp1 = dps
    # Frames encode their abs step; pick 0's window starts at the reveal (step 0),
    # pick 1's window starts only AFTER pick 0 completed (later steps).
    dp0_first_step = int(dp0.recent_frames[0][0, 0, 0])
    dp1_first_step = int(dp1.recent_frames[0][0, 0, 0])
    assert dp0_first_step == 0, "pick 0 window should include the earliest (reveal) frame"
    assert dp1_first_step > dp0_first_step, "pick 1 window should start after pick 0"


def test_history_excludes_current_pick():
    worker, _ = _make_worker()
    dps, _ = _drive(worker, ["pick up the container at <500, 600>",
                             "pick up the container at <700, 800>"])
    dp0, dp1 = dps
    # Pick 0: only the press is completed; the current pick is NOT in history.
    assert "press the first button" in dp0.history_subgoals
    assert not any("pick up the container" in h for h in dp0.history_subgoals)
    # Pick 1: pick 0 + put-down are completed and DO appear; pick 1 itself does not.
    assert any("put down" in h for h in dp1.history_subgoals)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all streaming_rollout tests passed")
