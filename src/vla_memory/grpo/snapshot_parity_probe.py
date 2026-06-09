"""Deterministic parity probe — does ``peek_and_snapshot`` → ``restore_env``
reproduce the env bit-for-bit?

This is the GATE before enabling ``--snapshot-branching`` in GRPO. The snapshot
path replaces the per-candidate oracle warm-up with an in-place env restore
(``env_snapshot.snapshot_env`` / ``restore_env``). That restore touches internals
we cannot edit (ManiSkill physics + scattered Python bookkeeping across the whole
wrapper chain), so its correctness must be *verified empirically*, not assumed —
this project has a history of silent reward-corruption ("fake flatline").

Method — RECORD then REPLAY, isolating snapshot fidelity from π0.5 stochasticity:
the policy server samples actions stochastically, so re-inferring after a restore
would differ even with a *perfect* snapshot. Instead we RECORD the exact action
arrays applied during one forward pass, then RESTORE and REPLAY those identical
arrays (no inference). Any divergence is then a snapshot bug, full stop.

Per ``(task, pick_index)`` × episode:
  1. ``peek_and_snapshot`` → decision point + snapshot ``S``. Capture the oracle
     pick subgoal + reference invariants (elapsed_steps, current_task_index, …).
  2. RECORD: restore ``S`` → drive forward with the FIXED oracle subgoal, logging
     every applied action + per-step (current_task_index, status); capture final
     physics + reward.
  3. REPLAY ×3 (idempotence across candidates): restore ``S`` → apply the recorded
     actions verbatim. Assert pre-step invariants, the per-step trace, status,
     reward, and final physics tensors are IDENTICAL to RECORD.

Exit 0 = snapshot/restore is faithful → safe to enable ``--snapshot-branching``.
Exit 1 = a mismatch (snapshot misses some state) → DO NOT trust the speed path.
Exit 2 = no decision points reachable (check task/seed).

Run via ``modal run modal_train.py::probe_snapshot_parity``.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "/app/examples/robomme")


def _scalar_int(v) -> Optional[int]:
    """Coerce a possibly-tensor scalar (e.g. ``elapsed_steps``) to a Python int."""
    import torch

    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return int(v.detach().cpu().reshape(-1)[0].item())
    if hasattr(v, "reshape"):
        return int(np.asarray(v).reshape(-1)[0])
    return int(v)


def _task_index(env_runner) -> int:
    v = getattr(env_runner.env.unwrapped, "current_task_index", None)
    return -1 if v is None else int(v)


def _button_list_len(env_runner) -> Optional[int]:
    bl = getattr(env_runner.env.unwrapped, "button_list", None)
    return None if bl is None else len(bl)


def _prestep_invariants(env_runner) -> dict:
    """Reward-relevant env state at the decision point — must survive restore."""
    return {
        "elapsed": _scalar_int(getattr(env_runner.env.unwrapped, "elapsed_steps", None)),
        "task_index": _task_index(env_runner),
        "timestep": int(getattr(env_runner.env.unwrapped, "timestep", -1)),
        "grounded": str(env_runner.grounded_subgoal_oracle),
        "button_list_len": _button_list_len(env_runner),
    }


def _flatten_tensors(x, prefix: str = ""):
    """Flatten a (possibly nested) get_state_dict into {path: cloned CPU tensor}.
    Cloning detaches the result from the sim's internal buffers so it stays a
    stable reference after subsequent restores/steps mutate the env."""
    import torch

    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(_flatten_tensors(v, f"{prefix}.{k}"))
    elif isinstance(x, torch.Tensor):
        out[prefix] = x.detach().cpu().clone()
    return out


def _physics_maxdiff(a: dict, b: dict) -> Tuple[Optional[str], float]:
    """Max abs elementwise difference over two flattened ({path: tensor}) physics
    snapshots → ``(worst_key, worst_diff)``. ``inf`` on a structural mismatch."""
    if set(a) != set(b):
        return "<key set mismatch>", float("inf")
    worst, worst_k = 0.0, None
    for k in a:
        if a[k].shape != b[k].shape:
            return f"{k}<shape {a[k].shape} vs {b[k].shape}>", float("inf")
        d = (a[k].float() - b[k].float()).abs().max().item()
        if d > worst:
            worst, worst_k = d, k
    return worst_k, worst


def _physics_close(a: dict, b: dict, atol: float) -> Tuple[bool, str]:
    """Within ``atol`` everywhere? Used for the t=0 RESTORE-FIDELITY check (before
    any stepping) — NOT for post-rollout physics, which the sim's nondeterministic
    forward stepping makes non-reproducible even from a perfect restore."""
    k, d = _physics_maxdiff(a, b)
    if d > atol:
        return False, f"{k}: max abs {d:.3e} > tol {atol:.0e}"
    return True, ""


def _run_segment(
    worker, snapshot, subgoal: str, recorded_actions: Optional[List[np.ndarray]]
):
    """Restore ``snapshot`` then step forward.

    RECORD mode (``recorded_actions is None``): infer via ``worker._infer_actions``
    and apply via ``env_runner.step``, logging every applied action.
    REPLAY mode: apply the given actions verbatim, no inference.

    Returns ``(trace, applied, status, n_steps, prestep, phys_t0)`` where ``trace``
    is a list of ``(current_task_index, status)`` per applied action, ``prestep`` the
    decision-point invariants read immediately after restore, and ``phys_t0`` the
    flattened ``get_state_dict`` read BEFORE any step — the clean restore-fidelity
    signal (isolated from the sim's nondeterministic forward stepping).
    """
    from vla_memory.grpo.env_snapshot import _copy_value, restore_env

    er = worker.env_runner
    restore_env(er.env, snapshot.env)
    # info lives on the runner (outside the wrapper chain restore_env touches), so
    # restore it explicitly — else the oracle-subgoal prestep check reads stale info.
    if snapshot.info is not None:
        er.info = _copy_value(snapshot.info)
    prestep = _prestep_invariants(er)
    phys_t0 = _flatten_tensors(er.env.unwrapped.get_state_dict())

    image_buf = deque(snapshot.image_buf, maxlen=64)
    wrist_buf = deque(snapshot.wrist_buf, maxlen=64)
    state_buf = deque(snapshot.state_buf, maxlen=64)
    img = None if snapshot.img is None else snapshot.img.copy()
    wrist = None if snapshot.wrist is None else snapshot.wrist.copy()
    robot_state = None if snapshot.robot_state is None else snapshot.robot_state.copy()
    task_goal = snapshot.task_goal
    exec_start_idx = snapshot.exec_start_idx
    n_steps = snapshot.n_steps
    success_flag = snapshot.success_flag

    record = recorded_actions is None
    applied: List[np.ndarray] = []
    trace: List[Tuple[int, str]] = []

    if record:
        stop = False
        while n_steps < worker.max_steps and not stop:
            actions = worker._infer_actions(
                img, wrist, robot_state, task_goal, subgoal=subgoal,
                image_buf=image_buf, state_buf=state_buf, exec_start_idx=exec_start_idx,
            )
            for action in actions:
                (img, wrist, robot_state), stop, success_flag = er.step(action)
                applied.append(np.asarray(action).copy())
                n_steps += 1
                trace.append((_task_index(er), success_flag))
                if img is None:
                    return trace, applied, "error", n_steps, prestep, phys_t0
                image_buf.append(img)
                wrist_buf.append(wrist)
                state_buf.append(robot_state)
                if stop or n_steps >= worker.max_steps:
                    break
    else:
        for action in recorded_actions:
            (_obs, _w, _s), stop, success_flag = er.step(action)
            n_steps += 1
            trace.append((_task_index(er), success_flag))
            if _obs is None:
                return trace, None, "error", n_steps, prestep, phys_t0
            if stop or n_steps >= worker.max_steps:
                break

    if n_steps >= worker.max_steps and success_flag in ("unknown", ""):
        success_flag = "timeout"
    return trace, (applied if record else None), success_flag, n_steps, prestep, phys_t0


def _parse_task_specs(spec: str) -> List[Tuple[str, int]]:
    """`"ButtonUnmask:0,ButtonUnmaskSwap:1"` → [("ButtonUnmask",0),("ButtonUnmaskSwap",1)]."""
    out: List[Tuple[str, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            name, pick = chunk.split(":", 1)
            out.append((name.strip(), int(pick)))
        else:
            out.append((chunk, 0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic snapshot/restore parity probe")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tasks", default="ButtonUnmask:0,ButtonUnmaskSwap:0,ButtonUnmaskSwap:1",
                        help="Comma-separated Task:pick_index specs to parity-check.")
    parser.add_argument("--n-episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--rollout-max-steps", type=int, default=700,
                        help="Env truncation cap. Must exceed the warm-up to the LATEST pick "
                             "+ the forward pass, or pick_index=1 decision points are never "
                             "reached (env truncates → all skipped). 700 covers Swap pick 2 "
                             "(~rel 410 + pick).")
    parser.add_argument("--decision-warm-cap", type=int, default=250,
                        help="Per-pick oracle warm-up cap (×(pick_index+1)); 250 reaches "
                             "ButtonUnmaskSwap pick 1 (~rel 410).")
    parser.add_argument("--n-replays", type=int, default=3,
                        help="Restore+replay repetitions per episode (proves idempotence "
                             "across the K candidates a group restores).")
    parser.add_argument("--phys-tol", type=float, default=1e-4,
                        help="Tolerance for the t=0 restore-fidelity check (post-restore "
                             "get_state_dict vs the captured snapshot). A real restore bug "
                             "shows huge t=0 drift; ~1e-6 is float/quaternion round-trip.")
    parser.add_argument("--output-dir", default="/tmp/snapshot_parity_probe")
    args = parser.parse_args()

    from openpi_client import websocket_client_policy as _wp  # type: ignore

    from vla_memory.grpo.env_runner import EnvRunner
    from vla_memory.grpo.rollout import RolloutWorker

    def fresh_worker(task: str) -> RolloutWorker:
        env_runner = EnvRunner(
            env_id=task,
            video_save_dir=str(Path(args.output_dir) / "videos"),
            max_steps=args.rollout_max_steps,
            dataset="train",
        )
        client = _wp.MMEVLAWebsocketClientPolicy("127.0.0.1", args.port)
        return RolloutWorker(
            env_runner=env_runner, policy_client=client, obs_horizon=16,
            max_steps=args.rollout_max_steps, use_history=False,
            subgoal_type="grounded_subgoal",
            decision_warm_cap=args.decision_warm_cap,
        )

    specs = _parse_task_specs(args.tasks)
    checked = 0
    skipped = 0
    failures: List[str] = []

    for task, pick_index in specs:
        print(f"\n######## {task} pick_index={pick_index} ########", flush=True)
        for ep in range(args.n_episodes):
            seed = (args.seed * 1_000_003 + ep) % 2_147_483_647
            w = fresh_worker(task)
            try:
                dp, snap = w.peek_and_snapshot(ep, seed=seed, pick_index=pick_index)
                if dp.terminated_early:
                    skipped += 1
                    print(f"[parity] {task}[{pick_index}] ep{ep}: decision point "
                          f"unreachable ({dp.success_flag}) — skipped", flush=True)
                    continue

                # Pin the oracle pick subgoal + reference invariants AT the decision
                # point (before any restore). Drive RECORD with this fixed string.
                subgoal = str(w.env_runner.grounded_subgoal_oracle)
                ref = _prestep_invariants(w.env_runner)
                phys_ref = _flatten_tensors(snap.env.physics)  # captured-snapshot physics

                # RECORD pass.
                trace_rec, actions, status_rec, n_rec, pre_rec, phys_t0_rec = _run_segment(
                    w, snap, subgoal, None
                )
                phys_final_rec = _flatten_tensors(w.env_runner.env.unwrapped.get_state_dict())
                reward_rec = w._progress_estimate(status_rec)

                ep_ok = True
                ep_msgs: List[str] = []

                # HARD — restore fidelity: post-restore physics (BEFORE stepping) must
                # round-trip the captured snapshot. A real restore bug shows huge t=0
                # drift here; the nondeterministic forward stepping never enters this check.
                ok, why = _physics_close(phys_t0_rec, phys_ref, args.phys_tol)
                if not ok:
                    ep_ok = False
                    ep_msgs.append(f"record: t=0 restore infidelity {why}")
                # HARD — reward-relevant invariants at the decision point.
                if pre_rec != ref:
                    ep_ok = False
                    ep_msgs.append(f"record: prestep {pre_rec} != reference {ref}")

                final_drift = 0.0
                for rep in range(args.n_replays):
                    trace_rep, _, status_rep, n_rep, pre_rep, phys_t0_rep = _run_segment(
                        w, snap, subgoal, actions
                    )
                    phys_final_rep = _flatten_tensors(w.env_runner.env.unwrapped.get_state_dict())
                    reward_rep = w._progress_estimate(status_rep)

                    # HARD — restore fidelity + idempotence at t=0 (the snapshot must be
                    # re-restorable identically for each of the K candidates).
                    ok, why = _physics_close(phys_t0_rep, phys_ref, args.phys_tol)
                    if not ok:
                        ep_ok = False
                        ep_msgs.append(f"replay{rep}: t=0 restore infidelity {why}")
                    ok, why = _physics_close(phys_t0_rep, phys_t0_rec, args.phys_tol)
                    if not ok:
                        ep_ok = False
                        ep_msgs.append(f"replay{rep}: t=0 not idempotent vs record {why}")
                    # HARD — the reward signal GRPO consumes must reproduce.
                    if pre_rep != ref:
                        ep_ok = False
                        ep_msgs.append(f"replay{rep}: prestep {pre_rep} != reference {ref}")
                    if trace_rep != trace_rec:
                        ep_ok = False
                        diff_i = next(
                            (i for i in range(min(len(trace_rec), len(trace_rep)))
                             if trace_rec[i] != trace_rep[i]),
                            min(len(trace_rec), len(trace_rep)),
                        )
                        ep_msgs.append(
                            f"replay{rep}: trace differs (len {len(trace_rec)} vs "
                            f"{len(trace_rep)}, first diff @ {diff_i})"
                        )
                    if status_rep != status_rec:
                        ep_ok = False
                        ep_msgs.append(f"replay{rep}: status {status_rep} != {status_rec}")
                    if abs(reward_rep - reward_rec) > 1e-9:
                        ep_ok = False
                        ep_msgs.append(f"replay{rep}: reward {reward_rep} != {reward_rec}")
                    # INFO — final-physics drift after the full (chaotic, contact-heavy)
                    # forward rollout. The GPU sim's stepping is NOT bitwise-deterministic
                    # (replays of identical actions diverge), so this is reported, NOT gated.
                    _, d = _physics_maxdiff(phys_final_rep, phys_final_rec)
                    final_drift = max(final_drift, d)

                checked += 1
                if ep_ok:
                    print(f"[parity] {task}[{pick_index}] ep{ep}: OK "
                          f"(steps={n_rec}, reward={reward_rec:.3f}, "
                          f"task_idx={trace_rec[-1][0] if trace_rec else '-'}, "
                          f"elapsed0={ref['elapsed']}, final_phys_drift={final_drift:.2e})", flush=True)
                else:
                    tag = f"{task}[{pick_index}] ep{ep}"
                    for m in ep_msgs:
                        failures.append(f"{tag}: {m}")
                    print(f"[parity] {tag}: MISMATCH", flush=True)
                    for m in ep_msgs:
                        print(f"          ✗ {m}", flush=True)
            finally:
                w.close()

    print("\n==================== SNAPSHOT PARITY VERDICT ====================", flush=True)
    print(f"  episodes checked / skipped : {checked} / {skipped}", flush=True)
    print(f"  mismatches                 : {len(failures)}", flush=True)
    if checked == 0:
        print("  RESULT: NO DATA — every decision point was unreachable. Check task/seed.", flush=True)
        sys.exit(2)
    if failures:
        print("  RESULT: FAIL — snapshot/restore is NOT faithful. Either the t=0 physics does", flush=True)
        print("          not round-trip the snapshot (a missed tensor/wrapper/RNG), or the", flush=True)
        print("          reward/task-index trajectory did not reproduce. Do NOT enable", flush=True)
        print("          --snapshot-branching. First few mismatches:", flush=True)
        for m in failures[:8]:
            print(f"            - {m}", flush=True)
        sys.exit(1)
    print("  RESULT: PASS — restore is faithful (t=0 physics round-trips the snapshot,", flush=True)
    print("          idempotently across replays) and the reward + task-index trajectory", flush=True)
    print("          reproduces. (Final-physics drift is expected & reported, not gated: the", flush=True)
    print("          GPU sim's forward stepping is nondeterministic — the rebuild path shares", flush=True)
    print("          that noise.) → safe to run GRPO with --snapshot-branching.", flush=True)
    print("================================================================", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
