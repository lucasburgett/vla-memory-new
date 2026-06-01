"""Phase 0 diagnostic: GroundSG ceiling, fixed-subgoal sufficiency, and causality.

Established by prior runs:
  * GroundSG + ONLINE oracle = ~83% success on ButtonUnmask → foundation sound.
  * The oracle PICK subgoal is CONSTANT through the pick (logged at 48-step
    granularity): "pick up the container at <85, 155> …" from t=96→192.
  * Yet a "warm-up then hold one fixed subgoal" rollout finished only the button
    (progress 0.5). Since the subgoal string is identical, the suspect is a HARNESS
    artifact: the probe/pipeline ran ``_warmup`` TWICE (once to read the subgoal,
    again inside ``rollout``), and the second reset on a mid-episode env didn't
    cleanly restore the cube/occlusion state.

This run isolates that with ``rollout_freeze_at_transition`` — a SINGLE continuous
rollout (one reset, exactly like the working oracle) that drives the press with
the oracle, then FREEZES the pick subgoal and holds it. Conditions (fresh env each):

  ORACLE        = oracle re-read every chunk (ceiling; logs subgoal every 16 on ep0)
  FROZEN_CORRECT= continuous, hold the oracle's pick subgoal (logs every 16 on ep0)
  FROZEN_WRONG  = continuous, hold it with the grounding point shifted ~100px

Interpretation:
  * FROZEN_CORRECT succeeds  → the earlier failure WAS the double-warm-up reset.
    Fix the pipeline to snapshot the decision state (don't re-warm); then
    FROZEN_CORRECT > FROZEN_WRONG also gives the causality result. PROCEED.
  * FROZEN_CORRECT fails too  → holding even the identical subgoal can't finish the
    pick; the pick needs the continuous oracle for a reason the string doesn't show.

Run via ``modal run modal_train.py::causality_probe``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "/app/examples/robomme")

_COORD_RE = re.compile(r"<\s*(-?\d+)\s*,\s*(-?\d+)\s*>")


def _shift_coord(subgoal: str, img_size: int = 256) -> tuple[str | None, tuple | None]:
    m = _COORD_RE.search(subgoal)
    if not m:
        return None, None
    x, y = int(m.group(1)), int(m.group(2))
    half = img_size // 2
    nx = max(0, min(img_size - 1, x - 100 if x > half else x + 100))
    ny = max(0, min(img_size - 1, y - 100 if y > half else y + 100))
    return subgoal[: m.start()] + f"<{nx}, {ny}>" + subgoal[m.end():], (x, y, nx, ny)


def _corrupt_shift(sg: str) -> str:
    alt, _ = _shift_coord(sg)
    return alt if alt else sg


def main() -> None:
    parser = argparse.ArgumentParser(description="GroundSG ceiling + fixed-subgoal sufficiency/causality")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task", default="ButtonUnmask")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-max-steps", type=int, default=400)
    parser.add_argument("--progress-margin", type=float, default=0.08)
    parser.add_argument("--output-dir", default="/tmp/causality_probe")
    args = parser.parse_args()

    from openpi_client import websocket_client_policy as _wp  # type: ignore

    from vla_memory.grpo.env_runner import EnvRunner
    from vla_memory.grpo.rollout import RolloutWorker

    def fresh_worker():
        env_runner = EnvRunner(
            env_id=args.task,
            video_save_dir=str(Path(args.output_dir) / "videos"),
            max_steps=args.rollout_max_steps,
            dataset="train",
        )
        client = _wp.MMEVLAWebsocketClientPolicy("127.0.0.1", args.port)
        return RolloutWorker(
            env_runner=env_runner, policy_client=client, obs_horizon=16,
            max_steps=args.rollout_max_steps, use_history=False,
            subgoal_type="grounded_subgoal",
        )

    rows: list[dict] = []
    for ep in range(args.episodes):
        seed = (args.seed * 1_000_003 + ep) % 2_147_483_647
        logn = 16 if ep == 0 else 0
        row: dict = {"ep": ep}

        w = fresh_worker()
        try:
            res_o = w.rollout_oracle(ep, seed=seed, log_every=logn)
            row["po"], row["so"] = res_o.progress, res_o.success_flag
        finally:
            w.close()

        w = fresh_worker()
        try:
            res_c, frozen_c = w.rollout_freeze_at_transition(ep, seed=seed, log_every=logn)
            row["pc"], row["sc"], row["fc"] = res_c.progress, res_c.success_flag, frozen_c
        finally:
            w.close()

        w = fresh_worker()
        try:
            res_w, frozen_w = w.rollout_freeze_at_transition(ep, seed=seed, corrupt=_corrupt_shift)
            row["pw"], row["sw"], row["fw"] = res_w.progress, res_w.success_flag, frozen_w
        finally:
            w.close()

        print(
            f"[probe] ep{ep} seed{seed}\n"
            f"        ORACLE         : progress={row['po']:.3f} status={row['so']}\n"
            f"        FROZEN_CORRECT : progress={row['pc']:.3f} status={row['sc']} frozen={row['fc']!r}\n"
            f"        FROZEN_WRONG   : progress={row['pw']:.3f} status={row['sw']} frozen={row['fw']!r}",
            flush=True,
        )
        rows.append(row)

    po = np.array([r["po"] for r in rows], float)
    pc = np.array([r["pc"] for r in rows], float)
    pw = np.array([r["pw"] for r in rows], float)
    oracle_succ = float(np.mean([r["so"] == "success" for r in rows]))
    correct_succ = float(np.mean([r["sc"] == "success" for r in rows]))
    gap = float(pc.mean() - pw.mean())
    wins = int(np.sum(pc > pw + 1e-9))
    losses = int(np.sum(pw > pc + 1e-9))

    correct_capable = correct_succ > 0 or pc.mean() > 0.6
    causal = gap >= args.progress_margin and wins >= max(1, losses)

    print("\n==================== DIAGNOSTIC VERDICT ====================", flush=True)
    print(f"  episodes                       : {len(rows)}", flush=True)
    print(f"  ORACLE         progress / succ : {po.mean():.3f} / {oracle_succ*100:.0f}%", flush=True)
    print(f"  FROZEN_CORRECT progress / succ : {pc.mean():.3f} / {correct_succ*100:.0f}%", flush=True)
    print(f"  FROZEN_WRONG   progress        : {pw.mean():.3f}   (gap {gap:+.3f}, wins/losses {wins}/{losses})", flush=True)
    print("  -----", flush=True)
    if correct_capable and causal:
        print("  RESULT: PASS — a single continuous rollout with a FROZEN correct subgoal", flush=True)
        print("          finishes the pick and beats wrong. The earlier failure was the", flush=True)
        print("          double-warm-up reset. Fix: snapshot the decision state (no re-warm).", flush=True)
        print("          → proceed to SFT/GRPO with a snapshot-based rollout.", flush=True)
        verdict = 0
    elif correct_capable and not causal:
        print(f"  RESULT: CAPABLE, WEAK CAUSALITY — frozen-correct works but gap is {gap:+.3f}.", flush=True)
        print("          Sharper wrong contrast may be needed; inspect rows.", flush=True)
        verdict = 1
    else:
        print("  RESULT: FROZEN SUBGOAL STILL INSUFFICIENT — even a single continuous rollout", flush=True)
        print("          holding the identical pick subgoal can't finish, while the re-read oracle", flush=True)
        print("          can. The difference isn't the string — inspect env reset / server state.", flush=True)
        verdict = 1
    print("===========================================================", flush=True)
    sys.exit(verdict)


if __name__ == "__main__":
    main()
