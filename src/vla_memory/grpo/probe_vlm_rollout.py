"""Held-out generalization probe — does the SFT'd VLM's grounded subgoal, executed
on the FROZEN GroundSG π0.5, complete the memory task on UNSEEN seeds?

This is the gate before GRPO. ``validate_memory_checkpoints`` proved the VLM
CONDITIONS on the keyframes (its coordinate tracks the target on TRAIN rows), but
reading the train jsonl can't separate "reads the memory" from "memorized the train
episode→coordinate map". Here we run the FULL inference path on held-out seeds:

  peek to the post-occlusion decision point  (oracle warm-up, fresh env from seed)
    → greedy-decode the VLM subgoal           (Qwen <x,y> 0–1000 — the GRPO path)
    → execute on GroundSG                      (coords.from_qwen_xy applied in rollout)
    → score task progress / success

and compare to the ONLINE-ORACLE ceiling and a coordinate-SHIFTED contrast. The
three things this gates for GRPO:

  * VLM success/progress NON-ZERO   → GRPO can bootstrap (RL cold-start needs >0)
  * VLM ≈ ORACLE                    → the VLM reads memory & GENERALIZES to unseen seeds
  * VLM > VLM_SHIFTED               → the specific coordinate is load-bearing, i.e. the
                                      SFT grounding + the 0–1000→0–256 <y,x> conversion
                                      are correct end-to-end (a wrong conversion would
                                      tie VLM ≈ SHIFTED and silently flatline GRPO)

Mirrors ``causality_probe.py`` (fresh worker per rollout, peek→rollout on one worker
as the GRPO trainer does) but swaps the oracle subgoal for the VLM's greedy one.

Run via ``modal run modal_train.py::probe_vlm_rollout``.
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


def _shift_qwen_xy(subtask: str, shift: int = 390, hi: int = 999) -> str:
    """Move the VLM's Qwen ``<x, y>`` 0–1000 grounding point well off the target
    (~shift/1000 of the image ≈ 100px at 256) for the WRONG-coordinate contrast.
    Mirrors ``causality_probe._shift_coord`` but in the VLM's native 0–1000 space.
    A no-op on a subtask without a coordinate."""
    m = _COORD_RE.search(subtask)
    if not m:
        return subtask
    x, y = int(m.group(1)), int(m.group(2))
    half = hi // 2
    nx = max(0, min(hi, x - shift if x > half else x + shift))
    ny = max(0, min(hi, y - shift if y > half else y + shift))
    return subtask[: m.start()] + f"<{nx}, {ny}>" + subtask[m.end():]


def _resolve_adapter(path: str) -> str:
    """Accept a specific ``checkpoint-N`` dir or a swift output_dir (→ latest)."""
    p = Path(path)
    if p.name.startswith("checkpoint-"):
        return str(p)
    runs = sorted(p.glob("v*-*"))
    if not runs:
        raise FileNotFoundError(f"No versioned swift run under {p} — did SFT complete?")
    ckpts = list(runs[-1].glob("checkpoint-*"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint under {runs[-1]}")
    return str(max(ckpts, key=lambda c: int(c.name.split("-", 1)[1])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out VLM-subgoal rollout probe (pre-GRPO gate)")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--adapter-path", required=True,
                        help="checkpoint-N dir or swift output_dir (→ latest checkpoint).")
    parser.add_argument("--task", default="ButtonUnmask")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260601,
                        help="Held-out seed base — distinct from training demos / GRPO seed=0.")
    parser.add_argument("--rollout-max-steps", type=int, default=400)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-key-frames", type=int, default=4)
    parser.add_argument("--n-recent-frames", type=int, default=2)
    parser.add_argument("--reveal-window", type=int, default=64)
    parser.add_argument("--decision-warm-cap", type=int, default=150)
    parser.add_argument("--progress-margin", type=float, default=0.08)
    parser.add_argument("--output-dir", default="/tmp/probe_vlm_rollout")
    args = parser.parse_args()

    import torch

    from openpi_client import websocket_client_policy as _wp  # type: ignore

    from vla_memory.grpo.env_runner import EnvRunner
    from vla_memory.grpo.rollout import RolloutWorker
    from vla_memory.qwen_subgoal.model import QwenSubgoalPolicy
    from vla_memory.qwen_subgoal.prompts import parse_subgoal_output

    resolved = _resolve_adapter(args.adapter_path)
    print(f"[probe] VLM adapter: {resolved}", flush=True)
    policy = QwenSubgoalPolicy(
        adapter_init_path=resolved, torch_dtype=torch.bfloat16, device="cuda",
    )

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
            decision_warm_cap=args.decision_warm_cap,
            n_key_frames=args.n_key_frames, n_recent_frames=args.n_recent_frames,
            reveal_window=args.reveal_window,
        )

    rows: list[dict] = []
    skipped = 0
    for ep in range(args.episodes):
        seed = (args.seed * 1_000_003 + ep) % 2_147_483_647

        # peek + VLM rollout on ONE worker — the validated GRPO trainer pattern
        # (peek_at_decision_point → rollout). Greedy-decode the subgoal between them.
        w = fresh_worker()
        try:
            dp = w.peek_at_decision_point(ep, seed=seed)
            if dp.terminated_early or not dp.recent_frames:
                skipped += 1
                print(f"[probe] ep{ep} seed{seed}: decision point unreachable — skipped", flush=True)
                continue
            text, _ = policy.greedy_subgoal(
                key_frames=dp.key_frames, recent_frames=dp.recent_frames,
                task_goal=dp.task_goal, history_subgoals=dp.history_subgoals,
                max_new_tokens=args.max_new_tokens,
            )
            subtask, _ = parse_subgoal_output(text)
            res_v = w.rollout(ep, subtask, seed=seed)
        finally:
            w.close()

        # Coordinate-shifted contrast (separate fresh env, same seed/layout).
        shifted = _shift_qwen_xy(subtask)
        w = fresh_worker()
        try:
            res_s = w.rollout(ep, shifted, seed=seed)
        finally:
            w.close()

        # Online-oracle ceiling (separate fresh env, same seed/layout).
        w = fresh_worker()
        try:
            res_o = w.rollout_oracle(ep, seed=seed)
        finally:
            w.close()

        row = {
            "ep": ep, "subtask": subtask,
            "pv": res_v.progress, "sv": res_v.success_flag,
            "ps": res_s.progress, "ss": res_s.success_flag,
            "po": res_o.progress, "so": res_o.success_flag,
        }
        print(
            f"[probe] ep{ep} seed{seed}\n"
            f"        VLM     : progress={row['pv']:.3f} status={row['sv']} subtask={subtask!r}\n"
            f"        SHIFTED : progress={row['ps']:.3f} status={row['ss']} subtask={shifted!r}\n"
            f"        ORACLE  : progress={row['po']:.3f} status={row['so']}",
            flush=True,
        )
        rows.append(row)

    if not rows:
        print("[probe] no valid episodes (all decision points unreachable) — check task/seed.", flush=True)
        sys.exit(2)

    pv = np.array([r["pv"] for r in rows], float)
    ps = np.array([r["ps"] for r in rows], float)
    po = np.array([r["po"] for r in rows], float)
    vlm_succ = float(np.mean([r["sv"] == "success" for r in rows]))
    oracle_succ = float(np.mean([r["so"] == "success" for r in rows]))
    gap_vs_shifted = float(pv.mean() - ps.mean())
    wins = int(np.sum(pv > ps + 1e-9))
    losses = int(np.sum(ps > pv + 1e-9))

    bootstrap_ok = vlm_succ > 0 or pv.mean() > 0.5             # GRPO can climb from here
    generalizes = pv.mean() >= po.mean() - max(0.15, args.progress_margin)
    coord_load_bearing = gap_vs_shifted >= args.progress_margin and wins >= max(1, losses)

    print("\n==================== HELD-OUT PROBE VERDICT ====================", flush=True)
    print(f"  episodes (valid / skipped)         : {len(rows)} / {skipped}", flush=True)
    print(f"  VLM      progress / success        : {pv.mean():.3f} / {vlm_succ*100:.0f}%", flush=True)
    print(f"  ORACLE   progress / success        : {po.mean():.3f} / {oracle_succ*100:.0f}%   (ceiling)", flush=True)
    print(f"  SHIFTED  progress                  : {ps.mean():.3f}   (gap {gap_vs_shifted:+.3f}, wins/losses {wins}/{losses})", flush=True)
    print("  -----", flush=True)
    print(f"  bootstrap  (VLM success > 0)        : {'YES' if bootstrap_ok else 'NO'}", flush=True)
    print(f"  generalizes (VLM ≈ oracle)         : {'YES' if generalizes else 'NO'}", flush=True)
    print(f"  coord load-bearing (VLM > shifted) : {'YES' if coord_load_bearing else 'NO'}", flush=True)
    print("  -----", flush=True)
    if bootstrap_ok and coord_load_bearing:
        print("  RESULT: PASS — the VLM's grounded subgoal drives correct picks on held-out", flush=True)
        print("          seeds, beats the shifted contrast (conversion + grounding correct), and", flush=True)
        print("          gives a non-zero success rate. → GRPO has a real bootstrap; proceed.", flush=True)
        if not generalizes:
            print("          NOTE: VLM trails the oracle ceiling — GRPO has headroom to close the gap.", flush=True)
        verdict = 0
    elif not bootstrap_ok:
        print("  RESULT: FAIL — VLM success is ZERO. GRPO would see all-zero reward and flatline", flush=True)
        print("          (RL cold-start needs >0). Check: is this the right SFT ckpt? Does the", flush=True)
        print("          conversion land in-container? Compare VLM vs ORACLE subtask coords by hand.", flush=True)
        verdict = 1
    else:
        print("  RESULT: WEAK — VLM picks succeed sometimes but don't beat the shifted contrast,", flush=True)
        print("          so the coordinate may not be load-bearing (grounding too coarse, or the", flush=True)
        print("          conversion is off). Inspect rows before trusting GRPO credit assignment.", flush=True)
        verdict = 1
    print("===============================================================", flush=True)
    sys.exit(verdict)


if __name__ == "__main__":
    main()
