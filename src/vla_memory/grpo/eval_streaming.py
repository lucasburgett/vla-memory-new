"""Full eval of the MemER-streaming hierarchy (GRPO-trained VLM + frozen GroundSG).

This is the FAITHFUL eval for a ``--streaming-memory`` GRPO checkpoint: it drives
the EXACT inference path the model was trained on (``RolloutWorker.rollout_streaming``,
the single-call MemER generator), not the one-shot ``probe_vlm_rollout`` path. The
distinction matters — the streaming model emits ONE ``{current_subtask,
keyframe_positions}`` per pick and a keyframe buffer accumulates across picks; the
one-shot probe makes a single decision with reveal-window keyframes and discards
``keyframe_positions`` entirely, so it would mis-measure a streaming adapter (prompt
parity violation, the project's recurring failure mode).

For each held-out seed:

  rollout_streaming generator:
    oracle drives the deterministic scaffolding (presses, put-downs)
      → at each PICK, GREEDY-decode the VLM ``{current_subtask, keyframe_positions}``
      → nominated frames merge into the keyframe buffer (MemER d=8/cap=8)
      → execute current_subtask on GroundSG (coords.from_qwen_xy in the rollout)
    → score success / task-completion progress

and compare to the ONLINE-ORACLE ceiling on the same seed/layout. Greedy decode (not
sampled) — the deterministic eval signal.

Reports aggregate VLM success rate / progress vs the oracle ceiling, the per-pick
count distribution, and writes a results JSON. Unlike ``probe_vlm_rollout`` this is a
MEASUREMENT, not a gate — it always exits 0 (the number is the product).

Run via ``modal run modal_train.py::eval_streaming``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "/app/examples/robomme")


def _resolve_adapter(path: str) -> str:
    """Resolve to a PEFT adapter dir. Accepts, in priority order:
      * a GRPO step dir holding ``policy/`` (e.g. ``.../step15``)         → ``.../step15/policy``
      * a direct adapter dir (has ``adapter_config.json``, e.g. SFT ckpt) → itself
      * a swift output_dir (``permanence_grounded``)                       → latest ``v*/checkpoint-*``
    """
    p = Path(path)
    if (p / "policy" / "adapter_config.json").exists():
        return str(p / "policy")
    if (p / "adapter_config.json").exists():
        return str(p)
    runs = sorted(p.glob("v*-*"))
    if runs:
        ckpts = list(runs[-1].glob("checkpoint-*"))
        if ckpts:
            return str(max(ckpts, key=lambda c: int(c.name.split("-", 1)[1])))
    raise FileNotFoundError(
        f"No adapter under {p} — expected a step dir with policy/, an adapter dir "
        "with adapter_config.json, or a swift output_dir with v*/checkpoint-*."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full streaming-hierarchy eval (GRPO VLM + GroundSG)")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--adapter-path", required=True,
                        help="GRPO step dir (.../stepN), adapter dir, or swift output_dir.")
    parser.add_argument("--task", default="ButtonUnmaskSwap")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260601,
                        help="Held-out seed base — distinct from the GRPO training seed=0.")
    parser.add_argument("--run-oracle", action=argparse.BooleanOptionalAction, default=True,
                        help="Also roll out the online-oracle ceiling per seed (doubles full episodes).")
    # Streaming hyperparameters — MUST match the training run (wandb config of the ckpt).
    parser.add_argument("--rollout-max-steps", type=int, default=700)
    parser.add_argument("--streaming-max-picks", type=int, default=4)
    parser.add_argument("--n-candidate-frames", type=int, default=12)
    parser.add_argument("--max-keyframes", type=int, default=4)
    parser.add_argument("--decision-warm-cap", type=int, default=250)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--subgoal-type", default="grounded_subgoal")
    parser.add_argument("--output-dir", default="/tmp/eval_streaming")
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=False,
                        help="Save an annotated mp4 per episode (VLM rollout, and oracle if "
                             "--run-oracle) to <output-dir>/videos. Use a small --episodes; "
                             "recording every frame of 50 full rollouts is wasteful.")
    args = parser.parse_args()

    import torch

    from openpi_client import websocket_client_policy as _wp  # type: ignore

    from vla_memory.grpo.env_runner import EnvRunner
    from vla_memory.grpo.rollout import DecisionPoint, RolloutWorker
    from vla_memory.qwen_subgoal.model import QwenSubgoalPolicy
    from vla_memory.qwen_subgoal.prompts import parse_subgoal_output

    # Submodule's annotated-frame video recorder (resolves via /app/examples/robomme).
    RolloutRecorder = None
    if args.record_video:
        from utils import RolloutRecorder  # type: ignore

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_adapter(args.adapter_path)
    print(f"[eval] VLM adapter        : {resolved}", flush=True)
    print(f"[eval] task / episodes    : {args.task} / {args.episodes}", flush=True)
    print(f"[eval] max_picks / cand   : {args.streaming_max_picks} / {args.n_candidate_frames}", flush=True)
    policy = QwenSubgoalPolicy(
        adapter_init_path=resolved, torch_dtype=torch.bfloat16, device="cuda",
    )

    def fresh_worker():
        env_runner = EnvRunner(
            env_id=args.task,
            video_save_dir=str(out_dir / "videos"),
            max_steps=args.rollout_max_steps,
            dataset="train",
        )
        client = _wp.MMEVLAWebsocketClientPolicy("127.0.0.1", args.port)
        return RolloutWorker(
            env_runner=env_runner, policy_client=client, obs_horizon=16,
            max_steps=args.rollout_max_steps, use_history=False,
            subgoal_type=args.subgoal_type,
            decision_warm_cap=args.decision_warm_cap,
            n_candidate_frames=args.n_candidate_frames,
            max_keyframes=args.max_keyframes,
        )

    rows: list[dict] = []
    skipped = 0
    for ep in range(args.episodes):
        seed = (args.seed * 1_000_003 + ep) % 2_147_483_647

        # --- VLM streaming rollout: drive the generator, greedy-decode each pick ---
        w = fresh_worker()
        picks: list[dict] = []
        vrec = RolloutRecorder(out_dir / "videos", task_goal=args.task, fps=30) if RolloutRecorder else None
        try:
            gen = w.rollout_streaming(ep, seed=seed, max_picks=args.streaming_max_picks, recorder=vrec)
            dp = next(gen)
            while isinstance(dp, DecisionPoint):
                text, gen_ids = policy.greedy_subgoal(
                    key_frames=dp.key_frames, recent_frames=dp.recent_frames,
                    task_goal=dp.task_goal, history_subgoals=dp.history_subgoals,
                    max_new_tokens=args.max_new_tokens, mode="use",
                )
                subtask, kf = parse_subgoal_output(text)
                picks.append({
                    "buf": len(dp.key_frames), "subtask": subtask,
                    "kf": kf, "ntok": int(gen_ids.numel()),
                })
                dp = gen.send((subtask, kf))
            res_v = dp  # final yield is the RolloutResult
            gen.close()
        finally:
            w.close()
        if vrec is not None:
            vrec.save_video(f"{args.task}_ep{ep}_{res_v.success_flag}_vlm.mp4")

        if not picks:
            # Episode ended in the deterministic scaffolding before any VLM pick —
            # nothing the VLM owned. Record + skip from the VLM aggregate.
            skipped += 1
            print(f"[eval] ep{ep} seed{seed}: no VLM decision point (scaffold-only) — skipped", flush=True)

        row: dict = {
            "ep": ep, "seed": seed, "n_picks": len(picks), "picks": picks,
            "pv": res_v.progress, "sv": res_v.success_flag,
        }

        # --- online-oracle ceiling on the same seed/layout (fresh env) ---
        if args.run_oracle:
            w = fresh_worker()
            orec = RolloutRecorder(out_dir / "videos", task_goal=args.task, fps=30) if RolloutRecorder else None
            try:
                res_o = w.rollout_oracle(ep, seed=seed, recorder=orec)
            finally:
                w.close()
            if orec is not None:
                orec.save_video(f"{args.task}_ep{ep}_{res_o.success_flag}_oracle.mp4")
            row["po"] = res_o.progress
            row["so"] = res_o.success_flag

        last_pick = picks[-1]["subtask"] if picks else "<none>"
        msg = (f"[eval] ep{ep} seed{seed} picks={len(picks)}\n"
               f"        VLM    : progress={row['pv']:.3f} status={row['sv']} last_subtask={last_pick!r}")
        if args.run_oracle:
            msg += f"\n        ORACLE : progress={row['po']:.3f} status={row['so']}"
        print(msg, flush=True)
        rows.append(row)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    vlm_rows = [r for r in rows if r["n_picks"] > 0]
    n_valid = len(vlm_rows)
    pv = np.array([r["pv"] for r in vlm_rows], float) if vlm_rows else np.array([0.0])
    vlm_succ = float(np.mean([r["sv"] == "success" for r in vlm_rows])) if vlm_rows else 0.0
    pick_counts = [r["n_picks"] for r in vlm_rows]

    summary = {
        "task": args.task,
        "adapter": resolved,
        "episodes_requested": args.episodes,
        "episodes_valid": n_valid,
        "episodes_skipped": skipped,
        "vlm_success_rate": vlm_succ,
        "vlm_mean_progress": float(pv.mean()),
        "mean_picks_per_episode": float(np.mean(pick_counts)) if pick_counts else 0.0,
        "seed_base": args.seed,
    }
    if args.run_oracle:
        po = np.array([r["po"] for r in vlm_rows], float) if vlm_rows else np.array([0.0])
        oracle_succ = float(np.mean([r["so"] == "success" for r in vlm_rows])) if vlm_rows else 0.0
        summary["oracle_success_rate"] = oracle_succ
        summary["oracle_mean_progress"] = float(po.mean())
        summary["vlm_vs_oracle_gap"] = float(pv.mean() - po.mean())

    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "eval_rows.json").write_text(json.dumps(rows, indent=2))

    print("\n==================== STREAMING EVAL SUMMARY ====================", flush=True)
    print(f"  task                            : {args.task}", flush=True)
    print(f"  adapter                         : {resolved}", flush=True)
    print(f"  episodes (valid / skipped)      : {n_valid} / {skipped}", flush=True)
    print(f"  mean picks / episode            : {summary['mean_picks_per_episode']:.2f}", flush=True)
    print(f"  VLM     success / progress      : {vlm_succ*100:.1f}% / {pv.mean():.3f}", flush=True)
    if args.run_oracle:
        print(f"  ORACLE  success / progress      : {summary['oracle_success_rate']*100:.1f}% / "
              f"{summary['oracle_mean_progress']:.3f}   (ceiling)", flush=True)
        print(f"  VLM − ORACLE progress gap       : {summary['vlm_vs_oracle_gap']:+.3f}", flush=True)
    print(f"  results                         : {out_dir}/eval_summary.json", flush=True)
    print("===============================================================", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
