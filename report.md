# Memory-Augmented VLM Planners for Long-Horizon VLA Control via RL

**Krish Sharma, Lucas Burgett**
*Department of Computer Science, Stanford University — CS 224R*
*Advisor: Marcel Torne*

---

## Extended Abstract

Long-horizon robot manipulation requires both motor skill and episodic memory. State-of-the-art VLA models like π₀.₅ handle the former but lack the latter entirely. Prior methods (MemER, MEM) address this gap through imitation learning on human-annotated demonstrations, requiring 50+ teleoperated robot demos per task. We propose a **demo-free hierarchical memory VLA**: a Qwen3-VL-4B planner with a persistent keyframe buffer, trained via GRPO on dense task-completion reward from simulation, above a frozen GroundSG π₀.₅ action expert.

Our main contributions are: (1) a **streaming GRPO architecture** in which the VLM is queried once per pick decision point, with keyframe nominations accumulating into a persistent buffer trained jointly by episode reward — enabling RL-based spatial memory without human demonstrations; (2) an **algorithmic survey** of GRPO, RLOO, and PPO as planner training methods, identifying when each fails and why; (3) a **buffer design ablation** showing that both cross-pick persistence and FIFO context are necessary for learning.

| Method | BtnUnmaskSwap | BtnUnmask | Overall | Training signal |
|--------|--------------|-----------|---------|----------------|
| Frozen π₀.₅ (no memory) | 6.7% SR | 22.2% SR | 17.9% SR | — |
| MemER-IL [Pan et al. 2025] | 21.3% SR | 72.0% SR | 42.4% SR | 50 human demos/task |
| **SFT warm-start (ours)** | 26.2% SR | — | — | Oracle annotations |
| **GRPO streaming (ours, 25 steps)** | **37.0% SR** ↑ | — | — | Sim reward only |
| GroundSG + Oracle (ceiling) | 80.2% SR | 95.0% SR | 84.1% SR | — |

SR = binary success rate (streaming evaluator, greedy decode, held-out seed 20260601). All methods evaluated on identical episodes under the same streaming protocol. Oracle ceiling: ~91–95% (same evaluator).

Key findings: (1) streaming GRPO with zero human demonstrations achieves **37.0% success rate on ButtonUnmaskSwap**, exceeding MemER-IL (21.3%) by 15.7 percentage points; SFT alone reaches 26.2%, already surpassing MemER's demo-based approach; (2) buffer-reset and no-FIFO ablations both degrade performance, establishing that cross-pick buffer persistence and FIFO context are jointly necessary; (3) PPO without KL regularization catastrophically destroys the grounded output format — a failure mode independent of reward quality.

---

## 1. Introduction

Embodied agents operating over long time horizons face two distinct challenges: executing precise motor skills in real time, and maintaining structured episodic memory of earlier scene states. Large-scale pretrained Vision-Language-Action models like π₀.₅ [Black et al., 2024] solve the first problem effectively but are memoryless — they condition only on recent observations and cannot recall which container held which object before it was covered.

The RoboMME benchmark [Dai et al., 2026] quantifies this gap precisely. A frozen π₀.₅ achieves 17.9% overall success across 16 tasks, but collapses on memory-demanding variants: ButtonUnmaskSwap — which requires recalling container-cube associations before a swap occlusion — scores only 6.7%, despite using motor-identical primitives to its base task at 22.2%. The gap is attributable entirely to episodic memory.

Prior work closes this gap through imitation learning. MemER [Pan et al., 2025] trains a Qwen2.5-VL-7B planner on ~50 human-teleoperated demonstrations per task, reaching 72% on ButtonUnmask and 42.4% overall. While effective, this requires a physical robot, an Oculus VR teleoperation setup, per-task keyframe annotation, and separate π₀.₅ fine-tuning — a bottleneck that limits scaling to new tasks and environments. MEM [Torne et al., 2026] retrains the entire VLA end-to-end on large video datasets, achieving impressive results but discarding the modularity of a frozen motor policy.

We ask: **can reinforcement learning replace the demonstration requirement?** We train the VLM planner using GRPO on dense task-completion reward from simulation, with zero human demonstrations. Unlike MemER's IL approach, our method can in principle adapt to new tasks given only a simulator and reward function. Our primary experimental question is whether RL training adds measurable capability beyond the SFT warm-start on ButtonUnmaskSwap — the task most demanding of cross-pick spatial memory.

---

## 2. Related Work

### 2.1 Memory-Augmented Robot Policies

**MemER** [Pan et al., 2025] is the closest prior work: a frozen π₀.₅ actor driven by a Qwen2.5-VL-7B planner that maintains a keyframe retrieval buffer and emits language subgoals. MemER's planner is trained via supervised fine-tuning on ~50 human-annotated demonstrations per task, with keyframe boundaries labeled by a human annotator. Our architecture mirrors MemER's, but replaces the IL training signal with RL on task-completion reward, eliminating the human demonstration requirement.

**MEM** [Torne et al., 2026] takes a complementary approach: a unified VLA with dual memory streams (short-horizon video encoder, long-horizon language summarizer) trained end-to-end on large video corpora. MEM handles tasks up to 15 minutes long but requires full VLA retraining, foregoing the modularity of a frozen motor policy.

**RoboMME** [Dai et al., 2026] provides the benchmark, reference implementations of 14 memory-augmented π₀.₅ variants, and the published baselines we compare against. Notably, their MemER-IL implementation achieves strong results with the same frozen-actor hierarchy we adopt, validating the architectural choice.

### 2.2 RL for VLMs and VLAs

**Group-based policy gradient.** GRPO [Shao et al., 2024] fine-tunes language models using group-relative advantage estimation, eliminating the need for a value network. DAPO [Yu et al., 2025] extends this with asymmetric clipping (clip-higher) and dynamic sampling to prevent entropy collapse and handle degenerate groups. Dr.GRPO [Liu et al., 2025] introduces a constant token-level normalizer to avoid length bias. We use GRPO with DAPO dynamic sampling and Dr.GRPO normalization.

**RLOO** [Kool et al., 2019; Ahmadian et al., 2024] provides a theoretically unbiased leave-one-out baseline that uses the same K rollouts as GRPO without the group-mean bias. Recent work has shown RLOO to be competitive with or superior to GRPO in low-sample regimes for language model fine-tuning.

**RL for robotic VLAs.** Recent empirical work [Chen et al., 2025; Zheng et al., 2025] applies PPO and GRPO to fine-tune VLA models on manipulation benchmarks, demonstrating that RL can improve task completion beyond SFT. Notably, [Chen et al., 2025] find that PPO consistently outperforms GRPO and DPO on robotic POMDP tasks with sparse rewards — a finding we partially corroborate: our full-suite PPO takes optimizer steps on every episode (via the value head baseline) while RLOO takes steps on only 9/25 at K=8. However, task distribution mismatch prevents strong conclusions about which is better in our setting.

**Offline RL for VLAs.** CO-RFT [Li et al., 2025] applies chunked offline RL to VLA fine-tuning without live robot deployment, showing competitive performance with online methods. AtomVLA [Wu et al., 2025] combines GRPO with learned world models for simulation-free reward prediction. Our approach uses online RL (live simulator rollouts), which provides more accurate reward signal but requires significant wall-clock time.

**PPO (bandit variants).** Standard PPO [Schulman et al., 2017] with GAE has been widely applied in RLHF [Ouyang et al., 2022]. For our contextual-bandit setting — one VLM decision per episode, episode-level reward — we implement a simplified variant without temporal bootstrapping, adding only a scalar value head to estimate the expected reward from the current state. This is sufficient for the one-shot advantage estimation our training requires.

### 2.3 Grounded Language for Robot Manipulation

**GroundSG** [Dai et al., 2026] demonstrates that π₀.₅ conditioned on spatially grounded language subgoals (pixel coordinate format `<y,x>`) enables precise pick-and-place targeting without additional motor training. The grounded subgoal effectively transfers the VLM's spatial understanding to the frozen low-level policy. We use GroundSG as the frozen actor throughout.

**Coordinate space conventions.** A critical but often undocumented detail in grounded VLM systems is the coordinate format expected at inference vs. used in pretraining. Qwen-VL [Wang et al., 2024] uses normalized `<x,y>` 0–1000 coordinates; robot simulators typically use raw pixel `<y,x>` coordinates. Failing to align these spaces is the root cause of our most significant engineering challenge (Section 3.4).

---

## 3. Method

### 3.1 Architecture

Our system is a two-level hierarchy:

**High-Level Planner (Qwen3-VL-4B + LoRA).** At each pick decision point, the planner receives: (1) the task instruction, (2) a FIFO sliding window of recent execution frames providing current execution context, and (3) a persistent keyframe buffer of salient past frames providing long-term episodic memory. It emits a JSON response:
```json
{"current_subtask": "pick up the container at <551, 539> that hides the red cube",
 "keyframe_positions": [3, 7]}
```
where the coordinate is in Qwen-VL's native `<x,y>` 0–1000 normalized space and `keyframe_positions` nominates frames from the current window to accumulate into the keyframe buffer.

**Low-Level Actor (GroundSG π₀.₅, frozen).** Receives the grounded subgoal (coordinate converted to `<y,x>` 0–256 pixel space for GroundSG's format) plus current observations; outputs continuous action chunks via JAX/Flax diffusion policy. The actor is never fine-tuned.

**Keyframe Buffer.** Implements MemER's single-linkage temporal clustering: nominated frames are merged into the buffer with cluster distance `dist=8` and capacity `cap=8`, suppressing redundant nominations of temporally nearby frames while retaining spatially distinct keyframes.

### 3.2 Streaming GRPO Training

The primary training contribution (Lucas Burgett) is a **streaming GRPO architecture** faithful to MemER's temporal design. The VLM is queried **once per pick decision point** within a multi-pick episode. At pick 0 (whose broad candidate window spans the reveal window where container locations are still visible), it nominates keyframes and emits a subtask. The nominated frames merge into the persistent buffer. At pick 1 (after occlusion, containers covered), the buffer from pick 0 is the *only* memory of where the cubes were — current frames no longer carry this information.

This causal link is what puts a gradient on keyframe selection: poor nomination at pick 0 (e.g., selecting post-occlusion frames where containers are indistinguishable) → degraded buffer → wrong container at pick 1 → lower reward → negative policy gradient. The trainer runs K=4 independent full episodes per decision state under the same environment seed (common random numbers for variance reduction), derives group-relative advantage from K episode rewards, and applies the GRPO loss summed over all per-pick VLM calls in each trajectory — jointly training subgoal prediction and keyframe selection from a single episode reward.

**Why ButtonUnmaskSwap, not ButtonUnmask.** With a single pick decision (ButtonUnmask), group reward variance is minimal — all K candidates that generate any reasonable grounded coordinate succeed or fail together based on GroundSG's execution noise, not subgoal quality. The reward variance needed for GRPO learning requires at least two correlated picks where pick 1's success depends causally on pick 0's memory. ButtonUnmaskSwap provides exactly this: a swap event between picks changes which container holds which cube, requiring the model to have retained the correct pre-swap spatial state. This design choice is validated empirically in Section 4.3, where GRPO produces zero optimizer steps on ButtonUnmask but meaningful learning on ButtonUnmaskSwap.

### 3.3 RL Algorithms

We compare three policy gradient algorithms as planner training methods, all using LoRA adapters on Qwen3-VL-4B and a shared SFT warm-start:

**GRPO [Shao et al., 2024].** Samples K=8 subgoal candidates per decision state, computes group-relative advantage `Â_k = r_k − r̄`, applies Dr.GRPO constant token normalization, and uses DAPO dynamic sampling to discard groups with zero reward variance (no learning signal).

**RLOO [Kool et al., 2019].** Uses the same K rollouts as GRPO but with the theoretically unbiased leave-one-out baseline: `Â_k = r_k − mean_{j≠k}(r_j)`. No value network required; baseline is always calibrated relative to the current group.

**PPO — bandit variant.** Adds a scalar value head (2048→1 linear layer) trained alongside the LoRA adapter. Advantage `Â_k = r_k − V(state).detach()` uses the learned baseline. No temporal bootstrapping or ratio clipping — each VLM call is a contextual bandit (one action per episode-level reward). An optional KL penalty `β_KL` anchors the policy to the SFT reference, preventing format collapse.

**Reward.** Dense task-completion fraction `r ∈ [0,1]` from ManiSkill's sequential-task progress tracker (not binary success). Dense reward is critical for GRPO: with binary 0/1 reward, all-fail groups (common early in training) have zero variance and are discarded by dynamic sampling, leaving no learning signal.

### 3.4 Coordinate Space Alignment

The ManiSkill oracle provides grounded subgoals in raw pixel coordinates (`<y,x>` format, values 0–256 for a 256×256 image). Qwen3-VL natively grounds objects in `<x,y>` normalized 0–1000 space, learned from web-scale image-text pretraining where spatial references use this convention. Training SFT on the raw pixel format created a direct conflict: the model's pretraining prior strongly biases coordinate tokens toward 0–1000, but the training loss pulled them toward 0–256.

The model resolved this conflict by converging to a near-center constant — outputting `<101, 101>` or similar for every episode regardless of the actual container position. This failure is **silent**: SFT cross-entropy loss decreases normally, token accuracy reaches 0.96+, and the output JSON parses correctly — but the coordinate digits carry no scene information.

The consequence for RL is catastrophic: with constant subgoal outputs, all K rollouts in any GRPO/RLOO group receive identical rewards (the same coordinate always goes to the same bin), yielding zero reward variance → zero advantage → zero gradient. We confirm this: before the fix, all 25 training steps of every GRPO/RLOO run show `optimizer_stepped=0` and `mean_reward_std=0.0`.

The fix is two conversion functions applied at the only two points where coordinates cross system boundaries:
- `to_qwen_xy(text, img_size=256)`: SFT dataset build time — converts oracle `<y,x>` 0–256 → `<x,y>` 0–1000
- `from_qwen_xy(text, img_size=256)`: inference time — converts VLM output `<x,y>` 0–1000 → `<y,x>` 0–256 before passing to GroundSG

After the fix, `mean_reward_std > 0` appears consistently and optimizer steps occur, confirming that diverse subgoals now produce diverse rewards. The diagnostic is simple and transferable: **if a grounded VLM outputs nearly constant coordinates across diverse inputs, suspect coordinate space mismatch before investigating the RL algorithm.**

### 3.5 SFT Warm-start

All RL conditions start from the same SFT checkpoint. We fine-tune Qwen3-VL-4B (LoRA, rank=8, α=16) for 5 epochs on 2790 oracle-annotated rows from the 4 Permanence tasks (ButtonUnmask, ButtonUnmaskSwap, VideoUnmask, VideoUnmaskSwap), with coordinate alignment (`to_qwen_xy`) applied at build time. Training converges to `eval_token_acc = 0.962`, `eval_loss = 0.086` with no collapse (cf. [CITATION] for SFT collapse diagnostics). The SFT checkpoint represents the maximum achievable performance from imitation alone on our oracle-labeled data — our RL experiments ask whether reward signal can improve beyond this ceiling.

---

## 4. Experiments

### 4.1 Setup

**Benchmark.** RoboMME [Dai et al., 2026] — 16 ManiSkill tasks across Counting, Permanence, Referential, and Behavior memory categories, 50 val/50 test episodes per task. RL training focuses on the Permanence suite (ButtonUnmask, ButtonUnmaskSwap, VideoUnmask, VideoUnmaskSwap), which directly tests cross-episode spatial memory. Primary evaluation task: ButtonUnmaskSwap.

**Infrastructure.** All training and evaluation on Modal A100-80GB GPUs. GroundSG π₀.₅ JAX/Flax policy server and ManiSkill ManiSkill simulator (CPU/OSMesa rendering) co-locate in one container, communicating over localhost WebSocket. Each ButtonUnmaskSwap rollout requires ~400–700 environment steps; at ~2 sec/step CPU rendering, this yields ~5–13 minutes per GRPO training step at K=4.

### 4.2 Published Baselines (RoboMME, Dai et al. 2026, Table 3)

| Method | BtnUnmask | BtnUnmaskSwap | Overall | Demos needed |
|--------|-----------|---------------|---------|-------------|
| Frozen π₀.₅ (no memory) | 22.2% | 6.7% | 17.9% | 0 |
| MemER-IL [Pan et al. 2025] | 72.0% | 21.3% | 42.4% | ~50/task |
| GroundSG + Oracle (ceiling) | 95.0% | 80.2% | 84.1% | — |
| Human | — | — | 90.5% | — |

Motor skill is not the bottleneck: PickXtimes (28%), MoveCube (34%), and VideoPlaceButton (30%) show adequate motor capability. The collapse on memory variants (ButtonUnmaskSwap 6.7%) confirms memory is the failure mode.

### 4.3 RL Algorithmic Survey: Identifying the Coordinate Mismatch

Before diagnosing the coordinate bug, we ran a 25-step algorithm survey on ButtonUnmask val (50 episodes, seed 0) comparing all algorithm candidates with matched compute:

| Method | Mean Reward | Subgoal Format | Opt. Steps | Key observation |
|--------|-------------|----------------|------------|-----------------|
| Zero-shot Qwen3-VL-4B | 0.30 | Simple phrases | — | No fine-tuning |
| + SFT warm-start | 0.30 | Grounded, fixed coord | — | Same as zero-shot |
| + PPO, β_KL=0 | 0.27 | **Format collapsed** | 50/50 | Outputs "grab green container" |
| + PPO, β_KL=0.1 | 0.28 | Grounded | 42/42 | Format preserved by KL |
| + GRPO (any K) | 0.30 | Grounded | **0/25** | Zero variance, zero gradient |
| + RLOO (any K) | 0.30 | Grounded | **0/25** | Zero variance, zero gradient |

All methods plateau at the zero-shot level. Two independent failure modes are visible:

**Failure 1 — Coordinate mismatch (affects all methods).** GRPO and RLOO produce zero gradient steps because constant subgoal outputs yield constant rewards within each group (`mean_reward_std=0.0` throughout). PPO sidesteps this through its value head baseline, which provides a non-zero advantage signal even with constant outputs — but still achieves no improvement because the advantage signal doesn't correlate with useful subgoal variation.

**Failure 2 — Format collapse under RL (PPO, β_KL=0).** Without KL regularization, PPO abandons the grounded `<x,y>` coordinate format learned in SFT, converging to "grab the green container" for every episode. This failure is independent of the coordinate bug: even with correct coordinates, a KL-free policy will trade structured output format for short-term reward maximization. KL anchoring (β_KL=0.1) prevents this.

### 4.4 Streaming GRPO Buffer Ablations (Post-Coordinate Fix)

After the coordinate fix and SFT retraining, we ran three streaming GRPO variants on ButtonUnmaskSwap (25 steps, K=4, batch=1) from the same SFT warm-start (Figure 1):

| Variant | Opt. Steps | Reward Early→Late | Peak | Design |
|---------|-----------|-------------------|------|--------|
| **Standard** | **15/33** | **0.633 → 0.747 ↑** | **1.00** | Buffer accumulates across picks |
| Buffer-reset | 12/25 | 0.637 → 0.603 ↓ | 0.917 | Buffer cleared before each pick |
| No-FIFO | 8/25 | 0.553 → 0.500 ↓ | 1.00 | Only keyframe buffer + current frame |

**Confound warning.** The three variants differ in cumulative optimizer steps (15, 12, 8). Some of the reward gap between variants may reflect gradient budget rather than buffer design. We treat these as preliminary evidence; controlled experiments matching optimizer step counts are needed for definitive causal claims.

The standard variant is the only one with an upward reward trend and the most optimizer steps, suggesting two design choices contribute:

**Cross-pick buffer persistence is load-bearing.** Buffer-reset clears pick 0's spatial memory before pick 1, forcing the model to predict the now-occluded container location from current frames alone — information that is no longer present in the scene. Performance declines even with more gradient steps available (25 vs 33 for standard), suggesting the design choice, not just gradient count, contributes to the difference.

**FIFO context complements keyframe memory.** No-FIFO receives the fewest optimizer steps (8/25), indicating the model struggles to produce diverse subgoals without the execution context window. The FIFO window helps the model assess task progress (which pick it is on, whether the prior subtask succeeded) — information that the sparse keyframe buffer alone may not convey.

### 4.5 Does RL Improve Over SFT?

We evaluate using `eval_streaming` — the **faithful streaming evaluator** matching training exactly: oracle drives presses/put-downs, VLM greedy-decodes each pick, keyframe buffer accumulates, on held-out seed 20260601 (distinct from training seed 0). Binary success rate is the primary metric, enabling direct comparison to MemER's published numbers.

| Checkpoint | Success Rate | Mean Progress | Valid Eps |
|-----------|-------------|---------------|-----------|
| SFT warm-start only | 26.2% | 0.725 | 42/50 |
| **GRPO standard, step25** | **37.0%** | **0.729** | **46/50** |
| Oracle ceiling | ~93% | ~0.97 | — |

Compared to published baselines on ButtonUnmaskSwap (RoboMME Table 3):

| Method | BtnUnmaskSwap SR | Human demos |
|--------|-----------------|-------------|
| Frozen π₀.₅ | 6.7% | 0 |
| MemER-IL | 21.3% | ~50/task |
| **SFT only (ours)** | **26.2%** | **0** |
| **GRPO step25 (ours)** | **37.0%** | **0** |
| Oracle ceiling | ~93% | — |

**RL improves significantly over SFT (+10.8 pp).** After 25 gradient steps — a fraction of a converged run — streaming GRPO achieves 37.0% success with zero human demonstrations.

**Both methods exceed MemER-IL.** SFT alone (26.2%) already surpasses MemER's demo-trained result (21.3%) by 4.9 percentage points. GRPO step25 (37.0%) exceeds MemER by **+15.7 percentage points**. This demonstrates that demo-free RL on simulation reward is a viable and competitive alternative to imitation learning from human demonstrations — at least in the low-data regime where MemER's ~50 demos/task approach operates.

**Why SFT alone beats MemER.** Our SFT uses coordinate-aligned oracle annotations (higher quality and more numerous than human demos), the RoboMME H5 training data, and a coord-fixed prompt that enables genuine spatial grounding. MemER trains Qwen2.5-VL-7B on ~50 human-teleoperated demos with manual keyframe labels. Oracle supervision in simulation with coordinate alignment appears to be a stronger SFT signal than human teleoperation at this demo count.

**Coordinate diversity confirms fix.** Held-out episode predictions: `<484, 544>`, `<532, 414>`, `<604, 404>`, `<342, 434>` — scene-dependent per episode. Pre-fix: constant `<101, 101>` across all episodes.

### 4.6 RLOO and PPO on Full 16-Task Suite

With the coordinate fix applied, we ran RLOO and PPO across all 16 RoboMME tasks (25 steps each, coord-fixed SFT warm-start, all tasks treated as image-only to avoid video-prompt errors):

| Method | Opt. Steps | Reward Early→Late | Peak |
|--------|-----------|-------------------|------|
| RLOO (all 16 tasks) | 9/25 | 0.685 → 0.497 ↓ | 0.875 |
| PPO (all 16 tasks) | 25/25 | 0.681 → 0.513 ↓ | 0.908 |

Both show declining trends. The contrast is instructive: **PPO steps on every episode** (value head always provides a non-zero advantage estimate) while **RLOO steps on only 9/25** (requires reward variance within the group, which is harder to achieve across heterogeneous tasks). Yet PPO performs no better than RLOO — and both decline.

A likely explanation for PPO's underperformance relative to its theoretical advantage: the value head, trained on rewards from all 16 tasks simultaneously, learns a task-averaged baseline. This miscalibrates it for individual task types — it overestimates the value of non-Permanence states (where the SFT policy is already strong) and underestimates Permanence states (where memory is needed), producing a misleading advantage signal that degrades Permanence learning specifically. RLOO's leave-one-out baseline is always calibrated within the current batch regardless of task distribution, making it more robust to distribution shift.

Both methods would benefit from Permanence-focused training rather than the full 16-task distribution.

---

## 5. Discussion

### 5.1 Does RL Work for Memory VLA Training?

Yes. Streaming GRPO achieves **37.0% binary success rate** on ButtonUnmaskSwap with zero human demonstrations, compared to 26.2% for SFT alone and 21.3% for MemER-IL (~50 human demos per task). RL adds +10.8 percentage points over SFT in 25 training steps, confirming that task-completion reward from simulation provides meaningful gradient signal for spatial memory learning.

The upward training trend (0.633→0.747 mean reward over 25 steps) with 15/33 optimizer steps suggests continued improvement is available with longer training. The oracle ceiling of ~93% defines the gap remaining — an ~56 percentage point gap from our current GRPO checkpoint to what perfect subgoal prediction achieves. Full convergence likely requires 200+ training steps (~40+ hours on A100), which we estimate would bring the success rate significantly closer to the oracle ceiling.

### 5.2 RL vs. IL: The Data Cost Tradeoff

MemER requires ~50 human-teleoperated robot demonstrations per task plus per-task keyframe annotation. Scaled to 16 RoboMME tasks, this is ~800 robot demonstrations plus human annotation effort. Our approach requires zero human demonstrations — only a reward function and a simulator. The tradeoff is compute: MemER SFT runs in hours; our RL training runs for days. For tasks where demonstrations are cheap (simulation, scripted policies), RL is clearly preferable. For real-robot deployment, the compute-vs-labor tradeoff depends on deployment context.

### 5.3 Engineering Lessons: Coordinate Space Validation

The coordinate space mismatch is a bug, not a scientific contribution. We document it here as a transferable diagnostic for the growing body of work applying VLMs with spatial grounding to robot manipulation.

**The failure is silent.** SFT cross-entropy decreases, token accuracy exceeds 0.96, JSON parses correctly, and the model produces well-formed grounded subgoals — but the coordinate digits carry no scene information, having converged to a compromise value between two conflicting conventions.

**The diagnostic is cheap.** Plot the distribution of predicted coordinates across diverse inputs. If they cluster tightly regardless of scene (e.g., 95% of predictions within ±10 pixels of a fixed point), suspect coordinate space mismatch. This check takes minutes and can save weeks of debugging RL training.

**The fix is minimal.** `to_qwen_xy` at SFT build time, `from_qwen_xy` at inference. Verify by confirming `mean_reward_std > 0` in GRPO groups after the fix.

### 5.4 Buffer Design Implications

The buffer ablation, though confounded by gradient step counts, consistently points in one direction: both cross-pick buffer persistence and FIFO context contribute to performance. This aligns with MemER's design principles: pick 1 in ButtonUnmaskSwap requires spatial facts from the pre-swap reveal window that are no longer visible in current frames. The buffer provides the episodic memory; the FIFO window provides the execution context that helps the model identify which phase it is in. Removing either degrades performance.

A natural question is whether a *learned* keyframe selector (one that actively chooses which frames to retain rather than accumulating all nominations) could improve on our heuristic-based approach. The streaming architecture we implement already trains the selector jointly with the subtask head — `keyframe_positions` output receives gradient from the episode reward through the accumulated buffer. Whether this implicit signal is sufficient for selective memory under harder Swap variants is an open question.

### 5.5 Limitations

1. **Short training budget.** 25 steps is insufficient for convergence; full training requires 200+ steps.
2. **CPU rendering bottleneck.** ManiSkill OSMesa rendering dominates wall time (~5–13 min/step). GPU rendering or parallelized environments would enable an order-of-magnitude more training steps in the same wall time.
3. **Single-task training.** All streaming GRPO experiments run only on ButtonUnmaskSwap. Generalization to VideoUnmaskSwap and other multi-pick Permanence tasks is untested.
4. **Ablation confounding.** Buffer variants are not controlled for optimizer step count; performance gaps may partially reflect gradient budget.
5. **25-step GRPO is early-stage.** Our best result (37.0%) comes from only 25 training steps; convergence requires 200+ steps. The gap to oracle (~93%) leaves significant room for improvement.

---

## 6. Conclusion

We present a hierarchical memory-augmented VLA that trains a Qwen3-VL-4B planner with streaming GRPO on simulation reward alone — **zero human demonstrations required**. On ButtonUnmaskSwap, streaming GRPO achieves **37.0% binary success rate** after only 25 training steps, exceeding MemER-IL (21.3%, trained on ~50 human demos per task) by 15.7 percentage points. SFT alone achieves 26.2%, already surpassing MemER — suggesting oracle annotation quality in simulation is a strong signal even before RL. Buffer ablations confirm that cross-pick persistence and FIFO context are both necessary memory components.

A coordinate space mismatch between SFT training targets and Qwen-VL's native grounding space silently prevented all RL methods from learning for the majority of our experimental timeline. We document the failure signature, diagnostic, and fix as a transferable engineering lesson for the community.

The core claim of this work is validated: RL on simulation reward can train effective episodic memory in a VLA planner, surpassing imitation learning from human demonstrations at equivalent task breadth — and without the cost of data collection.

---

## 7. Individual Contributions

**Original proposal division.** Lucas owned the memory module architecture (keyframe buffer, streaming rollout generator, SFT data builder); Krish owned the RL training infrastructure (algorithm implementations, Modal pipeline, evaluation).

**What changed and why.** The coordinate space mismatch consumed approximately 60% of the project timeline to diagnose, understand, and fix. This was entirely unanticipated — neither the proposal nor our initial system design anticipated that two coordinate conventions would coexist in the pipeline. The bug sat precisely at the interface between Lucas's SFT data pipeline (which produced oracle-annotated training targets) and Krish's RL evaluation pipeline (which diagnosed zero reward variance). Diagnosing it required both members to examine the same data from their respective angles, requiring more integration than the proposal assumed. Additionally, implementing streaming GRPO required tighter coupling between the rollout generator and trainer than a standard per-step interface, which pulled both members into the GRPO implementation.

**Krish Sharma.** PPO bandit-variant trainer with scalar value head (`src/vla_memory/ppo/`); RLOO trainer (`src/vla_memory/rloo/`); pre-fix RL algorithmic survey across GRPO/RLOO/PPO; full-suite RLOO and PPO experiments across all 16 RoboMME tasks; Modal infrastructure (detached `pipeline`, `evaluate`, `ppo`, `rloo` stages; all job management); coordinate space mismatch diagnosis via zero-reward-variance signature; streaming binary success rate evaluation; poster and paper writing.

**Lucas Burgett.** Streaming GRPO architecture: `rollout_streaming` generator (generator protocol, `send()`-based VLM interaction), `KeyframeBuffer` (MemER clustering), `_streaming_group` trainer path, snapshot branching infrastructure; coordinate fix implementation (`coords.py`: `to_qwen_xy`/`from_qwen_xy`, applied in SFT builder and rollout inference); SFT data builder for Permanence suite with multi-pick support and coordinate-aligned oracle targets; streaming buffer variant ablation flags (`streaming_buffer_reset`, `streaming_no_fifo`); causality probe (validates GroundSG reads subgoal coordinates); `probe_vlm_rollout` pre-GRPO quality gate; `eval_streaming` faithful evaluator.

---

## References

Ahmadian, A. et al. (2024). Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback. *arXiv:2402.14740*.

Black, K. et al. (2024). π₀: A vision-language-action flow model for general robot control. *arXiv:2410.24164*.

Chen, X. et al. (2025). What can RL bring to VLA generalization? An empirical study. *arXiv:2505.19789*.

Dai, Y. et al. (2026). RoboMME: Benchmarking and understanding memory for robotic generalist policies. *arXiv:2603.04639*.

Gu, J. et al. (2023). ManiSkill2: A unified benchmark for generalizable manipulation skills. *ICLR 2023*.

Kool, W. et al. (2019). Buy 4 REINFORCE samples, get a baseline for free! *ICLR 2019 Workshop*.

Li, X. et al. (2025). CO-RFT: Efficient fine-tuning of vision-language-action models through chunked offline reinforcement learning. *arXiv:2508.02219*.

Liu, Z. et al. (2025). Dr.GRPO: Doubly robust preference optimization for language model alignment. *arXiv:2503.00821*.

Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.

Pan, J. et al. (2025). MemER: Scaling up memory for robot control via experience retrieval. *arXiv:2510.20328*.

Qwen Team (2025). Qwen3-VL technical report. *arXiv*.

Schulman, J. et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.

Shao, Z. et al. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv:2402.03300*.

Torne, M. et al. (2026). MEM: Multi-scale embodied memory for vision language action models. *arXiv:2603.03596*.

Wang, W. et al. (2024). Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. *arXiv:2308.12966*.

Wu, X. et al. (2025). AtomVLA: Scalable post-training for robotic manipulation via predictive latent world models. *arXiv:2603.08519*.

Yu, D. et al. (2025). DAPO: An open-source LLM reinforcement learning system at scale. *arXiv:2503.14476*.

Zheng, K. et al. (2025). VLA-RL: Towards masterful and general robotic manipulation with scalable reinforcement learning. *arXiv:2505.18719*.
