"""Hierarchical-memory VLA: Qwen3-VL subgoal predictor over a frozen pi0.5 expert.

Subpackages:
  - ``qwen_subgoal``: prompts (CPU-importable) and the PEFT-wrapped Qwen-VL model.
  - ``grpo``: reward + dataset (CPU-importable) and the GRPO trainer + rollout.
  - ``data``: thin wrappers over the submodule's dataset builders.

Heavy classes (``QwenSubgoalPolicy``, ``GRPOTrainer``, ``RolloutWorker``) are
not re-exported here because they pull torch / transformers / openpi_client.
"""

__all__: list[str] = []
