"""GRPO subgoal trainer.

Heavy trainer / rollout imports require torch + openpi_client and are only
available inside the Modal container. To keep the package importable on CPU
dev machines, those classes are NOT re-exported here. Import them directly::

    from vla_memory.grpo.trainer import GRPOTrainer, GRPOConfig
    from vla_memory.grpo.rollout import RolloutWorker, RolloutResult
"""

from .reward import RewardConfig, compute_reward
from .state_dataset import StateDataset, StateSample

__all__ = ["RewardConfig", "compute_reward", "StateDataset", "StateSample"]
