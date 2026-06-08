#!/usr/bin/env bash
# PPO fine-tune (bandit variant with learned value baseline).
# Requires: SFT adapter at /mnt/robomme/ckpts/qwen_sft/buttonunmask_grounded
#           GroundSG π0.5 at /mnt/robomme/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999
set -euo pipefail
modal run modal_train.py --stage ppo "$@"
