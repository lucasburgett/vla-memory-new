#!/usr/bin/env bash
# RLOO fine-tune (REINFORCE leave-one-out ablation vs GRPO).
# Requires: same SFT adapter + GroundSG checkpoint as train_ppo.sh
set -euo pipefail
modal run modal_train.py --stage rloo "$@"
