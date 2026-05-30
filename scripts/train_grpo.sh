#!/usr/bin/env bash
# Run GRPO fine-tuning on top of an SFT'd Qwen-VL LoRA adapter on Modal.
#
# Prereqs:
#   - pi0.5 baseline checkpoint on the robomme-vla-data volume.
#         modal run modal_server.py --download-only
#   - SFT adapter at /mnt/robomme/ckpts/qwen_sft/...
#         ./scripts/train_sft.sh
set -euo pipefail
cd "$(dirname "$0")/.."

modal run modal_train.py --stage grpo
