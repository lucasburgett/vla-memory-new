#!/usr/bin/env bash
# Run the SFT warmstart on Modal.
#
# Prereqs:
#   - robomme-vla-data volume already exists (see modal_server.py).
#   - The SFT JSONL dataset has been built. Run once:
#         modal run modal_train.py --stage build_dataset
#
# Then:
#         ./scripts/train_sft.sh
set -euo pipefail
cd "$(dirname "$0")/.."

modal run modal_train.py --stage sft
