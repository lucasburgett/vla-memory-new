#!/usr/bin/env bash
# One-time setup for this repo.
# Run once after cloning before doing anything with Modal.
set -euo pipefail

echo "=== vla-memory-new setup ==="

# 1. Initialize the robomme_policy_learning submodule
echo ""
echo "1/4  Initializing robomme_policy_learning submodule …"
git submodule update --init robomme_policy_learning

# 2. Initialize its nested submodule (robomme_benchmark lives at
#    robomme_policy_learning/third_party/robomme_benchmark).
#    The upstream uses an SSH remote; override to HTTPS so this works
#    in environments without an SSH key.
echo ""
echo "2/4  Initializing third_party/robomme_benchmark (SSH → HTTPS override) …"
(
  cd robomme_policy_learning
  git config submodule.third_party/robomme_benchmark.url \
    https://github.com/RoboMME/robomme_benchmark.git
  git submodule update --init third_party/robomme_benchmark
)

# 3. Check that Modal is installed
echo ""
echo "3/4  Checking prerequisites …"
if ! command -v modal &>/dev/null; then
  echo "  ✗  'modal' not found. Install it:"
  echo "       pip install modal"
  echo "     or, if using uv in this project:"
  echo "       uv pip install modal"
  exit 1
fi
echo "  ✓  modal $(modal --version 2>/dev/null || echo '(version unknown)')"

if ! command -v git-lfs &>/dev/null; then
  echo "  ✗  git-lfs not found. Install it:"
  echo "       brew install git-lfs   (macOS)"
  echo "       apt install git-lfs    (Linux)"
  exit 1
fi
echo "  ✓  git-lfs $(git lfs version | head -1)"

# 4. Auth reminders
echo ""
echo "4/4  Auth setup …"
echo "  • Run 'modal setup' if you haven't authenticated with Modal yet."
echo "  • Copy .env.example → .env and fill in HF_TOKEN / WANDB_API_KEY."
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  Download the pi05_baseline checkpoint (~4 GB, one-time):"
echo "    modal run modal_server.py --download-only"
echo ""
echo "  Run a full RoboMME evaluation:"
echo "    modal run modal_server.py"
echo ""
echo "  Quick smoke-test (single task):"
echo "    modal run modal_server.py --only-tasks PickXtimes --overwrite"
