#!/usr/bin/env bash
# Verify the SFT JSONL build (stage 3) completed and the volume is consistent.
#
# Each RoboMME episode produces *multiple* subgoal training rows (one per
# subgoal transition, ~4 on average), so we don't check against a fixed
# episode count. Instead we verify internal consistency: simple and grounded
# JSONLs must have the same row count, and the image count must match.
#
# Exits 0 if both JSONLs exist, have matching row counts at or above a sane
# minimum, and the images dir has a matching number of files.
#
# Usage:
#   ./scripts/check_sft_dataset.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VOLUME="robomme-vla-data"
REMOTE_DIR="/data/preprocessed/qwenvl/vlm_subgoal"
MIN_ROWS=1600               # at least one subgoal per episode (100 × 16 tasks)
EXPECTED_TASKS=16           # RoboMME ships 16 task H5s — all must be represented
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

pass() { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; FAILED=1; }
info() { printf '  · %s\n' "$1"; }

FAILED=0

echo "[1/4] Listing $REMOTE_DIR on volume $VOLUME"
if ! LISTING=$(modal volume ls "$VOLUME" "$REMOTE_DIR" 2>&1); then
  fail "directory does not exist on the volume"
  echo "$LISTING"
  exit 1
fi
echo "$LISTING"

echo
echo "[2/4] Checking JSONL files"
declare -A ROW_COUNTS
for name in simple_subgoal_train.jsonl grounded_subgoal_train.jsonl; do
  remote="$REMOTE_DIR/$name"
  local="$TMP_DIR/$name"

  if ! modal volume get "$VOLUME" "$remote" "$local" >/dev/null 2>&1; then
    fail "$name missing on volume"
    continue
  fi

  rows=$(wc -l < "$local" | tr -d ' ')
  bytes=$(wc -c < "$local" | tr -d ' ')
  ROW_COUNTS["$name"]=$rows

  info "$name: $rows rows, $bytes bytes"

  if [ "$rows" -ge "$MIN_ROWS" ]; then
    pass "$name row count $rows ≥ minimum $MIN_ROWS"
  else
    fail "$name row count $rows below minimum $MIN_ROWS — build likely truncated"
  fi

  if head -n 1 "$local" | python3 -c "
import json, sys
row = json.loads(sys.stdin.read())
required = {'messages', 'images'}
missing = required - set(row.keys())
if missing:
    sys.exit(f'missing keys: {missing}')
" 2>/tmp/jsonl_err; then
    pass "$name first row parses with expected keys"
  else
    fail "$name first row malformed: $(cat /tmp/jsonl_err)"
  fi
done

simple_rows=${ROW_COUNTS[simple_subgoal_train.jsonl]:-0}
grounded_rows=${ROW_COUNTS[grounded_subgoal_train.jsonl]:-0}
if [ "$simple_rows" -gt 0 ] && [ "$simple_rows" -eq "$grounded_rows" ]; then
  pass "simple and grounded JSONLs have matching row counts ($simple_rows)"
else
  fail "row count mismatch: simple=$simple_rows grounded=$grounded_rows — builder did not finish symmetrically"
fi

# Task coverage: image filenames are `{TaskName}_ep{N}_step{M}.png`. Internal
# consistency between the three artifacts is not enough — a timeout mid-loop
# leaves a perfectly-consistent but partial dataset. Verify all 16 tasks made it.
if [ -s "$TMP_DIR/simple_subgoal_train.jsonl" ]; then
  task_count=$(python3 -c "
import json
tasks = set()
with open('$TMP_DIR/simple_subgoal_train.jsonl') as f:
    for line in f:
        img = json.loads(line)['images'][0].split('/')[-1]
        tasks.add(img.split('_ep')[0])
print(len(tasks))
")
  info "distinct tasks represented: $task_count"
  if [ "$task_count" -ge "$EXPECTED_TASKS" ]; then
    pass "all $EXPECTED_TASKS tasks present"
  else
    fail "only $task_count / $EXPECTED_TASKS tasks present — build timed out before processing the rest"
  fi
fi

echo
echo "[3/4] Checking images directory matches row count"
if IMG_LISTING=$(modal volume ls "$VOLUME" "$REMOTE_DIR/images" 2>&1); then
  img_count=$(printf '%s\n' "$IMG_LISTING" | grep -c -E '\.(png|jpg|jpeg)$' || true)
  info "found $img_count image files (top-level)"

  if [ "$img_count" -eq 0 ]; then
    # Builder might nest images per-task; recurse one level.
    info "no images at top level — recursing one subdir"
    first_sub=$(printf '%s\n' "$IMG_LISTING" | awk 'NR==1 {print $NF}')
    if [ -n "$first_sub" ]; then
      img_count=$(modal volume ls "$VOLUME" "$REMOTE_DIR/images/$first_sub" 2>/dev/null | grep -c -E '\.(png|jpg|jpeg)$' || true)
      info "subdir $first_sub: $img_count images"
    fi
  fi

  if [ "$img_count" -eq "$simple_rows" ] && [ "$simple_rows" -gt 0 ]; then
    pass "image count matches JSONL row count ($img_count)"
  elif [ "$img_count" -gt 0 ]; then
    fail "image count $img_count does not match JSONL row count $simple_rows — partial write"
  else
    fail "images directory empty"
  fi
else
  fail "images directory missing"
fi

echo
echo "[4/4] Summary"
if [ "$FAILED" -eq 0 ]; then
  pass "SFT dataset is consistent — safe to run ./scripts/train_sft.sh"
  exit 0
else
  fail "dataset is inconsistent or missing — re-run: modal run modal_train.py --stage build_dataset"
  exit 1
fi
