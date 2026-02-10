#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Paths
# ----------------------------
INPUT_BASE=""
OUTPUT_BASE=""

EVAL_SCRIPT="async_strong_reject_eval.py"   # evaluator script

# ----------------------------
# Runtime params
# ----------------------------
BATCH_SIZE=30
MAX_CONCURRENT=10
TIME_BETWEEN_BATCHES=2.0

# ----------------------------
# Files to process
# ----------------------------
FILES=(
  "mistral_random_60_lr_1e-4_epoch1.json"
  "mistral_pca_60_lr_1e-4_epoch1.json"
  "mistral_random_90_lr_1e-4_epoch1.json"
  "mistral_pca_90_lr_1e-4_epoch1.json"
)

# ----------------------------
# Ensure output dir exists
# ----------------------------
mkdir -p "${OUTPUT_BASE}"

# ----------------------------
# Run evaluation
# ----------------------------
for fname in "${FILES[@]}"; do
  INPUT_PATH="${INPUT_BASE}/${fname}"
  OUTPUT_PATH="${OUTPUT_BASE}/${fname}"

  echo "=============================================="
  echo "Evaluating: ${fname}"
  echo "Input : ${INPUT_PATH}"
  echo "Output: ${OUTPUT_PATH}"
  echo "=============================================="

  python "${EVAL_SCRIPT}" \
    "${INPUT_PATH}" \
    "${OUTPUT_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --max_concurrent "${MAX_CONCURRENT}" \
    --time_between_batches "${TIME_BETWEEN_BATCHES}"

  echo "✓ Finished ${fname}"
  echo
done

echo "🎉 All evaluations completed."
