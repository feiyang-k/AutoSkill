#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Paths
# ----------------------------
INPUT_BASE="/helios-storage/helios3-data/mahavirdabas18/auto_skill_data_v2/inference_outputs"
OUTPUT_BASE="/helios-storage/helios3-data/mahavirdabas18/auto_skill_data_v2/strong_reject_output"

EVAL_SCRIPT="async_strong_reject_eval.py"   # evaluator script

# ----------------------------
# Runtime params
# ----------------------------
BATCH_SIZE=30
MAX_CONCURRENT=10
TIME_BETWEEN_BATCHES=2.0

# ----------------------------
# Files to process (ORDER MATTERS)
# ----------------------------
FILES=(
  "mistral_random_50_lr_1e-4_epoch1.json"
  "mistral_pca_50_lr_1e-4_epoch1.json"

  "mistral_random_70_lr_1e-4_epoch1.json"
  "mistral_pca_70_lr_1e-4_epoch1.json"

  "mistral_random_80_lr_1e-4_epoch1.json"
  "mistral_pca_80_lr_1e-4_epoch1.json"

  "mistral_random_90_lr_1e-4_epoch1.json"
  "mistral_pca_90_lr_1e-4_epoch1.json"

  "mistral_random_100_lr_1e-4_epoch1.json"
  "mistral_pca_100_lr_1e-4_epoch1.json"

  "mistral_random_65_lr_1e-4_epoch1.json"
  "mistral_pca_65_lr_1e-4_epoch1.json"

  "mistral_random_75_lr_1e-4_epoch1.json"
  "mistral_pca_75_lr_1e-4_epoch1.json"

  "mistral_random_85_lr_1e-4_epoch1.json"
  "mistral_pca_85_lr_1e-4_epoch1.json"

  "mistral_random_95_lr_1e-4_epoch1.json"
  "mistral_pca_95_lr_1e-4_epoch1.json"
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