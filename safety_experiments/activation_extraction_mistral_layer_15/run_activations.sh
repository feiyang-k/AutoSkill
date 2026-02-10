#!/usr/bin/env bash
set -euo pipefail

# ================================
# GPU
# ================================
export CUDA_VISIBLE_DEVICES=0

# ================================
# Model / extraction config
# ================================
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
LAYER_IDX=15
SEED=0

SCRIPT="extract_activations.py"

# ================================
# Data paths
# ================================
INPUT_JSON="path to json"
COLUMN="mutated_prompt"

OUTPUT_DIR="your output path"

# ================================
# Run
# ================================
echo "======================================"
echo "[RUN] Mistral activation extraction"
echo "  model : ${MODEL_NAME}"
echo "  layer : ${LAYER_IDX}"
echo "  input : ${INPUT_JSON}"
echo "  column: ${COLUMN}"
echo "  output: ${OUTPUT_DIR}"
echo "======================================"

python "$SCRIPT" \
  --input_json "$INPUT_JSON" \
  --column "$COLUMN" \
  --model_name "$MODEL_NAME" \
  --layer_idx "$LAYER_IDX" \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED"

echo "[DONE] Activation extraction complete."
