#!/usr/bin/env bash
set -euo pipefail

# One-shot helper for H100 machines.
# Usage:
#   bash run_h100.sh
# Optional env overrides:
#   BACKBONE_MODE=mixed_antiberty_esm|shared_plm|kmer
#   FUSION_MODE=attention|concat
#   PLM_MODEL=facebook/esm2_t33_650M_UR50D
#   ANTIGEN_MODEL=facebook/esm2_t33_650M_UR50D
#   OUTPUT_DIR=csv/model_artifacts_h100

BACKBONE_MODE="${BACKBONE_MODE:-mixed_antiberty_esm}"
FUSION_MODE="${FUSION_MODE:-attention}"
PLM_MODEL="${PLM_MODEL:-facebook/esm2_t33_650M_UR50D}"
ANTIGEN_MODEL="${ANTIGEN_MODEL:-facebook/esm2_t33_650M_UR50D}"
OUTPUT_DIR="${OUTPUT_DIR:-csv/model_artifacts_h100}"

python setup_h100_run.py --run-train \
  --output-dir "${OUTPUT_DIR}" \
  --extra-args \
  --regressor-backend torch \
  --device auto \
  --epochs 40 \
  --batch-size 64 \
  --backbone-mode "${BACKBONE_MODE}" \
  --fusion-mode "${FUSION_MODE}" \
  --plm-model "${PLM_MODEL}" \
  --antigen-model "${ANTIGEN_MODEL}"
