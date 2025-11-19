#!/usr/bin/env bash
set -euo pipefail

SRC="eval_results"
OUT="score_results"

# 1) models via script args
if [ "$#" -gt 0 ]; then
  MODELS=("$@")
# 2) models via env var, space separated
elif [ -n "${MODELS:-}" ]; then
  # shellcheck disable=SC2206
  MODELS=(${MODELS})
# 3) fallback default list
else
  MODELS=(
   "Heoni/kanana_1.5_8b_it_tab1-ru-ab_20251118_5ep"
    # "Qwen/Qwen2.5-7B"
  )
fi

# dataset groups by evaluation style
MCQA_DATASETS=(
  "KMMLU_Redux"
  "KMMLU-HARD"
  "KMMLU-Pro"
  "CLIcK"
  "HRB1_0"
  "GPQA"
  "KorMedLawQA"
)

MCQA_RAW_DATASETS=(
  "ClinicalQA"
  "KoBALT-700"
)

OPEN_MATH_DATASETS=(
  "MCLM"
  "AIME2024"
  "AIME2025"
  "KSM"
)

mkdir -p "$OUT"

for model in "${MODELS[@]}"; do
  echo "Processing model: $model"

  # mcqa
  for ds in "${MCQA_DATASETS[@]}"; do
    echo "  dataset: $ds  style: mcqa"
    if python3 score.py -s "$SRC" -t "$OUT" -m "$model" -d "$ds" -y mcqa; then
      :
    else
      echo "    skipped or missing file for $ds and $model"
    fi
  done

  # mcqa raw
  for ds in "${MCQA_RAW_DATASETS[@]}"; do
    echo "  dataset: $ds  style: mcqa_raw"
    if python3 score.py -s "$SRC" -t "$OUT" -m "$model" -d "$ds" -y mcqa_raw; then
      :
    else
      echo "    skipped or missing file for $ds and $model"
    fi
  done

  # open ended math
  for ds in "${OPEN_MATH_DATASETS[@]}"; do
    echo "  dataset: $ds  style: open_math"
    if python3 score.py -s "$SRC" -t "$OUT" -m "$model" -d "$ds" -y open_math; then
      :
    else
      echo "    skipped or missing file for $ds and $model"
    fi
  done
done

echo "All done. Outputs are in $OUT"
