#!/usr/bin/env bash

# Run evaluation.py for multiple models and datasets

set -e

models=(
  "amphora/q2_5-1.5B-am-v02"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-Math-1.5B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "Qwen/Qwen3-1.7B"
  # add more models here
)

datasets=(
  "AIME2024"
  "AIME2025"
  "GPQA"
  "MMLU-Pro"
)

prompt='Solve the question and return you final answer in the following format: \boxed{N}.'

for m in "${models[@]}"; do
  for d in "${datasets[@]}"; do
    echo "Running model=$m dataset=$d"
    python evaluation.py \
      --model "$m" \
      --dataset "$d" \
      --system_prompt "$prompt"
  done
done


