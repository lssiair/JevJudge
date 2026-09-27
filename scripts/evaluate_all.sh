#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
for arm in base reference jev; do
 if [[ "$arm" == base ]]; then model="${MODEL_PATH:-cache/qwen3.5-4b-text}"; else model="outputs/pilot-$arm/final"; fi
 CUDA_VISIBLE_DEVICES=0 python -m src.generate --model "$model" --output "outputs/eval-$arm.jsonl" "$@"
 python -m src.evaluate --input "outputs/eval-$arm.jsonl" --name "$arm"
done
python -m src.analyze
