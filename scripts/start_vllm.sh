#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
exec env CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve --model "${MODEL_PATH:-cache/qwen3.5-4b-text}" --tensor-parallel-size 1 --dtype bfloat16 --max-model-len "${VLLM_MAX_MODEL_LEN:-2048}" --gpu-memory-utilization 0.85 --host 127.0.0.1 --port 8000 --enforce-eager
