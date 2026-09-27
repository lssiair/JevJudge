#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TMPDIR="${JEV_TMPDIR:-$PWD/.tmp}"
export UV_CACHE_DIR="$TMPDIR/uv-cache"
export HF_DATASETS_CACHE="$TMPDIR/datasets-cache"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export JEV_MAX_REQUESTS="${JEV_MAX_REQUESTS:-10000}"
export JEV_MAX_CONCURRENCY="${JEV_MAX_CONCURRENCY:-8}"
export JEV_REQUEST_TIMEOUT="${JEV_REQUEST_TIMEOUT:-60}"
export JEV_MAX_RETRIES="${JEV_MAX_RETRIES:-5}"
export PATH="$PWD/.venv/bin:$PATH"
mkdir -p "$TMPDIR" outputs results cache
