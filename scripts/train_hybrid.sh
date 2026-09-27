#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
test -f outputs/pilot-reference/completed.json
test -f outputs/pilot-jev/completed.json
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=7 -m src.train_grpo --config configs/grpo_hybrid.yaml --output outputs/pilot-hybrid "$@"
