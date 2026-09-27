#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=7 -m src.train_grpo --config configs/grpo_jev.yaml --output outputs/pilot-jev "$@"
