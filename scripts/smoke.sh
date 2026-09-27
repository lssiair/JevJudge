#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
for arm in reference jev; do
 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=7 -m src.train_grpo --config "configs/grpo_$arm.yaml" --output "outputs/smoke-$arm" --steps "${SMOKE_STEPS:-2}" --prompts 28
done
