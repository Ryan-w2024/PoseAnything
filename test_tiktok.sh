#!/bin/bash
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=180000000
export NCCL_ALGO=RING

SCRIPT_PATH="./examples/wanvideo/model_inference/test_ddp_without_ptc.py"

torchrun --nproc_per_node=1 --master_port=12345 \
    $SCRIPT_PATH \
    --metadata_path "./data/tiktok/metadata.csv" \
    --pretrained_path "./models/Wan2.2-TI2V-5B" \
    --model_path "./models/Pony/tiktok/diffusion_pytorch_model-tiktok.safetensors" \
    --output_path "./output"\
    --base_path "./data/tiktok"\
    --max_size 832\
    --disable_ptc