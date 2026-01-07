#!/bin/bash

MODEL_PATHS='["./models/Wan2.2-TI2V-5B", "./models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth", "./models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"]'

accelerate launch \
  --num_processes 1 \
  --main_process_ip "28.49.26.186" \
  --main_process_port 12345 \
  examples/wanvideo/model_training/train.py \
   --dataset_base_path "./data/example" \
   --dataset_metadata_path "./data/example/metadata.csv" \
   --max_size 832 \
   --num_frames 81 \
   --dataset_repeat 100 \
   --model_paths "$MODEL_PATHS" \
   --learning_rate 2e-5 \
   --num_epochs 20000 \
   --remove_prefix_in_ckpt "pipe.dit." \
   --output_path "./checkpoint" \
   --trainable_models "dit" \
   --extra_inputs "input_image" \
   --data_file_keys "image,video" \
   --steps_per_epoch 20480\
   --disable_ptc