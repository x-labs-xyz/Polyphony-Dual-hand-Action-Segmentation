#!/usr/bin/env bash
set -x

OUTPUT_DIR='output/IKEA'
DATA_ROOT='IKEA_mmaction/IKEA_mmaction'  
DATA_PATH='IKEA_mmaction/IKEA_mmaction'
MODEL_PATH='models/vit_b_k710_dl_from_giant.pth'

torchrun --nproc_per_node=4 --rdzv_backend=static --rdzv_endpoint=localhost:29501 run_class_finetuning.py \
  --model vit_base_patch16_224 \
  --data_path ${DATA_PATH} \
  --data_root ${DATA_ROOT} \
  --finetune ${MODEL_PATH} \
  --log_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 2 \
  --input_size 224 \
  --short_side_size 224 \
  --save_ckpt_freq 999 \
  --num_frames 16 \
  --sampling_rate 4 \
  --num_sample 1 \
  --num_workers 4 \
  --opt adamw \
  --lr 1e-3 \
  --drop_path 0.1 \
  --clip_grad 5.0 \
  --layer_decay 0.9 \
  --opt_betas 0.9 0.999 \
  --weight_decay 0.05 \
  --warmup_epochs 2 \
  --epochs 30 \
  --test_num_segment 5 \
  --test_num_crop 3 \
  --dist_eval \


