#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='output/IKEA-35-test'
DATA_ROOT='IKEA_mmaction/IKEA_mmaction' 
DATA_PATH='IKEA_mmaction/IKEA_mmaction'
MODEL_PATH='output/IKEA/checkpoint-best.pth'

torchrun --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 2 \
    --num_workers 8 \
    --opt adamw \
    --lr 1e-3 \
    --drop_path 0.3 \
    --clip_grad 5.0 \
    --layer_decay 0.9 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.1 \
    --warmup_epochs 5 \
    --epochs 35 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --dist_eval \
    ${@:1}

