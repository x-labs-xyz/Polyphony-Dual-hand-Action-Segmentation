#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='output/havid_alternating_hands'
LH_DATA_PATH='../data/havid/left_hand'
RH_DATA_PATH='../data/havid/right_hand'
MODEL_PATH='models/vit_b_k710_dl_from_giant.pth'

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    main.py \
    --model vit_base_patch16_224_alternating \
    --lh_data_path ${LH_DATA_PATH} \
    --rh_data_path ${RH_DATA_PATH} \
    --lh_num_classes 75 \
    --rh_num_classes 75 \
    --data_set HAVID \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
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
    --epochs 50 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --alternation_steps 50 \ 
    --dist_eval
    # --one_stream # Enable for single-stream training (only specify lh_data_path)

