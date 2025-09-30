#!/bin/bash

# Dual-hand VideoMAE fine-tuning with semantic feature alignment
# This script trains a dual-head model with semantic alignment for both left and right hands

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Basic training parameters
BATCH_SIZE=8
EPOCHS=50
LEARNING_RATE=1e-3
WARMUP_EPOCHS=5
WEIGHT_DECAY=0.05

# Model parameters
MODEL="vit_base_patch16_224_dual_semantic"
INPUT_SIZE=224
NUM_FRAMES=16
SAMPLING_RATE=4

# Semantic alignment parameters
SEMANTIC_MODEL="sentence-transformers/all-mpnet-base-v2"
SEMANTIC_ALIGNMENT_WEIGHT=0.1
SEMANTIC_LOSS_TYPE="adaptive"
TCN_HIDDEN_DIMS="512 256"

# Data paths (adjust these paths according to your setup)
LH_DATA_DIR="/home/hao/Polyphony/data/havid_mmaction/lh_v0"
RH_DATA_DIR="/home/hao/Polyphony/data/havid_mmaction/rh_v0"
LH_TRAIN_ANN="/home/hao/Polyphony/data/havid_mmaction/lh_v0/train_list_video.txt"
RH_TRAIN_ANN="/home/hao/Polyphony/data/havid_mmaction/rh_v0/train_list_video.txt"
LH_VAL_ANN="/home/hao/Polyphony/data/havid_mmaction/lh_v0/val_list_video.txt"
RH_VAL_ANN="/home/hao/Polyphony/data/havid_mmaction/rh_v0/val_list_video.txt"

# Semantic features paths
SEMANTIC_EMBEDDINGS_PATH="/home/hao/Polyphony/data/havid/semantic_embeddings/sentence-transformers_all-mpnet-base-v2.pt"
ACTION_MAPPING_PATH="/home/hao/Polyphony/data/havid/havid_description.txt"

# Output paths
OUTPUT_DIR="./output/dual_hand_semantic"
LOG_DIR="./logs/dual_hand_semantic"

# Pretrained model (optional)
PRETRAINED_MODEL=""

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

export CUDA_VISIBLE_DEVICES=1 

# Launch training
python run_dual_hand_finetuning_semantic.py \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --warmup_epochs $WARMUP_EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --input_size $INPUT_SIZE \
    --num_frames $NUM_FRAMES \
    --sampling_rate $SAMPLING_RATE \
    --lh_data_dir $LH_DATA_DIR \
    --rh_data_dir $RH_DATA_DIR \
    --lh_train_ann $LH_TRAIN_ANN \
    --rh_train_ann $RH_TRAIN_ANN \
    --lh_val_ann $LH_VAL_ANN \
    --rh_val_ann $RH_VAL_ANN \
    --lh_num_classes 75 \
    --rh_num_classes 75 \
    --semantic_model_name $SEMANTIC_MODEL \
    --semantic_embeddings_path $SEMANTIC_EMBEDDINGS_PATH \
    --action_mapping_path $ACTION_MAPPING_PATH \
    --semantic_alignment_weight $SEMANTIC_ALIGNMENT_WEIGHT \
    --semantic_loss_type $SEMANTIC_LOSS_TYPE \
    --tcn_hidden_dims $TCN_HIDDEN_DIMS \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --drop_path 0.1 \
    --layer_decay 0.75 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --num_sample 2 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --save_ckpt \
    --auto_resume

echo "Training completed! Check $OUTPUT_DIR for results and $LOG_DIR for logs."
