#!/usr/bin/env bash
set -euo pipefail
set -x  # print the commands

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='output/havid_alternating_breakfast'
LH_DATA_PATH='/home/hao/Polyphony/data/breakfast/breakfast_action_recognition'
LH_DATA_ROOT='/home/hao/Polyphony/data/breakfast/breakfast_action_recognition'
# RH_DATA_PATH='/home/hao/Polyphony/data/havid_mmaction/rh_v0'
# RH_DATA_ROOT='/home/hao/Polyphony/data/havid_mmaction/rh_v0'
MODEL_PATH='models/vit_b_k710_dl_from_giant.pth'

# Activate conda environment if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate videomaev2 || true

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}
export CUDA_VISIBLE_DEVICES=1

# Choose torchrun launcher or fallback
if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER="torchrun"
else
  LAUNCHER="python -m torch.distributed.run"
fi

${LAUNCHER} --nproc_per_node=1 \
  --master_port=$MASTER_PORT \
  run_alternating_hand_finetuning_one_stream.py \
  --model vit_base_patch16_224_alternating \
  --lh_data_path ${LH_DATA_PATH} \
  --lh_data_root ${LH_DATA_ROOT} \
  --lh_num_classes 48 \
  --data_set Breakfast \
  --finetune ${MODEL_PATH} \
  --log_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 6 \
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
  --dist_eval \
  --left_only

# Note: alternation_steps controls how often to switch between hands
# Try different values: 10 (very frequent), 50 (moderate), 200 (less frequent)