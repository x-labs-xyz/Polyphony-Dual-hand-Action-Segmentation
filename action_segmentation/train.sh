#!/bin/bash

# Training script for dual-hand action segmentation
# Usage: ./train.sh [config_file] [gpu_id]

# Default values
CONFIG_FILE="${1:-configs/example_config.json}"
GPU_ID="${2:-0}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Usage: ./train.sh [config_file] [gpu_id]"
    exit 1
fi

echo "=========================================="
echo "Dual-Hand Action Segmentation Training"
echo "=========================================="
echo "Configuration: $CONFIG_FILE"
echo "GPU ID: $GPU_ID"
echo "=========================================="
echo ""

# Run training
python -u main.py \
    --config "$CONFIG_FILE" \
    --device "$GPU_ID"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="

