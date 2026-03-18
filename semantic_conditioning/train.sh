#!/bin/bash
# Training script for semantic feature alignment

echo "Starting Semantic Feature Alignment Training"
echo "=============================================="
echo ""

# Activate conda environment (if needed)
# conda activate semantic

# Set CUDA device (optional)
export CUDA_VISIBLE_DEVICES=0

# Run training
python main.py

echo ""
echo "Training completed!"
echo ""
echo "Check results in:"
echo "  - checkpoints/best_model.pth"
echo "  - enhanced_features/"
echo "  - checkpoints/visualizations/"

