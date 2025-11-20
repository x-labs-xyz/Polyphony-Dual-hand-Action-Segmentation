#!/usr/bin/env python3
"""
Calculate FLOPs (Floating Point Operations) for Semantic Feature Alignment Model.

This script calculates the computational complexity of the semantic conditioning model
for reporting in research papers.

Usage:
    python calculate_flops.py
    python calculate_flops.py --visual_dim 768 --semantic_dim 384 --seq_len 100
"""

import argparse
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fvcore.nn import FlopCountMode, flop_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("Warning: fvcore not installed. Install with: pip install fvcore")
    print("Trying alternative: ptflops...")
    try:
        from ptflops import get_model_complexity_info
        HAS_PTFLOPS = True
    except ImportError:
        HAS_PTFLOPS = False
        print("Warning: ptflops not installed. Install with: pip install ptflops")
        print("Using manual calculation method (no dependencies needed).")

# Import model components
from main import SemanticFeatureAlignmentModel, TemporalConvNet


def calculate_flops_fvcore(model, input_shape):
    """Calculate FLOPs using fvcore."""
    model.eval()
    
    # Input shape: (batch_size, seq_len, visual_dim)
    batch_size, seq_len, visual_dim = input_shape
    dummy_input = torch.randn(batch_size, seq_len, visual_dim).to(next(model.parameters()).device)
    
    # Calculate FLOPs
    flop_dict, _ = flop_count(model, (dummy_input,), mode=FlopCountMode.TRAIN)
    
    total_flops = sum(flop_dict.values())
    
    return total_flops, flop_dict


def calculate_flops_manual(model, input_shape):
    """
    Manual FLOPs calculation for Semantic Feature Alignment Model.
    This provides a theoretical estimate based on the architecture.
    """
    batch_size, seq_len, visual_dim = input_shape
    
    # Get model configuration
    tcn_layers = model.tcn.layers
    num_tcn_layers = len(tcn_layers)
    
    # Get TCN output dimension (last layer's output channels)
    if num_tcn_layers > 0:
        tcn_output_dim = tcn_layers[-1].conv1.out_channels
    else:
        tcn_output_dim = visual_dim
    
    # Get semantic and alignment dimensions from projectors
    # Visual projector: Linear layers are at indices 0 and 4
    visual_proj_layers = [l for l in model.visual_projector if isinstance(l, nn.Linear)]
    semantic_proj_layers = [l for l in model.semantic_projector if isinstance(l, nn.Linear)]
    
    # TCN FLOPs calculation
    tcn_flops = 0
    prev_dim = visual_dim
    
    for i, layer in enumerate(tcn_layers):
        # Each TemporalBlock has 2 conv1d layers
        out_channels = layer.conv1.out_channels
        kernel_size = layer.conv1.kernel_size[0]
        dilation = layer.conv1.dilation[0]
        
        # Conv1d FLOPs: kernel_size * in_channels * out_channels * output_length
        # For dilated conv: effective_kernel = (kernel_size - 1) * dilation + 1
        effective_kernel = (kernel_size - 1) * dilation + 1
        output_length = seq_len  # After padding and chomp
        
        # First conv in TemporalBlock
        conv1_flops = effective_kernel * prev_dim * out_channels * output_length * batch_size
        # Second conv in TemporalBlock
        conv2_flops = effective_kernel * out_channels * out_channels * output_length * batch_size
        
        # BatchNorm and ReLU are negligible compared to conv
        # Residual connection (if downsample exists)
        if layer.downsample is not None:
            downsample_flops = prev_dim * out_channels * output_length * batch_size
        else:
            downsample_flops = 0
        
        layer_flops = conv1_flops + conv2_flops + downsample_flops
        tcn_flops += layer_flops
        
        prev_dim = out_channels
    
    # Visual projector FLOPs (MLP)
    visual_proj_flops = 0
    for layer in visual_proj_layers:
        in_dim = layer.in_features
        out_dim = layer.out_features
        # Linear layer FLOPs: batch_size * seq_len * in_dim * out_dim
        visual_proj_flops += batch_size * seq_len * in_dim * out_dim
    
    # Semantic projector FLOPs (MLP)
    semantic_proj_flops = 0
    for layer in semantic_proj_layers:
        in_dim = layer.in_features
        out_dim = layer.out_features
        semantic_proj_flops += batch_size * seq_len * in_dim * out_dim
    
    # Total FLOPs
    total_flops = tcn_flops + visual_proj_flops
    
    return total_flops, {
        'tcn': tcn_flops,
        'visual_projector': visual_proj_flops,
        'semantic_projector': semantic_proj_flops,
        'total': total_flops
    }


def get_model_info(model):
    """Get model architecture information."""
    info = {}
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info['total_params'] = total_params
    info['trainable_params'] = trainable_params
    
    # Get TCN configuration
    if hasattr(model, 'tcn') and hasattr(model.tcn, 'layers'):
        info['num_tcn_layers'] = len(model.tcn.layers)
        if len(model.tcn.layers) > 0:
            info['tcn_output_dim'] = model.tcn.layers[-1].conv1.out_channels
    
    # Get projector dimensions
    if hasattr(model, 'visual_projector'):
        for i, layer in enumerate(model.visual_projector):
            if isinstance(layer, nn.Linear):
                info[f'visual_proj_layer_{i}_in'] = layer.in_features
                info[f'visual_proj_layer_{i}_out'] = layer.out_features
    
    if hasattr(model, 'semantic_projector'):
        for i, layer in enumerate(model.semantic_projector):
            if isinstance(layer, nn.Linear):
                info[f'semantic_proj_layer_{i}_in'] = layer.in_features
                info[f'semantic_proj_layer_{i}_out'] = layer.out_features
    
    return info


def main():
    parser = argparse.ArgumentParser(description='Calculate FLOPs for Semantic Feature Alignment Model')
    parser.add_argument('--visual_dim', type=int, default=768,
                        help='Visual feature dimension')
    parser.add_argument('--semantic_dim', type=int, default=384,
                        help='Semantic feature dimension')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Sequence length (number of frames)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for FLOPs calculation')
    parser.add_argument('--tcn_hidden_dims', type=int, nargs='+', default=[512, 128, 64],
                        help='TCN hidden dimensions')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='TCN kernel size')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'fvcore', 'manual'],
                        help='Method to calculate FLOPs')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['inference', 'training', 'both'],
                        help='Calculate inference FLOPs, training FLOPs, or both')
    
    args = parser.parse_args()
    
    # Create model
    print(f"Creating Semantic Feature Alignment Model...")
    model = SemanticFeatureAlignmentModel(
        visual_dim=args.visual_dim,
        semantic_dim=args.semantic_dim,
        tcn_hidden_dims=args.tcn_hidden_dims,
        kernel_size=args.kernel_size,
        dropout=args.dropout
    )
    model = model.to(args.device)
    model.eval()
    
    # Get model info
    model_info = get_model_info(model)
    print("\n" + "="*60)
    print("Model Architecture Information")
    print("="*60)
    print(f"Visual Dimension: {args.visual_dim}")
    print(f"Semantic Dimension: {args.semantic_dim}")
    print(f"TCN Hidden Dimensions: {args.tcn_hidden_dims}")
    print(f"Number of TCN Layers: {model_info.get('num_tcn_layers', 'N/A')}")
    print(f"Total Parameters: {model_info['total_params']:,}")
    print(f"Trainable Parameters: {model_info['trainable_params']:,}")
    
    # Input shape: (batch_size, seq_len, visual_dim)
    input_shape = (args.batch_size, args.seq_len, args.visual_dim)
    
    print("\n" + "="*60)
    print("FLOPs Calculation")
    print("="*60)
    print(f"Input Shape: {input_shape}")
    print(f"Method: {args.method}")
    
    # Calculate FLOPs
    inference_flops = None
    training_flops = None
    inference_flops_dict = None
    training_flops_dict = None
    
    # Calculate inference FLOPs (forward pass only)
    if args.mode in ['inference', 'both']:
        print("\nCalculating Inference FLOPs (forward pass only)...")
        if args.method == 'auto':
            if HAS_FVCORE:
                print("Using fvcore...")
                inference_flops, inference_flops_dict = calculate_flops_fvcore(model, input_shape)
            else:
                print("Using manual calculation...")
                inference_flops, inference_flops_dict = calculate_flops_manual(model, input_shape)
        elif args.method == 'fvcore':
            if not HAS_FVCORE:
                print("Error: fvcore not installed. Install with: pip install fvcore")
                return
            inference_flops, inference_flops_dict = calculate_flops_fvcore(model, input_shape)
        elif args.method == 'manual':
            inference_flops, inference_flops_dict = calculate_flops_manual(model, input_shape)
    
    # Calculate training FLOPs (forward + backward pass)
    if args.mode in ['training', 'both']:
        print("\nCalculating Training FLOPs (forward + backward pass)...")
        if args.method == 'fvcore' and HAS_FVCORE:
            # fvcore can calculate training FLOPs directly
            model.train()
            training_flops, training_flops_dict = calculate_flops_fvcore(model, input_shape)
        else:
            # Estimate: training FLOPs ≈ 3 × inference FLOPs
            if inference_flops is not None:
                training_flops = inference_flops * 3
                training_flops_dict = {k: v * 3 for k, v in (inference_flops_dict or {}).items()}
                print("Note: Using estimated training FLOPs (3× inference)")
            else:
                print("Error: Cannot estimate training FLOPs without inference FLOPs")
    
    if inference_flops is None and training_flops is None:
        print("Error: Failed to calculate FLOPs")
        return
    
    # Print results
    print("\n" + "="*60)
    print("FLOPs Results")
    print("="*60)
    
    if inference_flops is not None:
        print("\n📊 INFERENCE FLOPs (Forward Pass Only)")
        print("-" * 60)
        print(f"Inference FLOPs: {inference_flops:,.0f}")
        print(f"Inference FLOPs (G): {inference_flops / 1e9:.2f} GFLOPs")
        print(f"Inference FLOPs (M): {inference_flops / 1e6:.2f} MFLOPs")
        
        if inference_flops_dict and len(inference_flops_dict) > 1:
            print("\nBreakdown:")
            for key, value in inference_flops_dict.items():
                if key != 'total' and isinstance(value, (int, float)):
                    print(f"  {key}: {value:,.0f} ({value / inference_flops * 100:.1f}%)")
        
        inference_flops_per_frame = inference_flops / args.seq_len
        print(f"\nInference FLOPs per Frame: {inference_flops_per_frame / 1e6:.2f} MFLOPs")
    
    if training_flops is not None:
        print("\n📊 TRAINING FLOPs (Forward + Backward Pass)")
        print("-" * 60)
        print(f"Training FLOPs: {training_flops:,.0f}")
        print(f"Training FLOPs (G): {training_flops / 1e9:.2f} GFLOPs")
        print(f"Training FLOPs (M): {training_flops / 1e6:.2f} MFLOPs")
        
        if inference_flops is not None:
            multiplier = training_flops / inference_flops
            print(f"Training/Inference Ratio: {multiplier:.2f}x")
        
        if training_flops_dict and len(training_flops_dict) > 1:
            print("\nBreakdown:")
            for key, value in training_flops_dict.items():
                if key != 'total' and isinstance(value, (int, float)):
                    print(f"  {key}: {value:,.0f} ({value / training_flops * 100:.1f}%)")
        
        training_flops_per_frame = training_flops / args.seq_len
        print(f"\nTraining FLOPs per Frame: {training_flops_per_frame / 1e6:.2f} MFLOPs")
    
    # Component breakdown
    if inference_flops_dict:
        tcn_flops = inference_flops_dict.get('tcn', 0)
        visual_proj_flops = inference_flops_dict.get('visual_projector', 0)
        semantic_proj_flops = inference_flops_dict.get('semantic_projector', 0)
        
        if tcn_flops > 0:
            print(f"\nTCN FLOPs (Inference): {tcn_flops / 1e9:.2f} GFLOPs ({tcn_flops / inference_flops * 100:.1f}%)")
        if visual_proj_flops > 0:
            print(f"Visual Projector FLOPs (Inference): {visual_proj_flops / 1e9:.2f} GFLOPs ({visual_proj_flops / inference_flops * 100:.1f}%)")
        if semantic_proj_flops > 0:
            print(f"Semantic Projector FLOPs (Inference): {semantic_proj_flops / 1e9:.2f} GFLOPs ({semantic_proj_flops / inference_flops * 100:.1f}%)")
    
    print("\n" + "="*60)
    print("Summary for Paper")
    print("="*60)
    print(f"Model: Semantic Feature Alignment Model")
    print(f"Parameters: {model_info['total_params']:,} ({model_info['total_params'] / 1e6:.2f}M)")
    if inference_flops is not None:
        print(f"Inference FLOPs: {inference_flops / 1e9:.2f} GFLOPs (forward pass only)")
    if training_flops is not None:
        print(f"Training FLOPs: {training_flops / 1e9:.2f} GFLOPs (forward + backward per step)")
    print(f"Input: {args.seq_len} frames × {args.visual_dim}D features")
    print(f"TCN Architecture: {args.tcn_hidden_dims}")
    print("="*60)


if __name__ == '__main__':
    main()

