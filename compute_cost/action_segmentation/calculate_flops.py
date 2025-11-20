#!/usr/bin/env python3
"""
Calculate FLOPs (Floating Point Operations) for Dual-Hand Action Segmentation Model.

This script calculates the computational complexity of the action segmentation model
for reporting in research papers.

Usage:
    python calculate_flops.py --config configs/example_config.json
    python calculate_flops.py --seq_len 200 --num_classes 75 --input_dim 768
"""

import argparse
import torch
import torch.nn as nn
import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fvcore.nn import FlopCountMode, flop_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("Warning: fvcore not installed. Install with: pip install fvcore")
    print("Using manual calculation method (no dependencies needed).")

# Import model components
try:
    from model import DualHandASDiffusionModel, HandFeatureFusion
    HAS_MODEL = True
except ImportError as e:
    HAS_MODEL = False
    print(f"Warning: Could not import model: {e}")
    print("Will use manual calculation based on architecture specifications.")


def calculate_flops_fvcore(model, input_shape, mode='train'):
    """
    Calculate FLOPs using fvcore.
    
    Args:
        model: Model to evaluate
        input_shape: Input shape tuple (batch_size, input_dim, seq_len)
        mode: 'train' for training FLOPs (forward+backward), 'eval' for inference FLOPs (forward only)
    """
    if mode == 'train':
        model.train()  # Enable training mode for backward pass
        flop_mode = FlopCountMode.TRAIN
    else:
        model.eval()  # Evaluation mode (forward only)
        flop_mode = FlopCountMode.EVAL
    
    # Input shape: (batch_size, input_dim, seq_len) for Conv1d
    batch_size, input_dim, seq_len = input_shape
    
    # Create dummy inputs for dual-hand model
    backbone_feats = torch.randn(batch_size, input_dim, seq_len).to(next(model.parameters()).device)
    t = torch.randint(0, model.num_timesteps, (batch_size,)).to(backbone_feats.device)
    
    # Create dummy noisy predictions
    num_classes = model.num_classes
    event_diffused_left = torch.randn(batch_size, num_classes, seq_len).to(backbone_feats.device)
    event_diffused_right = torch.randn(batch_size, num_classes, seq_len).to(backbone_feats.device)
    
    # Calculate FLOPs for forward pass
    flop_dict, _ = flop_count(
        model, 
        (backbone_feats, t, event_diffused_left, event_diffused_right),
        mode=flop_mode
    )
    
    total_flops = sum(flop_dict.values())
    
    return total_flops, flop_dict


def calculate_flops_manual(model, encoder_params, decoder_params, diffusion_params, 
                           input_shape, num_classes):
    """
    Manual FLOPs calculation for Dual-Hand Action Segmentation Model.
    This provides a theoretical estimate based on the architecture.
    """
    # Input shape: (batch_size, input_dim, seq_len) for Conv1d
    if len(input_shape) == 3:
        batch_size, input_dim, seq_len = input_shape
    else:
        # Fallback: assume (batch_size, seq_len, input_dim) and transpose
        batch_size, seq_len, input_dim = input_shape
    
    # Get model configuration
    num_f_maps_enc = encoder_params.get('num_f_maps', 64)
    num_layers_enc = encoder_params.get('num_layers', 10)
    num_f_maps_dec = decoder_params.get('num_f_maps', 64)
    num_layers_dec = decoder_params.get('num_layers', 10)
    feature_layer_indices = encoder_params.get('feature_layer_indices', [-1, 7, 9])
    
    # Calculate decoder input dimension
    decoder_input_dim = len([i for i in feature_layer_indices if i not in [-1, -2]]) * num_f_maps_enc
    if -1 in feature_layer_indices:
        decoder_input_dim += input_dim
    if -2 in feature_layer_indices:
        decoder_input_dim += num_classes
    
    # ===== Encoder FLOPs =====
    # MS-TCN style encoder: multiple 1D conv layers
    encoder_flops = 0
    prev_dim = input_dim
    
    for i in range(num_layers_enc):
        # Each layer: Conv1d + ReLU + Dropout
        # Conv1d FLOPs: kernel_size * in_channels * out_channels * seq_len * batch_size
        kernel_size = 3  # Typical MS-TCN kernel size
        out_dim = num_f_maps_enc
        
        # Conv1d FLOPs
        conv_flops = kernel_size * prev_dim * out_dim * seq_len * batch_size
        encoder_flops += conv_flops
        
        prev_dim = out_dim
    
    # Encoder output projection to num_classes
    encoder_output_flops = prev_dim * num_classes * seq_len * batch_size
    
    # ===== Feature Fusion FLOPs =====
    # HandFeatureFusion: Conv1d operations
    # Each fusion: 2 Conv1d layers (feature_dim*2 -> feature_dim -> feature_dim)
    fusion_flops = 0
    
    # Left hand fusion
    # Conv1d 1: (decoder_input_dim * 2) -> decoder_input_dim, kernel=1
    fusion_flops += 1 * (decoder_input_dim * 2) * decoder_input_dim * seq_len * batch_size
    # Conv1d 2: decoder_input_dim -> decoder_input_dim, kernel=1
    fusion_flops += 1 * decoder_input_dim * decoder_input_dim * seq_len * batch_size
    
    # Right hand fusion (same)
    fusion_flops *= 2
    
    # ===== Decoder FLOPs =====
    # MS-TCN style decoder: multiple 1D conv layers
    decoder_flops = 0
    prev_dim = decoder_input_dim
    
    for i in range(num_layers_dec):
        kernel_size = 3
        out_dim = num_f_maps_dec
        
        # Conv1d FLOPs
        conv_flops = kernel_size * prev_dim * out_dim * seq_len * batch_size
        decoder_flops += conv_flops
        
        prev_dim = out_dim
    
    # Decoder output projection to num_classes
    decoder_output_flops = prev_dim * num_classes * seq_len * batch_size
    
    # ===== Diffusion Operations FLOPs =====
    # These are mostly element-wise operations (negligible compared to conv)
    # But we account for normalization, softmax, etc.
    diffusion_ops_flops = 0
    
    # Normalize/denormalize: element-wise (negligible)
    # Softmax: O(num_classes * seq_len * batch_size) per hand
    softmax_flops = num_classes * seq_len * batch_size * 2  # For both hands
    diffusion_ops_flops += softmax_flops
    
    # ===== Total FLOPs =====
    # Note: We have separate decoders for left and right hands
    total_flops = (
        encoder_flops + encoder_output_flops +
        fusion_flops +
        (decoder_flops + decoder_output_flops) * 2 +  # Two decoders
        diffusion_ops_flops
    )
    
    return total_flops, {
        'encoder': encoder_flops + encoder_output_flops,
        'feature_fusion': fusion_flops,
        'decoder_left': decoder_flops + decoder_output_flops,
        'decoder_right': decoder_flops + decoder_output_flops,
        'diffusion_ops': diffusion_ops_flops,
        'total': total_flops
    }


def calculate_parameter_counts_manual(encoder_params, decoder_params, diffusion_params,
                                      input_dim, num_classes):
    """
    Estimate parameter counts when the full model cannot be instantiated.
    Uses the same architectural assumptions as calculate_flops_manual.
    """
    num_f_maps_enc = encoder_params.get('num_f_maps', 64)
    num_layers_enc = encoder_params.get('num_layers', 10)
    num_f_maps_dec = decoder_params.get('num_f_maps', 64)
    num_layers_dec = decoder_params.get('num_layers', 10)
    feature_layer_indices = encoder_params.get('feature_layer_indices', [-1, 7, 9])

    # Decoder input dimension matches manual FLOPs logic
    decoder_input_dim = len([i for i in feature_layer_indices if i not in [-1, -2]]) * num_f_maps_enc
    if -1 in feature_layer_indices:
        decoder_input_dim += input_dim
    if -2 in feature_layer_indices:
        decoder_input_dim += num_classes

    # ===== Encoder parameters =====
    encoder_params_count = 0
    prev_dim = input_dim
    kernel_size = 3
    for _ in range(num_layers_enc):
        out_dim = num_f_maps_enc
        weight = out_dim * prev_dim * kernel_size
        bias = out_dim
        encoder_params_count += weight + bias
        prev_dim = out_dim
    encoder_output_params = prev_dim * num_classes + num_classes
    encoder_total = encoder_params_count + encoder_output_params

    # ===== Fusion parameters =====
    fusion_params = 0
    # Each hand: two 1x1 conv layers
    fusion_single = 0
    in_channels = decoder_input_dim * 2
    out_channels = decoder_input_dim
    fusion_single += out_channels * in_channels + out_channels
    in_channels = decoder_input_dim
    out_channels = decoder_input_dim
    fusion_single += out_channels * in_channels + out_channels
    fusion_params = fusion_single * 2  # Left + Right hands

    # ===== Decoder parameters =====
    decoder_params_count = 0
    prev_dim = decoder_input_dim
    for _ in range(num_layers_dec):
        out_dim = num_f_maps_dec
        weight = out_dim * prev_dim * kernel_size
        bias = out_dim
        decoder_params_count += weight + bias
        prev_dim = out_dim
    decoder_output_params = prev_dim * num_classes + num_classes
    decoder_total_per_hand = decoder_params_count + decoder_output_params
    decoder_total = decoder_total_per_hand * 2  # Two decoders

    total_params = encoder_total + fusion_params + decoder_total

    param_dict = {
        'encoder_params': encoder_total,
        'fusion_params': fusion_params,
        'decoder_params_per_hand': decoder_total_per_hand,
        'decoder_params_total': decoder_total,
        'total_params': total_params
    }
    return param_dict


def get_model_info(model):
    """Get model architecture information."""
    info = {}
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info['total_params'] = total_params
    info['trainable_params'] = trainable_params
    
    # Get model configuration
    if hasattr(model, 'num_classes'):
        info['num_classes'] = model.num_classes
    if hasattr(model, 'num_timesteps'):
        info['num_timesteps'] = model.num_timesteps
    
    return info


def load_config(config_path):
    """Load configuration from JSON file or Python config file."""
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        # Load from JSON file
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.suffix == '.py':
        # Load from Python config file
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Get config using get_config() function if available
        if hasattr(config_module, 'get_config'):
            config = config_module.get_config()
        else:
            # Fall back to extracting from module attributes
            config = {
                'encoder_params': getattr(config_module, 'ENCODER_PARAMS', {}),
                'decoder_params': getattr(config_module, 'DECODER_PARAMS', {}),
                'diffusion_params': getattr(config_module, 'DIFFUSION_PARAMS', {}),
            }
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}. Use .json or .py")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Calculate FLOPs for Dual-Hand Action Segmentation Model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (JSON or Python .py file)')
    parser.add_argument('--input_dim', type=int, default=1227,
                        help='Input feature dimension')
    parser.add_argument('--seq_len', type=int, default=3000,
                        help='Sequence length (number of frames)')
    parser.add_argument('--num_classes', type=int, default=75,
                        help='Number of action classes')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for FLOPs calculation')
    parser.add_argument('--num_f_maps_enc', type=int, default=64,
                        help='Number of feature maps in encoder')
    parser.add_argument('--num_layers_enc', type=int, default=10,
                        help='Number of encoder layers')
    parser.add_argument('--num_f_maps_dec', type=int, default=64,
                        help='Number of feature maps in decoder')
    parser.add_argument('--num_layers_dec', type=int, default=10,
                        help='Number of decoder layers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'fvcore', 'manual'],
                        help='Method to calculate FLOPs')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['inference', 'training', 'both'],
                        help='Calculate inference FLOPs, training FLOPs, or both')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        encoder_params = config.get('encoder_params', {})
        decoder_params = config.get('decoder_params', {})
        diffusion_params = config.get('diffusion_params', {})
        args.input_dim = encoder_params.get('input_dim', args.input_dim)
        args.num_classes = config.get('num_classes', args.num_classes)
        args.num_f_maps_enc = encoder_params.get('num_f_maps', args.num_f_maps_enc)
        args.num_layers_enc = encoder_params.get('num_layers', args.num_layers_enc)
        args.num_f_maps_dec = decoder_params.get('num_f_maps', args.num_f_maps_dec)
        args.num_layers_dec = decoder_params.get('num_layers', args.num_layers_dec)
    else:
        encoder_params = {
            'input_dim': args.input_dim,
            'num_f_maps': args.num_f_maps_enc,
            'num_layers': args.num_layers_enc,
            'feature_layer_indices': [-1, 7, 9],
            'use_instance_norm': True
        }
        decoder_params = {
            'num_f_maps': args.num_f_maps_dec,
            'num_layers': args.num_layers_dec
        }
        diffusion_params = {
            'timesteps': 1000,
            'sampling_timesteps': 25,
            'ddim_sampling_eta': 1.0,
            'snr_scale': 2.0,
            'detach_decoder': False,
            'cond_types': ['full']
        }
    
    # Create model if possible
    model = None
    model_info = None
    manual_param_info = None
    if HAS_MODEL:
        try:
            print("Creating Dual-Hand Action Segmentation Model...")
            model = DualHandASDiffusionModel(
                encoder_params=encoder_params,
                decoder_params=decoder_params,
                diffusion_params=diffusion_params,
                num_classes=args.num_classes,
                device=torch.device(args.device)
            )
            model = model.to(args.device)
            model.eval()
            
            # Get model info
            model_info = get_model_info(model)
            print("\n" + "="*60)
            print("Model Architecture Information")
            print("="*60)
            print(f"Input Dimension: {args.input_dim}")
            print(f"Number of Classes: {args.num_classes}")
            print(f"Encoder Layers: {args.num_layers_enc}")
            print(f"Encoder Feature Maps: {args.num_f_maps_enc}")
            print(f"Decoder Layers: {args.num_layers_dec}")
            print(f"Decoder Feature Maps: {args.num_f_maps_dec}")
            print(f"Total Parameters: {model_info['total_params']:,}")
            print(f"Trainable Parameters: {model_info['trainable_params']:,}")
            if 'num_timesteps' in model_info:
                print(f"Diffusion Timesteps: {model_info['num_timesteps']}")
        except Exception as e:
            print(f"Warning: Could not create model: {e}")
            print("Using manual calculation based on architecture specifications.")
            model = None
            manual_param_info = calculate_parameter_counts_manual(
                encoder_params, decoder_params, diffusion_params,
                args.input_dim, args.num_classes
            )
            print("\nEstimated Parameter Counts (Manual Calculation)")
            print("="*60)
            print(f"Total Parameters: {manual_param_info['total_params']:,} ({manual_param_info['total_params']/1e6:.2f}M)")
            print(f"Encoder Parameters: {manual_param_info['encoder_params']:,}")
            print(f"Fusion Parameters: {manual_param_info['fusion_params']:,}")
            print(f"Decoder Parameters (per hand): {manual_param_info['decoder_params_per_hand']:,}")
    else:
        print("Using manual calculation based on architecture specifications.")
        manual_param_info = calculate_parameter_counts_manual(
            encoder_params, decoder_params, diffusion_params,
            args.input_dim, args.num_classes
        )
        print("\nEstimated Parameter Counts (Manual Calculation)")
        print("="*60)
        print(f"Total Parameters: {manual_param_info['total_params']:,} ({manual_param_info['total_params']/1e6:.2f}M)")
        print(f"Encoder Parameters: {manual_param_info['encoder_params']:,}")
        print(f"Fusion Parameters: {manual_param_info['fusion_params']:,}")
        print(f"Decoder Parameters (per hand): {manual_param_info['decoder_params_per_hand']:,}")
    
    # Input shape: (batch_size, input_dim, seq_len) for Conv1d models
    input_shape = (args.batch_size, args.input_dim, args.seq_len)
    
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
            if HAS_FVCORE and model is not None:
                print("Using fvcore...")
                try:
                    inference_flops, inference_flops_dict = calculate_flops_fvcore(model, input_shape, mode='eval')
                except Exception as e:
                    print(f"fvcore calculation failed: {e}")
                    print("Falling back to manual calculation...")
                    inference_flops, inference_flops_dict = calculate_flops_manual(
                        model, encoder_params, decoder_params, diffusion_params,
                        input_shape, args.num_classes
                    )
            else:
                print("Using manual calculation...")
                inference_flops, inference_flops_dict = calculate_flops_manual(
                    model, encoder_params, decoder_params, diffusion_params,
                    input_shape, args.num_classes
                )
        elif args.method == 'fvcore':
            if not HAS_FVCORE or model is None:
                print("Error: fvcore not available or model creation failed")
                print("Please install fvcore and ensure model can be created")
                return
            inference_flops, inference_flops_dict = calculate_flops_fvcore(model, input_shape, mode='eval')
        elif args.method == 'manual':
            inference_flops, inference_flops_dict = calculate_flops_manual(
                model, encoder_params, decoder_params, diffusion_params,
                input_shape, args.num_classes
            )
    
    # Calculate training FLOPs (forward + backward pass)
    if args.mode in ['training', 'both']:
        print("\nCalculating Training FLOPs (forward + backward pass)...")
        if args.method == 'fvcore' and HAS_FVCORE and model is not None:
            try:
                model.train()  # Enable training mode
                training_flops, training_flops_dict = calculate_flops_fvcore(model, input_shape, mode='train')
            except Exception as e:
                print(f"Training FLOPs calculation failed: {e}")
                # Fall back to estimation
                if inference_flops is not None:
                    training_flops = inference_flops * 3
                    training_flops_dict = {k: v * 3 for k, v in (inference_flops_dict or {}).items()}
                    print("Note: Using estimated training FLOPs (3× inference)")
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
    
    # Component breakdown (using inference FLOPs)
    if inference_flops_dict:
        encoder_flops = inference_flops_dict.get('encoder', 0)
        fusion_flops = inference_flops_dict.get('feature_fusion', 0)
        decoder_lh_flops = inference_flops_dict.get('decoder_left', 0)
        decoder_rh_flops = inference_flops_dict.get('decoder_right', 0)
        diffusion_flops = inference_flops_dict.get('diffusion_ops', 0)
        
        if encoder_flops > 0:
            print(f"\nEncoder FLOPs (Inference): {encoder_flops / 1e9:.2f} GFLOPs ({encoder_flops / inference_flops * 100:.1f}%)")
        if fusion_flops > 0:
            print(f"Feature Fusion FLOPs (Inference): {fusion_flops / 1e9:.2f} GFLOPs ({fusion_flops / inference_flops * 100:.1f}%)")
        if decoder_lh_flops > 0:
            print(f"Decoder (Left Hand) FLOPs (Inference): {decoder_lh_flops / 1e9:.2f} GFLOPs ({decoder_lh_flops / inference_flops * 100:.1f}%)")
        if decoder_rh_flops > 0:
            print(f"Decoder (Right Hand) FLOPs (Inference): {decoder_rh_flops / 1e9:.2f} GFLOPs ({decoder_rh_flops / inference_flops * 100:.1f}%)")
        if diffusion_flops > 0:
            print(f"Diffusion Operations FLOPs (Inference): {diffusion_flops / 1e9:.2f} GFLOPs ({diffusion_flops / inference_flops * 100:.1f}%)")
    
    print("\n" + "="*60)
    print("Summary for Paper")
    print("="*60)
    print(f"Model: Dual-Hand Action Segmentation (Diffusion-based)")
    if model_info:
        print(f"Parameters: {model_info['total_params']:,} ({model_info['total_params'] / 1e6:.2f}M)")
    elif manual_param_info:
        print(f"Parameters (estimated): {manual_param_info['total_params']:,} ({manual_param_info['total_params'] / 1e6:.2f}M)")
    if inference_flops is not None:
        print(f"Inference FLOPs: {inference_flops / 1e9:.2f} GFLOPs (forward pass only)")
    if training_flops is not None:
        print(f"Training FLOPs: {training_flops / 1e9:.2f} GFLOPs (forward + backward per step)")
    print(f"Input: {args.seq_len} frames × {args.input_dim}D features")
    print(f"Architecture: Encoder ({args.num_layers_enc} layers) + Fusion + Dual Decoders ({args.num_layers_dec} layers each)")
    print("="*60)


if __name__ == '__main__':
    main()

