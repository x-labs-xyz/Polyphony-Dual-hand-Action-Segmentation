#!/usr/bin/env python3
"""
Calculate FLOPs (Floating Point Operations) for ADH-ViT model.

This script calculates the computational complexity of the ADH-ViT model
for reporting in research papers.

Usage:
    python calculate_flops.py --model vit_base_patch16_224_alternating
    python calculate_flops.py --model vit_base_patch16_224_alternating --input_size 224 --num_frames 16
"""

import argparse
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fvcore.nn import flop_count
    # Try to import FlopCountMode (available in newer versions)
    try:
        from fvcore.nn import FlopCountMode
        HAS_FLOP_COUNT_MODE = True
    except ImportError:
        # Older versions of fvcore don't have FlopCountMode
        # We'll use a workaround
        HAS_FLOP_COUNT_MODE = False
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    HAS_FLOP_COUNT_MODE = False
    print("Warning: fvcore not installed. Install with: pip install fvcore")
    print("Trying alternative: ptflops...")
    try:
        from ptflops import get_model_complexity_info
        HAS_PTFLOPS = True
    except ImportError:
        HAS_PTFLOPS = False
        print("Warning: ptflops not installed. Install with: pip install ptflops")
        print("Using manual calculation method (no dependencies needed).")

# Import models only if needed - use try/except to handle missing modules
try:
    import models.modeling_finetune_alternating  # Register alternating models
except ImportError:
    # If import fails, models will be registered via timm
    pass


def calculate_flops_fvcore(model, input_shape, mode='train'):
    """
    Calculate FLOPs using fvcore.
    
    Args:
        model: Model to evaluate
        input_shape: Input shape tuple
        mode: 'train' for training FLOPs (forward+backward), 'eval' for inference FLOPs (forward only)
    """
    if mode == 'train':
        model.train()  # Enable training mode for backward pass
    else:
        model.eval()  # Evaluation mode (forward only)
    
    # Input shape: (T, C, H, W) -> (B, C, T, H, W) for video models
    if len(input_shape) == 4:
        T, C, H, W = input_shape
        dummy_input = torch.randn(1, C, T, H, W).to(next(model.parameters()).device)
    else:
        dummy_input = torch.randn(1, *input_shape).to(next(model.parameters()).device)
    
    # Create a wrapper to ensure both heads are traced
    # fvcore has issues with conditional logic in forward pass
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            # Call both heads to ensure full tracing
            if hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(x)
                # Use both heads to get complete FLOPs
                lh_out = self.model.lh_head(features)
                rh_out = self.model.rh_head(features)
                # Return average (doesn't affect FLOPs calculation)
                return (lh_out + rh_out) / 2
            else:
                return self.model(x)
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval() if mode == 'eval' else wrapped_model.train()
    
    # Calculate FLOPs
    # Handle different fvcore versions
    if HAS_FLOP_COUNT_MODE:
        # Newer version with FlopCountMode
        flop_mode = FlopCountMode.TRAIN if mode == 'train' else FlopCountMode.EVAL
        flop_dict, _ = flop_count(wrapped_model, (dummy_input,), mode=flop_mode)
    else:
        # Older version - flop_count doesn't have mode parameter
        # It only calculates forward pass FLOPs
        flop_dict, _ = flop_count(wrapped_model, (dummy_input,))
        # For training mode, estimate as 3x inference
        if mode == 'train':
            flop_dict = {k: v * 3 for k, v in flop_dict.items()}
    
    total_flops = sum(flop_dict.values())
    
    # Validate: if fvcore gives suspiciously low results, fall back to manual
    # Check if total is less than 1M FLOPs (clearly wrong for a ViT)
    if total_flops < 1e6:
        print(f"Warning: fvcore returned suspiciously low FLOPs ({total_flops:,.0f}).")
        print("This may be due to incomplete model tracing. Falling back to manual calculation.")
        return None, None
    
    return total_flops, flop_dict


def calculate_flops_ptflops(model, input_shape):
    """Calculate FLOPs using ptflops."""
    model.eval()
    
    # ptflops expects input in format (C, H, W) for images
    # For video, we need to reshape: (T, C, H, W) -> (C, T, H, W)
    if len(input_shape) == 4:  # (T, C, H, W)
        T, C, H, W = input_shape
        input_shape_ptflops = (C, T, H, W)
    else:
        input_shape_ptflops = input_shape
    
    # Note: ptflops may not work perfectly with video models
    # We'll use a workaround by creating a wrapper
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            # Reshape if needed
            if x.dim() == 4:  # (C, T, H, W)
                x = x.permute(1, 0, 2, 3)  # (T, C, H, W)
            return self.model(x)
    
    wrapped_model = ModelWrapper(model)
    
    try:
        macs, params = get_model_complexity_info(
            wrapped_model,
            input_shape_ptflops,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        # MACs (Multiply-Accumulate) to FLOPs: typically 2 FLOPs per MAC
        flops = macs * 2
        return flops, {'total': flops}
    except Exception as e:
        print(f"Error with ptflops: {e}")
        return None, None


def calculate_flops_manual(model, input_shape):
    """
    Manual FLOPs calculation for Vision Transformer.
    This provides a theoretical estimate based on the architecture.
    """
    # Get model configuration
    if hasattr(model, 'embed_dim'):
        embed_dim = model.embed_dim
        depth = model.depth if hasattr(model, 'depth') else len(model.blocks)
        num_heads = model.num_heads if hasattr(model, 'num_heads') else 12
        mlp_ratio = model.mlp_ratio if hasattr(model, 'mlp_ratio') else 4.0
    else:
        # Try to get from model structure
        embed_dim = model.embed_dim if hasattr(model, 'embed_dim') else 768
        depth = len(model.blocks) if hasattr(model, 'blocks') else 12
        num_heads = 12  # Default
        mlp_ratio = 4.0  # Default
    
    # Input shape: (T, C, H, W) or (C, T, H, W)
    if len(input_shape) == 4:
        if input_shape[0] == 3:  # (C, H, W, T) format
            C, H, W, T = input_shape[0], input_shape[1], input_shape[2], 16
        else:  # (T, C, H, W) format
            T, C, H, W = input_shape
    else:
        T, C, H, W = 16, 3, 224, 224
    
    # Calculate number of patches
    patch_size = 16  # Default for ViT-Base
    num_patches_spatial = (H // patch_size) * (W // patch_size)
    tubelet_size = 2  # Default
    num_patches_temporal = T // tubelet_size
    num_patches = num_patches_spatial * num_patches_temporal
    
    mlp_dim = int(embed_dim * mlp_ratio)
    
    # Patch embedding FLOPs
    patch_embed_flops = num_patches * C * patch_size * patch_size * tubelet_size * embed_dim
    
    # Self-attention FLOPs per layer
    # QKV projection: 3 * num_patches * embed_dim^2
    # Attention: num_patches^2 * embed_dim
    # Output projection: num_patches * embed_dim^2
    attn_flops_per_layer = (
        3 * num_patches * embed_dim * embed_dim +  # QKV projection
        num_patches * num_patches * embed_dim +     # Attention matrix
        num_patches * embed_dim * embed_dim         # Output projection
    )
    
    # MLP FLOPs per layer
    mlp_flops_per_layer = (
        2 * num_patches * embed_dim * mlp_dim  # Two linear layers
    )
    
    # Total FLOPs for transformer blocks
    block_flops_per_layer = attn_flops_per_layer + mlp_flops_per_layer
    total_block_flops = depth * block_flops_per_layer
    
    # Classification head FLOPs (for one head)
    head_flops = num_patches * embed_dim * 75  # 75 classes
    
    # Total FLOPs
    total_flops = patch_embed_flops + total_block_flops + head_flops
    
    return total_flops, {
        'patch_embedding': patch_embed_flops,
        'transformer_blocks': total_block_flops,
        'classification_head': head_flops,
        'total': total_flops
    }


def get_model_info(model):
    """Get model architecture information."""
    info = {}
    
    if hasattr(model, 'embed_dim'):
        info['embed_dim'] = model.embed_dim
    if hasattr(model, 'blocks'):
        info['depth'] = len(model.blocks)
    if hasattr(model, 'num_heads'):
        info['num_heads'] = model.num_heads
    elif hasattr(model, 'blocks') and len(model.blocks) > 0:
        if hasattr(model.blocks[0], 'attn'):
            info['num_heads'] = model.blocks[0].attn.num_heads
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info['total_params'] = total_params
    info['trainable_params'] = trainable_params
    
    return info


def main():
    parser = argparse.ArgumentParser(description='Calculate FLOPs for ADH-ViT model')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_alternating',
                        help='Model name')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames')
    parser.add_argument('--tubelet_size', type=int, default=2,
                        help='Tubelet size')
    parser.add_argument('--lh_num_classes', type=int, default=75,
                        help='Left hand number of classes')
    parser.add_argument('--rh_num_classes', type=int, default=75,
                        help='Right hand number of classes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'fvcore', 'ptflops', 'manual'],
                        help='Method to calculate FLOPs')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['inference', 'training', 'both'],
                        help='Calculate inference FLOPs, training FLOPs, or both')
    
    args = parser.parse_args()
    
    # Import create_model from timm
    from timm.models import create_model
    
    # Try to import and register alternating models
    try:
        import models.modeling_finetune_alternating
    except (ImportError, ModuleNotFoundError):
        # Models will be registered via @register_model decorator when create_model is called
        pass
    
    # Create model
    print(f"Creating model: {args.model}")
    try:
        model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        lh_num_classes=args.lh_num_classes,
        rh_num_classes=args.rh_num_classes,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        head_drop_rate=0.0,
        use_mean_pooling=True,
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        print("\nTrying to import models explicitly...")
        # Try importing models module to register them
        try:
            import models
            model = create_model(
                args.model,
                img_size=args.input_size,
                pretrained=False,
                lh_num_classes=args.lh_num_classes,
                rh_num_classes=args.rh_num_classes,
                all_frames=args.num_frames,
                tubelet_size=args.tubelet_size,
                drop_rate=0.0,
                drop_path_rate=0.0,
                attn_drop_rate=0.0,
                head_drop_rate=0.0,
                use_mean_pooling=True,
            )
        except Exception as e2:
            print(f"Failed to create model: {e2}")
            print("\nPlease make sure:")
            print("1. You are in the ADH-ViT directory")
            print("2. The model name is correct")
            print("3. All required files are present")
            sys.exit(1)
    
    model = model.to(args.device)
    model.eval()
    
    # Get model info
    model_info = get_model_info(model)
    print("\n" + "="*60)
    print("Model Architecture Information")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Embedding Dimension: {model_info.get('embed_dim', 'N/A')}")
    print(f"Depth (Number of Layers): {model_info.get('depth', 'N/A')}")
    print(f"Number of Heads: {model_info.get('num_heads', 'N/A')}")
    print(f"Total Parameters: {model_info['total_params']:,}")
    print(f"Trainable Parameters: {model_info['trainable_params']:,}")
    print(f"Left Hand Classes: {args.lh_num_classes}")
    print(f"Right Hand Classes: {args.rh_num_classes}")
    
    # Input shape: (batch, channels, frames, height, width)
    # For video models, we typically use (T, C, H, W) format
    input_shape = (args.num_frames, 3, args.input_size, args.input_size)
    
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
                inference_flops, inference_flops_dict = calculate_flops_fvcore(model, input_shape, mode='eval')
                # If fvcore failed (returned None), fall back to manual
                if inference_flops is None:
                    print("Falling back to manual calculation...")
                    inference_flops, inference_flops_dict = calculate_flops_manual(model, input_shape)
            elif HAS_PTFLOPS:
                print("Using ptflops...")
                inference_flops, inference_flops_dict = calculate_flops_ptflops(model, input_shape)
            else:
                print("Using manual calculation...")
                inference_flops, inference_flops_dict = calculate_flops_manual(model, input_shape)
        elif args.method == 'fvcore':
            if not HAS_FVCORE:
                print("Error: fvcore not installed. Install with: pip install fvcore")
                return
            inference_flops, inference_flops_dict = calculate_flops_fvcore(model, input_shape, mode='eval')
            # If fvcore failed (returned None), fall back to manual
            if inference_flops is None:
                print("fvcore calculation failed. Falling back to manual calculation...")
                inference_flops, inference_flops_dict = calculate_flops_manual(model, input_shape)
        elif args.method == 'ptflops':
            if not HAS_PTFLOPS:
                print("Error: ptflops not installed. Install with: pip install ptflops")
                return
            inference_flops, inference_flops_dict = calculate_flops_ptflops(model, input_shape)
        elif args.method == 'manual':
            inference_flops, inference_flops_dict = calculate_flops_manual(model, input_shape)
    
    # Calculate training FLOPs (forward + backward pass)
    if args.mode in ['training', 'both']:
        print("\nCalculating Training FLOPs (forward + backward pass)...")
        if args.method == 'auto' or args.method == 'fvcore':
            if HAS_FVCORE:
                training_flops, training_flops_dict = calculate_flops_fvcore(model, input_shape, mode='train')
                # If fvcore failed (returned None), estimate from inference
                if training_flops is None and inference_flops is not None:
                    training_flops = inference_flops * 3
                    training_flops_dict = {k: v * 3 for k, v in (inference_flops_dict or {}).items()}
                    print("Note: Using estimated training FLOPs (3× inference) since fvcore failed")
            else:
                # Estimate: training FLOPs ≈ 3 × inference FLOPs
                if inference_flops is not None:
                    training_flops = inference_flops * 3
                    training_flops_dict = {k: v * 3 for k, v in (inference_flops_dict or {}).items()}
                    print("Note: Using estimated training FLOPs (3× inference) since fvcore is not available")
                else:
                    print("Error: Cannot estimate training FLOPs without inference FLOPs")
        elif args.method == 'manual':
            # Estimate: training FLOPs ≈ 3 × inference FLOPs
            if inference_flops is not None:
                training_flops = inference_flops * 3
                training_flops_dict = {k: v * 3 for k, v in (inference_flops_dict or {}).items()}
                print("Note: Using estimated training FLOPs (3× inference) for manual calculation")
    
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
        
        if inference_flops_dict and len(inference_flops_dict) > 1:
            print("\nBreakdown:")
            for key, value in inference_flops_dict.items():
                if key != 'total' and isinstance(value, (int, float)):
                    print(f"  {key}: {value:,.0f} ({value / inference_flops * 100:.1f}%)")
        
        # Calculate FLOPs per frame
        inference_flops_per_frame = inference_flops / args.num_frames
        print(f"\nInference FLOPs per Frame: {inference_flops_per_frame / 1e9:.2f} GFLOPs")
    
    if training_flops is not None:
        print("\n📊 TRAINING FLOPs (Forward + Backward Pass)")
        print("-" * 60)
        print(f"Training FLOPs: {training_flops:,.0f}")
        print(f"Training FLOPs (G): {training_flops / 1e9:.2f} GFLOPs")
        
        if inference_flops is not None:
            multiplier = training_flops / inference_flops
            print(f"Training/Inference Ratio: {multiplier:.2f}x")
        
        if training_flops_dict and len(training_flops_dict) > 1:
            print("\nBreakdown:")
            for key, value in training_flops_dict.items():
                if key != 'total' and isinstance(value, (int, float)):
                    print(f"  {key}: {value:,.0f} ({value / training_flops * 100:.1f}%)")
        
        # Calculate FLOPs per frame
        training_flops_per_frame = training_flops / args.num_frames
        print(f"\nTraining FLOPs per Frame: {training_flops_per_frame / 1e9:.2f} GFLOPs")
    
    # Calculate FLOPs for shared backbone vs heads (if available)
    if inference_flops_dict:
        backbone_flops = inference_flops - (inference_flops_dict.get('lh_head', 0) + inference_flops_dict.get('rh_head', 0))
        print(f"\nShared Backbone FLOPs (Inference): {backbone_flops / 1e9:.2f} GFLOPs")
        if 'lh_head' in inference_flops_dict:
            print(f"Left Hand Head FLOPs (Inference): {inference_flops_dict['lh_head'] / 1e9:.2f} GFLOPs")
        if 'rh_head' in inference_flops_dict:
            print(f"Right Hand Head FLOPs (Inference): {inference_flops_dict['rh_head'] / 1e9:.2f} GFLOPs")
    
    print("\n" + "="*60)
    print("Summary for Paper")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Parameters: {model_info['total_params']:,} ({model_info['total_params'] / 1e6:.2f}M)")
    if inference_flops is not None:
        print(f"Inference FLOPs: {inference_flops / 1e9:.2f} GFLOPs (forward pass only)")
    if training_flops is not None:
        print(f"Training FLOPs: {training_flops / 1e9:.2f} GFLOPs (forward + backward per step)")
    print(f"Input: {args.num_frames} frames × {args.input_size}×{args.input_size}")
    print("="*60)


if __name__ == '__main__':
    # Check if timm is installed
    try:
        import timm
    except ImportError:
        print("Error: timm not installed. Install with: pip install timm")
        sys.exit(1)
    
    main()

