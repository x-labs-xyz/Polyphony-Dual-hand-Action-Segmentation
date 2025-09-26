#!/usr/bin/env python3
"""
Alternating Hands VideoMAEv2 Feature Extraction Script

This script extracts features from videos using a trained alternating hands VideoMAEv2 model.
It can extract:
1. Shared encoder features (common backbone features)
2. Left hand specific features (from lh_head)
3. Right hand specific features (from rh_head)
4. Both hand features simultaneously

Usage:
    python extract_alternating_hand_features.py \
        --checkpoint output/havid_alternating_hands/checkpoint-best.pth \
        --video_dir /path/to/videos \
        --output_dir /path/to/output \
        --hand_type both \
        --feature_type shared
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2

# Import the alternating hands model
from models.modeling_finetune_alternating import AlternatingDualHeadVisionTransformer
import models  # Register models

# Import video loading utilities
from dataset.transforms import *
from dataset.video_transforms import *


class AlternatingHandFeatureExtractor:
    """Feature extractor for alternating hands VideoMAEv2 model"""
    
    def __init__(self,
                 checkpoint_path: str,
                 device: str = 'cuda',
                 input_size: int = 224,
                 num_frames: int = 16,
                 sampling_rate: int = 4,
                 model_name: str = 'vit_base_patch16_224_alternating',
                 lh_num_classes: int = 75,
                 rh_num_classes: int = 75):
        """
        Initialize feature extractor
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on ('cuda' or 'cpu')
            input_size: Input image size
            num_frames: Number of frames to sample
            sampling_rate: Frame sampling rate
            model_name: Model architecture name
            lh_num_classes: Number of left hand classes
            rh_num_classes: Number of right hand classes
        """
        self.device = torch.device(device)
        self.input_size = input_size
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        
        # Initialize model
        self.model = self._create_model(model_name, lh_num_classes, rh_num_classes)
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup video transforms
        self.transform = self._setup_transforms()
        
    def _create_model(self, model_name: str, lh_num_classes: int, rh_num_classes: int):
        """Create alternating hands model"""
        from timm.models import create_model
        
        model = create_model(
            model_name,
            pretrained=False,
            lh_num_classes=lh_num_classes,
            rh_num_classes=rh_num_classes,
            all_frames=self.num_frames,
            tubelet_size=2,
            use_mean_pooling=True,
            init_scale=0.001,
        )
        
        return model
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DDP training)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint: {msg}")
    
    def _setup_transforms(self):
        """Setup video preprocessing transforms"""
        return Compose([
            RandomResizedCrop(self.input_size, scale=(0.2, 1.0)),
            ToTensorVideo(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _resize_and_center_crop(self, frames):
        """Resize and center crop frames"""
        # frames: [T, H, W, C]
        T, H, W, C = frames.shape
        
        # Resize short side to 256
        short_side_size = 256
        if H < W:
            new_H, new_W = short_side_size, int(W * short_side_size / H)
        else:
            new_H, new_W = int(H * short_side_size / W), short_side_size
        
        frames = torch.nn.functional.interpolate(
            frames.permute(3, 0, 1, 2),  # [C, T, H, W]
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        ).permute(1, 2, 3, 0)  # [T, H, W, C]
        
        # Center crop to input_size
        crop_h = (new_H - self.input_size) // 2
        crop_w = (new_W - self.input_size) // 2
        frames = frames[:, crop_h:crop_h+self.input_size, crop_w:crop_w+self.input_size, :]
        
        return frames
    
    def load_video(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Could not load video: {video_path}")
        
        # Convert to numpy and sample frames
        frames = np.array(frames)  # [T, H, W, C]
        
        # Sample frames
        total_frames = len(frames)
        if total_frames < self.num_frames:
            # Repeat last frame if not enough frames
            indices = np.arange(total_frames)
            indices = np.concatenate([indices, np.repeat(indices[-1], self.num_frames - total_frames)])
        else:
            # Sample uniformly
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = frames[indices]  # [num_frames, H, W, C]
        
        # Convert to tensor and preprocess
        frames = torch.from_numpy(frames).float()
        frames = self._resize_and_center_crop(frames)
        
        # Normalize to [0, 1]
        frames = frames / 255.0
        
        # Apply standard normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
        frames = (frames - mean) / std
        
        # Rearrange to [C, T, H, W]
        frames = frames.permute(3, 0, 1, 2)
        
        return frames
    
    def extract_features(self,
                        video_path: str,
                        feature_type: str = 'shared',
                        hand_type: str = 'both') -> Dict[str, np.ndarray]:
        """
        Extract features from video
        
        Args:
            video_path: Path to video file
            feature_type: Type of features to extract ('shared', 'head', 'final')
            hand_type: Which hand predictions to get ('lh', 'rh', 'both')
            
        Returns:
            Dictionary containing extracted features
        """
        # Load video
        video = self.load_video(video_path)
        video = video.unsqueeze(0).to(self.device)  # Add batch dimension
        
        features = {}
        
        with torch.no_grad():
            if feature_type == 'shared':
                # Extract shared backbone features (both pooled and unpooled)
                # First get the raw patch features before pooling
                x = video
                B = x.size(0)
                x = self.model.patch_embed(x)
                if self.model.pos_embed is not None:
                    x = x + self.model.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
                x = self.model.pos_drop(x)
                for blk in self.model.blocks:
                    x = blk(x)
                x = self.model.norm(x)  # [B, num_patches, embed_dim]
                
                # Save unpooled patch features
                features['patch_features'] = x.cpu().numpy()
                
                # Get pooled features (what forward_features normally returns)
                if hasattr(self.model, 'fc_norm') and self.model.fc_norm is not None:
                    pooled_features = self.model.fc_norm(x.mean(1))  # [B, embed_dim]
                else:
                    pooled_features = x.mean(1)  # [B, embed_dim] - mean pooling
                
                features['shared_features'] = pooled_features.cpu().numpy()
                
            elif feature_type == 'patch':
                # Extract only patch features (before pooling)
                x = video
                B = x.size(0)
                x = self.model.patch_embed(x)
                if self.model.pos_embed is not None:
                    x = x + self.model.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
                x = self.model.pos_drop(x)
                for blk in self.model.blocks:
                    x = blk(x)
                x = self.model.norm(x)  # [B, num_patches, embed_dim]
                
                features['patch_features'] = x.cpu().numpy()
                
            elif feature_type == 'head':
                # Extract features from classification heads
                shared_features = self.model.forward_features(video)
                
                if hand_type in ['lh', 'both']:
                    lh_features = self.model.lh_head(shared_features)
                    features['lh_head_features'] = lh_features.cpu().numpy()
                
                if hand_type in ['rh', 'both']:
                    rh_features = self.model.rh_head(shared_features)
                    features['rh_head_features'] = rh_features.cpu().numpy()
                    
            elif feature_type == 'final':
                # Extract final predictions
                if hand_type == 'both':
                    predictions = self.model(video, hand_type='both')
                    features['lh_predictions'] = predictions['lh_pred'].cpu().numpy()
                    features['rh_predictions'] = predictions['rh_pred'].cpu().numpy()
                else:
                    predictions = self.model(video, hand_type=hand_type)
                    features[f'{hand_type}_predictions'] = predictions.cpu().numpy()
                    
            elif feature_type == 'all':
                # Extract all types of features
                # First get the raw patch features before pooling
                x = video
                B = x.size(0)
                x = self.model.patch_embed(x)
                if self.model.pos_embed is not None:
                    x = x + self.model.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
                x = self.model.pos_drop(x)
                for blk in self.model.blocks:
                    x = blk(x)
                x = self.model.norm(x)  # [B, num_patches, embed_dim]
                
                # Save unpooled patch features
                features['patch_features'] = x.cpu().numpy()
                
                # Get pooled features
                if hasattr(self.model, 'fc_norm') and self.model.fc_norm is not None:
                    pooled_features = self.model.fc_norm(x.mean(1))  # [B, embed_dim]
                else:
                    pooled_features = x.mean(1)  # [B, embed_dim] - mean pooling
                
                features['shared_features'] = pooled_features.cpu().numpy()
                
                # Head features (using pooled features)
                lh_head_features = self.model.lh_head(pooled_features)
                rh_head_features = self.model.rh_head(pooled_features)
                features['lh_head_features'] = lh_head_features.cpu().numpy()
                features['rh_head_features'] = rh_head_features.cpu().numpy()
                
                # Final predictions (same as head features for this model)
                features['lh_predictions'] = lh_head_features.cpu().numpy()
                features['rh_predictions'] = rh_head_features.cpu().numpy()
        
        return features
    
    def extract_batch_features(self,
                              video_paths: List[str],
                              feature_type: str = 'shared',
                              hand_type: str = 'both',
                              output_dir: str = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract features from multiple videos
        
        Args:
            video_paths: List of video file paths
            feature_type: Type of features to extract
            hand_type: Which hand predictions to get
            output_dir: Directory to save features (optional)
            
        Returns:
            Dictionary mapping video names to their features
        """
        all_features = {}
        
        for video_path in tqdm(video_paths, desc="Extracting features"):
            try:
                video_name = Path(video_path).stem
                features = self.extract_features(video_path, feature_type, hand_type)
                all_features[video_name] = features
                
                # Save individual feature file if output_dir is provided
                if output_dir:
                    output_path = Path(output_dir) / f"{video_name}_{feature_type}_{hand_type}.npz"
                    np.savez_compressed(output_path, **features)
                    
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        
        return all_features


def main():
    parser = argparse.ArgumentParser(description='Extract features from alternating hands VideoMAEv2 model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224_alternating',
                        help='Model architecture name')
    parser.add_argument('--lh_num_classes', type=int, default=75,
                        help='Number of left hand classes')
    parser.add_argument('--rh_num_classes', type=int, default=75,
                        help='Number of right hand classes')
    
    # Video arguments
    parser.add_argument('--video_dir', type=str,
                        help='Directory containing videos')
    parser.add_argument('--video_path', type=str,
                        help='Path to single video file')
    parser.add_argument('--video_list', type=str,
                        help='Text file containing list of video paths')
    
    # Feature extraction arguments
    parser.add_argument('--feature_type', type=str, default='shared',
                        choices=['shared', 'patch', 'head', 'final', 'all'],
                        help='Type of features to extract')
    parser.add_argument('--hand_type', type=str, default='both',
                        choices=['lh', 'rh', 'both'],
                        help='Which hand features to extract')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted features')
    
    # Model parameters
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to sample')
    parser.add_argument('--sampling_rate', type=int, default=4,
                        help='Frame sampling rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize feature extractor
    print("Initializing feature extractor...")
    extractor = AlternatingHandFeatureExtractor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        input_size=args.input_size,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        model_name=args.model_name,
        lh_num_classes=args.lh_num_classes,
        rh_num_classes=args.rh_num_classes
    )
    
    # Collect video paths
    video_paths = []
    
    if args.video_path:
        video_paths = [args.video_path]
    elif args.video_dir:
        video_dir = Path(args.video_dir)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for ext in video_extensions:
            video_paths.extend(video_dir.glob(f'**/*{ext}'))
        video_paths = [str(p) for p in video_paths]
    elif args.video_list:
        with open(args.video_list, 'r') as f:
            video_paths = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must specify either --video_path, --video_dir, or --video_list")
    
    print(f"Found {len(video_paths)} videos to process")
    
    # Extract features
    print(f"Extracting {args.feature_type} features for {args.hand_type} hand(s)...")
    all_features = extractor.extract_batch_features(
        video_paths=video_paths,
        feature_type=args.feature_type,
        hand_type=args.hand_type,
        output_dir=args.output_dir
    )
    
    # Save summary
    summary = {
        'checkpoint': args.checkpoint,
        'model_name': args.model_name,
        'feature_type': args.feature_type,
        'hand_type': args.hand_type,
        'num_videos': len(all_features),
        'video_names': list(all_features.keys()),
        'feature_shapes': {}
    }
    
    # Add feature shape information
    if all_features:
        sample_features = next(iter(all_features.values()))
        for key, value in sample_features.items():
            summary['feature_shapes'][key] = value.shape
    
    # Save summary
    summary_path = Path(args.output_dir) / f"extraction_summary_{args.feature_type}_{args.hand_type}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Feature extraction completed!")
    print(f"Processed {len(all_features)} videos")
    print(f"Features saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
