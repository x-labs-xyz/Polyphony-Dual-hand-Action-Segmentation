#!/usr/bin/env python3
"""
Alternating Hands VideoMAEv2 Frame-wise Feature Extraction Script

This script extracts frame-wise features from untrimmed videos using a fine-tuned alternating hands VideoMAEv2 model.
Features are extracted at high temporal resolution using a sliding window approach and saved as 
[video_length, feature_dim] tensors for each feature type.

Usage:
    python extract_alternating_frame_features.py --video_path /path/to/video.mp4 --checkpoint /path/to/checkpoint.pth
    python extract_alternating_frame_features.py --video_dir /path/to/videos --checkpoint /path/to/checkpoint.pth --batch_mode
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from timm.models import create_model
from tqdm import tqdm
import cv2

# Import VideoMAEv2 models
import models  # noqa: F401
from models.modeling_finetune_alternating import AlternatingDualHeadVisionTransformer


class AlternatingHandsFrameFeatureExtractor:
    """Frame-wise feature extractor for alternating hands VideoMAEv2 models"""
    
    def __init__(self, 
                 model_name: str = 'vit_base_patch16_224_alternating',
                 checkpoint_path: str = None,
                 device: str = 'cuda',
                 num_frames: int = 16,
                 sampling_rate: int = 1,  # Default to 1 for frame-wise
                 input_size: int = 224,
                 tubelet_size: int = 2,
                 lh_num_classes: int = 75,
                 rh_num_classes: int = 75,
                 use_mean_pooling: bool = True,
                 drop_path_rate: float = 0.3,
                 overlap_frames: int = 0):
        """
        Initialize the frame-wise feature extractor
        
        Args:
            model_name: Name of the model architecture
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run inference on
            num_frames: Number of frames in each window
            sampling_rate: Frame sampling rate within each window
            input_size: Input image size
            tubelet_size: Tubelet size for VideoMAE
            lh_num_classes: Number of left hand classes
            rh_num_classes: Number of right hand classes
            use_mean_pooling: Whether to use mean pooling
            drop_path_rate: Drop path rate
            overlap_frames: Number of overlapping frames between windows
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.input_size = input_size
        self.tubelet_size = tubelet_size
        self.lh_num_classes = lh_num_classes
        self.rh_num_classes = rh_num_classes
        self.overlap_frames = overlap_frames
        
        # Initialize model
        print(f"Loading model: {model_name}")
        self.model = self._create_model()
        
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = self._setup_transforms()
        
    def _create_model(self):
        """Create the alternating hands model"""
        model = create_model(
            self.model_name,
            pretrained=False,
            lh_num_classes=self.lh_num_classes,
            rh_num_classes=self.rh_num_classes,
            all_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            use_mean_pooling=True,
            init_scale=0.001,
        )
        return model
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from: {checkpoint_path}")
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
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint: {msg}")
    
    def _setup_transforms(self):
        """Setup video preprocessing transforms"""
        # We'll apply normalization manually for video tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        return None
    
    def _load_video(self, video_path: str) -> List[np.ndarray]:
        """Load video frames"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _extract_frame_window(self, frames: List[np.ndarray], start_idx: int) -> torch.Tensor:
        """Extract a window of frames and preprocess them"""
        # Calculate frame indices for this window
        end_idx = start_idx + self.num_frames * self.sampling_rate
        frame_indices = list(range(start_idx, min(end_idx, len(frames)), self.sampling_rate))
        
        # Pad if not enough frames
        while len(frame_indices) < self.num_frames:
            frame_indices.append(frame_indices[-1])  # Repeat last frame
        
        # Extract frames
        window_frames = []
        for idx in frame_indices[:self.num_frames]:
            frame = frames[idx]
            # Resize and center crop
            frame = self._resize_and_center_crop(frame)
            window_frames.append(frame)
        
        # Convert to tensor
        window_tensor = torch.stack(window_frames)  # [T, H, W, C]
        window_tensor = window_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
        window_tensor = window_tensor.float() / 255.0
        
        # Apply normalization manually for video tensors
        window_tensor = (window_tensor - self.mean) / self.std
        
        return window_tensor
    
    def _resize_and_center_crop(self, frame: np.ndarray) -> torch.Tensor:
        """Resize and center crop a single frame"""
        frame = torch.from_numpy(frame)
        H, W = frame.shape[:2]
        
        # Resize short side to 256
        short_side_size = 256
        if H < W:
            new_H, new_W = short_side_size, int(W * short_side_size / H)
        else:
            new_H, new_W = int(H * short_side_size / W), short_side_size
        
        frame = F.interpolate(
            frame.permute(2, 0, 1).unsqueeze(0).float(),  # [1, C, H, W]
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)  # [H, W, C]
        
        # Center crop to input_size
        crop_h = (new_H - self.input_size) // 2
        crop_w = (new_W - self.input_size) // 2
        frame = frame[crop_h:crop_h+self.input_size, crop_w:crop_w+self.input_size, :]
        
        return frame
    
    def extract_frame_features(self, 
                             video_path: str,
                             save_path: str = None,
                             stride: int = 1,
                             feature_types: List[str] = ['shared', 'head', 'patch']) -> Dict[str, Any]:
        """
        Extract frame-wise features from an untrimmed video
        
        Args:
            video_path: Path to the video file
            save_path: Path to save features (optional)
            stride: Stride for frame sampling (1 for every frame)
            feature_types: Types of features to extract ['shared', 'head', 'patch', 'predictions']
            
        Returns:
            Dictionary containing features and metadata
        """
        print(f"Processing video: {video_path}")
        
        # Load video
        try:
            original_frames = self._load_video(video_path)
            original_length = len(original_frames)
            print(f"Original video length: {original_length} frames")
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
        
        if original_length == 0:
            print(f"Video has no frames")
            return None
        
        # Add padding to ensure we can extract features for every frame
        # We need (num_frames - 1) padding frames at the beginning and end
        padding_needed = (self.num_frames * self.sampling_rate - 1) // 2
        
        # Pad at the beginning (repeat first frame)
        padded_frames = [original_frames[0]] * padding_needed + original_frames
        
        # Pad at the end (repeat last frame) 
        padded_frames = padded_frames + [original_frames[-1]] * padding_needed
        
        padded_length = len(padded_frames)
        print(f"Padded video length: {padded_length} frames (added {padding_needed} frames at each end)")
        
        # Calculate frame indices for feature extraction
        # Now we can extract features for every original frame
        if stride == 1:
            # For stride=1, we want exactly original_length features (one per original frame)
            frame_indices = list(range(padding_needed, padding_needed + original_length, stride))
        else:
            # For other strides, use the standard approach but account for padding
            max_start_idx = padded_length - self.num_frames * self.sampling_rate
            frame_indices = list(range(padding_needed, max_start_idx + 1, stride))
        
        if not frame_indices:
            frame_indices = [padding_needed]  # At least one sample
        
        print(f"Extracting features for {len(frame_indices)} frame positions")
        print(f"Frame indices range: {frame_indices[0]} to {frame_indices[-1]}")
        
        # Use padded frames for extraction
        frames = padded_frames
        
        # Initialize feature containers
        features_dict = {}
        for feature_type in feature_types:
            features_dict[feature_type] = []
        
        frame_positions = []
        
        # Extract features
        with torch.no_grad():
            for frame_idx in tqdm(frame_indices, desc="Extracting frame features"):
                try:
                    # Extract frame window
                    window_tensor = self._extract_frame_window(frames, frame_idx)
                    window_tensor = window_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
                    
                    # Extract different types of features
                    frame_features = self._extract_window_features(window_tensor, feature_types)
                    
                    # Store features
                    for feature_type in feature_types:
                        if feature_type in frame_features:
                            features_dict[feature_type].append(frame_features[feature_type])
                    
                    frame_positions.append(frame_idx)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    continue
        
        # Convert to numpy arrays and stack
        result_features = {}
        for feature_type in feature_types:
            if features_dict[feature_type]:
                # Stack features: [num_windows, feature_dim] or [num_windows, num_patches, feature_dim]
                stacked = torch.stack(features_dict[feature_type], dim=0).cpu().numpy()
                result_features[feature_type] = stacked
        
        # Adjust frame positions to correspond to original video frames
        original_frame_positions = [pos - padding_needed for pos in frame_positions]
        
        # Create result dictionary
        result = {
            'video_path': video_path,
            'original_video_length': original_length,
            'padded_video_length': padded_length,
            'padding_frames': padding_needed,
            'num_windows': len(frame_positions),
            'frame_positions': original_frame_positions,  # Positions relative to original video
            'padded_frame_positions': frame_positions,    # Positions in padded video
            'features': result_features,
            'model_info': {
                'model_name': self.model_name,
                'num_frames': self.num_frames,
                'sampling_rate': self.sampling_rate,
                'input_size': self.input_size,
                'stride': stride,
                'lh_num_classes': self.lh_num_classes,
                'rh_num_classes': self.rh_num_classes,
                'padding_strategy': 'repeat_boundary_frames'
            }
        }
        
        # Add feature dimensions (original extraction shapes, before transposing for save)
        for feature_type, feature_data in result_features.items():
            result[f'{feature_type}_shape'] = feature_data.shape
        
        # Save features if path provided
        if save_path:
            base_save_path = Path(save_path)
            base_dir = base_save_path.parent
            video_name = base_save_path.stem
            
            # Create feature type directories
            shared_dir = base_dir / 'shared_features'
            lh_dir = base_dir / 'lh_features'
            rh_dir = base_dir / 'rh_features'
            metadata_dir = base_dir / 'metadata'
            
            # Create directories
            for dir_path in [shared_dir, lh_dir, rh_dir, metadata_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Save features in separate folders by type
            # All features are transposed to [feature_dim, T] format
            saved_files = []
            
            # Save shared features - transpose from [T, 768] to [768, T]
            if 'shared' in result_features:
                shared_path = shared_dir / f"{video_name}.npy"
                shared_transposed = result_features['shared'].T  # [768, T]
                np.save(shared_path, shared_transposed)
                saved_files.append(str(shared_path))
                print(f"Shared features saved to: {shared_path} with shape {shared_transposed.shape}")
            
            # Save patch features - transpose from [T, N, 768] to [768, T*N] (flattened spatial-temporal)
            if 'patch' in result_features:
                patch_path = shared_dir / f"{video_name}_patch.npy"
                patch_features = result_features['patch']  # [T, N, 768]
                # Reshape to [T*N, 768] then transpose to [768, T*N]
                patch_reshaped = patch_features.reshape(-1, patch_features.shape[-1])  # [T*N, 768]
                patch_transposed = patch_reshaped.T  # [768, T*N]
                np.save(patch_path, patch_transposed)
                saved_files.append(str(patch_path))
                print(f"Patch features saved to: {patch_path} with shape {patch_transposed.shape}")
            
            # Save LH head features separately
            if 'lh_head' in result_features:
                lh_head_path = lh_dir / f"{video_name}_head.npy"
                lh_head_transposed = result_features['lh_head'].T  # [lh_classes, T]
                np.save(lh_head_path, lh_head_transposed)
                saved_files.append(str(lh_head_path))
                print(f"LH head features saved to: {lh_head_path} with shape {lh_head_transposed.shape}")
            
            # Save RH head features separately
            if 'rh_head' in result_features:
                rh_head_path = rh_dir / f"{video_name}_head.npy"
                rh_head_transposed = result_features['rh_head'].T  # [rh_classes, T]
                np.save(rh_head_path, rh_head_transposed)
                saved_files.append(str(rh_head_path))
                print(f"RH head features saved to: {rh_head_path} with shape {rh_head_transposed.shape}")
            
            # Save left hand predictions - transpose from [T, lh_classes] to [lh_classes, T]
            if 'lh_predictions' in result_features:
                lh_path = lh_dir / f"{video_name}.npy"
                lh_transposed = result_features['lh_predictions'].T  # [lh_classes, T]
                np.save(lh_path, lh_transposed)
                saved_files.append(str(lh_path))
                print(f"LH predictions saved to: {lh_path} with shape {lh_transposed.shape}")
            
            # Save right hand predictions - transpose from [T, rh_classes] to [rh_classes, T]
            if 'rh_predictions' in result_features:
                rh_path = rh_dir / f"{video_name}.npy"
                rh_transposed = result_features['rh_predictions'].T  # [rh_classes, T]
                np.save(rh_path, rh_transposed)
                saved_files.append(str(rh_path))
                print(f"RH predictions saved to: {rh_path} with shape {rh_transposed.shape}")
            
            # Save metadata
            metadata_path = metadata_dir / f"{video_name}.json"
            metadata = {k: v for k, v in result.items() if k != 'features'}
            # Convert numpy arrays to lists for JSON serialization
            metadata['frame_positions'] = [int(x) for x in metadata['frame_positions']]
            metadata['padded_frame_positions'] = [int(x) for x in metadata['padded_frame_positions']]
            metadata['saved_files'] = saved_files  # Track which files were saved
            
            # Add saved feature shapes (transposed format [feature_dim, T])
            metadata['saved_feature_shapes'] = {}
            if 'shared' in result_features:
                metadata['saved_feature_shapes']['shared'] = [768, metadata['num_windows']]
            if 'patch' in result_features:
                T, N, D = result_features['patch'].shape
                metadata['saved_feature_shapes']['patch'] = [D, T * N]  # [768, T*N]
            if 'lh_head' in result_features:
                metadata['saved_feature_shapes']['lh_head'] = [result_features['lh_head'].shape[1], metadata['num_windows']]
            if 'rh_head' in result_features:
                metadata['saved_feature_shapes']['rh_head'] = [result_features['rh_head'].shape[1], metadata['num_windows']]
            if 'lh_predictions' in result_features:
                metadata['saved_feature_shapes']['lh_predictions'] = [result_features['lh_predictions'].shape[1], metadata['num_windows']]
            if 'rh_predictions' in result_features:
                metadata['saved_feature_shapes']['rh_predictions'] = [result_features['rh_predictions'].shape[1], metadata['num_windows']]
            
            # Convert original extraction shapes to lists
            for key in list(metadata.keys()):
                if key.endswith('_shape'):
                    metadata[key] = list(metadata[key])
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Metadata saved to: {metadata_path}")
            print(f"Total files saved: {len(saved_files)}")
        
        return result
    
    def _extract_window_features(self, video_tensor: torch.Tensor, feature_types: List[str]) -> Dict[str, torch.Tensor]:
        """Extract features from a single video window"""
        features = {}
        
        # Get backbone features (patch-level before pooling)
        x = video_tensor
        B = x.size(0)
        x = self.model.patch_embed(x)
        if self.model.pos_embed is not None:
            x = x + self.model.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)  # [B, num_patches, embed_dim]
        
        # Patch features (before pooling)
        if 'patch' in feature_types:
            features['patch'] = x.squeeze(0)  # [num_patches, embed_dim]
        
        # Pooled shared features
        if hasattr(self.model, 'fc_norm') and self.model.fc_norm is not None:
            pooled_features = self.model.fc_norm(x.mean(1))  # [B, embed_dim]
        else:
            pooled_features = x.mean(1)  # [B, embed_dim]
        
        if 'shared' in feature_types:
            features['shared'] = pooled_features.squeeze(0)  # [embed_dim]
        
        # Head features - save LH and RH separately
        if 'head' in feature_types:
            lh_head_features = self.model.lh_head(pooled_features)
            rh_head_features = self.model.rh_head(pooled_features)
            # Store as separate features instead of concatenating
            features['lh_head'] = lh_head_features.squeeze(0)  # [lh_classes]
            features['rh_head'] = rh_head_features.squeeze(0)  # [rh_classes]
        
        # Separate hand predictions
        if 'predictions' in feature_types:
            predictions = self.model(video_tensor, hand_type='both')
            lh_pred = predictions['lh_pred'].squeeze(0)  # [lh_classes]
            rh_pred = predictions['rh_pred'].squeeze(0)  # [rh_classes]
            # Store as separate features
            features['lh_predictions'] = lh_pred
            features['rh_predictions'] = rh_pred
        
        return features
    
    def extract_batch_features(self, 
                             video_dir: str,
                             output_dir: str,
                             stride: int = 1,
                             feature_types: List[str] = ['shared', 'head'],
                             video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv']) -> Dict[str, Any]:
        """
        Extract frame-wise features from a directory of videos
        
        Args:
            video_dir: Directory containing videos
            output_dir: Directory to save features
            stride: Stride for frame sampling
            feature_types: Types of features to extract
            video_extensions: Video file extensions to process
            
        Returns:
            Dictionary containing processing results
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.rglob(f"*{ext}"))
        
        print(f"Found {len(video_files)} videos in {video_dir}")
        
        results = {}
        failed_videos = []
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            try:
                # Generate output path structure
                relative_path = video_file.relative_to(video_dir)
                # Create subdirectory structure in output
                output_subdir = output_dir / relative_path.parent
                video_name = video_file.stem
                # Use a dummy path since we'll save to organized folders
                dummy_output_path = output_subdir / f"{video_name}.npy"
                
                # Extract features
                result = self.extract_frame_features(
                    str(video_file), 
                    str(dummy_output_path),
                    stride,
                    feature_types
                )
                
                if result is not None:
                    results[str(video_file)] = result
                    print(f"Successfully processed: {video_file}")
                else:
                    failed_videos.append(str(video_file))
                    
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                failed_videos.append(str(video_file))
        
        # Save summary
        summary = {
            'total_videos': len(video_files),
            'successful': len(results),
            'failed': len(failed_videos),
            'failed_videos': failed_videos,
            'output_directory': str(output_dir),
            'feature_types': feature_types,
            'model_info': {
                'model_name': self.model_name,
                'num_frames': self.num_frames,
                'sampling_rate': self.sampling_rate,
                'input_size': self.input_size,
                'stride': stride,
                'lh_num_classes': self.lh_num_classes,
                'rh_num_classes': self.rh_num_classes
            }
        }
        
        summary_path = output_dir / 'alternating_frame_extraction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nFrame-wise feature extraction completed!")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Summary saved to: {summary_path}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Extract frame-wise features from alternating hands VideoMAEv2 model')
    
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
    parser.add_argument('--video_path', type=str,
                        help='Path to single video file')
    parser.add_argument('--video_dir', type=str,
                        help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted features')
    
    # Feature extraction arguments
    parser.add_argument('--feature_types', type=str, nargs='+', 
                        default=['shared', 'head'],
                        choices=['shared', 'patch', 'head', 'predictions'],
                        help='Types of features to extract')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for frame sampling (1 for every frame)')
    
    # Model parameters
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames in each window')
    parser.add_argument('--sampling_rate', type=int, default=1,
                        help='Frame sampling rate within window')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process all videos in directory')
    
    args = parser.parse_args()
    
    # Initialize feature extractor
    print("Initializing alternating hands frame feature extractor...")
    extractor = AlternatingHandsFrameFeatureExtractor(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        input_size=args.input_size,
        lh_num_classes=args.lh_num_classes,
        rh_num_classes=args.rh_num_classes
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.batch_mode or args.video_dir:
        # Batch processing
        if not args.video_dir:
            raise ValueError("--video_dir required for batch processing")
        
        print(f"Processing videos in: {args.video_dir}")
        summary = extractor.extract_batch_features(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            stride=args.stride,
            feature_types=args.feature_types
        )
        
    elif args.video_path:
        # Single video processing
        print(f"Processing single video: {args.video_path}")
        output_path = Path(args.output_dir) / f"{Path(args.video_path).stem}.npz"
        
        result = extractor.extract_frame_features(
            video_path=args.video_path,
            save_path=str(output_path),
            stride=args.stride,
            feature_types=args.feature_types
        )
        
        if result:
            print(f"\nFeature extraction completed!")
            print(f"Video length: {result['video_length']} frames")
            print(f"Number of windows: {result['num_windows']}")
            for feature_type in args.feature_types:
                if feature_type in result['features']:
                    shape = result['features'][feature_type].shape
                    print(f"{feature_type} features shape: {shape}")
        else:
            print("Feature extraction failed!")
    
    else:
        raise ValueError("Must specify either --video_path or --video_dir")
    
    print(f"Features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
