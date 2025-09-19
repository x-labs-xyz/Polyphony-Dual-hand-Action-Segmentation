#!/usr/bin/env python3
"""
VideoMAEv2 Frame-wise Feature Extraction Script

This script extracts frame-wise features from untrimmed videos using a fine-tuned VideoMAEv2 model.
Features are extracted at high temporal resolution and saved as [feature_dim, video_length] tensors.

Usage:
    python extract_frame_features.py --video_path /path/to/video.mp4 --model_path /path/to/checkpoint.pth
    python extract_frame_features.py --video_dir /path/to/videos --model_path /path/to/checkpoint.pth --batch_mode
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

# Import VideoMAEv2 models
import models  # noqa: F401
from dataset.loader import get_video_loader


class VideoMAEv2FrameFeatureExtractor:
    """Frame-wise feature extractor for VideoMAEv2 models"""
    
    def __init__(self, 
                 model_name: str = 'vit_base_patch16_224',
                 checkpoint_path: str = None,
                 device: str = 'cuda',
                 num_frames: int = 16,
                 sampling_rate: int = 1,  # Default to 1 for frame-wise
                 input_size: int = 224,
                 tubelet_size: int = 2,
                 num_classes: int = 400,
                 use_mean_pooling: bool = True,
                 drop_path_rate: float = 0.3,
                 overlap_frames: int = 0):
        """
        Initialize the frame-wise feature extractor
        
        Args:
            model_name: Name of the VideoMAEv2 model
            checkpoint_path: Path to the fine-tuned checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            num_frames: Number of frames to process in each window
            sampling_rate: Temporal sampling rate (1 for frame-wise)
            input_size: Input spatial size
            tubelet_size: Size of temporal tubelets
            num_classes: Number of classes in the fine-tuned model
            use_mean_pooling: Whether to use mean pooling
            drop_path_rate: Drop path rate for regularization
            overlap_frames: Number of overlapping frames between windows
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.input_size = input_size
        self.tubelet_size = tubelet_size
        self.overlap_frames = overlap_frames
        
        # Initialize video loader
        self.video_loader = get_video_loader()
        
        # Initialize transforms
        self.transform = self._get_transforms()
        
        # Load model
        self.model = self._load_model(num_classes, use_mean_pooling, drop_path_rate)
        
    def _get_transforms(self):
        """Get video transformation pipeline for frame-wise processing"""
        return transforms.Compose([
            transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)),  # T H W C -> C T H W
            transforms.Lambda(lambda x: x.float() / 255.0),      # Normalize to [0, 1]
            transforms.Lambda(lambda x: F.interpolate(x, size=(self.input_size, self.input_size), 
                                                    mode='bilinear', align_corners=False)),
            # Keep as C T H W format for the model
        ])
    
    def _load_model(self, num_classes: int, use_mean_pooling: bool, drop_path_rate: float):
        """Load the VideoMAEv2 model"""
        print(f"Loading model: {self.model_name}")
        
        # Create model
        model = create_model(
            self.model_name,
            img_size=self.input_size,
            pretrained=False,
            num_classes=num_classes,
            all_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            drop_path_rate=drop_path_rate,
            use_mean_pooling=use_mean_pooling
        )
        
        # Load checkpoint if provided
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'module' in checkpoint:
                state_dict = checkpoint['module']
            else:
                state_dict = checkpoint
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully")
        else:
            print("No checkpoint provided, using random weights")
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _extract_frame_window(self, video_reader, start_idx: int) -> torch.Tensor:
        """Extract a window of frames starting at start_idx"""
        # Calculate frame indices for the window
        frame_indices = np.arange(start_idx, start_idx + self.num_frames * self.sampling_rate, self.sampling_rate)
        frame_indices = np.clip(frame_indices, 0, len(video_reader) - 1)
        
        # Extract frames
        frames = video_reader.get_batch(frame_indices).asnumpy()
        
        # Convert to tensor and apply transforms
        frames_tensor = torch.from_numpy(frames)  # Shape: [T, H, W, C]
        frames_transformed = self.transform(frames_tensor)  # Shape: [C, T, H, W]
        
        return frames_transformed
    
    def extract_frame_features(self, 
                              video_path: str, 
                              save_path: Optional[str] = None,
                              stride: int = 1) -> Dict[str, Any]:
        """
        Extract frame-wise features from an untrimmed video
        
        Args:
            video_path: Path to the video file
            save_path: Path to save features (optional)
            stride: Stride for frame sampling (1 for every frame)
            
        Returns:
            Dictionary containing features and metadata
        """
        print(f"Processing video: {video_path}")
        
        # Load video
        try:
            video_reader = self.video_loader(video_path)
            video_length = len(video_reader)
            print(f"Video length: {video_length} frames")
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
        
        # Calculate frame indices for feature extraction
        # We'll extract features for frames that can form complete windows
        max_start_idx = max(0, video_length - self.num_frames * self.sampling_rate)
        
        if max_start_idx < 0:
            print(f"Video too short for {self.num_frames} frames with sampling rate {self.sampling_rate}")
            return None
        
        # Generate frame indices for feature extraction
        frame_indices = list(range(0, max_start_idx + 1, stride))
        
        if not frame_indices:
            frame_indices = [0]  # At least one sample
        
        print(f"Extracting features for {len(frame_indices)} frame positions")
        
        # Extract features
        features_list = []
        frame_positions = []
        
        with torch.no_grad():
            for frame_idx in tqdm(frame_indices, desc="Extracting frame features"):
                try:
                    # Extract frame window
                    frames = self._extract_frame_window(video_reader, frame_idx)
                    frames = frames.unsqueeze(0).to(self.device)  # Add batch dimension
                    
                    # Extract features for this window
                    features = self.model.forward_features(frames)
                    features = features.cpu().numpy()
                    
                    # Anchor feature to the center frame of the window
                    center_offset = ((self.num_frames - 1) // 2) * self.sampling_rate
                    center_idx = min(max(frame_idx + center_offset, 0), video_length - 1)
                    
                    features_list.append(features)
                    frame_positions.append(center_idx)
                    
                except Exception as e:
                    print(f"Error processing frame at {frame_idx}: {e}")
                    continue
        
        if not features_list:
            print(f"No features extracted from {video_path}")
            return None
        
        # Stack features
        features_array = np.vstack(features_list)  # Shape: [N, D] where N is number of windows, D is feature dimension
        
        # Create dense feature representation
        # Initialize full video feature tensor (video_length, feature_dim)
        feature_dim = features_array.shape[1]
        full_video_features = np.zeros((video_length, feature_dim), dtype=np.float32)
        
        # Fill in the features at their center-frame positions
        for i, pos in enumerate(frame_positions):
            full_video_features[pos] = features_array[i]
        
        # For frames without direct features, use interpolation or nearest neighbor
        if len(frame_positions) > 1:
            # Interpolate features for frames between extracted positions
            for fidx in range(video_length):
                if fidx not in frame_positions:
                    # Find nearest extracted frame
                    distances = [abs(fidx - pos) for pos in frame_positions]
                    nearest_idx = distances.index(min(distances))
                    nearest_pos = frame_positions[nearest_idx]
                    nearest_feature_idx = frame_positions.index(nearest_pos)
                    full_video_features[fidx] = features_array[nearest_feature_idx]
        
        # Transpose to requested layout: (feature_dim, video_length)
        features_transposed = full_video_features.T
        
        # Prepare output
        result = {
            'video_path': video_path,
            'video_length': video_length,
            'num_extracted_frames': len(frame_positions),
            'feature_dim': feature_dim,
            'features': features_transposed,  # Shape: [feature_dim, video_length]
            'extracted_positions': frame_positions,
            'model_name': self.model_name,
            'num_frames': self.num_frames,
            'sampling_rate': self.sampling_rate,
            'input_size': self.input_size,
            'stride': stride
        }
        
        # Save features if requested
        if save_path:
            self._save_features(result, save_path)
        
        return result
    
    def extract_frame_features_batch(self, 
                                    video_dir: str, 
                                    output_dir: str,
                                    stride: int = 1,
                                    file_extensions: List[str] = None) -> Dict[str, Any]:
        """
        Extract frame-wise features from multiple videos in a directory
        
        Args:
            video_dir: Directory containing video files
            output_dir: Directory to save extracted features
            stride: Stride for frame sampling
            file_extensions: List of video file extensions to process
            
        Returns:
            Dictionary containing processing results
        """
        if file_extensions is None:
            file_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find video files
        video_files = []
        for ext in file_extensions:
            video_files.extend(Path(video_dir).glob(f"*{ext}"))
        
        print(f"Found {len(video_files)} video files")
        
        # Process videos
        results = {}
        failed_videos = []
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            try:
                # Generate output path
                relative_path = video_file.relative_to(video_dir)
                output_path = Path(output_dir) / relative_path.with_suffix('.npy')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Extract features
                result = self.extract_frame_features(
                    str(video_file), 
                    str(output_path),
                    stride
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
            'output_directory': output_dir,
            'model_info': {
                'model_name': self.model_name,
                'num_frames': self.num_frames,
                'sampling_rate': self.sampling_rate,
                'input_size': self.input_size,
                'stride': stride
            }
        }
        
        summary_path = Path(output_dir) / 'frame_extraction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nFrame-wise feature extraction completed!")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Summary saved to: {summary_path}")
        
        return summary
    
    def _save_features(self, result: Dict[str, Any], save_path: str):
        """Save extracted features to NumPy file"""
        # Ensure .npy extension
        if not save_path.endswith('.npy'):
            save_path = str(Path(save_path).with_suffix('.npy'))
        
        # Save features as NumPy array (feature_dim, video_length)
        np.save(save_path, result['features'])
        
        # # Save metadata as JSON
        # metadata_path = save_path.replace('.npy', '_metadata.json')
        # metadata = {k: v for k, v in result.items() if k != 'features'}
        
        # # Convert numpy arrays to lists for JSON serialization
        # if 'extracted_positions' in metadata:
        #     metadata['extracted_positions'] = metadata['extracted_positions']
        
        # with open(metadata_path, 'w') as f:
        #     json.dump(metadata, f, indent=2)
        
        print(f"Features saved to: {save_path}")
        # print(f"Metadata saved to: {metadata_path}")
        print(f"Feature array shape: {result['features'].shape}  # [feature_dim, video_length]")
    
    def get_feature_dimension(self) -> int:
        """Get the dimension of extracted features"""
        # Create a dummy input to get feature dimension
        dummy_input = torch.randn(1, 3, self.num_frames, self.input_size, self.input_size).to(self.device)
        
        with torch.no_grad():
            features = self.model.forward_features(dummy_input)
            feature_dim = features.shape[-1]
        
        return feature_dim


def main():
    parser = argparse.ArgumentParser(description='Extract frame-wise features from videos using VideoMAEv2')
    
    # Model parameters
    parser.add_argument('--model_name', default='vit_base_patch16_224', 
                       choices=['vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224', 
                               'vit_huge_patch16_224', 'vit_giant_patch14_224'],
                       help='VideoMAEv2 model to use')
    parser.add_argument('--checkpoint_path', required=True, help='Path to fine-tuned checkpoint')
    
    # Video processing parameters
    parser.add_argument('--video_path', help='Path to single video file')
    parser.add_argument('--video_dir', help='Directory containing videos (for batch processing)')
    parser.add_argument('--output_dir', help='Output directory for batch processing')
    parser.add_argument('--stride', type=int, default=1, 
                       help='Stride for frame sampling (1 for every frame)')
    
    # Model configuration
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames in each window')
    parser.add_argument('--sampling_rate', type=int, default=1, help='Temporal sampling rate within window')
    parser.add_argument('--input_size', type=int, default=224, help='Input spatial size')
    parser.add_argument('--tubelet_size', type=int, default=2, help='Temporal tubelet size')
    parser.add_argument('--num_classes', type=int, default=400, help='Number of classes in model')
    parser.add_argument('--use_mean_pooling', action='store_true', help='Use mean pooling')
    parser.add_argument('--drop_path_rate', type=float, default=0.3, help='Drop path rate')
    
    # System parameters
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_mode', action='store_true', help='Enable batch processing mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video_path and not args.video_dir:
        parser.error("Either --video_path or --video_dir must be specified")
    
    if args.batch_mode and not args.output_dir:
        parser.error("--output_dir must be specified for batch mode")
    
    # Initialize feature extractor
    print("Initializing VideoMAEv2 Frame-wise Feature Extractor...")
    extractor = VideoMAEv2FrameFeatureExtractor(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        input_size=args.input_size,
        tubelet_size=args.tubelet_size,
        num_classes=args.num_classes,
        use_mean_pooling=args.use_mean_pooling,
        drop_path_rate=args.drop_path_rate
    )
    
    # Print model information
    feature_dim = extractor.get_feature_dimension()
    print(f"Model: {args.model_name}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Input frames: {args.num_frames}")
    print(f"Sampling rate: {args.sampling_rate}")
    print(f"Input size: {args.input_size}x{args.input_size}")
    print(f"Frame stride: {args.stride}")
    
    # Process videos
    if args.batch_mode or args.video_dir:
        # Batch processing
        print(f"\nStarting batch processing...")
        extractor.extract_frame_features_batch(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            stride=args.stride
        )
    else:
        # Single video processing
        print(f"\nProcessing single video...")
        result = extractor.extract_frame_features(
            video_path=args.video_path,
            stride=args.stride
        )
        
        if result:
            print(f"Frame-wise feature extraction completed!")
            print(f"Video: {result['video_path']}")
            print(f"Video length: {result['video_length']} frames")
            print(f"Extracted features for: {result['num_extracted_frames']} frame positions")
            print(f"Feature shape: {result['features'].shape}  # [feature_dim, video_length]")
            print(f"Feature dimension: {result['feature_dim']}")
            print(f"Output format: [feature_dim, video_length] = [{result['feature_dim']}, {result['video_length']}]")


if __name__ == '__main__':
    main()
