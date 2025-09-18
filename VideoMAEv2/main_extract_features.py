#!/usr/bin/env python3
"""
Example script for frame-wise feature extraction using VideoMAEv2
"""

import torch
import os
from extract_frame_features import VideoMAEv2FrameFeatureExtractor

def main():
    # Example 1: Extract frame-wise features from a single video
    print("=== Frame-wise Feature Extraction ===")
    
    # Initialize extractor with your fine-tuned model
    extractor = VideoMAEv2FrameFeatureExtractor(
        model_name='vit_base_patch16_224',
        checkpoint_path='output/havid_lh_v0_whole/checkpoint-best.pth',  # Your checkpoint path
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_frames=16,        # Window size for feature extraction
        sampling_rate=1,      # Sample every frame within the window
        input_size=224,
        num_classes=400       # Your checkpoint was trained for 400 classes
    )
    
    # # Extract frame-wise features from a single video
    # video_path = '/home/hao/mmaction2/VideoMAEv2/IKEA_mmaction/IKEA_mmaction/videos_val/align_leg/0001_align_leg_001.mp4'
    
    # if os.path.exists(video_path):
    #     print(f"Processing video: {video_path}")
        
    #     # Extract features with stride=1 (every frame)
    #     result = extractor.extract_frame_features(
    #         video_path=video_path,
    #         stride=1  # Extract features for every frame position
    #     )
        
    #     if result:
    #         print(f"Frame-wise feature extraction completed!")
    #         print(f"Video length: {result['video_length']} frames")
    #         print(f"Feature tensor shape: {result['features'].shape}")
    #         print(f"Feature dimension: {result['feature_dim']}")
    #         print(f"Output format: [video_length, feature_dim] = [{result['video_length']}, {result['feature_dim']}]")
            
    #         # Access the features
    #         features = result['features']  # Shape: [video_length, feature_dim]
    #         extracted_positions = result['extracted_positions']
            
    #         print(f"\nFeature details:")
    #         print(f"  - Total frames: {result['video_length']}")
    #         print(f"  - Directly extracted frames: {len(extracted_positions)}")
    #         print(f"  - Feature dimension: {result['feature_dim']}")
    #         print(f"  - Features for frame 0: {features[0][:10]}...")  # First 10 values
    #         print(f"  - Features for frame 10: {features[10][:10]}...")  # First 10 values
            
    #         # Save features to PT file
    #         output_path = "frame_features_example.pt"
    #         extractor._save_features(result, output_path)
    #         print(f"\nFeatures saved to: {output_path}")
            
    #         # Verify the saved file
    #         loaded_features = torch.load(output_path)
    #         print(f"Loaded features shape: {loaded_features.shape}")
    #         print(f"Features match: {torch.allclose(torch.from_numpy(features), loaded_features)}")
    
    # # Example 2: Batch processing for multiple videos
    # print("\n=== Batch Frame-wise Processing ===")
    
    video_directory = '/home/hao/Polyphony/data/havid/videos'
    output_directory = '/home/hao/Polyphony/data/havid/videomae_features/lh_v0'
    
    if os.path.exists(video_directory):
        print(f"Processing videos in: {video_directory}")
        
        # Process with stride=1
        summary = extractor.extract_frame_features_batch(
            video_dir=video_directory,
            output_dir=output_directory,
            stride=1  # Extract features every frames
        )
        
        print(f"Batch processing completed!")
        print(f"Total videos: {summary['total_videos']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        
        # Show some output files
        if os.path.exists(output_directory):
            pt_files = list(Path(output_directory).glob("**/*.pt"))
            print(f"\nGenerated PT files:")
            for pt_file in pt_files[:5]:  # Show first 5 files
                features = torch.load(pt_file)
                print(f"  {pt_file}: {features.shape}")
    
    # # Example 3: Different sampling strategies
    # print("\n=== Sampling Strategy Examples ===")
    
    # Dense sampling (every frame)
    # print("Dense sampling (stride=1):")
    # if os.path.exists(video_path):
    #     result_dense = extractor.extract_frame_features(
    #         video_path=video_path,
    #         stride=1
    #     )
    #     if result_dense:
    #         print(f"  Features: {result_dense['features'].shape}")
    #         print(f"  Extracted positions: {len(result_dense['extracted_positions'])}")
    
    # Sparse sampling (every 4th frame)
    # print("Sparse sampling (stride=4):")
    # if os.path.exists(video_path):
    #     result_sparse = extractor.extract_frame_features(
    #         video_path=video_path,
    #         stride=4
    #     )
    #     if result_sparse:
    #         print(f"  Features: {result_sparse['features'].shape}")
    #         print(f"  Extracted positions: {len(result_sparse['extracted_positions'])}")
    
    print("\n=== Frame-wise Feature Extraction Complete ===")
    print("Key benefits of this approach:")
    print("1. Dense temporal representation: [video_length, feature_dim]")
    print("2. Every frame gets a feature vector")
    print("3. Suitable for temporal action localization")
    print("4. Compatible with sliding window approaches")
    print("5. Features saved as PyTorch tensors (.pt files)")

if __name__ == '__main__':
    from pathlib import Path
    main()
