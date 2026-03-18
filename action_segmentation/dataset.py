"""
Dataset classes for dual-hand action segmentation
Handles loading visual features and labels for both hands with temporal augmentation
"""

import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
from utils import get_labels_start_end_time, get_boundary_seq


# ============================================================================
# Data Loading
# ============================================================================

def get_dual_hand_data_dict(
    feature_dir_lh: str,
    feature_dir_rh: str,
    label_dir_lh: str,
    label_dir_rh: str,
    video_list: List[str],
    event_list: List[str],
    sample_rate: int = 4,
    temporal_aug: bool = True,
    boundary_smooth: Optional[float] = None
) -> Dict:
    """
    Load data for dual-hand action segmentation with hand-specific features
    
    Args:
        feature_dir_lh: Directory containing left hand video features (.npy)
        feature_dir_rh: Directory containing right hand video features (.npy)
        label_dir_lh: Directory containing left hand labels (.txt)
        label_dir_rh: Directory containing right hand labels (.txt)
        video_list: List of video names (without extension)
        event_list: List of action class names
        sample_rate: Temporal sampling rate (default: 4)
        temporal_aug: Whether to apply temporal augmentation (default: True)
        boundary_smooth: Gaussian smoothing sigma for boundaries (None = no smoothing)
        
    Returns:
        Dictionary mapping video names to their features and labels for both hands
    """
    assert sample_rate > 0, "Sample rate must be positive"
    
    # Initialize data dictionary
    data_dict = {
        video: {
            'feature_lh': None,
            'feature_rh': None,
            'event_seq_raw_lh': None,
            'event_seq_ext_lh': None,
            'boundary_seq_raw_lh': None,
            'boundary_seq_ext_lh': None,
            'event_seq_raw_rh': None,
            'event_seq_ext_rh': None,
            'boundary_seq_raw_rh': None,
            'boundary_seq_ext_rh': None,
        }
        for video in video_list
    }
    
    print(f'Loading Dual-Hand Dataset ({len(video_list)} videos)...')
    
    for video in tqdm(video_list):
        # File paths
        feature_file_lh = os.path.join(feature_dir_lh, f'{video}.npy')
        feature_file_rh = os.path.join(feature_dir_rh, f'{video}.npy')
        event_file_lh = os.path.join(label_dir_lh, f'{video}.txt')
        event_file_rh = os.path.join(label_dir_rh, f'{video}.txt')
        
        # Load labels for both hands
        event_lh = np.loadtxt(event_file_lh, dtype=str)
        event_rh = np.loadtxt(event_file_rh, dtype=str)
        frame_num = len(event_lh)
        
        # Ensure both hands have same number of frames
        assert len(event_lh) == len(event_rh), \
            f"Frame mismatch for video {video}: LH={len(event_lh)}, RH={len(event_rh)}"
        
        # Process left hand labels
        event_seq_raw_lh = np.zeros((frame_num,))
        for i in range(frame_num):
            if event_lh[i] in event_list:
                event_seq_raw_lh[i] = event_list.index(event_lh[i])
            else:
                event_seq_raw_lh[i] = -100  # Ignore index for background
        
        # Process right hand labels
        event_seq_raw_rh = np.zeros((frame_num,))
        for i in range(frame_num):
            if event_rh[i] in event_list:
                event_seq_raw_rh[i] = event_list.index(event_rh[i])
            else:
                event_seq_raw_rh[i] = -100  # Ignore index for background
        
        # Generate boundary sequences
        boundary_seq_raw_lh = get_boundary_seq(event_seq_raw_lh, boundary_smooth)
        boundary_seq_raw_rh = get_boundary_seq(event_seq_raw_rh, boundary_smooth)
        
        # Load features for both hands
        feature_lh = np.load(feature_file_lh, allow_pickle=True)
        feature_rh = np.load(feature_file_rh, allow_pickle=True)
        
        # Process left hand features - normalize shape to [batch, T, F]
        if len(feature_lh.shape) == 3:
            feature_lh = np.swapaxes(feature_lh, 0, 1)  # [T, batch, F] -> [batch, T, F]
        elif len(feature_lh.shape) == 2:
            feature_lh = np.swapaxes(feature_lh, 0, 1)  # [F, T] -> [T, F]
            feature_lh = np.expand_dims(feature_lh, 0)  # [T, F] -> [1, T, F]
        else:
            raise ValueError(f'Invalid LH Feature shape: {feature_lh.shape}')
        
        # Process right hand features
        if len(feature_rh.shape) == 3:
            feature_rh = np.swapaxes(feature_rh, 0, 1)
        elif len(feature_rh.shape) == 2:
            feature_rh = np.swapaxes(feature_rh, 0, 1)
            feature_rh = np.expand_dims(feature_rh, 0)
        else:
            raise ValueError(f'Invalid RH Feature shape: {feature_rh.shape}')
        
        # Verify shapes match
        assert feature_lh.shape[1] == event_seq_raw_lh.shape[0], \
            f"LH feature-label mismatch: {feature_lh.shape[1]} vs {event_seq_raw_lh.shape[0]}"
        assert feature_rh.shape[1] == event_seq_raw_rh.shape[0], \
            f"RH feature-label mismatch: {feature_rh.shape[1]} vs {event_seq_raw_rh.shape[0]}"
        
        # Temporal augmentation: create multiple temporal views
        if temporal_aug:
            # Sample at different offsets to create temporal augmentation
            feature_lh = [
                feature_lh[:, offset::sample_rate, :]
                for offset in range(sample_rate)
            ]
            feature_rh = [
                feature_rh[:, offset::sample_rate, :]
                for offset in range(sample_rate)
            ]
            
            event_seq_ext_lh = [
                event_seq_raw_lh[offset::sample_rate]
                for offset in range(sample_rate)
            ]
            event_seq_ext_rh = [
                event_seq_raw_rh[offset::sample_rate]
                for offset in range(sample_rate)
            ]
            
            boundary_seq_ext_lh = [
                boundary_seq_raw_lh[offset::sample_rate]
                for offset in range(sample_rate)
            ]
            boundary_seq_ext_rh = [
                boundary_seq_raw_rh[offset::sample_rate]
                for offset in range(sample_rate)
            ]
        else:
            # No temporal augmentation - single sampling
            feature_lh = [feature_lh[:, ::sample_rate, :]]
            feature_rh = [feature_rh[:, ::sample_rate, :]]
            event_seq_ext_lh = [event_seq_raw_lh[::sample_rate]]
            event_seq_ext_rh = [event_seq_raw_rh[::sample_rate]]
            boundary_seq_ext_lh = [boundary_seq_raw_lh[::sample_rate]]
            boundary_seq_ext_rh = [boundary_seq_raw_rh[::sample_rate]]
        
        # Convert to tensors and store
        data_dict[video]['feature_lh'] = [torch.from_numpy(f).float() for f in feature_lh]
        data_dict[video]['feature_rh'] = [torch.from_numpy(f).float() for f in feature_rh]
        
        data_dict[video]['event_seq_raw_lh'] = torch.from_numpy(event_seq_raw_lh).float()
        data_dict[video]['event_seq_ext_lh'] = [torch.from_numpy(e).float() for e in event_seq_ext_lh]
        data_dict[video]['boundary_seq_raw_lh'] = torch.from_numpy(boundary_seq_raw_lh).float()
        data_dict[video]['boundary_seq_ext_lh'] = [torch.from_numpy(b).float() for b in boundary_seq_ext_lh]
        
        data_dict[video]['event_seq_raw_rh'] = torch.from_numpy(event_seq_raw_rh).float()
        data_dict[video]['event_seq_ext_rh'] = [torch.from_numpy(e).float() for e in event_seq_ext_rh]
        data_dict[video]['boundary_seq_raw_rh'] = torch.from_numpy(boundary_seq_raw_rh).float()
        data_dict[video]['boundary_seq_ext_rh'] = [torch.from_numpy(b).float() for b in boundary_seq_ext_rh]
    
    print(f'✓ Loaded {len(video_list)} videos')
    return data_dict


def restore_full_sequence(x: np.ndarray, full_len: int, left_offset: int, 
                         right_offset: int, sample_rate: int) -> np.ndarray:
    """
    Restore full sequence from temporally sampled predictions using interpolation
    
    Args:
        x: Sampled predictions
        full_len: Length of original full sequence
        left_offset: Left padding offset
        right_offset: Right padding offset
        sample_rate: Temporal sampling rate
        
    Returns:
        Full sequence predictions
    """
    # Generate frame ticks for sampled and full sequences
    frame_ticks = np.arange(left_offset, full_len - right_offset, sample_rate)
    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1] + 1, 1)
    
    # Nearest neighbor interpolation
    interp_func = interp1d(frame_ticks, x, kind='nearest')
    
    assert len(frame_ticks) == len(x), "Frame ticks length mismatch"
    
    # Restore full sequence
    out = np.zeros((full_len))
    out[:frame_ticks[0]] = x[0]  # Pad left
    out[frame_ticks[0]:frame_ticks[-1] + 1] = interp_func(full_ticks)  # Interpolate
    out[frame_ticks[-1] + 1:] = x[-1]  # Pad right
    
    return out


# ============================================================================
# Dataset Class
# ============================================================================

class DualHandVideoFeatureDataset(Dataset):
    """
    PyTorch Dataset for dual-hand action segmentation
    
    Supports two modes:
    - train: Returns single augmented view with random temporal/spatial selection
    - test: Returns all temporal views for averaging
    """
    
    def __init__(self, data_dict: Dict, class_num: int, mode: str = 'train'):
        """
        Args:
            data_dict: Dictionary from get_dual_hand_data_dict()
            class_num: Number of action classes
            mode: 'train' or 'test'
        """
        super().__init__()
        
        assert mode in ['train', 'test'], f"Invalid mode: {mode}"
        
        self.data_dict = data_dict
        self.class_num = class_num
        self.mode = mode
        self.video_list = list(self.data_dict.keys())
    
    def get_class_weights(self, hand: str = 'lh') -> np.ndarray:
        """
        Compute class weights for balancing loss
        
        Args:
            hand: 'lh' for left hand, 'rh' for right hand
            
        Returns:
            Class weights array
        """
        assert hand in ['lh', 'rh'], f"Invalid hand: {hand}"
        
        # Concatenate all event sequences for the specified hand
        if hand == 'lh':
            full_event_seq = np.concatenate([
                self.data_dict[v]['event_seq_raw_lh'].numpy()
                for v in self.video_list
            ])
        else:
            full_event_seq = np.concatenate([
                self.data_dict[v]['event_seq_raw_rh'].numpy()
                for v in self.video_list
            ])
        
        # Count occurrences of each class
        class_counts = np.zeros((self.class_num,))
        for c in range(self.class_num):
            class_counts[c] = (full_event_seq == c).sum()
        
        # Compute inverse frequency weights (add 10 for smoothing)
        class_weights = class_counts.sum() / ((class_counts + 10) * self.class_num)
        
        return class_weights
    
    def __len__(self) -> int:
        return len(self.video_list)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single video sample
        
        Returns:
            Tuple of (feature_lh, feature_rh, label_lh, boundary_lh, label_rh, boundary_rh, video_name)
        """
        video = self.video_list[idx]
        
        if self.mode == 'train':
            # Training mode: random temporal and spatial augmentation
            feature_lh = self.data_dict[video]['feature_lh']
            feature_rh = self.data_dict[video]['feature_rh']
            label_lh = self.data_dict[video]['event_seq_ext_lh']
            boundary_lh = self.data_dict[video]['boundary_seq_ext_lh']
            label_rh = self.data_dict[video]['event_seq_ext_rh']
            boundary_rh = self.data_dict[video]['boundary_seq_ext_rh']
            
            # Random temporal augmentation selection
            temporal_aug_num = len(feature_lh)
            temporal_rid = random.randint(0, temporal_aug_num - 1)
            feature_lh = feature_lh[temporal_rid]
            feature_rh = feature_rh[temporal_rid]
            label_lh = label_lh[temporal_rid]
            boundary_lh = boundary_lh[temporal_rid]
            label_rh = label_rh[temporal_rid]
            boundary_rh = boundary_rh[temporal_rid]
            
            # Random spatial augmentation selection (if multiple views exist)
            spatial_aug_num = feature_lh.shape[0]
            spatial_rid = random.randint(0, spatial_aug_num - 1)
            feature_lh = feature_lh[spatial_rid]  # [T, F]
            feature_rh = feature_rh[spatial_rid]  # [T, F]
            
            # Transpose to [F, T] for model input
            feature_lh = feature_lh.T
            feature_rh = feature_rh.T
            
            # Normalize boundaries
            boundary_lh = boundary_lh.unsqueeze(0)
            boundary_lh /= boundary_lh.max() if boundary_lh.max() > 0 else 1
            
            boundary_rh = boundary_rh.unsqueeze(0)
            boundary_rh /= boundary_rh.max() if boundary_rh.max() > 0 else 1
        
        elif self.mode == 'test':
            # Test mode: return all temporal views for averaging
            feature_lh = self.data_dict[video]['feature_lh']
            feature_rh = self.data_dict[video]['feature_rh']
            label_lh = self.data_dict[video]['event_seq_raw_lh']
            boundary_lh = self.data_dict[video]['boundary_seq_ext_lh']
            label_rh = self.data_dict[video]['event_seq_raw_rh']
            boundary_rh = self.data_dict[video]['boundary_seq_ext_rh']
            
            # Transpose features: [batch, T, F] -> [batch, F, T]
            feature_lh = [torch.swapaxes(f, 1, 2) for f in feature_lh]
            feature_rh = [torch.swapaxes(f, 1, 2) for f in feature_rh]
            
            # Add batch dimension to labels
            label_lh = label_lh.unsqueeze(0)  # [1, T]
            label_rh = label_rh.unsqueeze(0)  # [1, T]
            
            # Add batch and channel dimensions to boundaries
            boundary_lh = [b.unsqueeze(0).unsqueeze(0) for b in boundary_lh]  # [1, 1, T]
            boundary_rh = [b.unsqueeze(0).unsqueeze(0) for b in boundary_rh]  # [1, 1, T]
        
        return (feature_lh, feature_rh,
                label_lh, boundary_lh,
                label_rh, boundary_rh,
                video)

