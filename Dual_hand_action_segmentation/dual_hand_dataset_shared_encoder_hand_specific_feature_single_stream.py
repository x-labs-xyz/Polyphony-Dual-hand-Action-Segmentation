import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from utils import get_labels_start_end_time
from scipy.ndimage import gaussian_filter1d

def get_dual_hand_data_dict(feature_dir_lh, feature_dir_rh, label_dir_lh, label_dir_rh, video_list, event_list, 
                           sample_rate=4, temporal_aug=True, boundary_smooth=None, single_stream=False):
    """
    Load data for dual-hand action segmentation with hand-specific features.
    
    Args:
        feature_dir_lh: Directory containing left hand video features
        feature_dir_rh: Directory containing right hand video features
        label_dir_lh: Directory containing left hand labels  
        label_dir_rh: Directory containing right hand labels
        video_list: List of video names
        event_list: List of action classes
        sample_rate: Temporal sampling rate
        temporal_aug: Whether to apply temporal augmentation
        boundary_smooth: Gaussian smoothing parameter for boundaries
        single_stream: If True, only load LH data and skip RH entirely
    """
    
    assert(sample_rate > 0)
        
    data_dict = {k: {
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
        } for k in video_list
    }
    
    if single_stream:
        print(f'Loading Single-Stream (LH only) Dataset ...')
    else:
        print(f'Loading Dual-Hand Dataset ...')
    
    for video in tqdm(video_list):
        
        feature_file_lh = os.path.join(feature_dir_lh, '{}.npy'.format(video))
        event_file_lh = os.path.join(label_dir_lh, '{}.txt'.format(video))

        # Load left hand labels
        event_lh = np.loadtxt(event_file_lh, dtype=str)
        frame_num = len(event_lh)
        
        # Only load RH if not single_stream
        if not single_stream:
            feature_file_rh = os.path.join(feature_dir_rh, '{}.npy'.format(video))
            event_file_rh = os.path.join(label_dir_rh, '{}.txt'.format(video))
            event_rh = np.loadtxt(event_file_rh, dtype=str)
            # Ensure both hands have same number of frames
            assert len(event_lh) == len(event_rh), f"Frame mismatch for video {video}: LH={len(event_lh)}, RH={len(event_rh)}"
                
        # Process left hand labels
        event_seq_raw_lh = np.zeros((frame_num,))
        for i in range(frame_num):
            if event_lh[i] in event_list:
                event_seq_raw_lh[i] = event_list.index(event_lh[i])
            else:
                event_seq_raw_lh[i] = -100  # background

        # Process right hand labels (only if not single_stream)
        if not single_stream:
            event_seq_raw_rh = np.zeros((frame_num,))
            for i in range(frame_num):
                if event_rh[i] in event_list:
                    event_seq_raw_rh[i] = event_list.index(event_rh[i])
                else:
                    event_seq_raw_rh[i] = -100  # background
        else:
            # In single_stream mode, RH data will mirror LH
            event_seq_raw_rh = None

        # Generate boundary sequences
        boundary_seq_raw_lh = get_boundary_seq(event_seq_raw_lh, boundary_smooth)
        if not single_stream:
            boundary_seq_raw_rh = get_boundary_seq(event_seq_raw_rh, boundary_smooth)
        else:
            boundary_seq_raw_rh = None

        # Load features
        feature_lh = np.load(feature_file_lh, allow_pickle=True)
        if not single_stream:
            feature_rh = np.load(feature_file_rh, allow_pickle=True)
        else:
            feature_rh = None
        
        # Process left hand features
        if len(feature_lh.shape) == 3:
            feature_lh = np.swapaxes(feature_lh, 0, 1)  
        elif len(feature_lh.shape) == 2:
            feature_lh = np.swapaxes(feature_lh, 0, 1)
            feature_lh = np.expand_dims(feature_lh, 0)
        else:
            raise Exception('Invalid LH Feature.')
            
        # Process right hand features (only if not single_stream)
        if not single_stream:
            if len(feature_rh.shape) == 3:
                feature_rh = np.swapaxes(feature_rh, 0, 1)  
            elif len(feature_rh.shape) == 2:
                feature_rh = np.swapaxes(feature_rh, 0, 1)
                feature_rh = np.expand_dims(feature_rh, 0)
            else:
                raise Exception('Invalid RH Feature.')
                    
        assert(feature_lh.shape[1] == event_seq_raw_lh.shape[0])
        assert(feature_lh.shape[1] == boundary_seq_raw_lh.shape[0])
        if not single_stream:
            assert(feature_rh.shape[1] == event_seq_raw_rh.shape[0])
            assert(feature_rh.shape[1] == boundary_seq_raw_rh.shape[0])

        # Temporal augmentation: create multiple temporal views                                 
        if temporal_aug:
            feature_lh = [
                feature_lh[:,offset::sample_rate,:]
                for offset in range(sample_rate)
            ]
            
            event_seq_ext_lh = [
                event_seq_raw_lh[offset::sample_rate]
                for offset in range(sample_rate)
            ]

            boundary_seq_ext_lh = [
                boundary_seq_raw_lh[offset::sample_rate]
                for offset in range(sample_rate)
            ]
            
            # Only process RH if not single_stream
            if not single_stream:
                feature_rh = [
                    feature_rh[:,offset::sample_rate,:]
                    for offset in range(sample_rate)
                ]
                event_seq_ext_rh = [
                    event_seq_raw_rh[offset::sample_rate]
                    for offset in range(sample_rate)
                ]
                boundary_seq_ext_rh = [
                    boundary_seq_raw_rh[offset::sample_rate]
                    for offset in range(sample_rate)
                ]
            else:
                feature_rh = None
                event_seq_ext_rh = None
                boundary_seq_ext_rh = None
                        
        else:
            feature_lh = [feature_lh[:,::sample_rate,:]]
            event_seq_ext_lh = [event_seq_raw_lh[::sample_rate]]
            boundary_seq_ext_lh = [boundary_seq_raw_lh[::sample_rate]]
            
            if not single_stream:
                feature_rh = [feature_rh[:,::sample_rate,:]]
                event_seq_ext_rh = [event_seq_raw_rh[::sample_rate]]
                boundary_seq_ext_rh = [boundary_seq_raw_rh[::sample_rate]]
            else:
                feature_rh = None
                event_seq_ext_rh = None
                boundary_seq_ext_rh = None

        # Store processed data - LH always
        data_dict[video]['feature_lh'] = [torch.from_numpy(i).float() for i in feature_lh]
        data_dict[video]['event_seq_raw_lh'] = torch.from_numpy(event_seq_raw_lh).float()
        data_dict[video]['event_seq_ext_lh'] = [torch.from_numpy(i).float() for i in event_seq_ext_lh]
        data_dict[video]['boundary_seq_raw_lh'] = torch.from_numpy(boundary_seq_raw_lh).float()
        data_dict[video]['boundary_seq_ext_lh'] = [torch.from_numpy(i).float() for i in boundary_seq_ext_lh]
        
        # Store RH data only if not single_stream
        if not single_stream:
            data_dict[video]['feature_rh'] = [torch.from_numpy(i).float() for i in feature_rh]
            data_dict[video]['event_seq_raw_rh'] = torch.from_numpy(event_seq_raw_rh).float()
            data_dict[video]['event_seq_ext_rh'] = [torch.from_numpy(i).float() for i in event_seq_ext_rh]
            data_dict[video]['boundary_seq_raw_rh'] = torch.from_numpy(boundary_seq_raw_rh).float()
            data_dict[video]['boundary_seq_ext_rh'] = [torch.from_numpy(i).float() for i in boundary_seq_ext_rh]
        else:
            # In single_stream, RH data will be mirrored from LH at runtime
            data_dict[video]['feature_rh'] = None
            data_dict[video]['event_seq_raw_rh'] = None
            data_dict[video]['event_seq_ext_rh'] = None
            data_dict[video]['boundary_seq_raw_rh'] = None
            data_dict[video]['boundary_seq_ext_rh'] = None
        
    return data_dict

def get_boundary_seq(event_seq, boundary_smooth=None):
    """Generate boundary sequence from event sequence"""
    boundary_seq = np.zeros_like(event_seq)

    _, start_times, end_times = get_labels_start_end_time([str(int(i)) for i in event_seq])
    boundaries = start_times[1:]
    if len(boundaries) > 0:
        assert min(boundaries) > 0
        boundary_seq[boundaries] = 1
        boundary_seq[[i-1 for i in boundaries]] = 1

    if boundary_smooth is not None:
        boundary_seq = gaussian_filter1d(boundary_seq, boundary_smooth)
        
        # Normalize. This is ugly.
        temp_seq = np.zeros_like(boundary_seq)
        temp_seq[temp_seq.shape[0] // 2] = 1
        temp_seq[temp_seq.shape[0] // 2 - 1] = 1
        norm_z = gaussian_filter1d(temp_seq, boundary_smooth).max()
        boundary_seq[boundary_seq > norm_z] = norm_z
        boundary_seq /= boundary_seq.max()

    return boundary_seq

def restore_full_sequence(x, full_len, left_offset, right_offset, sample_rate):
    """Restore full sequence from sampled sequence"""
    frame_ticks = np.arange(left_offset, full_len-right_offset, sample_rate)
    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1]+1, 1)

    interp_func = interp1d(frame_ticks, x, kind='nearest')
    
    assert(len(frame_ticks) == len(x)) # Rethink this
    
    out = np.zeros((full_len))
    out[:frame_ticks[0]] = x[0]
    out[frame_ticks[0]:frame_ticks[-1]+1] = interp_func(full_ticks)
    out[frame_ticks[-1]+1:] = x[-1]

    return out

class DualHandVideoFeatureDataset(Dataset):
    """Dataset for dual-hand action segmentation"""
    
    def __init__(self, data_dict, class_num, mode):
        super(DualHandVideoFeatureDataset, self).__init__()
        
        assert(mode in ['train', 'test'])
        
        self.data_dict = data_dict
        self.class_num = class_num
        self.mode = mode
        self.video_list = [i for i in self.data_dict.keys()]
        
    def get_class_weights(self, hand='lh'):
        """Get class weights for balancing (can specify which hand to use)"""
        if hand == 'lh':
            full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw_lh'] for v in self.video_list])
        else:
            # Check if RH data exists (not single_stream mode)
            if self.data_dict[self.video_list[0]]['event_seq_raw_rh'] is not None:
                full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw_rh'] for v in self.video_list])
            else:
                # In single_stream, use LH weights for RH as well
                full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw_lh'] for v in self.video_list])
            
        class_counts = np.zeros((self.class_num,))
        for c in range(self.class_num):
            class_counts[c] = (full_event_seq == c).sum()
                    
        class_weights = class_counts.sum() / ((class_counts + 10) * self.class_num)
        return class_weights

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video = self.video_list[idx]

        if self.mode == 'train':
            feature_lh = self.data_dict[video]['feature_lh']
            label_lh = self.data_dict[video]['event_seq_ext_lh']
            boundary_lh = self.data_dict[video]['boundary_seq_ext_lh']
            
            # Check if RH data exists (not single_stream mode)
            feature_rh = self.data_dict[video]['feature_rh']
            label_rh = self.data_dict[video]['event_seq_ext_rh']
            boundary_rh = self.data_dict[video]['boundary_seq_ext_rh']

            temporal_aug_num = len(feature_lh)
            temporal_rid = random.randint(0, temporal_aug_num - 1)
            feature_lh = feature_lh[temporal_rid]
            label_lh = label_lh[temporal_rid]
            boundary_lh = boundary_lh[temporal_rid]
            
            # Handle RH data - mirror from LH if None (single_stream mode)
            if feature_rh is not None:
                feature_rh = feature_rh[temporal_rid]
                label_rh = label_rh[temporal_rid]
                boundary_rh = boundary_rh[temporal_rid]
            else:
                # Mirror LH data to RH (will be handled by trainer, just pass LH)
                feature_rh = feature_lh.clone()
                label_rh = label_lh.clone()
                boundary_rh = boundary_lh.clone()

            spatial_aug_num = feature_lh.shape[0]
            spatial_rid = random.randint(0, spatial_aug_num - 1)
            feature_lh = feature_lh[spatial_rid]
            feature_rh = feature_rh[spatial_rid]
            
            feature_lh = feature_lh.T   # F x T
            feature_rh = feature_rh.T   # F x T

            boundary_lh = boundary_lh.unsqueeze(0)
            boundary_lh /= boundary_lh.max() if boundary_lh.max() > 0 else 1  # normalize again
            
            boundary_rh = boundary_rh.unsqueeze(0)
            boundary_rh /= boundary_rh.max() if boundary_rh.max() > 0 else 1  # normalize again
            
        elif self.mode == 'test':
            feature_lh = self.data_dict[video]['feature_lh']
            label_lh = self.data_dict[video]['event_seq_raw_lh']
            boundary_lh = self.data_dict[video]['boundary_seq_ext_lh']
            
            # Check if RH data exists (not single_stream mode)
            feature_rh = self.data_dict[video]['feature_rh']
            label_rh = self.data_dict[video]['event_seq_raw_rh']
            boundary_rh = self.data_dict[video]['boundary_seq_ext_rh']

            feature_lh = [torch.swapaxes(i, 1, 2) for i in feature_lh]  # [10 x F x T]
            label_lh = label_lh.unsqueeze(0)   # 1 X T'  
            boundary_lh = [i.unsqueeze(0).unsqueeze(0) for i in boundary_lh]   # [1 x 1 x T]
            
            # Handle RH data - mirror from LH if None (single_stream mode)
            if feature_rh is not None:
                feature_rh = [torch.swapaxes(i, 1, 2) for i in feature_rh]  # [10 x F x T]
                label_rh = label_rh.unsqueeze(0)   # 1 X T'  
                boundary_rh = [i.unsqueeze(0).unsqueeze(0) for i in boundary_rh]   # [1 x 1 x T]
            else:
                # Mirror LH data to RH
                feature_rh = [f.clone() for f in feature_lh]
                label_rh = label_lh.clone()
                boundary_rh = [b.clone() for b in boundary_lh]  

        return (feature_lh, feature_rh,
                label_lh, boundary_lh, 
                label_rh, boundary_rh, 
                video)