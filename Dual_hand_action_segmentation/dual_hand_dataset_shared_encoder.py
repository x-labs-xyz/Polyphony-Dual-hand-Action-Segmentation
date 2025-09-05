import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from utils import get_labels_start_end_time
from scipy.ndimage import gaussian_filter1d

def get_dual_hand_data_dict(feature_dir, label_dir_lh, label_dir_rh, video_list, event_list, 
                           sample_rate=4, temporal_aug=True, boundary_smooth=None):
    """
    Load data for dual-hand action segmentation.
    
    Args:
        feature_dir: Directory containing video features
        label_dir_lh: Directory containing left hand labels  
        label_dir_rh: Directory containing right hand labels
        video_list: List of video names
        event_list: List of action classes
        sample_rate: Temporal sampling rate
        temporal_aug: Whether to apply temporal augmentation
        boundary_smooth: Gaussian smoothing parameter for boundaries
    """
    
    assert(sample_rate > 0)
        
    data_dict = {k: {
        'feature': None,
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
    
    print(f'Loading Dual-Hand Dataset ...')
    
    for video in tqdm(video_list):
        
        feature_file = os.path.join(feature_dir, '{}.npy'.format(video))
        event_file_lh = os.path.join(label_dir_lh, '{}.txt'.format(video))
        event_file_rh = os.path.join(label_dir_rh, '{}.txt'.format(video))

        # Load labels for both hands
        event_lh = np.loadtxt(event_file_lh, dtype=str)
        event_rh = np.loadtxt(event_file_rh, dtype=str)
        frame_num = len(event_lh)
        
        # Ensure both hands have same number of frames
        assert len(event_lh) == len(event_rh), f"Frame mismatch for video {video}: LH={len(event_lh)}, RH={len(event_rh)}"
                
        # Process left hand labels
        event_seq_raw_lh = np.zeros((frame_num,))
        for i in range(frame_num):
            if event_lh[i] in event_list:
                event_seq_raw_lh[i] = event_list.index(event_lh[i])
            else:
                event_seq_raw_lh[i] = -100  # background

        # Process right hand labels
        event_seq_raw_rh = np.zeros((frame_num,))
        for i in range(frame_num):
            if event_rh[i] in event_list:
                event_seq_raw_rh[i] = event_list.index(event_rh[i])
            else:
                event_seq_raw_rh[i] = -100  # background

        # Generate boundary sequences
        boundary_seq_raw_lh = get_boundary_seq(event_seq_raw_lh, boundary_smooth)
        boundary_seq_raw_rh = get_boundary_seq(event_seq_raw_rh, boundary_smooth)

        # Load features
        feature = np.load(feature_file, allow_pickle=True)
        
        if len(feature.shape) == 3:
            feature = np.swapaxes(feature, 0, 1)  
        elif len(feature.shape) == 2:
            feature = np.swapaxes(feature, 0, 1)
            feature = np.expand_dims(feature, 0)
        else:
            raise Exception('Invalid Feature.')
                    
        assert(feature.shape[1] == event_seq_raw_lh.shape[0])
        assert(feature.shape[1] == event_seq_raw_rh.shape[0])
        assert(feature.shape[1] == boundary_seq_raw_lh.shape[0])
        assert(feature.shape[1] == boundary_seq_raw_rh.shape[0])

        # Temporal augmentation: create multiple temporal views                                 
        if temporal_aug:
            feature = [
                feature[:,offset::sample_rate,:]
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
            feature = [feature[:,::sample_rate,:]]
            event_seq_ext_lh = [event_seq_raw_lh[::sample_rate]]
            event_seq_ext_rh = [event_seq_raw_rh[::sample_rate]]
            boundary_seq_ext_lh = [boundary_seq_raw_lh[::sample_rate]]
            boundary_seq_ext_rh = [boundary_seq_raw_rh[::sample_rate]]

        # Store processed data
        data_dict[video]['feature'] = [torch.from_numpy(i).float() for i in feature]
        
        data_dict[video]['event_seq_raw_lh'] = torch.from_numpy(event_seq_raw_lh).float()
        data_dict[video]['event_seq_ext_lh'] = [torch.from_numpy(i).float() for i in event_seq_ext_lh]
        data_dict[video]['boundary_seq_raw_lh'] = torch.from_numpy(boundary_seq_raw_lh).float()
        data_dict[video]['boundary_seq_ext_lh'] = [torch.from_numpy(i).float() for i in boundary_seq_ext_lh]
        
        data_dict[video]['event_seq_raw_rh'] = torch.from_numpy(event_seq_raw_rh).float()
        data_dict[video]['event_seq_ext_rh'] = [torch.from_numpy(i).float() for i in event_seq_ext_rh]
        data_dict[video]['boundary_seq_raw_rh'] = torch.from_numpy(boundary_seq_raw_rh).float()
        data_dict[video]['boundary_seq_ext_rh'] = [torch.from_numpy(i).float() for i in boundary_seq_ext_rh]
        
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
            full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw_rh'] for v in self.video_list])
            
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
            feature = self.data_dict[video]['feature']
            label_lh = self.data_dict[video]['event_seq_ext_lh']
            boundary_lh = self.data_dict[video]['boundary_seq_ext_lh']
            label_rh = self.data_dict[video]['event_seq_ext_rh']
            boundary_rh = self.data_dict[video]['boundary_seq_ext_rh']

            temporal_aug_num = len(feature)
            temporal_rid = random.randint(0, temporal_aug_num - 1)
            feature = feature[temporal_rid]
            label_lh = label_lh[temporal_rid]
            boundary_lh = boundary_lh[temporal_rid]
            label_rh = label_rh[temporal_rid]
            boundary_rh = boundary_rh[temporal_rid]

            spatial_aug_num = feature.shape[0]
            spatial_rid = random.randint(0, spatial_aug_num - 1)
            feature = feature[spatial_rid]
            
            feature = feature.T   # F x T

            boundary_lh = boundary_lh.unsqueeze(0)
            boundary_lh /= boundary_lh.max() if boundary_lh.max() > 0 else 1  # normalize again
            
            boundary_rh = boundary_rh.unsqueeze(0)
            boundary_rh /= boundary_rh.max() if boundary_rh.max() > 0 else 1  # normalize again
            
        elif self.mode == 'test':
            feature = self.data_dict[video]['feature']
            label_lh = self.data_dict[video]['event_seq_raw_lh']
            boundary_lh = self.data_dict[video]['boundary_seq_ext_lh']
            label_rh = self.data_dict[video]['event_seq_raw_rh']
            boundary_rh = self.data_dict[video]['boundary_seq_ext_rh']

            feature = [torch.swapaxes(i, 1, 2) for i in feature]  # [10 x F x T]
            label_lh = label_lh.unsqueeze(0)   # 1 X T'  
            label_rh = label_rh.unsqueeze(0)   # 1 X T'  
            boundary_lh = [i.unsqueeze(0).unsqueeze(0) for i in boundary_lh]   # [1 x 1 x T]  
            boundary_rh = [i.unsqueeze(0).unsqueeze(0) for i in boundary_rh]   # [1 x 1 x T]  

        return (feature, 
                label_lh, boundary_lh, 
                label_rh, boundary_rh, 
                video)