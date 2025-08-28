import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from dataset import get_data_dict, VideoFeatureDataset


class DualHandVideoFeatureDataset(Dataset):
    def __init__(self, lh_data_dict, rh_data_dict, class_num, mode):
        super(DualHandVideoFeatureDataset, self).__init__()
        
        assert(mode in ['train', 'test'])
        
        self.lh_data_dict = lh_data_dict
        self.rh_data_dict = rh_data_dict
        self.class_num = class_num
        self.mode = mode
        
        # Ensure both datasets have the same videos
        lh_videos = set(self.lh_data_dict.keys())
        rh_videos = set(self.rh_data_dict.keys())
        common_videos = lh_videos.intersection(rh_videos)
        
        if len(common_videos) != len(lh_videos) or len(common_videos) != len(rh_videos):
            print(f"Warning: Left hand has {len(lh_videos)} videos, right hand has {len(rh_videos)} videos")
            print(f"Using {len(common_videos)} common videos")
        
        self.video_list = list(common_videos)
        
    def get_class_weights(self):
        """
        Get class weights by combining both left and right hand data
        """
        lh_event_seq = np.concatenate([self.lh_data_dict[v]['event_seq_raw'] for v in self.video_list])
        rh_event_seq = np.concatenate([self.rh_data_dict[v]['event_seq_raw'] for v in self.video_list])
        
        full_event_seq = np.concatenate([lh_event_seq, rh_event_seq])
        
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
            # Left hand data
            lh_feature = self.lh_data_dict[video]['feature']
            lh_label = self.lh_data_dict[video]['event_seq_ext']
            lh_boundary = self.lh_data_dict[video]['boundary_seq_ext']

            # Right hand data
            rh_feature = self.rh_data_dict[video]['feature']
            rh_label = self.rh_data_dict[video]['event_seq_ext']
            rh_boundary = self.rh_data_dict[video]['boundary_seq_ext']

            # Apply same temporal augmentation to both hands
            temporal_aug_num = len(lh_feature)
            temporal_rid = random.randint(0, temporal_aug_num - 1)
            
            lh_feature = lh_feature[temporal_rid]
            lh_label = lh_label[temporal_rid]
            lh_boundary = lh_boundary[temporal_rid]
            
            rh_feature = rh_feature[temporal_rid]
            rh_label = rh_label[temporal_rid]
            rh_boundary = rh_boundary[temporal_rid]

            # Apply same spatial augmentation to both hands
            lh_spatial_aug_num = lh_feature.shape[0]
            rh_spatial_aug_num = rh_feature.shape[0]
            
            # Use the same random seed for both hands to ensure consistency
            spatial_rid = random.randint(0, min(lh_spatial_aug_num, rh_spatial_aug_num) - 1)
            
            lh_feature = lh_feature[spatial_rid]
            rh_feature = rh_feature[spatial_rid]
            
            # Transpose to F x T format
            lh_feature = lh_feature.T
            rh_feature = rh_feature.T

            # Normalize boundaries
            lh_boundary = lh_boundary.unsqueeze(0)
            lh_boundary /= lh_boundary.max() if lh_boundary.max() > 0 else 1
            
            rh_boundary = rh_boundary.unsqueeze(0)
            rh_boundary /= rh_boundary.max() if rh_boundary.max() > 0 else 1
            
            return (lh_feature, lh_label, lh_boundary, 
                   rh_feature, rh_label, rh_boundary, video)
            
        elif self.mode == 'test':
            # Left hand data
            lh_feature = self.lh_data_dict[video]['feature']
            lh_label = self.lh_data_dict[video]['event_seq_raw']
            lh_boundary = self.lh_data_dict[video]['boundary_seq_ext']

            # Right hand data
            rh_feature = self.rh_data_dict[video]['feature']
            rh_label = self.rh_data_dict[video]['event_seq_raw']
            rh_boundary = self.rh_data_dict[video]['boundary_seq_ext']

            # Format for test mode
            lh_feature = [torch.swapaxes(i, 1, 2) for i in lh_feature]  # [N x F x T]
            lh_label = lh_label.unsqueeze(0)   # 1 X T'  
            lh_boundary = [i.unsqueeze(0).unsqueeze(0) for i in lh_boundary]   # [1 x 1 x T]  

            rh_feature = [torch.swapaxes(i, 1, 2) for i in rh_feature]  # [N x F x T]
            rh_label = rh_label.unsqueeze(0)   # 1 X T'  
            rh_boundary = [i.unsqueeze(0).unsqueeze(0) for i in rh_boundary]   # [1 x 1 x T]  

            return (lh_feature, lh_label, lh_boundary,
                   rh_feature, rh_label, rh_boundary, video)


def get_dual_hand_data_dicts(root_data_dir, dataset_name, split_id, event_list, 
                            sample_rate, temporal_aug, boundary_smooth, view_id):
    """
    Helper function to load both left and right hand data dictionaries
    """
    feature_dir = os.path.join(root_data_dir, dataset_name, 'features')
    
    # Left hand data
    lh_label_dir = os.path.join(root_data_dir, dataset_name, f'groundTruth/{view_id}/lh_pt')
    lh_train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, f'splits/{view_id}/lh_pt', f'train.split{split_id}.bundle'), dtype=str)
    lh_test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, f'splits/{view_id}/lh_pt', f'test.split{split_id}.bundle'), dtype=str)
    
    lh_train_video_list = [i.split('.')[0] for i in lh_train_video_list]
    lh_test_video_list = [i.split('.')[0] for i in lh_test_video_list]
    
    # Right hand data  
    rh_label_dir = os.path.join(root_data_dir, dataset_name, f'groundTruth/{view_id}/rh_pt')
    rh_train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, f'splits/{view_id}/rh_pt', f'train.split{split_id}.bundle'), dtype=str)
    rh_test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, f'splits/{view_id}/rh_pt', f'test.split{split_id}.bundle'), dtype=str)
    
    rh_train_video_list = [i.split('.')[0] for i in rh_train_video_list]
    rh_test_video_list = [i.split('.')[0] for i in rh_test_video_list]
    
    # Load data dictionaries
    lh_train_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=lh_label_dir, 
        video_list=lh_train_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

    lh_test_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=lh_label_dir, 
        video_list=lh_test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )
    
    rh_train_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=rh_label_dir, 
        video_list=rh_train_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

    rh_test_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=rh_label_dir, 
        video_list=rh_test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )
    
    return (lh_train_data_dict, lh_test_data_dict, 
            rh_train_data_dict, rh_test_data_dict) 