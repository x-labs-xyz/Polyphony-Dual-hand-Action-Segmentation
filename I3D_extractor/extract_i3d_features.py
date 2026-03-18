import cv2
from models.i3d.extract_i3d import ExtractI3D
from utils.utils import build_cfg_path
from omegaconf import OmegaConf
import torch
import numpy as np
import os
import os.path
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Select the feature type
#feature_type = 'r21d'
feature_type = 'i3d'

# Load and patch the config
args = OmegaConf.load(build_cfg_path(feature_type))
# args.show_pred = True
# args.stack_size = 24
# args.step_size = 24

path = '../videos_downsampled'

for video_name in os.listdir(path):
    print('Extracting features of ' + video_name)

    video_path = path + '/' + video_name
    feature_path = '../i3d_features/' + video_name.split('.')[0] + '.npy'
    if os.path.exists(feature_path):
        pass
    else:
        args.video_paths = [video_path]
        extractor = ExtractI3D(args)
        model, class_head = extractor.load_model(device)

            #feature_folder = '/features' + '/' + folder_name 
            #if not Path(feature_folder).exists():
            #    os.makedirs(feature_folder)

        #feature_path = '/features' + '/' + folder_name + '/' + video_name.split('.')[0] + '.npy'
            
        features = extractor.extract(device, model, class_head, video_path)
        features_cat = np.concatenate((features['rgb'],features['flow']),axis=1) 
        feature_save = features_cat.T
        print(feature_save.shape)
        np.save(feature_path,feature_save)
