import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dual_hand_dataset_shared_encoder_hand_specific_feature import restore_full_sequence
from dual_hand_dataset_shared_encoder_hand_specific_feature import get_dual_hand_data_dict
from dual_hand_dataset_shared_encoder_hand_specific_feature import DualHandVideoFeatureDataset
from dual_hand_model_shared_encoder_v2_hand_specifc_feature import DualHandASDiffusionModel
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter


class DualHandTrainer:
    """Trainer for dual-hand action segmentation model"""
    
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
                 event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess

        self.model = DualHandASDiffusionModel(
            encoder_params, decoder_params, diffusion_params, 
            self.num_classes, self.device
        )
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, 
              loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, 
              label_dir_lh, label_dir_rh, result_dir, log_freq, log_train_results=True):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        # Resume training if checkpoint exists
        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        # Setup loss functions
        if class_weighting:
            class_weights = train_train_dataset.get_class_weights('lh')  # Use left hand for weights
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)
        
        for epoch in range(restore_epoch+1, num_epochs):
            self.model.train()
            epoch_running_loss = 0
            
            for _, data in enumerate(train_train_loader):
                feature, label_lh, boundary_lh, label_rh, boundary_rh, video = data
                
                # Move to device
                feature = feature.to(device)
                label_lh = label_lh.to(device)
                boundary_lh = boundary_lh.to(device)
                label_rh = label_rh.to(device)
                boundary_rh = boundary_rh.to(device)
                
                # Convert labels to one-hot
                event_gt_lh = F.one_hot(label_lh.long(), num_classes=self.num_classes).permute(0, 2, 1)
                event_gt_rh = F.one_hot(label_rh.long(), num_classes=self.num_classes).permute(0, 2, 1)
                
                # Compute losses
                loss_dict = self.model.get_training_loss(
                    feature,
                    event_gt_lh, boundary_lh,
                    event_gt_rh, boundary_rh,
                    encoder_ce_criterion=ce_criterion, 
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                # ##############
                # # feature         torch.Size([1, F, T])
                # # label_lh/rh      torch.Size([1, T])
                # # boundary_lh/rh   torch.Size([1, 1, T])
                # ##############

                total_loss = 0
                for k, v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k, v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)
            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
        
            if result_dir:
                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:
                if result_dir:
                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')
        
                # Evaluation
                for mode in ['decoder-agg']:  # Can add other modes if needed
                    # Test evaluation
                    test_result_dict_lh, test_result_dict_rh = self.test(
                        test_test_dataset, mode, device, label_dir_lh, label_dir_rh,
                        result_dir=result_dir, model_path=None)

                    if result_dir:
                        # Log left hand results
                        for k, v in test_result_dict_lh.items():
                            logger.add_scalar(f'Test-{mode}-LH-{k}', v, epoch)
                        # Log right hand results
                        for k, v in test_result_dict_rh.items():
                            logger.add_scalar(f'Test-{mode}-RH-{k}', v, epoch)

                        np.save(os.path.join(result_dir, f'test_results_{mode}_lh_epoch{epoch}.npy'), test_result_dict_lh)
                        np.save(os.path.join(result_dir, f'test_results_{mode}_rh_epoch{epoch}.npy'), test_result_dict_rh)

                    # Print results
                    for k, v in test_result_dict_lh.items():
                        print(f'Epoch {epoch} - {mode}-Test-LH-{k} {v}')
                    for k, v in test_result_dict_rh.items():
                        print(f'Epoch {epoch} - {mode}-Test-RH-{k} {v}')

                    # Train evaluation
                    if log_train_results:
                        train_result_dict_lh, train_result_dict_rh = self.test(
                            train_test_dataset, mode, device, label_dir_lh, label_dir_rh,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            # Log left hand results
                            for k, v in train_result_dict_lh.items():
                                logger.add_scalar(f'Train-{mode}-LH-{k}', v, epoch)
                            # Log right hand results  
                            for k, v in train_result_dict_rh.items():
                                logger.add_scalar(f'Train-{mode}-RH-{k}', v, epoch)
                                 
                            np.save(os.path.join(result_dir, f'train_results_{mode}_lh_epoch{epoch}.npy'), train_result_dict_lh)
                            np.save(os.path.join(result_dir, f'train_results_{mode}_rh_epoch{epoch}.npy'), train_result_dict_rh)
                            
                        # Print results
                        for k, v in train_result_dict_lh.items():
                            print(f'Epoch {epoch} - {mode}-Train-LH-{k} {v}')
                        for k, v in train_result_dict_rh.items():
                            print(f'Epoch {epoch} - {mode}-Train-RH-{k} {v}')
                        
        if result_dir:
            logger.close()

    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        """Test single video and return predictions for both hands"""
        
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():
            feature, label_lh, _, label_rh, _, video = test_dataset[video_idx]

            # feature:     [torch.Size([1, F, Sampled T])]
            # label_lh/rh: torch.Size([1, Original T])

            if mode == 'encoder':
                encoder_out = [self.model.encoder(feature[i].to(device)) 
                              for i in range(len(feature))]
                # Use encoder output for both hands (shared prediction)
                output_lh = [F.softmax(i, 1).cpu() for i in encoder_out]
                output_rh = [F.softmax(i, 1).cpu() for i in encoder_out]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            elif mode == 'decoder-agg':
                # Use dual-hand DDIM sampling
                outputs = [self.model.ddim_sample(feature[i].to(device), seed) 
                          for i in range(len(feature))]
                output_lh = [i[0].cpu() for i in outputs]  # Left hand outputs
                output_rh = [i[1].cpu() for i in outputs]  # Right hand outputs
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            elif mode == 'decoder-noagg':  # temporal aug must be true
                output_lh_single, output_rh_single = self.model.ddim_sample(
                    feature[len(feature)//2].to(device), seed)
                output_lh = [output_lh_single.cpu()]
                output_rh = [output_rh_single.cpu()]
                left_offset = self.sample_rate // 2
                right_offset = 0

            # Process outputs for both hands
            def process_output(output_list, label):
                assert(output_list[0].shape[0] == 1)
                min_len = min([i.shape[2] for i in output_list])
                output = [i[:,:,:min_len] for i in output_list]
                output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
                output = output.mean(0).numpy()

                if self.postprocess['type'] == 'median':
                    smoothed_output = np.zeros_like(output)
                    for c in range(output.shape[0]):
                        smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                    output = smoothed_output / smoothed_output.sum(0, keepdims=True)

                output = np.argmax(output, 0)
                output = restore_full_sequence(output, 
                    full_len=label.shape[-1], 
                    left_offset=left_offset, 
                    right_offset=right_offset, 
                    sample_rate=self.sample_rate
                )

                if self.postprocess['type'] == 'mode':
                    output = mode_filter(output, self.postprocess['value'])

                if self.postprocess['type'] == 'purge':
                    trans, starts, ends = get_labels_start_end_time(output)
                    
                    for e in range(0, len(trans)):
                        duration = ends[e] - starts[e]
                        if duration <= self.postprocess['value']:
                            
                            if e == 0:
                                output[starts[e]:ends[e]] = trans[e+1]
                            elif e == len(trans) - 1:
                                output[starts[e]:ends[e]] = trans[e-1]
                            else:
                                mid = starts[e] + duration // 2
                                output[starts[e]:mid] = trans[e-1]
                                output[mid:ends[e]] = trans[e+1]

                label_np = label.squeeze(0).cpu().numpy()
                assert(output.shape == label_np.shape)
                return output

            output_lh_processed = process_output(output_lh, label_lh)
            output_rh_processed = process_output(output_rh, label_rh)
            
            return video, output_lh_processed, output_rh_processed, label_lh.squeeze(0).cpu().numpy(), label_rh.squeeze(0).cpu().numpy()

    def test(self, test_dataset, mode, device, label_dir_lh, label_dir_rh, result_dir=None, model_path=None):
        """Evaluate model on test dataset"""
        
        assert(test_dataset.mode == 'test')
        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        with torch.no_grad():
            for video_idx in tqdm(range(len(test_dataset))):
                
                video, pred_lh, pred_rh, label_lh, label_rh = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                pred_lh_str = [self.event_list[int(i)] for i in pred_lh]
                pred_rh_str = [self.event_list[int(i)] for i in pred_rh]
                
                # Create prediction directories
                if not os.path.exists(os.path.join(result_dir, 'prediction_lh')):
                    os.makedirs(os.path.join(result_dir, 'prediction_lh'))
                if not os.path.exists(os.path.join(result_dir, 'prediction_rh')):
                    os.makedirs(os.path.join(result_dir, 'prediction_rh'))

                # Save left hand predictions
                file_name_lh = os.path.join(result_dir, 'prediction_lh', f'{video}.txt')
                with open(file_name_lh, 'w') as f:
                    f.write('### Frame level recognition: ###\n')
                    f.write(' '.join(pred_lh_str))

                # Save right hand predictions
                file_name_rh = os.path.join(result_dir, 'prediction_rh', f'{video}.txt')
                with open(file_name_rh, 'w') as f:
                    f.write('### Frame level recognition: ###\n')
                    f.write(' '.join(pred_rh_str))

        # Evaluate both hands
        acc_lh, edit_lh, f1s_lh = func_eval(
            label_dir_lh, os.path.join(result_dir, 'prediction_lh'), test_dataset.video_list)
        
        acc_rh, edit_rh, f1s_rh = func_eval(
            label_dir_rh, os.path.join(result_dir, 'prediction_rh'), test_dataset.video_list)

        result_dict_lh = {
            'Acc': acc_lh,
            'Edit': edit_lh,
            'F1@10': f1s_lh[0],
            'F1@25': f1s_lh[1],
            'F1@50': f1s_lh[2]
        }
        
        result_dict_rh = {
            'Acc': acc_rh,
            'Edit': edit_rh,
            'F1@10': f1s_rh[0],
            'F1@25': f1s_rh[1],
            'F1@50': f1s_rh[2]
        }
        
        return result_dict_lh, result_dict_rh


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    feature_dir_lh = os.path.join(root_data_dir, dataset_name, 'videomae_features/lh_v0')
    feature_dir_rh = os.path.join(root_data_dir, dataset_name, 'videomae_features/rh_v0')    
    label_dir_lh = os.path.join(root_data_dir, dataset_name, 'groundTruth/View0/lh_pt')
    label_dir_rh = os.path.join(root_data_dir, dataset_name, 'groundTruth/View0/rh_pt')
    mapping_file = os.path.join(root_data_dir, dataset_name, 'task_mapping.txt')

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits/View0/lh_pt', f'train.split{split_id}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits/View0/lh_pt', f'test.split{split_id}.bundle'), dtype=str)

    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]

    # Load dual-hand data
    train_data_dict = get_dual_hand_data_dict(
        feature_dir_lh=feature_dir_lh,
        feature_dir_rh=feature_dir_rh,
        label_dir_lh=label_dir_lh,
        label_dir_rh=label_dir_rh,
        video_list=train_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

    test_data_dict = get_dual_hand_data_dict(
        feature_dir_lh=feature_dir_lh,
        feature_dir_rh=feature_dir_rh,
        label_dir_lh=label_dir_lh,
        label_dir_rh=label_dir_rh,
        video_list=test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )
    
    train_train_dataset = DualHandVideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset = DualHandVideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = DualHandVideoFeatureDataset(test_data_dict, num_classes, mode='test')

    # Update loss weights for dual-hand model
    dual_hand_loss_weights = {
        'encoder_ce_loss': loss_weights.get('encoder_ce_loss', 0.5),
        'encoder_mse_loss': loss_weights.get('encoder_mse_loss', 0.025),
        'encoder_boundary_loss': loss_weights.get('encoder_boundary_loss', 0.0),
        'decoder_left_ce_loss': loss_weights.get('decoder_ce_loss', 0.5) / 2,
        'decoder_left_mse_loss': loss_weights.get('decoder_mse_loss', 0.025) / 2,
        'decoder_left_boundary_loss': loss_weights.get('decoder_boundary_loss', 0.1) / 2,
        'decoder_right_ce_loss': loss_weights.get('decoder_ce_loss', 0.5) / 2,
        'decoder_right_mse_loss': loss_weights.get('decoder_mse_loss', 0.025) / 2,
        'decoder_right_boundary_loss': loss_weights.get('decoder_boundary_loss', 0.1) / 2,
    }

    trainer = DualHandTrainer(
        dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    trainer.train(
        train_train_dataset, train_test_dataset, test_test_dataset, 
        dual_hand_loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        label_dir_lh=label_dir_lh, label_dir_rh=label_dir_rh,
        result_dir=os.path.join(result_dir, naming), 
        log_freq=log_freq, log_train_results=log_train_results
    )