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
from dataset import restore_full_sequence
from dual_hand_model import DualHandASDiffusionModel
from dual_hand_dataset import DualHandVideoFeatureDataset, get_dual_hand_data_dicts
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter


class DualHandCoordinationLoss:
    """
    Loss functions to capture coordination between dual hands
    """
    
    def __init__(self, device, num_classes):
        self.device = device
        self.num_classes = num_classes
    
    def temporal_synchronization_loss(self, lh_predictions, rh_predictions, weight=0.1):
        """
        Encourage temporal synchronization between hands
        
        Args:
            lh_predictions: Left hand predictions [B, C, T]
            rh_predictions: Right hand predictions [B, C, T]
        """
        # Get action confidence for each hand
        lh_confidence = torch.max(F.softmax(lh_predictions, dim=1), dim=1)[0]  # [B, T]
        rh_confidence = torch.max(F.softmax(rh_predictions, dim=1), dim=1)[0]  # [B, T]
        
        # Encourage similar confidence patterns (synchronized actions)
        sync_loss = F.mse_loss(lh_confidence, rh_confidence)
        
        return weight * sync_loss
    
    # def cross_hand_consistency_loss(self, lh_predictions, rh_predictions, weight=0.08):
    #     """
    #     Encourage consistency in prediction confidence between hands
    #     """
    #     # Compute entropy for each hand (lower entropy = higher confidence)
    #     lh_probs = F.softmax(lh_predictions, dim=1)
    #     rh_probs = F.softmax(rh_predictions, dim=1)
        
    #     lh_entropy = -torch.sum(lh_probs * torch.log(lh_probs + 1e-8), dim=1)  # [B, T]
    #     rh_entropy = -torch.sum(rh_probs * torch.log(rh_probs + 1e-8), dim=1)  # [B, T]
        
    #     # Encourage similar confidence levels
    #     consistency_loss = F.mse_loss(lh_entropy, rh_entropy)
        
    #     return weight * consistency_loss
    
    def cross_hand_consistency_loss(self, lh_predictions, rh_predictions, weight=0.08):
        """
        Encourage consistency in prediction confidence between hands
        """
        # Compute entropy for each hand (lower entropy = higher confidence)
        lh_probs = F.softmax(lh_predictions, dim=1)
        rh_probs = F.softmax(rh_predictions, dim=1)
        
        lh_entropy = -torch.sum(lh_probs * torch.log(lh_probs + 1e-8), dim=1)  # [B, T]
        rh_entropy = -torch.sum(rh_probs * torch.log(rh_probs + 1e-8), dim=1)  # [B, T]
        
        # Encourage similar confidence levels
        consistency_loss = F.mse_loss(lh_entropy, rh_entropy)
        
        return weight * consistency_loss
    
    def boundary_synchronization_loss(self, lh_predictions, rh_predictions, weight=0.06):
        """
        Encourage action boundaries to be synchronized between hands
        """
        # Detect action transitions (boundaries) for each hand
        lh_classes = torch.argmax(F.softmax(lh_predictions, dim=1), dim=1)  # [B, T]
        rh_classes = torch.argmax(F.softmax(rh_predictions, dim=1), dim=1)  # [B, T]
        
        # Compute boundaries (action transitions)
        lh_boundaries = (lh_classes[:, 1:] != lh_classes[:, :-1]).float()  # [B, T-1]
        rh_boundaries = (rh_classes[:, 1:] != rh_classes[:, :-1]).float()  # [B, T-1]
        
        # Encourage boundaries to occur at similar times
        boundary_sync_loss = F.mse_loss(lh_boundaries, rh_boundaries)
        
        return weight * boundary_sync_loss
    
    # Add action transition loss, improved version
    def action_transition_loss(self, lh_predictions, rh_predictions, lh_labels, rh_labels, weight=0.05):
        """
        Encourage coordinated action transitions between hands
        """
        # Get predicted classes
        lh_classes = torch.argmax(F.softmax(lh_predictions, dim=1), dim=1)  # [B, T]
        rh_classes = torch.argmax(F.softmax(rh_predictions, dim=1), dim=1)  # [B, T]
        
        # Detect transitions
        lh_transitions = (lh_classes[:, 1:] != lh_classes[:, :-1]).float()
        rh_transitions = (rh_classes[:, 1:] != rh_classes[:, :-1]).float()
        lh_gt_transitions = (lh_labels[:, 1:] != lh_labels[:, :-1]).float()
        rh_gt_transitions = (rh_labels[:, 1:] != rh_labels[:, :-1]).float()
        
        # Loss for coordinated transitions
        transition_coord_loss = F.mse_loss(lh_transitions, rh_transitions)
        
        # Loss for matching ground truth transitions
        lh_transition_acc = F.mse_loss(lh_transitions, lh_gt_transitions)
        rh_transition_acc = F.mse_loss(rh_transitions, rh_gt_transitions)
        
        total_loss = transition_coord_loss + 0.5 * (lh_transition_acc + rh_transition_acc)
        
        return weight * total_loss
    
    def _get_coordinated_action_pairs(self):
        """
        Define which action pairs should be coordinated
        Customize this based on your specific dataset
        """
        # Example coordination patterns (you should customize this)
        coordinated_pairs = set()
        
        # Both hands doing the same action (common coordination)
        for i in range(self.num_classes):
            coordinated_pairs.add((i, i))
        
        # Add specific action pair coordinations here
        # For example: (cut_action_id, cut_action_id), (pour_left_id, hold_right_id), etc.
        
        return coordinated_pairs
    
    def _get_action_groups(self):
        """
        Define semantic action groups
        Customize this based on your dataset's action taxonomy
        """
        # Example action groups - you should customize this
        if self.num_classes >= 10:
            return [
                list(range(0, 3)),      # Group 1: Basic actions
                list(range(3, 6)),      # Group 2: Manipulation actions  
                list(range(6, 10)),     # Group 3: Tool usage actions
                # Add more groups as needed
            ]
        else:
            # Simple grouping for smaller action sets
            mid = self.num_classes // 2
            return [
                list(range(0, mid)),
                list(range(mid, self.num_classes))
            ]


class DualHandTrainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device, args=None):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess
        self.args = args  # Store args for coordination weights

        self.model = DualHandASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Dual Hand Model Size: ', sum(p.numel() for p in self.model.parameters()))
        
        # Initialize dual-hand coordination loss
        self.coordination_loss = DualHandCoordinationLoss(self.device, self.num_classes)

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, lh_label_dir, rh_label_dir, result_dir, log_freq, log_train_results=True):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Add learning rate scheduler, improved version
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
        )
        
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1
        best_avg_acc = 0.0  # Track best validation accuracy, improved version

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
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

                (lh_feature, lh_label, lh_boundary, 
                 rh_feature, rh_label, rh_boundary, video) = data
                
                lh_feature, lh_label, lh_boundary = lh_feature.to(device), lh_label.to(device), lh_boundary.to(device)
                rh_feature, rh_label, rh_boundary = rh_feature.to(device), rh_label.to(device), rh_boundary.to(device)
                
                loss_dict = self.model.get_training_loss(
                    lh_feature, rh_feature,
                    lh_event_gt=F.one_hot(lh_label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    rh_event_gt=F.one_hot(rh_label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    lh_boundary_gt=lh_boundary,
                    rh_boundary_gt=rh_boundary,
                    encoder_ce_criterion=ce_criterion,
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                total_loss = 0

                for k, v in loss_dict.items():
                    # Use the same weights for both hands
                    base_key = k.replace('lh_', '').replace('rh_', '')
                    if base_key in loss_weights:
                        total_loss += loss_weights[base_key] * v

                # Add dual-hand coordination losses
                # Get encoder outputs for coordination loss computation
                lh_encoder_out, _ = self.model.left_hand_model.encoder(lh_feature, get_features=True)
                rh_encoder_out, _ = self.model.right_hand_model.encoder(rh_feature, get_features=True)
                
                # Compute coordination losses with adaptive weights
                # Reduce coordination loss importance as training progresses
                coord_decay = max(0.5, 1.0 - epoch / (num_epochs * 0.8))  # Decay after 80% of training
                coord_weights = [w * coord_decay for w in self.args.coordination_weights]
                
                temporal_sync_loss = self.coordination_loss.temporal_synchronization_loss(
                    lh_encoder_out, rh_encoder_out, weight=coord_weights[0])
                
                cross_consistency_loss = self.coordination_loss.cross_hand_consistency_loss(
                    lh_encoder_out, rh_encoder_out, weight=coord_weights[1])
                
                boundary_sync_loss = self.coordination_loss.boundary_synchronization_loss(
                    lh_encoder_out, rh_encoder_out, weight=coord_weights[2]) 

                # Add new transition loss, improved version
                action_transition_loss = self.coordination_loss.action_transition_loss(
                    lh_encoder_out, rh_encoder_out, lh_label, rh_label, weight=coord_weights[3])
                
                # Add coordination losses to total loss, improved version
                coordination_total = (temporal_sync_loss + cross_consistency_loss + 
                                    boundary_sync_loss + action_transition_loss)
                
                # Ensure coordination_total is a proper tensor
                if not isinstance(coordination_total, torch.Tensor):
                    coordination_total = torch.tensor(coordination_total, device=device, requires_grad=True)
                
                total_loss += coordination_total

                if result_dir:
                    for k, v in loss_dict.items():
                        base_key = k.replace('lh_', '').replace('rh_', '')
                        if base_key in loss_weights:
                            logger.add_scalar(f'Train-{k}', loss_weights[base_key] * v.item() / batch_size, step)
                    
                    # Log coordination losses
                    logger.add_scalar('Train-Coordination-Temporal-Sync', 
                                     temporal_sync_loss.item() if isinstance(temporal_sync_loss, torch.Tensor) else temporal_sync_loss, step)
                    logger.add_scalar('Train-Coordination-Cross-Consistency', 
                                     cross_consistency_loss.item() if isinstance(cross_consistency_loss, torch.Tensor) else cross_consistency_loss, step)
                    logger.add_scalar('Train-Coordination-Boundary-Sync', 
                                     boundary_sync_loss.item() if isinstance(boundary_sync_loss, torch.Tensor) else boundary_sync_loss, step)
                    logger.add_scalar('Train-Coordination-Action-Transition', 
                                     action_transition_loss.item() if isinstance(action_transition_loss, torch.Tensor) else action_transition_loss, step)
                    logger.add_scalar('Train-Coordination-Total', 
                                     coordination_total.item() if isinstance(coordination_total, torch.Tensor) else coordination_total, step)
                    
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()
                
                # Gradient clipping for stability, improved version
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
            
            # Step the scheduler with validation loss，improved version
            scheduler.step(epoch_running_loss)
        
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
        
                # Test both hands
                for mode in ['decoder-agg']: 

                    test_result_dict = self.test(
                        test_test_dataset, mode, device, lh_label_dir, rh_label_dir,
                        result_dir=result_dir, model_path=None)

                    if result_dir:
                        for k, v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                        np.save(os.path.join(result_dir, 
                            f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                    for k, v in test_result_dict.items():
                        print(f'Epoch {epoch} - {mode}-Test-{k} {v}')

                    # Save best model, improved version
                    current_avg_acc = test_result_dict.get('Avg_Acc', 0.0)
                    if current_avg_acc > best_avg_acc:
                        best_avg_acc = current_avg_acc
                        if result_dir:
                            torch.save(self.model.state_dict(), f'{result_dir}/best_model.model')
                            print(f'🎯 New best model saved! Avg_Acc: {best_avg_acc:.4f}')

                    if log_train_results:

                        train_result_dict = self.test(
                            train_test_dataset, mode, device, lh_label_dir, rh_label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k, v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                                 
                            np.save(os.path.join(result_dir, 
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                            
                        for k, v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
                        
        if result_dir:
            logger.close()

    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        
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

            (lh_feature, lh_label, lh_boundary,
             rh_feature, rh_label, rh_boundary, video) = test_dataset[video_idx]

            if mode == 'encoder':
                lh_output, rh_output = self.model.get_encoders_output(
                    torch.stack([f.to(device) for f in lh_feature]),
                    torch.stack([f.to(device) for f in rh_feature])
                )
                lh_output = [F.softmax(lh_output, 1).cpu()]
                rh_output = [F.softmax(rh_output, 1).cpu()]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            elif mode == 'decoder-agg':
                lh_outputs = []
                rh_outputs = []
                for i in range(len(lh_feature)):
                    lh_out, rh_out = self.model.ddim_sample_both_hands(
                        lh_feature[i].to(device), rh_feature[i].to(device), seed)
                    lh_outputs.append(lh_out.cpu())
                    rh_outputs.append(rh_out.cpu())
                
                lh_output = lh_outputs
                rh_output = rh_outputs
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            elif mode == 'decoder-noagg':  # temporal aug must be true
                lh_out, rh_out = self.model.ddim_sample_both_hands(
                    lh_feature[len(lh_feature)//2].to(device), 
                    rh_feature[len(rh_feature)//2].to(device), seed)
                lh_output = [lh_out.cpu()]
                rh_output = [rh_out.cpu()]
                left_offset = self.sample_rate // 2
                right_offset = 0

            # Process left hand output
            assert(lh_output[0].shape[0] == 1)
            min_len = min([i.shape[2] for i in lh_output])
            lh_output = [i[:,:,:min_len] for i in lh_output]
            lh_output = torch.cat(lh_output, 0).mean(0).numpy()

            # Process right hand output  
            assert(rh_output[0].shape[0] == 1)
            min_len = min([i.shape[2] for i in rh_output])
            rh_output = [i[:,:,:min_len] for i in rh_output]
            rh_output = torch.cat(rh_output, 0).mean(0).numpy()

            # Apply postprocessing to both hands
            def apply_postprocessing(output, label):
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
                return output

            lh_output = apply_postprocessing(lh_output, lh_label)
            rh_output = apply_postprocessing(rh_output, rh_label)
            
            lh_label = lh_label.squeeze(0).cpu().numpy()
            rh_label = rh_label.squeeze(0).cpu().numpy()

            assert(lh_output.shape == lh_label.shape)
            assert(rh_output.shape == rh_label.shape)
            
            return video, lh_output, lh_label, rh_output, rh_label

    def test(self, test_dataset, mode, device, lh_label_dir, rh_label_dir, result_dir=None, model_path=None):
        
        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):
                
                video, lh_pred, lh_label, rh_pred, rh_label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                lh_pred = [self.event_list[int(i)] for i in lh_pred]
                rh_pred = [self.event_list[int(i)] for i in rh_pred]
                
                # Save left hand predictions
                if not os.path.exists(os.path.join(result_dir, 'lh_prediction')):
                    os.makedirs(os.path.join(result_dir, 'lh_prediction'))

                lh_file_name = os.path.join(result_dir, 'lh_prediction', f'{video}.txt')
                with open(lh_file_name, 'w') as file_ptr:
                    file_ptr.write('### Frame level recognition: ###\n')
                    file_ptr.write(' '.join(lh_pred))

                # Save right hand predictions
                if not os.path.exists(os.path.join(result_dir, 'rh_prediction')):
                    os.makedirs(os.path.join(result_dir, 'rh_prediction'))

                rh_file_name = os.path.join(result_dir, 'rh_prediction', f'{video}.txt')
                with open(rh_file_name, 'w') as file_ptr:
                    file_ptr.write('### Frame level recognition: ###\n')
                    file_ptr.write(' '.join(rh_pred))

        # Evaluate both hands
        lh_acc, lh_edit, lh_f1s = func_eval(
            lh_label_dir, os.path.join(result_dir, 'lh_prediction'), test_dataset.video_list)
        
        rh_acc, rh_edit, rh_f1s = func_eval(
            rh_label_dir, os.path.join(result_dir, 'rh_prediction'), test_dataset.video_list)

        result_dict = {
            'LH_Acc': lh_acc,
            'LH_Edit': lh_edit,
            'LH_F1@10': lh_f1s[0],
            'LH_F1@25': lh_f1s[1],
            'LH_F1@50': lh_f1s[2],
            'RH_Acc': rh_acc,
            'RH_Edit': rh_edit,
            'RH_F1@10': rh_f1s[0],
            'RH_F1@25': rh_f1s[1],
            'RH_F1@50': rh_f1s[2],
            'Avg_Acc': (lh_acc + rh_acc) / 2,
            'Avg_Edit': (lh_edit + rh_edit) / 2,
            'Avg_F1@10': (lh_f1s[0] + rh_f1s[0]) / 2,
            'Avg_F1@25': (lh_f1s[1] + rh_f1s[1]) / 2,
            'Avg_F1@50': (lh_f1s[2] + rh_f1s[2]) / 2,
        }
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--coordination_weights', type=float, nargs=4, 
                       default=[0, 0, 0, 0],
                       help='Weights for coordination losses: [temporal_sync, cross_consistency, boundary_sync, action transition]')
    parser.add_argument('--result_suffix', type=str)
    args = parser.parse_args()
    view_id = "View0"

    all_params = load_config_file(args.config)
    locals().update(all_params)

    # - `dataset_name: "havid"`
    # - `encoder_params: {num_layers: 12, num_f_maps: 256, ...}`
    # - `diffusion_params: {timesteps: 1000, sampling_timesteps: 25, ...}`
    # - `loss_weights: {encoder_ce_loss: 0.5, decoder_ce_loss: 0.5, ...}`
    print(args.config)
    print(all_params)
    print(f"Coordination loss weights: {args.coordination_weights}")
    print(f"  - Temporal Sync: {args.coordination_weights[0]}")
    print(f"  - Cross Consistency: {args.coordination_weights[1]}")
    print(f"  - Boundary Sync: {args.coordination_weights[2]}")
    print(f"  - Action Transition: {args.coordination_weights[5]}")
    print(f"Result suffix: {args.result_suffix}")

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    mapping_file = os.path.join(root_data_dir, dataset_name, 'task_mapping.txt')
    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    # Load dual hand data
    (lh_train_data_dict, lh_test_data_dict, 
     rh_train_data_dict, rh_test_data_dict) = get_dual_hand_data_dicts(
        root_data_dir, dataset_name, split_id, event_list,
        sample_rate, temporal_aug, boundary_smooth, view_id
    )

    # Wraps both hand datasets and ensures video alignment
    train_train_dataset = DualHandVideoFeatureDataset(
        lh_train_data_dict, rh_train_data_dict, num_classes, mode='train')
    train_test_dataset = DualHandVideoFeatureDataset(
        lh_train_data_dict, rh_train_data_dict, num_classes, mode='test')
    test_test_dataset = DualHandVideoFeatureDataset(
        lh_test_data_dict, rh_test_data_dict, num_classes, mode='test')

    trainer = DualHandTrainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        args=args
    )    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Get label directories for evaluation
    lh_label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth/View0/lh_pt')
    rh_label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth/View0/rh_pt')

    trainer.train(train_train_dataset, train_test_dataset, test_test_dataset, 
        loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        lh_label_dir=lh_label_dir, rh_label_dir=rh_label_dir,
        result_dir=os.path.join(result_dir, naming + "_coordination_loss_"+ args.result_suffix), 
        log_freq=log_freq, log_train_results=log_train_results
    ) 