"""
Main training script for dual-hand action segmentation
Includes adaptive loss weighting and comprehensive evaluation
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple

from model import DualHandASDiffusionModel
from dataset import get_dual_hand_data_dict, DualHandVideoFeatureDataset, restore_full_sequence
from utils import (
    load_config_file, func_eval, set_random_seed, 
    get_labels_start_end_time, mode_filter
)


# ============================================================================
# Adaptive Loss Weight Manager
# ============================================================================

class AdaptiveLossWeightManager:
    """
    Manages adaptive loss weighting based on performance gap between hands
    Dynamically adjusts loss weights to balance training between left and right hands
    """
    
    def __init__(self, base_weights: Dict[str, float], performance_window: int = 5,
                 min_boost: float = 1.0, max_boost: float = 2.0):
        """
        Args:
            base_weights: Base loss weights dictionary
            performance_window: Number of epochs to track for moving average
            min_boost: Minimum boost factor
            max_boost: Maximum boost factor
        """
        self.base_weights = base_weights.copy()
        self.performance_window = performance_window
        self.min_boost = min_boost
        self.max_boost = max_boost
        self.performance_history = {'lh': [], 'rh': []}
    
    def update_performance(self, lh_performance: float, rh_performance: float) -> Dict[str, float]:
        """
        Update performance history and compute adaptive weights
        
        Args:
            lh_performance: Left hand performance metric (e.g., accuracy)
            rh_performance: Right hand performance metric
            
        Returns:
            Updated adaptive weights dictionary
        """
        self.performance_history['lh'].append(lh_performance)
        self.performance_history['rh'].append(rh_performance)
        
        # Keep only recent history
        if len(self.performance_history['lh']) > self.performance_window:
            self.performance_history['lh'] = self.performance_history['lh'][-self.performance_window:]
            self.performance_history['rh'] = self.performance_history['rh'][-self.performance_window:]
        
        return self._compute_adaptive_weights()
    
    def _compute_adaptive_weights(self) -> Dict[str, float]:
        """
        Compute adaptive weights based on recent performance
        Bidirectional adjustment: boost whichever hand is performing worse
        """
        if len(self.performance_history['lh']) < 2:
            return self.base_weights.copy()
        
        # Use average performance over recent history
        avg_lh = np.mean(self.performance_history['lh'])
        avg_rh = np.mean(self.performance_history['rh'])
        
        # Calculate performance ratios
        if avg_lh > 0 and avg_rh > 0:
            lh_to_rh_ratio = avg_lh / avg_rh
            rh_to_lh_ratio = avg_rh / avg_lh
        else:
            lh_to_rh_ratio = 1.0
            rh_to_lh_ratio = 1.0
        
        # Initialize boost factors
        lh_boost_factor = 1.0
        rh_boost_factor = 1.0
        
        # Boost left hand if it's significantly worse than right hand
        if lh_to_rh_ratio < 0.95:  # If LH is >5% worse than RH
            lh_boost_factor = min(self.max_boost, max(self.min_boost, 1.0 / lh_to_rh_ratio))
        
        # Boost right hand if it's significantly worse than left hand
        if rh_to_lh_ratio < 0.95:  # If RH is >5% worse than LH
            rh_boost_factor = min(self.max_boost, max(self.min_boost, 1.0 / rh_to_lh_ratio))
        
        # Apply boost factors to respective hand weights
        adaptive_weights = self.base_weights.copy()
        
        # Left hand weights
        adaptive_weights['decoder_left_ce_loss'] *= lh_boost_factor
        adaptive_weights['decoder_left_mse_loss'] *= lh_boost_factor
        adaptive_weights['decoder_left_boundary_loss'] *= lh_boost_factor
        
        # Right hand weights
        adaptive_weights['decoder_right_ce_loss'] *= rh_boost_factor
        adaptive_weights['decoder_right_mse_loss'] *= rh_boost_factor
        adaptive_weights['decoder_right_boundary_loss'] *= rh_boost_factor
        
        return adaptive_weights


# ============================================================================
# Dual-Hand Trainer
# ============================================================================

class DualHandTrainer:
    """
    Trainer for dual-hand action segmentation model
    Supports adaptive loss weighting, checkpointing, and comprehensive evaluation
    Also supports single-stream mode with label perturbation
    """
    
    def __init__(self, encoder_params: Dict, decoder_params: Dict, diffusion_params: Dict,
                 event_list: list, sample_rate: int, temporal_aug: bool,
                 set_sampling_seed: bool, postprocess: Dict, device: torch.device,
                 one_stream: bool = False, perturbation_config: Optional[Dict] = None,
                 boundary_smooth: Optional[float] = None):
        """
        Args:
            encoder_params: Encoder configuration
            decoder_params: Decoder configuration
            diffusion_params: Diffusion hyperparameters
            event_list: List of action class names
            sample_rate: Temporal sampling rate
            temporal_aug: Whether to use temporal augmentation
            set_sampling_seed: Whether to set deterministic seed for sampling
            postprocess: Post-processing configuration
            device: Device (cpu or cuda)
            one_stream: Whether to use single-stream mode (copy LH to RH with perturbation)
            perturbation_config: Configuration for label perturbation in single-stream mode
            boundary_smooth: Gaussian smoothing sigma for boundaries
        """
        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess
        self.one_stream = one_stream
        self.boundary_smooth = boundary_smooth
        
        # Default perturbation config for single-stream mode
        if perturbation_config is None:
            perturbation_config = {
                'max_shift_frames': 5,
                'max_perturbations': 3,
                'min_segment_len': 3
            }
        self.perturbation_config = perturbation_config
        
        # Create model
        self.model = DualHandASDiffusionModel(
            encoder_params, decoder_params, diffusion_params,
            self.num_classes, self.device
        )
        
        print(f'Model Size: {sum(p.numel() for p in self.model.parameters()):,} parameters')
        if self.one_stream:
            print('Running in SINGLE-STREAM mode with label perturbation')
            print(f'Perturbation config: {self.perturbation_config}')
    
    def _perturb_labels_by_shifting_boundaries(self, label_1d_np: np.ndarray, 
                                                max_shift_frames: int, 
                                                max_perturbations: int, 
                                                min_segment_len: int = 3) -> np.ndarray:
        """
        Randomly shift up to N internal boundaries by up to K frames.
        Keeps total length unchanged and labels contiguous.
        
        Args:
            label_1d_np: Numpy array of shape [T] with int class IDs
            max_shift_frames: Maximum frames to shift each boundary
            max_perturbations: Maximum number of boundaries to perturb
            min_segment_len: Minimum segment length to preserve
            
        Returns:
            Perturbed label sequence of shape [T]
        """
        T = label_1d_np.shape[0]
        if T <= 2:
            return label_1d_np
        
        # Get segments (labels and start/end indices)
        labels, starts, ends = get_labels_start_end_time([str(int(i)) for i in label_1d_np])
        if len(starts) <= 1:
            return label_1d_np
        
        # Build boundaries list (internal ends equal next starts)
        boundaries = [e for e in ends[:-1]]  # Indices where a segment ends
        if len(boundaries) == 0:
            return label_1d_np
        
        # Choose boundaries to perturb
        num_to_perturb = min(max_perturbations, len(boundaries))
        
        # Protect edges and very short neighbors
        candidate_indices = []
        for bi, b in enumerate(boundaries):
            left_len = ends[bi] - starts[bi]
            right_len = ends[bi + 1] - starts[bi + 1]
            if left_len >= (min_segment_len + 1) and right_len >= (min_segment_len + 1):
                candidate_indices.append(bi)
        
        if len(candidate_indices) == 0:
            return label_1d_np
        
        chosen = np.random.choice(candidate_indices, 
                                 size=min(num_to_perturb, len(candidate_indices)), 
                                 replace=False)
        
        # Apply shifts to chosen boundaries
        new_starts = starts.copy()
        new_ends = ends.copy()
        
        for bi in chosen:
            shift = np.random.randint(-max_shift_frames, max_shift_frames + 1)
            if shift == 0:
                continue
            
            # Proposed new boundary is ends[bi] + shift; adjust adjacent segment ends/starts
            new_end_left = max(new_starts[bi] + min_segment_len, 
                              min(new_ends[bi] + shift, new_ends[bi + 1] - min_segment_len))
            new_start_right = new_end_left
            
            # Enforce ordering
            if new_end_left <= new_starts[bi] + min_segment_len - 1:
                new_end_left = new_starts[bi] + min_segment_len
                new_start_right = new_end_left
            if new_ends[bi + 1] - new_start_right < min_segment_len:
                new_start_right = new_ends[bi + 1] - min_segment_len
                new_end_left = new_start_right
            
            new_ends[bi] = new_end_left
            new_starts[bi + 1] = new_start_right
        
        # Rebuild label sequence
        out = np.zeros_like(label_1d_np)
        for si, (s, e) in enumerate(zip(new_starts, new_ends)):
            out[s:e] = int(labels[si])
        
        # Fill any potential gaps due to rounding
        if new_starts[0] > 0:
            out[:new_starts[0]] = int(labels[0])
        if new_ends[-1] < T:
            out[new_ends[-1]:] = int(labels[-1])
        
        return out
    
    def train(self, train_train_dataset: DualHandVideoFeatureDataset,
              train_test_dataset: DualHandVideoFeatureDataset,
              test_test_dataset: DualHandVideoFeatureDataset,
              loss_weights: Dict[str, float], class_weighting: bool, soft_label: Optional[float],
              num_epochs: int, batch_size: int, learning_rate: float, weight_decay: float,
              label_dir_lh: str, label_dir_rh: str, result_dir: str,
              log_freq: int, log_train_results: bool = True):
        """
        Train the model
        
        Args:
            train_train_dataset: Training dataset (train mode)
            train_test_dataset: Training dataset (test mode for evaluation)
            test_test_dataset: Test dataset (test mode)
            loss_weights: Loss weight dictionary
            class_weighting: Whether to use class weights
            soft_label: Soft label smoothing sigma (None = no smoothing)
            num_epochs: Number of training epochs
            batch_size: Batch size (simulated via gradient accumulation)
            learning_rate: Learning rate
            weight_decay: Weight decay
            label_dir_lh: Left hand label directory (for evaluation)
            label_dir_rh: Right hand label directory (for evaluation)
            result_dir: Directory to save results
            log_freq: Logging frequency (epochs)
            log_train_results: Whether to log train evaluation results
        """
        device = self.device
        self.model.to(device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()
        
        restore_epoch = -1
        step = 1
        
        # Resume training if checkpoint exists
        if os.path.exists(result_dir):
            latest_checkpoint = os.path.join(result_dir, 'latest.pt')
            if os.path.exists(latest_checkpoint) and os.path.getsize(latest_checkpoint) > 0:
                print(f'Resuming from checkpoint: {latest_checkpoint}')
                saved_state = torch.load(latest_checkpoint, map_location=device)
                self.model.load_state_dict(saved_state['model'])
                optimizer.load_state_dict(saved_state['optimizer'])
                restore_epoch = saved_state['epoch']
                step = saved_state['step']
                print(f'Resumed from epoch {restore_epoch + 1}')
        
        # Setup loss functions with hand-specific class weights
        if class_weighting:
            # Compute class weights for both hands separately
            class_weights_lh = train_train_dataset.get_class_weights('lh')
            class_weights_rh = train_train_dataset.get_class_weights('rh')
            
            class_weights_lh = torch.from_numpy(class_weights_lh).float().to(device)
            class_weights_rh = torch.from_numpy(class_weights_rh).float().to(device)
            
            # Create separate loss functions for each hand
            ce_criterion_lh = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights_lh, reduction='none')
            ce_criterion_rh = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights_rh, reduction='none')
            
            # Use left hand weights for encoder (shared)
            ce_criterion_encoder = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights_lh, reduction='none')
        else:
            ce_criterion_lh = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            ce_criterion_rh = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            ce_criterion_encoder = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        
        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        # Data loader
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4
        )
        
        # TensorBoard logger
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            logger = SummaryWriter(result_dir)
        
        # Initialize adaptive loss weight manager
        adaptive_weight_manager = AdaptiveLossWeightManager(loss_weights)
        
        # Training loop
        for epoch in range(restore_epoch + 1, num_epochs):
            self.model.train()
            epoch_running_loss = 0
            
            for _, data in enumerate(train_train_loader):
                feature_lh, feature_rh, label_lh, boundary_lh, label_rh, boundary_rh, video = data
                
                # Move to device
                feature_lh = feature_lh.to(device)
                feature_rh = feature_rh.to(device)
                label_lh = label_lh.to(device)
                boundary_lh = boundary_lh.to(device)
                label_rh = label_rh.to(device)
                boundary_rh = boundary_rh.to(device)
                
                # Single-stream mode: copy LH to RH with perturbation
                if self.one_stream:
                    # Copy left hand features to right hand (they're already the same from dataset)
                    feature_rh = feature_lh.clone()
                    
                    # Perturb right hand labels by shifting boundaries
                    max_shift = int(self.perturbation_config.get('max_shift_frames', 5))
                    max_perturb = int(self.perturbation_config.get('max_perturbations', 3))
                    min_seg = int(self.perturbation_config.get('min_segment_len', 3))
                    
                    # Work in numpy ints, ensure 1D shape [T]
                    label_rh_np = label_rh.detach().cpu().numpy()
                    label_rh_np = np.asarray(label_rh_np).squeeze().astype(np.int64)
                    label_rh_np = self._perturb_labels_by_shifting_boundaries(
                        label_rh_np, max_shift, max_perturb, min_seg
                    )
                    
                    # Restore batch dimension [1, T]
                    label_rh = torch.from_numpy(label_rh_np).to(device).unsqueeze(0)
                    
                    # Recompute boundary for RH and normalize
                    from utils import get_boundary_seq
                    boundary_rh_np = get_boundary_seq(label_rh_np, self.boundary_smooth)
                    boundary_rh = torch.from_numpy(boundary_rh_np).float().to(device).unsqueeze(0).unsqueeze(0)
                    boundary_rh /= boundary_rh.max() if boundary_rh.max() > 0 else 1
                
                # Convert labels to one-hot
                event_gt_lh = F.one_hot(label_lh.long(), num_classes=self.num_classes).permute(0, 2, 1)
                event_gt_rh = F.one_hot(label_rh.long(), num_classes=self.num_classes).permute(0, 2, 1)
                
                # Compute losses with hand-specific criteria
                loss_dict = self.model.get_training_loss(
                    feature_lh, feature_rh,
                    event_gt_lh, boundary_lh,
                    event_gt_rh, boundary_rh,
                    encoder_ce_criterion=ce_criterion_encoder,
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion_left=ce_criterion_lh,
                    decoder_ce_criterion_right=ce_criterion_rh,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )
                
                # Get current adaptive weights
                current_weights = adaptive_weight_manager.base_weights
                
                # Compute total loss
                total_loss = 0
                for k, v in loss_dict.items():
                    total_loss += current_weights[k] * v
                
                # Log individual losses
                if result_dir:
                    for k, v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', current_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)
                
                # Backward pass
                total_loss /= batch_size
                total_loss.backward()
                
                epoch_running_loss += total_loss.item()
                
                # Gradient accumulation
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                step += 1
            
            # Epoch summary
            epoch_running_loss /= len(train_train_dataset)
            print(f'Epoch {epoch:3d} - Running Loss: {epoch_running_loss:.6f}')
            
            # Save checkpoint
            if result_dir:
                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }
                torch.save(state, os.path.join(result_dir, 'latest.pt'))
            
            # Periodic evaluation
            if epoch % log_freq == 0:
                # Save model checkpoint
                if result_dir:
                    torch.save(self.model.state_dict(), os.path.join(result_dir, f'epoch-{epoch}.model'))
                
                # Test evaluation
                for mode in ['decoder-agg']:
                    print(f'\nEvaluating epoch {epoch} ({mode})...')
                    test_result_dict_lh, test_result_dict_rh = self.test(
                        test_test_dataset, mode, device, label_dir_lh, label_dir_rh,
                        result_dir=result_dir, model_path=None
                    )
                    
                    # Update adaptive weights based on performance
                    lh_acc = test_result_dict_lh.get('Acc', 0)
                    rh_acc = test_result_dict_rh.get('Acc', 0)
                    adaptive_weights = adaptive_weight_manager.update_performance(lh_acc, rh_acc)
                    
                    # Log adaptive weight changes
                    if result_dir:
                        boost_factor_lh = adaptive_weights['decoder_left_ce_loss'] / loss_weights['decoder_left_ce_loss']
                        boost_factor_rh = adaptive_weights['decoder_right_ce_loss'] / loss_weights['decoder_right_ce_loss']
                        logger.add_scalar('Adaptive/LeftHandBoost', boost_factor_lh, epoch)
                        logger.add_scalar('Adaptive/RightHandBoost', boost_factor_rh, epoch)
                        logger.add_scalar('Adaptive/PerformanceRatio', rh_acc / lh_acc if lh_acc > 0 else 1.0, epoch)
                        
                        # Log test results
                        for k, v in test_result_dict_lh.items():
                            logger.add_scalar(f'Test-{mode}-LH-{k}', v, epoch)
                        for k, v in test_result_dict_rh.items():
                            logger.add_scalar(f'Test-{mode}-RH-{k}', v, epoch)
                        
                        np.save(os.path.join(result_dir, f'test_results_{mode}_lh_epoch{epoch}.npy'), test_result_dict_lh)
                        np.save(os.path.join(result_dir, f'test_results_{mode}_rh_epoch{epoch}.npy'), test_result_dict_rh)
                    
                    # Print results
                    print(f'\n{"="*70}')
                    print(f'Epoch {epoch} - Test Results ({mode}):')
                    print(f'{"="*70}')
                    print(f'{"Metric":<20} {"Left Hand":>15} {"Right Hand":>15} {"Average":>15}')
                    print(f'{"-"*70}')
                    for k in test_result_dict_lh.keys():
                        avg_val = (test_result_dict_lh[k] + test_result_dict_rh[k]) / 2
                        print(f'{k:<20} {test_result_dict_lh[k]:>15.2f} {test_result_dict_rh[k]:>15.2f} {avg_val:>15.2f}')
                    print(f'{"="*70}')
                    print(f'Adaptive Boost - LH: {boost_factor_lh:.3f}, RH: {boost_factor_rh:.3f}\n')
                    
                    # Train evaluation (optional)
                    if log_train_results:
                        train_result_dict_lh, train_result_dict_rh = self.test(
                            train_test_dataset, mode, device, label_dir_lh, label_dir_rh,
                            result_dir=result_dir, model_path=None
                        )
                        
                        if result_dir:
                            for k, v in train_result_dict_lh.items():
                                logger.add_scalar(f'Train-{mode}-LH-{k}', v, epoch)
                            for k, v in train_result_dict_rh.items():
                                logger.add_scalar(f'Train-{mode}-RH-{k}', v, epoch)
                            
                            np.save(os.path.join(result_dir, f'train_results_{mode}_lh_epoch{epoch}.npy'), train_result_dict_lh)
                            np.save(os.path.join(result_dir, f'train_results_{mode}_rh_epoch{epoch}.npy'), train_result_dict_rh)
        
        if result_dir:
            logger.close()
        
        print('\n✓ Training completed!')
    
    def test_single_video(self, video_idx: int, test_dataset: DualHandVideoFeatureDataset,
                          mode: str, device: torch.device, model_path: Optional[str] = None) -> Tuple:
        """
        Test single video and return predictions for both hands
        
        Args:
            video_idx: Video index in dataset
            test_dataset: Test dataset
            mode: Evaluation mode ('encoder', 'decoder-noagg', 'decoder-agg')
            device: Device
            model_path: Path to model checkpoint (optional)
            
        Returns:
            Tuple of (video_name, pred_lh, pred_rh, label_lh, label_rh)
        """
        assert test_dataset.mode == 'test'
        assert mode in ['encoder', 'decoder-noagg', 'decoder-agg']
        assert self.postprocess['type'] in ['median', 'mode', 'purge', None]
        
        self.model.eval()
        self.model.to(device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
        
        with torch.no_grad():
            feature_lh, feature_rh, label_lh, _, label_rh, _, video = test_dataset[video_idx]
            
            if mode == 'encoder':
                # Use encoder predictions
                encoder_out_lh = [self.model.encoder(feature_lh[i].to(device)) for i in range(len(feature_lh))]
                encoder_out_rh = [self.model.encoder(feature_rh[i].to(device)) for i in range(len(feature_rh))]
                output_lh = [F.softmax(i, 1).cpu() for i in encoder_out_lh]
                output_rh = [F.softmax(i, 1).cpu() for i in encoder_out_rh]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2
            
            elif mode == 'decoder-agg':
                # Use DDIM sampling with aggregation
                outputs = [self.model.ddim_sample(feature_lh[i].to(device), feature_rh[i].to(device), seed)
                          for i in range(len(feature_lh))]
                output_lh = [i[0].cpu() for i in outputs]
                output_rh = [i[1].cpu() for i in outputs]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2
            
            elif mode == 'decoder-noagg':
                # Single temporal view (no aggregation)
                output_lh_single, output_rh_single = self.model.ddim_sample(
                    feature_lh[len(feature_lh) // 2].to(device),
                    feature_rh[len(feature_rh) // 2].to(device), seed
                )
                output_lh = [output_lh_single.cpu()]
                output_rh = [output_rh_single.cpu()]
                left_offset = self.sample_rate // 2
                right_offset = 0
            
            # Process outputs
            def process_output(output_list, label):
                assert output_list[0].shape[0] == 1
                min_len = min([i.shape[2] for i in output_list])
                output = [i[:, :, :min_len] for i in output_list]
                output = torch.cat(output, 0)  # [sample_rate, C, T]
                output = output.mean(0).numpy()  # Average over temporal views
                
                # Post-processing: median filter
                if self.postprocess['type'] == 'median':
                    smoothed_output = np.zeros_like(output)
                    for c in range(output.shape[0]):
                        smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                    output = smoothed_output / smoothed_output.sum(0, keepdims=True)
                
                # Get predictions
                output = np.argmax(output, 0)
                output = restore_full_sequence(output,
                    full_len=label.shape[-1],
                    left_offset=left_offset,
                    right_offset=right_offset,
                    sample_rate=self.sample_rate
                )
                
                # Post-processing: mode filter
                if self.postprocess['type'] == 'mode':
                    output = mode_filter(output, self.postprocess['value'])
                
                # Post-processing: purge short segments
                if self.postprocess['type'] == 'purge':
                    trans, starts, ends = get_labels_start_end_time(output)
                    for e in range(len(trans)):
                        duration = ends[e] - starts[e]
                        if duration <= self.postprocess['value']:
                            if e == 0:
                                output[starts[e]:ends[e]] = trans[e + 1]
                            elif e == len(trans) - 1:
                                output[starts[e]:ends[e]] = trans[e - 1]
                            else:
                                mid = starts[e] + duration // 2
                                output[starts[e]:mid] = trans[e - 1]
                                output[mid:ends[e]] = trans[e + 1]
                
                label_np = label.squeeze(0).cpu().numpy()
                assert output.shape == label_np.shape
                return output
            
            output_lh_processed = process_output(output_lh, label_lh)
            output_rh_processed = process_output(output_rh, label_rh)
            
            return video, output_lh_processed, output_rh_processed, label_lh.squeeze(0).cpu().numpy(), label_rh.squeeze(0).cpu().numpy()
    
    def test(self, test_dataset: DualHandVideoFeatureDataset, mode: str, device: torch.device,
             label_dir_lh: str, label_dir_rh: str, result_dir: Optional[str] = None,
             model_path: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Evaluate model on test dataset
        
        Args:
            test_dataset: Test dataset
            mode: Evaluation mode
            device: Device
            label_dir_lh: Left hand ground truth label directory
            label_dir_rh: Right hand ground truth label directory
            result_dir: Directory to save predictions
            model_path: Path to model checkpoint (optional)
            
        Returns:
            Tuple of (result_dict_lh, result_dict_rh)
        """
        assert test_dataset.mode == 'test'
        self.model.eval()
        self.model.to(device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Create prediction directories
        if result_dir:
            pred_dir_lh = os.path.join(result_dir, 'prediction_lh')
            pred_dir_rh = os.path.join(result_dir, 'prediction_rh')
            os.makedirs(pred_dir_lh, exist_ok=True)
            os.makedirs(pred_dir_rh, exist_ok=True)
        
        # Predict on all videos
        with torch.no_grad():
            for video_idx in tqdm(range(len(test_dataset)), desc='Evaluating'):
                video, pred_lh, pred_rh, label_lh, label_rh = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path
                )
                
                # Convert to string labels
                pred_lh_str = [self.event_list[int(i)] for i in pred_lh]
                pred_rh_str = [self.event_list[int(i)] for i in pred_rh]
                
                # Save predictions
                if result_dir:
                    # Left hand
                    with open(os.path.join(pred_dir_lh, f'{video}.txt'), 'w') as f:
                        f.write('### Frame level recognition: ###\n')
                        f.write(' '.join(pred_lh_str))
                    
                    # Right hand
                    with open(os.path.join(pred_dir_rh, f'{video}.txt'), 'w') as f:
                        f.write('### Frame level recognition: ###\n')
                        f.write(' '.join(pred_rh_str))
        
        # Evaluate
        if result_dir:
            acc_lh, edit_lh, f1s_lh = func_eval(label_dir_lh, pred_dir_lh, test_dataset.video_list)
            acc_rh, edit_rh, f1s_rh = func_eval(label_dir_rh, pred_dir_rh, test_dataset.video_list)
        else:
            # If no result_dir, cannot evaluate
            acc_lh = edit_lh = acc_rh = edit_rh = 0
            f1s_lh = f1s_rh = np.zeros(3)
        
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


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dual-Hand Action Segmentation Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--device', type=int, default=-1, help='GPU device (-1 for CPU)')
    parser.add_argument('--one_stream', action='store_true', default=False,
                       help='Single-stream mode: copy LH to RH with label perturbation')
    parser.add_argument('--max_shift_frames', type=int, default=5,
                       help='Max frames to shift boundaries in single-stream mode')
    parser.add_argument('--max_perturbations', type=int, default=3,
                       help='Max number of boundaries to perturb in single-stream mode')
    parser.add_argument('--min_segment_len', type=int, default=3,
                       help='Minimum segment length to preserve in single-stream mode')
    args = parser.parse_args()
    
    # Load configuration
    all_params = load_config_file(args.config)
    locals().update(all_params)
    
    print(f'\n{"="*70}')
    print(f'Configuration: {args.config}')
    print(f'{"="*70}')
    for k, v in all_params.items():
        print(f'{k:<30} : {v}')
    print(f'{"="*70}')
    
    if args.one_stream:
        print(f'\n{"!"*70}')
        print(f'SINGLE-STREAM MODE ENABLED')
        print(f'{"!"*70}')
        print(f'Left-hand data will be copied to right-hand with label perturbation:')
        print(f'  - Max shift frames:   {args.max_shift_frames}')
        print(f'  - Max perturbations:  {args.max_perturbations}')
        print(f'  - Min segment length: {args.min_segment_len}')
        print(f'{"!"*70}\n')
    else:
        print()
    
    # Set device
    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != -1 else 'cpu')
    print(f'Using device: {device}\n')
    
    # Set paths
    feature_dir_lh = os.path.join(root_data_dir, dataset_name, feature_subdir_lh)
    feature_dir_rh = os.path.join(root_data_dir, dataset_name, feature_subdir_rh)
    label_dir_lh = os.path.join(root_data_dir, dataset_name, label_subdir_lh)
    label_dir_rh = os.path.join(root_data_dir, dataset_name, label_subdir_rh)
    mapping_file = os.path.join(root_data_dir, dataset_name, 'task_mapping.txt')
    
    # Load action list
    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)
    print(f'Number of action classes: {num_classes}')
    
    # Load video lists
    train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, split_dir, f'train.split{split_id}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, split_dir, f'test.split{split_id}.bundle'), dtype=str)
    
    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]
    
    print(f'Train videos: {len(train_video_list)}')
    print(f'Test videos: {len(test_video_list)}\n')
    
    # Load data
    print('Loading training data...')
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
    
    print('Loading test data...')
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
    
    # Create datasets
    train_train_dataset = DualHandVideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset = DualHandVideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = DualHandVideoFeatureDataset(test_data_dict, num_classes, mode='test')
    
    # Setup loss weights
    dual_hand_loss_weights = {
        'encoder_lh_ce_loss': loss_weights.get('encoder_ce_loss', 0.5) / 2,
        'encoder_lh_mse_loss': loss_weights.get('encoder_mse_loss', 0.025) / 2,
        'encoder_lh_boundary_loss': loss_weights.get('encoder_boundary_loss', 0.0) / 2,
        'encoder_rh_ce_loss': loss_weights.get('encoder_ce_loss', 0.5) / 2,
        'encoder_rh_mse_loss': loss_weights.get('encoder_mse_loss', 0.025) / 2,
        'encoder_rh_boundary_loss': loss_weights.get('encoder_boundary_loss', 0.0) / 2,
        'decoder_left_ce_loss': loss_weights.get('decoder_ce_loss', 0.5) / 2,
        'decoder_left_mse_loss': loss_weights.get('decoder_mse_loss', 0.025) / 2,
        'decoder_left_boundary_loss': loss_weights.get('decoder_boundary_loss', 0.1) / 2,
        'decoder_right_ce_loss': loss_weights.get('decoder_ce_loss', 0.5) / 2,
        'decoder_right_mse_loss': loss_weights.get('decoder_mse_loss', 0.025) / 2,
        'decoder_right_boundary_loss': loss_weights.get('decoder_boundary_loss', 0.1) / 2,
    }
    
    # Prepare perturbation config
    perturbation_config = {
        'max_shift_frames': args.max_shift_frames,
        'max_perturbations': args.max_perturbations,
        'min_segment_len': args.min_segment_len
    }
    
    # Create trainer
    trainer = DualHandTrainer(
        dict(encoder_params), dict(decoder_params), dict(diffusion_params),
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=device,
        one_stream=args.one_stream,
        perturbation_config=perturbation_config,
        boundary_smooth=boundary_smooth if 'boundary_smooth' in locals() else None
    )
    
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)
    final_result_dir = os.path.join(result_dir, naming)
    os.makedirs(final_result_dir, exist_ok=True)
    
    print(f'\nResults will be saved to: {final_result_dir}\n')
    
    # Train
    print('Starting training...\n')
    trainer.train(
        train_train_dataset, train_test_dataset, test_test_dataset,
        dual_hand_loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        label_dir_lh=label_dir_lh, label_dir_rh=label_dir_rh,
        result_dir=final_result_dir,
        log_freq=log_freq, log_train_results=log_train_results
    )

