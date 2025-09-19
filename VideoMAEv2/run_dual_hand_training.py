#!/usr/bin/env python3
"""
VideoMAEv2 Dual-Hand Action Recognition Training Script

This script trains a VideoMAEv2 model with:
- Shared encoder for both hands
- Separate decoders for left-hand and right-hand actions
- Online dataset loading for flexible video processing
- Support for extracting both shared and hand-specific features

Architecture:
    Input Video -> Shared Encoder -> [LH Decoder, RH Decoder] -> [LH Logits, RH Logits]
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import numpy as np
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma, accuracy
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

# Import existing modules
import models  # noqa: F401
import utils
from dataset.loader import get_video_loader
from dataset.build import build_dual_hand_datasets
from optim_factory import LayerDecayValueAssigner, create_optimizer, get_parameter_groups
from utils import NativeScalerWithGradNormCount as NativeScaler


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Using original VideoClsDataset approach for better performance and robustness


class DualHandVideoMAE(nn.Module):
        """
        Args:
            lh_data_dir: Directory containing left-hand videos
            rh_data_dir: Directory containing right-hand videos  
            lh_annotation_file: Left-hand annotation CSV file
            rh_annotation_file: Right-hand annotation CSV file
            mode: 'train', 'val', or 'test'
            num_frames: Number of frames per clip
            sampling_rate: Temporal sampling rate
            input_size: Input image size
            short_side_size: Short side resize size
            crop_size: Final crop size
            num_clips: Number of clips per video during training
            test_num_clips: Number of clips per video during testing
        """
        self.lh_data_dir = Path(lh_data_dir)
        self.rh_data_dir = Path(rh_data_dir)
        self.mode = mode
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.input_size = input_size
        self.short_side_size = short_side_size
        self.crop_size = crop_size
        self.num_clips = num_clips if mode == 'train' else test_num_clips
        self.num_sample = num_sample
        
        # Load annotations
        self.lh_samples = self._load_annotations(lh_annotation_file, 'lh')
        self.rh_samples = self._load_annotations(rh_annotation_file, 'rh')
        
        # Video loader
        self.video_loader = get_video_loader()
        
        # Validate video files exist (during initialization for both train and val)
        self._validate_video_files()
        
        print(f"Loaded {len(self.lh_samples)} left-hand samples")
        print(f"Loaded {len(self.rh_samples)} right-hand samples")
    
    def _load_annotations(self, annotation_file: str, hand_type: str) -> List[Dict]:
        """Load annotation file and return list of samples"""
        samples = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_path = parts[0]
                    label = int(parts[1])
                    samples.append({
                        'video_path': video_path,
                        'label': label,
                        'hand_type': hand_type
                    })
        return samples
    
    def _validate_video_files(self):
        """Validate that video files exist and report missing ones"""
        print(f"Validating video files for {self.mode} mode...")
        
        lh_missing = 0
        rh_missing = 0
        
        # Determine video subdirectory based on mode
        video_subdir = 'videos_train' if self.mode == 'train' else 'videos_val'
        
        # Check left-hand videos
        for sample in self.lh_samples[:50]:  # Check first 50 to avoid long delays
            video_path = Path(self.lh_data_dir) / video_subdir / sample['video_path']
            if not video_path.exists():
                lh_missing += 1
        
        # Check right-hand videos  
        for sample in self.rh_samples[:50]:  # Check first 50 to avoid long delays
            video_path = Path(self.rh_data_dir) / video_subdir / sample['video_path']
            if not video_path.exists():
                rh_missing += 1
        
        if lh_missing > 0 or rh_missing > 0:
            print(f"WARNING: Found missing videos in first 50 {self.mode} samples:")
            print(f"  Left-hand missing: {lh_missing}/50")
            print(f"  Right-hand missing: {rh_missing}/50")
            print(f"  {self.mode.capitalize()} will continue with dummy frames for missing videos")
        else:
            print(f"Video validation passed - all checked {self.mode} files exist")
    
    def _apply_transforms(self, frames):
        """Apply transformation pipeline based on mode"""
        if self.mode == 'train':
            frames = self._random_crop(frames)
            frames = self._random_horizontal_flip(frames)
        else:
            frames = self._center_crop(frames)
        frames = self._normalize(frames)
        return frames
    
    def _random_crop(self, frames):
        """Random crop for training"""
        # frames: [T, H, W, C]
        T, H, W, C = frames.shape
        
        # Convert to float for interpolation
        frames = frames.float()
        
        # Resize short side
        if H < W:
            new_H, new_W = self.short_side_size, int(W * self.short_side_size / H)
        else:
            new_H, new_W = int(H * self.short_side_size / W), self.short_side_size
        
        frames = torch.nn.functional.interpolate(
            frames.permute(3, 0, 1, 2),  # [C, T, H, W]
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        ).permute(1, 2, 3, 0)  # [T, H, W, C]
        
        # Random crop
        _, H, W, _ = frames.shape
        top = random.randint(0, max(0, H - self.crop_size))
        left = random.randint(0, max(0, W - self.crop_size))
        
        frames = frames[:, top:top+self.crop_size, left:left+self.crop_size, :]
        return frames
    
    def _center_crop(self, frames):
        """Center crop for validation/testing"""
        # frames: [T, H, W, C]
        T, H, W, C = frames.shape
        
        # Convert to float for interpolation
        frames = frames.float()
        
        # Resize short side
        if H < W:
            new_H, new_W = self.short_side_size, int(W * self.short_side_size / H)
        else:
            new_H, new_W = int(H * self.short_side_size / W), self.short_side_size
        
        frames = torch.nn.functional.interpolate(
            frames.permute(3, 0, 1, 2),  # [C, T, H, W]
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        ).permute(1, 2, 3, 0)  # [T, H, W, C]
        
        # Center crop
        _, H, W, _ = frames.shape
        top = (H - self.crop_size) // 2
        left = (W - self.crop_size) // 2
        
        frames = frames[:, top:top+self.crop_size, left:left+self.crop_size, :]
        return frames
    
    def _random_horizontal_flip(self, frames):
        """Random horizontal flip for training"""
        if random.random() < 0.5:
            frames = torch.flip(frames, dims=[2])  # Flip width dimension
        return frames
    
    def _normalize(self, frames):
        """Normalize frames"""
        # Convert to [0, 1] and normalize (frames should already be float from crop functions)
        frames = frames / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        frames = (frames - mean) / std
        return frames.permute(3, 0, 1, 2)  # [C, T, H, W]
    
    def _load_video_frames(self, video_path: str, hand_type: str) -> torch.Tensor:
        """Load video frames using online processing"""
        # Determine video subdirectory based on mode
        video_subdir = 'videos_train' if self.mode == 'train' else 'videos_val'
        
        # Determine full video path
        if hand_type == 'lh':
            full_path = Path(self.lh_data_dir) / video_subdir / video_path
        else:
            full_path = Path(self.rh_data_dir) / video_subdir / video_path
        
        # Check if file exists first
        if not full_path.exists():
            # Silently return dummy frames for missing videos
            return torch.zeros(self.num_frames, 224, 224, 3, dtype=torch.uint8)
        
        try:
            # Load video
            video_reader = self.video_loader(str(full_path))
            video_length = len(video_reader)
            
            # Sample frames
            if video_length <= self.num_frames * self.sampling_rate:
                # Video too short, repeat frames
                frame_indices = np.linspace(0, video_length - 1, self.num_frames).astype(int)
            else:
                # Random start position for training, center for validation/testing
                if self.mode == 'train':
                    max_start = video_length - self.num_frames * self.sampling_rate
                    start_idx = random.randint(0, max_start)
                else:
                    start_idx = (video_length - self.num_frames * self.sampling_rate) // 2
                
                frame_indices = np.arange(start_idx, start_idx + self.num_frames * self.sampling_rate, self.sampling_rate)
                frame_indices = np.clip(frame_indices, 0, video_length - 1)
            
            # Extract frames
            frames = video_reader.get_batch(frame_indices).asnumpy()
            frames = torch.from_numpy(frames)  # [T, H, W, C]
            
            return frames
            
        except Exception as e:
            # Silently return dummy frames for corrupted videos
            return torch.zeros(self.num_frames, 224, 224, 3, dtype=torch.uint8)
    
    def __len__(self):
        # Return length of the larger dataset
        return max(len(self.lh_samples), len(self.rh_samples))
    
    def __getitem__(self, idx):
        # Sample from both hands (cycle through smaller dataset if needed)
        lh_idx = idx % len(self.lh_samples)
        rh_idx = idx % len(self.rh_samples)
        
        lh_sample = self.lh_samples[lh_idx]
        rh_sample = self.rh_samples[rh_idx]
        
        # Load video frames
        lh_frames = self._load_video_frames(lh_sample['video_path'], 'lh')
        rh_frames = self._load_video_frames(rh_sample['video_path'], 'rh')
        
        # Apply transforms
        lh_frames = self._apply_transforms(lh_frames)
        rh_frames = self._apply_transforms(rh_frames)
        
        return {
            'lh_frames': lh_frames,  # [C, T, H, W]
            'rh_frames': rh_frames,  # [C, T, H, W]
            'lh_label': lh_sample['label'],
            'rh_label': rh_sample['label'],
            'lh_video_path': lh_sample['video_path'],
            'rh_video_path': rh_sample['video_path']
        }


class DualHandVideoMAE(nn.Module):
    """VideoMAEv2 with shared encoder and separate decoders for dual-hand action recognition"""
    
    def __init__(self,
                 encoder_model_name: str = 'vit_base_patch16_224',
                 lh_num_classes: int = 400,
                 rh_num_classes: int = 400,
                 input_size: int = 224,
                 num_frames: int = 16,
                 tubelet_size: int = 2,
                 drop_path_rate: float = 0.1,
                 use_mean_pooling: bool = True):
        """
        Initialize dual-hand VideoMAE model
        
        Args:
            encoder_model_name: Base encoder model name
            lh_num_classes: Number of left-hand action classes
            rh_num_classes: Number of right-hand action classes
            input_size: Input image size
            num_frames: Number of frames
            tubelet_size: Temporal tubelet size
            drop_path_rate: Drop path rate
            use_mean_pooling: Use mean pooling for features
        """
        super().__init__()
        
        # Create shared encoder (with temporary classification head, will be replaced)
        self.shared_encoder = create_model(
            encoder_model_name,
            img_size=input_size,
            pretrained=False,
            num_classes=1000,  # Temporary head to avoid Identity layer issues
            all_frames=num_frames,
            tubelet_size=tubelet_size,
            drop_path_rate=drop_path_rate,
            use_mean_pooling=use_mean_pooling
        )
        
        # Remove the classification head to use only the feature extractor
        self.shared_encoder.head = nn.Identity()
        
        # Get encoder output dimension
        self.embed_dim = self.shared_encoder.embed_dim
        
        # Separate decoders for each hand
        self.lh_decoder = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim // 2, lh_num_classes)
        )
        
        self.rh_decoder = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim // 2, rh_num_classes)
        )
        
        self.lh_num_classes = lh_num_classes
        self.rh_num_classes = rh_num_classes
        
        # Initialize decoder weights
        self._init_decoder_weights()
    
    def _init_decoder_weights(self):
        """Initialize decoder weights with smaller std to prevent instability"""
        for decoder in [self.lh_decoder, self.rh_decoder]:
            for m in decoder.modules():
                if isinstance(m, nn.Linear):
                    # Use smaller std to prevent gradient explosion
                    trunc_normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward_shared_features(self, x):
        """Extract shared features from encoder"""
        return self.shared_encoder.forward_features(x)
    
    def forward_lh_classifier(self, shared_features):
        """Forward through left-hand decoder"""
        return self.lh_decoder(shared_features)
    
    def forward_rh_classifier(self, shared_features):
        """Forward through right-hand decoder"""
        return self.rh_decoder(shared_features)
    
    def forward(self, lh_frames, rh_frames):
        """Forward pass for both hands"""
        # Extract shared features for both hands
        lh_shared_features = self.forward_shared_features(lh_frames)
        rh_shared_features = self.forward_shared_features(rh_frames)
        
        # Get hand-specific predictions
        lh_logits = self.forward_lh_classifier(lh_shared_features)
        rh_logits = self.forward_rh_classifier(rh_shared_features)
        
        return {
            'lh_logits': lh_logits,
            'rh_logits': rh_logits,
            'lh_shared_features': lh_shared_features,
            'rh_shared_features': rh_shared_features
        }
    
    def extract_features(self, frames, hand_type='lh'):
        """Extract both shared and hand-specific features"""
        with torch.no_grad():
            shared_features = self.forward_shared_features(frames)
            
            if hand_type == 'lh':
                # Get intermediate features from LH decoder
                decoder = self.lh_decoder
            else:
                # Get intermediate features from RH decoder
                decoder = self.rh_decoder
            
            # Extract features from decoder layers
            x = shared_features
            hand_specific_features = []
            
            for i, layer in enumerate(decoder):
                x = layer(x)
                if i < len(decoder) - 1:  # Don't include final classification layer
                    hand_specific_features.append(x.clone())
            
            return {
                'shared_features': shared_features,
                'hand_specific_features': hand_specific_features,
                'final_logits': x
            }


def train_one_epoch(model: DualHandVideoMAE,
                   data_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   device: torch.device,
                   loss_scaler,
                   max_norm: float = 0,
                   log_writer=None,
                   lr_scheduler=None,
                   start_steps=None,
                   lr_schedule_values=None,
                   wd_schedule_values=None):
    """Train for one epoch"""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    loss_fn = nn.CrossEntropyLoss()
    
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move to device
        lh_frames = batch['lh_frames'].to(device, non_blocking=True)
        rh_frames = batch['rh_frames'].to(device, non_blocking=True)
        lh_labels = batch['lh_label'].to(device, non_blocking=True)
        rh_labels = batch['rh_label'].to(device, non_blocking=True)
        
        # Update learning rate
        if lr_schedule_values is not None:
            it = start_steps + step if start_steps else step
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(lh_frames, rh_frames)
            
            # Calculate losses
            lh_loss = loss_fn(outputs['lh_logits'], lh_labels)
            rh_loss = loss_fn(outputs['rh_logits'], rh_labels)
            total_loss = lh_loss + rh_loss
        
        loss_value = total_loss.item()
        lh_loss_value = lh_loss.item()
        rh_loss_value = rh_loss.item()
        
        # Check for NaN/Inf in individual losses
        if not (math.isfinite(loss_value) and math.isfinite(lh_loss_value) and math.isfinite(rh_loss_value)):
            print(f"Non-finite loss detected: total={loss_value}, lh={lh_loss_value}, rh={rh_loss_value}")
            print("Skipping this batch...")
            continue
        
        # Backward pass with loss scaling
        optimizer.zero_grad()
        
        # Scale the loss and backward pass
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(total_loss, optimizer, clip_grad=max_norm,
                   parameters=model.parameters(), create_graph=is_second_order)
        
        torch.cuda.synchronize()
        
        # Calculate accuracies
        lh_acc1, lh_acc5 = accuracy(outputs['lh_logits'], lh_labels, topk=(1, 5))
        rh_acc1, rh_acc5 = accuracy(outputs['rh_logits'], rh_labels, topk=(1, 5))
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(lh_loss=lh_loss_value)
        metric_logger.update(rh_loss=rh_loss_value)
        metric_logger.update(lh_acc1=lh_acc1.item())
        metric_logger.update(lh_acc5=lh_acc5.item())
        metric_logger.update(rh_acc1=rh_acc1.item())
        metric_logger.update(rh_acc5=rh_acc5.item())
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lh_loss=lh_loss.item(), head="loss")
            log_writer.update(rh_loss=rh_loss.item(), head="loss")
            log_writer.update(lh_acc1=lh_acc1.item(), head="acc")
            log_writer.update(lh_acc5=lh_acc5.item(), head="acc")
            log_writer.update(rh_acc1=rh_acc1.item(), head="acc")
            log_writer.update(rh_acc5=rh_acc5.item(), head="acc")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.set_step()
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: DualHandVideoMAE, data_loader: DataLoader, device: torch.device):
    """Evaluate model"""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    loss_fn = nn.CrossEntropyLoss()
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        lh_frames = batch['lh_frames'].to(device, non_blocking=True)
        rh_frames = batch['rh_frames'].to(device, non_blocking=True)
        lh_labels = batch['lh_label'].to(device, non_blocking=True)
        rh_labels = batch['rh_label'].to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(lh_frames, rh_frames)
            lh_loss = loss_fn(outputs['lh_logits'], lh_labels)
            rh_loss = loss_fn(outputs['rh_logits'], rh_labels)
            total_loss = lh_loss + rh_loss
        
        # Calculate accuracies
        lh_acc1, lh_acc5 = accuracy(outputs['lh_logits'], lh_labels, topk=(1, 5))
        rh_acc1, rh_acc5 = accuracy(outputs['rh_logits'], rh_labels, topk=(1, 5))
        
        batch_size = lh_frames.shape[0]
        metric_logger.update(loss=total_loss.item())
        metric_logger.update(lh_loss=lh_loss.item())
        metric_logger.update(rh_loss=rh_loss.item())
        metric_logger.meters['lh_acc1'].update(lh_acc1.item(), n=batch_size)
        metric_logger.meters['lh_acc5'].update(lh_acc5.item(), n=batch_size)
        metric_logger.meters['rh_acc1'].update(rh_acc1.item(), n=batch_size)
        metric_logger.meters['rh_acc5'].update(rh_acc5.item(), n=batch_size)
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser('VideoMAEv2 Dual-Hand Action Recognition Training')
    
    # Data parameters
    parser.add_argument('--lh_data_dir', required=True, help='Left-hand video directory')
    parser.add_argument('--rh_data_dir', required=True, help='Right-hand video directory')
    parser.add_argument('--lh_train_ann', required=True, help='Left-hand training annotation file')
    parser.add_argument('--rh_train_ann', required=True, help='Right-hand training annotation file')
    parser.add_argument('--lh_val_ann', required=True, help='Left-hand validation annotation file')
    parser.add_argument('--rh_val_ann', required=True, help='Right-hand validation annotation file')
    
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', help='Base encoder model')
    parser.add_argument('--lh_num_classes', type=int, default=400, help='Number of left-hand classes')
    parser.add_argument('--rh_num_classes', type=int, default=400, help='Number of right-hand classes')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--sampling_rate', type=int, default=4, help='Temporal sampling rate')
    parser.add_argument('--tubelet_size', type=int, default=2, help='Temporal tubelet size')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--layer_decay', type=float, default=0.9, help='Layer decay for different LR per layer')
    parser.add_argument('--opt_betas', type=float, nargs='+', default=[0.9, 0.999], help='Optimizer betas')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-8, help='Warmup learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='Gradient clipping')
    parser.add_argument('--opt', type=str, default='adamw', help='Optimizer type')
    parser.add_argument('--opt_eps', type=float, default=1e-8, help='Optimizer epsilon')
    parser.add_argument('--num_sample', type=int, default=2, help='Repeated augmentation samples')
    parser.add_argument('--test_num_segment', type=int, default=5, help='Number of segments for testing')
    parser.add_argument('--test_num_crop', type=int, default=3, help='Number of crops for testing')
    
    # System parameters
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--log_dir', help='Log directory')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data workers')
    parser.add_argument('--pin_mem', action='store_true', help='Pin memory')
    parser.add_argument('--save_ckpt_freq', type=int, default=10, help='Save checkpoint frequency')
    
    # Pre-trained model
    parser.add_argument('--finetune', help='Path to pre-trained checkpoint')
    parser.add_argument('--resume', help='Resume from checkpoint')
    
    # Distributed training
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = get_args()
    
    # Initialize distributed training
    utils.init_distributed_mode(args)
    
    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"args: {args}")
    
    device = torch.device(args.device)
    
    # Fix random seeds
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    cudnn.benchmark = True
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = DualHandVideoDataset(
        lh_data_dir=args.lh_data_dir,
        rh_data_dir=args.rh_data_dir,
        lh_annotation_file=args.lh_train_ann,
        rh_annotation_file=args.rh_train_ann,
        mode='train',
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        input_size=args.input_size,
        num_sample=args.num_sample
    )
    
    val_dataset = DualHandVideoDataset(
        lh_data_dir=args.lh_data_dir,
        rh_data_dir=args.rh_data_dir,
        lh_annotation_file=args.lh_val_ann,
        rh_annotation_file=args.rh_val_ann,
        mode='val',
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        input_size=args.input_size,
        test_num_clips=args.test_num_segment,
        num_sample=1  # No repeated augmentation for validation
    )
    
    # Create data loaders
    if utils.get_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=(train_sampler is None)
    )
    
    val_loader = DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    # Create model
    print(f"Creating model: {args.model}")
    model = DualHandVideoMAE(
        encoder_model_name=args.model,
        lh_num_classes=args.lh_num_classes,
        rh_num_classes=args.rh_num_classes,
        input_size=args.input_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        drop_path_rate=args.drop_path
    )
    
    # Load pre-trained weights
    if args.finetune:
        print(f"Loading pre-trained weights from: {args.finetune}")
        try:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            
            # Load encoder weights (excluding head)
            encoder_state_dict = {}
            for k, v in checkpoint_model.items():
                if k.startswith('head'):
                    continue  # Skip classification head
                encoder_state_dict[k] = v
            
            msg = model.shared_encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"Loading encoder weights: {msg}")
            
            # Check if we loaded any weights successfully
            if len(msg.missing_keys) == len(list(model.shared_encoder.state_dict().keys())):
                print("WARNING: No weights were loaded successfully. Training from scratch.")
            else:
                print(f"Successfully loaded {len(encoder_state_dict)} checkpoint parameters")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from scratch...")
    
    model.to(device)
    
    # Wrap model for distributed training
    if utils.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu] if hasattr(args, 'gpu') else None)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Calculate scaled learning rate (like original training)
    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    
    print(f"Scaled LR = {args.lr:.8f}")
    print(f"Batch size = {total_batch_size}")
    
    # Layer decay functionality
    num_layers = model_without_ddp.shared_encoder.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay**(num_layers + 1 - i) 
                 for i in range(num_layers + 2))
        )
        print(f"Layer decay values = {assigner.values}")
    else:
        assigner = None
    
    # Create optimizer with layer decay
    skip_weight_decay_list = getattr(model_without_ddp.shared_encoder, 'no_weight_decay', lambda: [])()
    print("Skip weight decay list: ", skip_weight_decay_list)
    
    if assigner is not None:
        optimizer = create_optimizer(
            args, model_without_ddp, 
            get_num_layer=assigner.get_layer_id,
            get_layer_scale=assigner.get_scale,
            skip_list=skip_weight_decay_list
        )
    else:
        param_groups = get_parameter_groups(
            model_without_ddp, 
            args.weight_decay,
            skip_list=skip_weight_decay_list
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=args.opt_betas)
    loss_scaler = NativeScaler()
    
    # Learning rate scheduler
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    max_accuracy = 0.0
    
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train one epoch
        train_stats = train_one_epoch(
            model, train_loader, optimizer, epoch, device, loss_scaler,
            args.clip_grad, lr_schedule_values=lr_schedule_values,
            start_steps=epoch * len(train_loader)
        )
        
        # Evaluate
        val_stats = evaluate(model, val_loader, device)
        
        print(f"Accuracy of the network on the validation set: LH {val_stats['lh_acc1']:.1f}%, RH {val_stats['rh_acc1']:.1f}%")
        
        # Save checkpoint
        if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )
        
        # Log stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'val_{k}': v for k, v in val_stats.items()},
                    'epoch': epoch}
        
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
