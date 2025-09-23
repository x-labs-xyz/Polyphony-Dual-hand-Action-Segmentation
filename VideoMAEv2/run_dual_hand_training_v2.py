#!/usr/bin/env python3
"""
Dual-hand VideoMAEv2 training script using original dataset building approach.
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _pair

from timm.models import create_model
from timm.utils import ModelEma
from timm.models.layers import trunc_normal_

# Import existing modules
import models  # noqa: F401
import utils
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
    
    def get_num_layers(self):
        return self.shared_encoder.get_num_layers()
    
    def no_weight_decay(self):
        return self.shared_encoder.no_weight_decay()


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
        # Assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        # Handle both single and multiple samples (repeated augmentation)
        # Convert tensor to boolean if needed
        multiple_samples = batch['multiple_samples']
        if isinstance(multiple_samples, torch.Tensor):
            # If it's a tensor, check the first element or use any() for multi-element tensors
            if multiple_samples.numel() == 1:
                multiple_samples = multiple_samples.item()
            else:
                multiple_samples = multiple_samples[0].item() if multiple_samples.dim() > 0 else False
        
        if multiple_samples:
            # Repeated augmentation case - average over multiple augmented versions
            lh_frames_list = batch['lh_frames']
            rh_frames_list = batch['rh_frames']
            lh_labels = batch['lh_label'][0]  # Same labels for all augmentations
            rh_labels = batch['rh_label'][0]
            
            total_lh_loss = 0
            total_rh_loss = 0
            num_augs = len(lh_frames_list)
            
            for aug_idx in range(num_augs):
                lh_frames = lh_frames_list[aug_idx].to(device, non_blocking=True)
                rh_frames = rh_frames_list[aug_idx].to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(lh_frames, rh_frames)
                    lh_loss = loss_fn(outputs['lh_logits'], lh_labels.to(device))
                    rh_loss = loss_fn(outputs['rh_logits'], rh_labels.to(device))
                
                total_lh_loss += lh_loss
                total_rh_loss += rh_loss
            
            # Average losses across augmentations
            lh_loss = total_lh_loss / num_augs
            rh_loss = total_rh_loss / num_augs
            total_loss = lh_loss + rh_loss
            
        else:
            # Single sample case
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
        
        # Calculate accuracies for the last sample (in case of repeated augmentation)
        if multiple_samples:
            # Use the last augmented sample for accuracy calculation
            lh_acc1, lh_acc5 = accuracy(outputs['lh_logits'], lh_labels.to(device), topk=(1, 5))
            rh_acc1, rh_acc5 = accuracy(outputs['rh_logits'], rh_labels.to(device), topk=(1, 5))
        else:
            lh_acc1, lh_acc5 = accuracy(outputs['lh_logits'], lh_labels, topk=(1, 5))
            rh_acc1, rh_acc5 = accuracy(outputs['rh_logits'], rh_labels, topk=(1, 5))
        
        batch_size = lh_frames.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lh_loss=lh_loss_value)
        metric_logger.update(rh_loss=rh_loss_value)
        metric_logger.meters['lh_acc1'].update(lh_acc1.item(), n=batch_size)
        metric_logger.meters['lh_acc5'].update(lh_acc5.item(), n=batch_size)
        metric_logger.meters['rh_acc1'].update(rh_acc1.item(), n=batch_size)
        metric_logger.meters['rh_acc5'].update(rh_acc5.item(), n=batch_size)
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
    
    # Gather the stats from all processes
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
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* LH Acc@1 {lh_acc1.global_avg:.3f} LH Acc@5 {lh_acc5.global_avg:.3f} '
          'RH Acc@1 {rh_acc1.global_avg:.3f} RH Acc@5 {rh_acc5.global_avg:.3f} '
          'loss {losses.global_avg:.3f}'.format(
              lh_acc1=metric_logger.lh_acc1, lh_acc5=metric_logger.lh_acc5,
              rh_acc1=metric_logger.rh_acc1, rh_acc5=metric_logger.rh_acc5,
              losses=metric_logger.loss))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_args():
    parser = argparse.ArgumentParser('Dual-hand VideoMAEv2 fine-tuning', add_help=False)
    
    # Data parameters
    parser.add_argument('--lh_data_dir', required=True, help='Left-hand data directory')
    parser.add_argument('--rh_data_dir', required=True, help='Right-hand data directory')
    parser.add_argument('--lh_train_ann', required=True, help='Left-hand training annotation file')
    parser.add_argument('--rh_train_ann', required=True, help='Right-hand training annotation file')
    parser.add_argument('--lh_val_ann', required=True, help='Left-hand validation annotation file')
    parser.add_argument('--rh_val_ann', required=True, help='Right-hand validation annotation file')
    
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', help='Name of model to train')
    parser.add_argument('--lh_num_classes', type=int, default=400, help='Number of left-hand classes')
    parser.add_argument('--rh_num_classes', type=int, default=400, help='Number of right-hand classes')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--short_side_size', type=int, default=256, help='Short side size for resizing')
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
    
    # Additional arguments required by VideoClsDataset
    parser.add_argument('--reprob', type=float, default=0.25, help='Random erasing probability')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erasing mode')
    parser.add_argument('--recount', type=int, default=1, help='Random erasing count')
    parser.add_argument('--resplit', action='store_true', help='Random erasing resplit')
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup alpha')
    parser.add_argument('--cutmix', type=float, default=0.0, help='CutMix alpha')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='CutMix min/max')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of applying mixup or cutmix')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching between mixup and cutmix')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--nb_classes', type=int, default=400, help='Number of classes (set automatically)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic', help='Training interpolation')
    parser.add_argument('--sparse_sample', action='store_true', help='Use sparse sampling')
    parser.add_argument('--dense_sample', action='store_true', help='Use dense sampling')
    parser.add_argument('--num_segments', type=int, default=1, help='Number of segments')
    parser.add_argument('--update_freq', default=1, type=int, help='Update frequency')
    parser.add_argument('--use_mean_pooling', action='store_true', default=True, help='Use mean pooling')
    parser.add_argument('--init_scale', default=0.001, type=float, help='Init scale')
    parser.add_argument('--model_key', default='model|module', type=str, help='Model key')
    parser.add_argument('--model_prefix', default='', type=str, help='Model prefix')
    
    # Additional augmentation arguments required by VideoClsDataset
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', help='Auto augment policy')
    parser.add_argument('--color_jitter', type=float, default=0.4, help='Color jitter')
    parser.add_argument('--auto_augment', type=str, default='rand-m9-mstd0.5-inc1', help='Auto augment policy')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Training interpolation')
    parser.add_argument('--crop_pct', type=float, default=None, help='Input image center crop percent')
    parser.add_argument('--data_set', type=str, default='DUAL_HAND', help='Dataset name')
    
    # System parameters
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--log_dir', help='Log directory')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--finetune', help='Fine-tune from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='Start epoch')
    parser.add_argument('--eval', action='store_true', help='Only evaluate')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory for DataLoader')
    parser.add_argument('--save_ckpt_freq', default=10, type=int, help='Save checkpoint frequency')
    
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Initialize distributed mode
    utils.init_distributed_mode(args)
    
    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print("{}".format(args).replace(', ', ',\n'))
    
    device = torch.device(args.device)
    
    # Fix random seeds for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # Create datasets using original approach
    print("Creating datasets...")
    
    # Set nb_classes for compatibility with original dataset
    args.nb_classes = args.lh_num_classes
    
    train_dataset, args.lh_num_classes = build_dual_hand_datasets(
        is_train=True, test_mode=False, args=args)
    
    val_dataset, _ = build_dual_hand_datasets(
        is_train=False, test_mode=False, args=args)
    
    # Create data loaders
    if utils.get_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = DualHandVideoMAE(
        encoder_model_name=args.model,
        lh_num_classes=args.lh_num_classes,
        rh_num_classes=args.rh_num_classes,
        input_size=args.input_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        drop_path_rate=args.drop_path,
        use_mean_pooling=True
    )
    
    # Load pre-trained weights
    if args.finetune:
        print(f"Loading pre-trained weights from: {args.finetune}")
        try:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            
            # Get the model state dict
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
    
    for epoch in range(args.start_epoch, args.epochs):
        if utils.get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, train_loader, optimizer, epoch, device, loss_scaler,
            args.clip_grad, None, None,
            start_steps=epoch * len(train_loader),
            lr_schedule_values=lr_schedule_values
        )
        
        if val_loader is not None:
            test_stats = evaluate(model, val_loader, device)
            print(f"Accuracy of the network on validation videos: "
                  f"LH: {test_stats['lh_acc1']:.1f}%, RH: {test_stats['rh_acc1']:.1f}%")
            max_accuracy = max(max_accuracy, (test_stats['lh_acc1'] + test_stats['rh_acc1']) / 2)
            print(f'Max accuracy: {max_accuracy:.2f}%')
        
        # Log stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}
        
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # Save checkpoint
        if args.output_dir and (epoch % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, 
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
