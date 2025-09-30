# Engine for dual-hand video action recognition training with semantic alignment
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
import numpy as np
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import utils
from models.modeling_finetune_dual_semantic import SemanticAlignmentLoss


def train_dual_hand_batch_with_semantic(model, samples_dict, targets_dict, criterion, semantic_criterion, semantic_weight=0.1):
    """Train batch for dual hand model with semantic alignment"""
    # Model expects dict input for dual mode
    outputs = model(samples_dict, return_semantic_features=True)
    
    # Compute action recognition loss for each hand
    lh_loss = criterion(outputs['lh_pred'], targets_dict['lh_label'])
    rh_loss = criterion(outputs['rh_pred'], targets_dict['rh_label'])
    
    # Combine action recognition losses (equal weighting)
    action_loss = (lh_loss + rh_loss) / 2.0
    
    # Compute semantic alignment loss if semantic features are available
    semantic_loss = 0.0
    semantic_loss_dict = {}
    
    if 'lh_semantic' in samples_dict and 'lh_aligned_features' in outputs:
        # Left hand semantic alignment
        lh_aligned = outputs['lh_aligned_features']
        # Handle both 2D and 3D aligned features
        if lh_aligned.dim() == 3:
            lh_aligned = lh_aligned.mean(dim=1)  # Average over sequence [B, T, D] -> [B, D]
        
        lh_semantic_loss = semantic_criterion(lh_aligned, samples_dict['lh_semantic'])
        semantic_loss += lh_semantic_loss
        semantic_loss_dict['lh_semantic_loss'] = lh_semantic_loss.item()
    
    if 'rh_semantic' in samples_dict and 'rh_aligned_features' in outputs:
        # Right hand semantic alignment
        rh_aligned = outputs['rh_aligned_features']
        # Handle both 2D and 3D aligned features
        if rh_aligned.dim() == 3:
            rh_aligned = rh_aligned.mean(dim=1)  # Average over sequence [B, T, D] -> [B, D]
        
        rh_semantic_loss = semantic_criterion(rh_aligned, samples_dict['rh_semantic'])
        semantic_loss += rh_semantic_loss
        semantic_loss_dict['rh_semantic_loss'] = rh_semantic_loss.item()
    
    # Combine losses
    total_loss = action_loss + semantic_weight * semantic_loss
    
    loss_dict = {
        'lh_loss': lh_loss.item(),
        'rh_loss': rh_loss.item(),
        'action_loss': action_loss.item(),
        'semantic_loss': semantic_loss.item() if isinstance(semantic_loss, torch.Tensor) else semantic_loss,
        **semantic_loss_dict
    }
    
    return total_loss, outputs, loss_dict


def train_one_epoch_dual_with_semantic(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    semantic_weight: float = 0.1,
    semantic_loss_type: str = 'adaptive'
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # Initialize semantic alignment loss
    semantic_criterion = SemanticAlignmentLoss(loss_type=semantic_loss_type)

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step

        # Update LR & WD
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Handle different batch formats
        if isinstance(batch_data, dict):
            # Handle multiple samples first (check before calling .to())
            multiple_samples = batch_data.get('multiple_samples', False)
            if hasattr(multiple_samples, 'item'):  # Convert tensor to Python bool
                if multiple_samples.numel() == 1:  # Single element tensor
                    multiple_samples = multiple_samples.item()
                else:  # Batched tensor - check if all elements are the same
                    multiple_samples = multiple_samples[0].item()  # Take first element
            
            if multiple_samples and isinstance(batch_data['lh_frames'], list):
                # For repeated augmentation - take first augmentation for now
                lh_frames = batch_data['lh_frames'][0].to(device, non_blocking=True)
                rh_frames = batch_data['rh_frames'][0].to(device, non_blocking=True)
                lh_labels = batch_data['lh_label']
                rh_labels = batch_data['rh_label']
                if isinstance(lh_labels, list):
                    lh_labels = lh_labels[0]
                if isinstance(rh_labels, list):
                    rh_labels = rh_labels[0]
                lh_labels = lh_labels.to(device, non_blocking=True)
                rh_labels = rh_labels.to(device, non_blocking=True)
                
                # Handle semantic features
                lh_semantic = batch_data.get('lh_semantic', None)
                rh_semantic = batch_data.get('rh_semantic', None)
                if lh_semantic is not None:
                    lh_semantic = lh_semantic.to(device, non_blocking=True)
                if rh_semantic is not None:
                    rh_semantic = rh_semantic.to(device, non_blocking=True)
            else:
                # Single sample case
                lh_frames = batch_data['lh_frames'].to(device, non_blocking=True)
                rh_frames = batch_data['rh_frames'].to(device, non_blocking=True)
                lh_labels = batch_data['lh_label'].to(device, non_blocking=True)
                rh_labels = batch_data['rh_label'].to(device, non_blocking=True)
                
                # Handle semantic features
                lh_semantic = batch_data.get('lh_semantic', None)
                rh_semantic = batch_data.get('rh_semantic', None)
                if lh_semantic is not None:
                    lh_semantic = lh_semantic.to(device, non_blocking=True)
                if rh_semantic is not None:
                    rh_semantic = rh_semantic.to(device, non_blocking=True)
        else:
            # Fallback for standard format (shouldn't happen with DualHandDataset)
            raise ValueError("Expected dict format from DualHandDataset")

        samples_dict = {
            'lh_frames': lh_frames, 
            'rh_frames': rh_frames,
            'lh_semantic': lh_semantic,
            'rh_semantic': rh_semantic
        }
        targets_dict = {'lh_label': lh_labels, 'rh_label': rh_labels}

        # Apply mixup if enabled (apply to both hands)
        if mixup_fn is not None:
            B, C, T, H, W = lh_frames.shape
            # Apply mixup to left hand
            lh_frames_2d = lh_frames.view(B, C * T, H, W)
            lh_frames_mixed, lh_labels_mixed = mixup_fn(lh_frames_2d, lh_labels)
            samples_dict['lh_frames'] = lh_frames_mixed.view(B, C, T, H, W)
            targets_dict['lh_label'] = lh_labels_mixed
            
            # Apply mixup to right hand
            rh_frames_2d = rh_frames.view(B, C * T, H, W)
            rh_frames_mixed, rh_labels_mixed = mixup_fn(rh_frames_2d, rh_labels)
            samples_dict['rh_frames'] = rh_frames_mixed.view(B, C, T, H, W)
            targets_dict['rh_label'] = rh_labels_mixed

        if loss_scaler is None:
            samples_dict['lh_frames'] = samples_dict['lh_frames'].half()
            samples_dict['rh_frames'] = samples_dict['rh_frames'].half()
            loss, outputs, loss_dict = train_dual_hand_batch_with_semantic(
                model, samples_dict, targets_dict, criterion, semantic_criterion, semantic_weight
            )
        else:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, outputs, loss_dict = train_dual_hand_batch_with_semantic(
                    model, samples_dict, targets_dict, criterion, semantic_criterion, semantic_weight
                )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            grad_norm = model.get_global_grad_norm()
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(
                loss, optimizer, clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # Compute accuracy for both hands
        if mixup_fn is None:
            lh_acc = (outputs['lh_pred'].max(-1)[-1] == lh_labels).float().mean()
            rh_acc = (outputs['rh_pred'].max(-1)[-1] == rh_labels).float().mean()
        else:
            lh_acc = None
            rh_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(lh_loss=loss_dict['lh_loss'])
        metric_logger.update(rh_loss=loss_dict['rh_loss'])
        metric_logger.update(action_loss=loss_dict['action_loss'])
        metric_logger.update(semantic_loss=loss_dict['semantic_loss'])
        metric_logger.update(lh_acc=lh_acc)
        metric_logger.update(rh_acc=rh_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        
        # Update semantic-specific metrics
        if 'lh_semantic_loss' in loss_dict:
            metric_logger.update(lh_semantic_loss=loss_dict['lh_semantic_loss'])
        if 'rh_semantic_loss' in loss_dict:
            metric_logger.update(rh_semantic_loss=loss_dict['rh_semantic_loss'])
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lh_loss=loss_dict['lh_loss'], head="loss")
            log_writer.update(rh_loss=loss_dict['rh_loss'], head="loss")
            log_writer.update(action_loss=loss_dict['action_loss'], head="loss")
            log_writer.update(semantic_loss=loss_dict['semantic_loss'], head="loss")
            log_writer.update(lh_acc=lh_acc, head="loss")
            log_writer.update(rh_acc=rh_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch_dual_with_semantic(data_loader, model, device, semantic_weight=0.1, semantic_loss_type='adaptive'):
    criterion = torch.nn.CrossEntropyLoss()
    semantic_criterion = SemanticAlignmentLoss(loss_type=semantic_loss_type)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    model.eval()

    for batch_data in metric_logger.log_every(data_loader, 10, header):
        if isinstance(batch_data, dict):
            # Handle multiple samples first (check before calling .to())
            multiple_samples = batch_data.get('multiple_samples', False)
            if hasattr(multiple_samples, 'item'):  # Convert tensor to Python bool
                if multiple_samples.numel() == 1:  # Single element tensor
                    multiple_samples = multiple_samples.item()
                else:  # Batched tensor - check if all elements are the same
                    multiple_samples = multiple_samples[0].item()  # Take first element
            
            if multiple_samples and isinstance(batch_data['lh_frames'], list):
                # For repeated augmentation - take first augmentation for now
                lh_frames = batch_data['lh_frames'][0].to(device, non_blocking=True)
                rh_frames = batch_data['rh_frames'][0].to(device, non_blocking=True)
                lh_labels = batch_data['lh_label']
                rh_labels = batch_data['rh_label']
                if isinstance(lh_labels, list):
                    lh_labels = lh_labels[0]
                if isinstance(rh_labels, list):
                    rh_labels = rh_labels[0]
                lh_labels = lh_labels.to(device, non_blocking=True)
                rh_labels = rh_labels.to(device, non_blocking=True)
                
                # Handle semantic features
                lh_semantic = batch_data.get('lh_semantic', None)
                rh_semantic = batch_data.get('rh_semantic', None)
                if lh_semantic is not None:
                    lh_semantic = lh_semantic.to(device, non_blocking=True)
                if rh_semantic is not None:
                    rh_semantic = rh_semantic.to(device, non_blocking=True)
            else:
                # Single sample case
                lh_frames = batch_data['lh_frames'].to(device, non_blocking=True)
                rh_frames = batch_data['rh_frames'].to(device, non_blocking=True)
                lh_labels = batch_data['lh_label'].to(device, non_blocking=True)
                rh_labels = batch_data['rh_label'].to(device, non_blocking=True)
                
                # Handle semantic features
                lh_semantic = batch_data.get('lh_semantic', None)
                rh_semantic = batch_data.get('rh_semantic', None)
                if lh_semantic is not None:
                    lh_semantic = lh_semantic.to(device, non_blocking=True)
                if rh_semantic is not None:
                    rh_semantic = rh_semantic.to(device, non_blocking=True)
            
            samples_dict = {
                'lh_frames': lh_frames, 
                'rh_frames': rh_frames,
                'lh_semantic': lh_semantic,
                'rh_semantic': rh_semantic
            }
            
            with torch.cuda.amp.autocast():
                outputs = model(samples_dict, return_semantic_features=True)
                lh_loss = criterion(outputs['lh_pred'], lh_labels)
                rh_loss = criterion(outputs['rh_pred'], rh_labels)
                action_loss = (lh_loss + rh_loss) / 2.0
                
                # Compute semantic alignment loss
                semantic_loss = 0.0
                if lh_semantic is not None and 'lh_aligned_features' in outputs:
                    lh_aligned = outputs['lh_aligned_features']
                    # Handle both 2D and 3D aligned features
                    if lh_aligned.dim() == 3:
                        lh_aligned = lh_aligned.mean(dim=1)  # Average over sequence [B, T, D] -> [B, D]
                    
                    lh_semantic_loss = semantic_criterion(lh_aligned, lh_semantic)
                    semantic_loss += lh_semantic_loss
                
                if rh_semantic is not None and 'rh_aligned_features' in outputs:
                    rh_aligned = outputs['rh_aligned_features']
                    # Handle both 2D and 3D aligned features
                    if rh_aligned.dim() == 3:
                        rh_aligned = rh_aligned.mean(dim=1)  # Average over sequence [B, T, D] -> [B, D]
                    
                    rh_semantic_loss = semantic_criterion(rh_aligned, rh_semantic)
                    semantic_loss += rh_semantic_loss
                
                total_loss = action_loss + semantic_weight * semantic_loss

            lh_acc1, lh_acc5 = accuracy(outputs['lh_pred'], lh_labels, topk=(1, 5))
            rh_acc1, rh_acc5 = accuracy(outputs['rh_pred'], rh_labels, topk=(1, 5))
            
            batch_size = lh_frames.shape[0]
            metric_logger.update(loss=total_loss.item())
            metric_logger.update(lh_loss=lh_loss.item())
            metric_logger.update(rh_loss=rh_loss.item())
            metric_logger.update(action_loss=action_loss.item())
            metric_logger.update(semantic_loss=semantic_loss.item() if isinstance(semantic_loss, torch.Tensor) else semantic_loss)
            metric_logger.meters['lh_acc1'].update(lh_acc1.item(), n=batch_size)
            metric_logger.meters['lh_acc5'].update(lh_acc5.item(), n=batch_size)
            metric_logger.meters['rh_acc1'].update(rh_acc1.item(), n=batch_size)
            metric_logger.meters['rh_acc5'].update(rh_acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    
    print('* Left Hand - Acc@1 {:.3f} Acc@5 {:.3f}'.format(
        metric_logger.lh_acc1.global_avg,
        metric_logger.lh_acc5.global_avg
    ))
    print('* Right Hand - Acc@1 {:.3f} Acc@5 {:.3f}'.format(
        metric_logger.rh_acc1.global_avg,
        metric_logger.rh_acc5.global_avg
    ))
    print('* Overall Loss: {:.3f} (Action: {:.3f}, Semantic: {:.3f})'.format(
        metric_logger.loss.global_avg,
        metric_logger.action_loss.global_avg,
        metric_logger.semantic_loss.global_avg
    ))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test_dual_with_semantic(data_loader, model, device, file_prefix, semantic_weight=0.1, semantic_loss_type='adaptive'):
    """Test and save predictions for both hands with semantic alignment"""
    criterion = torch.nn.CrossEntropyLoss()
    semantic_criterion = SemanticAlignmentLoss(loss_type=semantic_loss_type)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    lh_results = []
    rh_results = []

    for batch_data in metric_logger.log_every(data_loader, 10, header):
        if 'lh_name' in batch_data:  # Test mode format
            # Handle multiple samples first (check before calling .to())
            multiple_samples = batch_data.get('multiple_samples', False)
            if hasattr(multiple_samples, 'item'):  # Convert tensor to Python bool
                if multiple_samples.numel() == 1:  # Single element tensor
                    multiple_samples = multiple_samples.item()
                else:  # Batched tensor - check if all elements are the same
                    multiple_samples = multiple_samples[0].item()  # Take first element
            
            if multiple_samples and isinstance(batch_data['lh_frames'], list):
                # For repeated augmentation - take first augmentation for now
                lh_frames = batch_data['lh_frames'][0].to(device, non_blocking=True)
                rh_frames = batch_data['rh_frames'][0].to(device, non_blocking=True)
                lh_labels = batch_data['lh_label']
                rh_labels = batch_data['rh_label']
                if isinstance(lh_labels, list):
                    lh_labels = lh_labels[0]
                if isinstance(rh_labels, list):
                    rh_labels = rh_labels[0]
                lh_labels = lh_labels.to(device, non_blocking=True)
                rh_labels = rh_labels.to(device, non_blocking=True)
                
                # Handle semantic features
                lh_semantic = batch_data.get('lh_semantic', None)
                rh_semantic = batch_data.get('rh_semantic', None)
                if lh_semantic is not None:
                    lh_semantic = lh_semantic.to(device, non_blocking=True)
                if rh_semantic is not None:
                    rh_semantic = rh_semantic.to(device, non_blocking=True)
            else:
                # Single sample case
                lh_frames = batch_data['lh_frames'].to(device, non_blocking=True)
                rh_frames = batch_data['rh_frames'].to(device, non_blocking=True)
                lh_labels = batch_data['lh_label'].to(device, non_blocking=True)
                rh_labels = rh_labels.to(device, non_blocking=True)
                
                # Handle semantic features
                lh_semantic = batch_data.get('lh_semantic', None)
                rh_semantic = batch_data.get('rh_semantic', None)
                if lh_semantic is not None:
                    lh_semantic = lh_semantic.to(device, non_blocking=True)
                if rh_semantic is not None:
                    rh_semantic = rh_semantic.to(device, non_blocking=True)
            
            lh_names = batch_data['lh_name']
            rh_names = batch_data['rh_name']
            
            samples_dict = {
                'lh_frames': lh_frames, 
                'rh_frames': rh_frames,
                'lh_semantic': lh_semantic,
                'rh_semantic': rh_semantic
            }
            
            with torch.cuda.amp.autocast():
                outputs = model(samples_dict, return_semantic_features=True)
                lh_loss = criterion(outputs['lh_pred'], lh_labels)
                rh_loss = criterion(outputs['rh_pred'], rh_labels)

            # Save predictions
            for i in range(outputs['lh_pred'].size(0)):
                lh_string = "{} {} {}\n".format(
                    lh_names[i],
                    str(outputs['lh_pred'].data[i].cpu().numpy().tolist()),
                    str(int(lh_labels[i].cpu().numpy()))
                )
                lh_results.append(lh_string)
                
                rh_string = "{} {} {}\n".format(
                    rh_names[i],
                    str(outputs['rh_pred'].data[i].cpu().numpy().tolist()),
                    str(int(rh_labels[i].cpu().numpy()))
                )
                rh_results.append(rh_string)

            lh_acc1, lh_acc5 = accuracy(outputs['lh_pred'], lh_labels, topk=(1, 5))
            rh_acc1, rh_acc5 = accuracy(outputs['rh_pred'], rh_labels, topk=(1, 5))
            
            batch_size = lh_frames.shape[0]
            metric_logger.meters['lh_acc1'].update(lh_acc1.item(), n=batch_size)
            metric_logger.meters['lh_acc5'].update(lh_acc5.item(), n=batch_size)
            metric_logger.meters['rh_acc1'].update(rh_acc1.item(), n=batch_size)
            metric_logger.meters['rh_acc5'].update(rh_acc5.item(), n=batch_size)

    # Save results
    lh_file = file_prefix + '_lh.txt'
    rh_file = file_prefix + '_rh.txt'
    
    with open(lh_file, 'w') as f:
        f.write("{}, {}\n".format(metric_logger.lh_acc1.global_avg, metric_logger.lh_acc5.global_avg))
        for line in lh_results:
            f.write(line)
    
    with open(rh_file, 'w') as f:
        f.write("{}, {}\n".format(metric_logger.rh_acc1.global_avg, metric_logger.rh_acc5.global_avg))
        for line in rh_results:
            f.write(line)

    metric_logger.synchronize_between_processes()
    
    print('* Test Results:')
    print('  Left Hand - Acc@1 {:.3f} Acc@5 {:.3f}'.format(
        metric_logger.lh_acc1.global_avg,
        metric_logger.lh_acc5.global_avg
    ))
    print('  Right Hand - Acc@1 {:.3f} Acc@5 {:.3f}'.format(
        metric_logger.rh_acc1.global_avg,
        metric_logger.rh_acc5.global_avg
    ))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
