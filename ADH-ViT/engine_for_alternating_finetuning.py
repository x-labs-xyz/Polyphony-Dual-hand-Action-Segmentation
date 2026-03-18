# Engine for alternating dual-hand video action recognition training
import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import utils


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch_alternating(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lh_data_loader: Iterable,
    rh_data_loader: Iterable,
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
    alternation_steps=10,  # Switch between hands every N steps
):
    """
    Alternating training: switches between left and right hand datasets every `alternation_steps` steps.
    """
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # Create iterators for both datasets
    lh_iterator = iter(metric_logger.log_every(lh_data_loader, print_freq * 2, header + ' [LH]'))
    rh_iterator = iter(metric_logger.log_every(rh_data_loader, print_freq * 2, header + ' [RH]'))
    
    # Track which hand we're training
    current_hand = 'lh'
    steps_since_switch = 0
    
    # Set initial training mode
    if hasattr(model, 'module'):  # DDP
        model.module.set_training_mode(current_hand)
    else:
        model.set_training_mode(current_hand)
    
    for data_iter_step in range(len(lh_data_loader)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        
        it = start_steps + step
        
        # Switch hands every alternation_steps
        if steps_since_switch >= alternation_steps:
            current_hand = 'rh' if current_hand == 'lh' else 'lh'
            steps_since_switch = 0
            # Update model's training mode
            if hasattr(model, 'module'):  # DDP
                model.module.set_training_mode(current_hand)
            else:
                model.set_training_mode(current_hand)
            print(f"Switched to training {current_hand.upper()} hand at step {data_iter_step}")
        
        # Get data from appropriate iterator
        try:
            if current_hand == 'lh':
                samples, targets, _, _ = next(lh_iterator)
            else:
                samples, targets, _, _ = next(rh_iterator)
        except StopIteration:
            # Restart iterator if exhausted
            if current_hand == 'lh':
                lh_iterator = iter(lh_data_loader)
                samples, targets, _, _ = next(lh_iterator)
            else:
                rh_iterator = iter(rh_data_loader)
                samples, targets, _, _ = next(rh_iterator)
        
        # Update LR & WD
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup if enabled
        if mixup_fn is not None:
            B, C, T, H, W = samples.shape
            samples = samples.view(B, C * T, H, W)
            samples, targets = mixup_fn(samples, targets)
            samples = samples.view(B, C, T, H, W)

        # Forward pass - model knows which head to use based on training_mode
        if loss_scaler is None:
            samples = samples.half()
            outputs = model(samples, hand_type=current_hand)
            loss = criterion(outputs, targets)
        else:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(samples, hand_type=current_hand)
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Backward pass
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            grad_norm = model.get_global_grad_norm()
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = get_loss_scale_for_deepspeed(model)
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

        # Compute accuracy
        if mixup_fn is None:
            class_acc = (outputs.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        # Update metrics based on current hand
        metric_logger.update(loss=loss_value)
        if current_hand == 'lh':
            metric_logger.update(lh_loss=loss_value)
            metric_logger.update(lh_acc=class_acc)
        else:
            metric_logger.update(rh_loss=loss_value)
            metric_logger.update(rh_acc=class_acc)
        
        metric_logger.update(loss_scale=loss_scale_value)
        
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
            if current_hand == 'lh':
                log_writer.update(lh_loss=loss_value, head="loss")
                log_writer.update(lh_acc=class_acc, head="loss")
            else:
                log_writer.update(rh_loss=loss_value, head="loss")
                log_writer.update(rh_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
        
        steps_since_switch += 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch_alternating(lh_data_loader, rh_data_loader, model, device):
    """Validate both hands separately"""
    criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    
    model.eval()
    
    # Set model to evaluation mode for both hands
    if hasattr(model, 'module'):
        model.module.set_training_mode('both')
    else:
        model.set_training_mode('both')
    
    # Evaluate left hand
    print("Evaluating Left Hand...")
    for batch in metric_logger.log_every(lh_data_loader, 10, header + ' [LH]'):
        images = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(images, hand_type='lh')
            loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = images.shape[0]
        metric_logger.meters['lh_loss'].update(loss.item(), n=batch_size)
        metric_logger.meters['lh_acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['lh_acc5'].update(acc5.item(), n=batch_size)
    
    # Evaluate right hand
    print("Evaluating Right Hand...")
    for batch in metric_logger.log_every(rh_data_loader, 10, header + ' [RH]'):
        images = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(images, hand_type='rh')
            loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = images.shape[0]
        metric_logger.meters['rh_loss'].update(loss.item(), n=batch_size)
        metric_logger.meters['rh_acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['rh_acc5'].update(acc5.item(), n=batch_size)
    
    # Calculate average loss
    avg_loss = (metric_logger.meters['lh_loss'].global_avg + 
                metric_logger.meters['rh_loss'].global_avg) / 2.0
    metric_logger.meters['loss'].update(avg_loss, n=1)
    
    metric_logger.synchronize_between_processes()
    
    print('* Left Hand - Acc@1 {:.3f} Acc@5 {:.3f} Loss {:.3f}'.format(
        metric_logger.lh_acc1.global_avg,
        metric_logger.lh_acc5.global_avg,
        metric_logger.lh_loss.global_avg
    ))
    print('* Right Hand - Acc@1 {:.3f} Acc@5 {:.3f} Loss {:.3f}'.format(
        metric_logger.rh_acc1.global_avg,
        metric_logger.rh_acc5.global_avg,
        metric_logger.rh_loss.global_avg
    ))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test_alternating(lh_data_loader, rh_data_loader, model, device, file_prefix):
    """Test both hands and save results separately"""
    criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    model.eval()
    
    # Results storage
    lh_results = []
    rh_results = []
    
    # Test left hand
    print("Testing Left Hand...")
    for batch in metric_logger.log_every(lh_data_loader, 10, header + ' [LH]'):
        images = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(images, hand_type='lh')
            loss = criterion(output, target)
        
        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(
                ids[i], 
                str(output.data[i].cpu().numpy().tolist()),
                str(int(target[i].cpu().numpy())),
                str(int(chunk_nb[i].cpu().numpy())),
                str(int(split_nb[i].cpu().numpy()))
            )
            lh_results.append(string)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.meters['lh_acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['lh_acc5'].update(acc5.item(), n=batch_size)
    
    # Test right hand
    print("Testing Right Hand...")
    for batch in metric_logger.log_every(rh_data_loader, 10, header + ' [RH]'):
        images = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(images, hand_type='rh')
            loss = criterion(output, target)
        
        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(
                ids[i],
                str(output.data[i].cpu().numpy().tolist()),
                str(int(target[i].cpu().numpy())),
                str(int(chunk_nb[i].cpu().numpy())),
                str(int(split_nb[i].cpu().numpy()))
            )
            rh_results.append(string)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.meters['rh_acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['rh_acc5'].update(acc5.item(), n=batch_size)
    
    # Save results
    lh_file = file_prefix + '_lh.txt'
    rh_file = file_prefix + '_rh.txt'
    
    with open(lh_file, 'w') as f:
        f.write("{}, {}\n".format(
            metric_logger.lh_acc1.global_avg, 
            metric_logger.lh_acc5.global_avg))
        for line in lh_results:
            f.write(line)
    
    with open(rh_file, 'w') as f:
        f.write("{}, {}\n".format(
            metric_logger.rh_acc1.global_avg,
            metric_logger.rh_acc5.global_avg))
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