# Alternating dual-hand video action recognition training script
import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma

import models  # noqa: F401
import models.modeling_finetune_alternating  # Import alternating model
import utils
from dataset.build import build_dataset
from engine_for_alternating_finetuning import (
    train_one_epoch_alternating,
    validation_one_epoch_alternating,
    final_test_alternating,
)
from engine_for_finetuning import merge
from optim_factory import (
    LayerDecayValueAssigner,
    create_optimizer,
    get_parameter_groups,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate


def get_args():
    parser = argparse.ArgumentParser(
        'Alternating dual-hand VideoMAE fine-tuning script', add_help=False)
    
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224_alternating', type=str,
                        help='Name of alternating dual-head model to train')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--with_checkpoint', action='store_true', default=False)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--head_drop_rate', type=float, default=0.0, metavar='PCT')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)
    
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--warmup_lr', type=float, default=1e-8, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')

    # Alternating training specific
    parser.add_argument('--alternation_steps', type=int, default=50,
                        help='Number of steps before switching hands')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT')
    parser.add_argument('--num_sample', type=int, default=2)
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--train_interpolation', type=str, default='bicubic')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=10)
    parser.add_argument('--test_num_crop', type=int, default=3)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT')
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dual-hand dataset parameters
    parser.add_argument('--lh_data_path', required=True, type=str, 
                        help='Left hand data path (contains train/val_list_video.txt and videos/)')
    parser.add_argument('--lh_data_root', required=False, default='', type=str, 
                        help='Left hand data root (defaults to lh_data_path if not specified)')
    parser.add_argument('--rh_data_path', required=False, default='', type=str, 
                        help='Right hand data path (optional, only needed for dual-hand training)')
    parser.add_argument('--rh_data_root', required=False, default='', type=str, 
                        help='Right hand data root (defaults to rh_data_path if not specified)')
    parser.add_argument('--lh_num_classes', default=75, type=int)
    parser.add_argument('--rh_num_classes', default=75, type=int)
    parser.add_argument('--one_stream', action='store_true', default=False,
                        help='Train on single-stream data. Right-hand will mirror left-hand data.')
    
    # Standard dataset params
    parser.add_argument('--data_set', default='HAVID', type=str)
    parser.add_argument('--nb_classes', default=75, type=int)  # Used for compatibility
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--sparse_sample', default=False, action='store_true')
    parser.add_argument('--fname_tmpl', default='img_{:05}.jpg', type=str)
    parser.add_argument('--start_idx', default=1, type=int)

    # Output
    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--log_dir', default=None, help='tensorboard log directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser.parse_args()


def main(args):
    utils.init_distributed_mode(args)
    
    print(args)
    
    device = torch.device(args.device)
    
    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Set default data_root values if not specified
    if not args.lh_data_root:
        args.lh_data_root = args.lh_data_path
    if not args.rh_data_root and args.rh_data_path:
        args.rh_data_root = args.rh_data_path

    # If single-stream mode, mirror left-hand data to right-hand
    if args.one_stream:
        args.rh_data_path = args.lh_data_path
        args.rh_data_root = args.lh_data_root
        args.rh_num_classes = args.lh_num_classes

    # Build separate datasets for left and right hands
    # Left hand datasets
    args_lh = argparse.Namespace(**vars(args))
    args_lh.data_path = args.lh_data_path
    args_lh.data_root = args.lh_data_root
    args_lh.nb_classes = args.lh_num_classes
    
    lh_dataset_train, _ = build_dataset(is_train=True, test_mode=False, args=args_lh)
    lh_dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args_lh) if not args.disable_eval_during_finetuning else (None, None)
    lh_dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args_lh)
    
    # Right hand datasets (mirror LH when --one_stream)
    args_rh = argparse.Namespace(**vars(args))
    args_rh.data_path = args.rh_data_path
    args_rh.data_root = args.rh_data_root
    args_rh.nb_classes = args.rh_num_classes
    
    if args.one_stream:
        rh_dataset_train, rh_dataset_val, rh_dataset_test = lh_dataset_train, (lh_dataset_val if not args.disable_eval_during_finetuning else None), lh_dataset_test
    else:
        rh_dataset_train, _ = build_dataset(is_train=True, test_mode=False, args=args_rh)
        rh_dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args_rh) if not args.disable_eval_during_finetuning else (None, None)
        rh_dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args_rh)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    # Create samplers (RH optional for one-stream mode)
    lh_sampler_train = torch.utils.data.DistributedSampler(
        lh_dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    if args.one_stream:
        rh_sampler_train = None
    else:
        rh_sampler_train = torch.utils.data.DistributedSampler(
            rh_dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    print("LH Sampler_train = %s" % str(lh_sampler_train))
    if not args.one_stream:
        print("RH Sampler_train = %s" % str(rh_sampler_train))
    
    if args.dist_eval:
        lh_sampler_val = torch.utils.data.DistributedSampler(
            lh_dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if lh_dataset_val else None
        lh_sampler_test = torch.utils.data.DistributedSampler(
            lh_dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        rh_sampler_val = torch.utils.data.DistributedSampler(
            rh_dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if rh_dataset_val else None
        rh_sampler_test = torch.utils.data.DistributedSampler(
            rh_dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        lh_sampler_val = torch.utils.data.SequentialSampler(lh_dataset_val) if lh_dataset_val else None
        lh_sampler_test = torch.utils.data.SequentialSampler(lh_dataset_test)
        rh_sampler_val = torch.utils.data.SequentialSampler(rh_dataset_val) if rh_dataset_val else None
        rh_sampler_test = torch.utils.data.SequentialSampler(rh_dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # Collate function for multiple samples
    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    # Create separate data loaders
    lh_data_loader_train = torch.utils.data.DataLoader(
        lh_dataset_train, sampler=lh_sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True, collate_fn=collate_func,
        persistent_workers=True)
    
    if args.one_stream:
        rh_data_loader_train = None
    else:
        rh_data_loader_train = torch.utils.data.DataLoader(
            rh_dataset_train, sampler=rh_sampler_train,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=True, collate_fn=collate_func,
            persistent_workers=True)

    if lh_dataset_val is not None:
        lh_data_loader_val = torch.utils.data.DataLoader(
            lh_dataset_val, sampler=lh_sampler_val,
            batch_size=int(1.5 * args.batch_size), num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False, persistent_workers=True)
        if not args.one_stream:
            rh_data_loader_val = torch.utils.data.DataLoader(
                rh_dataset_val, sampler=rh_sampler_val,
                batch_size=int(1.5 * args.batch_size), num_workers=args.num_workers,
                pin_memory=args.pin_mem, drop_last=False, persistent_workers=True)
        else:
            rh_data_loader_val = None
    else:
        lh_data_loader_val = None
        rh_data_loader_val = None

    lh_data_loader_test = torch.utils.data.DataLoader(
        lh_dataset_test, sampler=lh_sampler_test,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False, persistent_workers=True)
    
    if not args.one_stream:
        rh_data_loader_test = torch.utils.data.DataLoader(
            rh_dataset_test, sampler=rh_sampler_test,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False, persistent_workers=True)
    else:
        rh_data_loader_test = None

    # Mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=max(args.lh_num_classes, args.rh_num_classes))

    # Create alternating dual-head model
    model = create_model(
        args.model,
        img_size=args.input_size, pretrained=False,
        lh_num_classes=args.lh_num_classes, rh_num_classes=args.rh_num_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop, drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate, head_drop_rate=args.head_drop_rate,
        drop_block_rate=None, use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale, with_cp=args.with_checkpoint,
    )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size

    # Load pretrained weights
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        
        print("Load pretrained ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        
        # Clean up keys
        for old_key in list(checkpoint_model.keys()):
            if old_key.startswith('_orig_mod.'):
                new_key = old_key[10:]
                checkpoint_model[new_key] = checkpoint_model.pop(old_key)
        
        # Remove single head weights
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model:
                print(f"Removing single head key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        # Handle prefixes
        all_keys = list(checkpoint_model.keys())
        new_dict = {}
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
        
        # Interpolate position embedding if needed
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // 
                            (args.num_frames // model.patch_embed.tubelet_size))**0.5)
            new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size))**0.5)
            
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size,
                                               orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    -1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
        
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model, decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '', resume='')

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model = %s" % str(model_without_ddp))
    print('Number of params:', n_parameters)

    # Setup distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # Learning rate scaling
    total_batch_size = args.batch_size * args.update_freq * num_tasks
    if args.one_stream:
        num_training_steps_per_epoch = len(lh_dataset_train) // total_batch_size
    else:
        num_training_steps_per_epoch = max(len(lh_dataset_train), len(rh_dataset_train)) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)
    print("Alternation steps = %d" % args.alternation_steps)

    # Layer decay
    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay**(num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []

    # Optimizer
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    # Learning rate schedule
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # Loss criterion
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # Auto resume
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # Validation only mode
    if args.validation:
        test_stats = validation_one_epoch_alternating(
            lh_data_loader_val, rh_data_loader_val, model, device)
        exit(0)

    # Evaluation only mode
    if args.eval:
        file_prefix = os.path.join(args.output_dir, str(global_rank))
        test_stats = final_test_alternating(
            lh_data_loader_test, rh_data_loader_test, model, device, file_prefix)
        exit(0)

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_lh_accuracy = 0.0
    max_rh_accuracy = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            lh_data_loader_train.sampler.set_epoch(epoch)
            if not args.one_stream:
                rh_data_loader_train.sampler.set_epoch(epoch)
        
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        
        # Alternating training
        train_stats = train_one_epoch_alternating(
            model, criterion, lh_data_loader_train, rh_data_loader_train,
            optimizer, device, epoch, loss_scaler, args.clip_grad,
            model_ema, mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            alternation_steps=args.alternation_steps,
        )
        
        # Save checkpoints
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or (epoch + 1) == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                    model_ema=model_ema)
        
        # Validation
        if lh_data_loader_val is not None:
            # For one-stream mode, skip RH validation (pass None) and ignore RH metrics
            test_stats = validation_one_epoch_alternating(
                lh_data_loader_val, rh_data_loader_val if not args.one_stream else None, model, device)
            
            # Check for best accuracy
            save_best = False
            if test_stats.get('lh_acc1', 0) > max_lh_accuracy:
                max_lh_accuracy = test_stats['lh_acc1']
                save_best = True
            if not args.one_stream and test_stats.get('rh_acc1', 0) > max_rh_accuracy:
                max_rh_accuracy = test_stats['rh_acc1']
                save_best = True
            
            if save_best and args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch="best",
                    model_ema=model_ema)
            
            if not args.one_stream:
                print(f'Max accuracy - LH: {max_lh_accuracy:.2f}%, RH: {max_rh_accuracy:.2f}%')
            else:
                print(f'Max accuracy - LH: {max_lh_accuracy:.2f}%')
            
            if log_writer is not None:
                log_writer.update(val_lh_acc1=test_stats['lh_acc1'], head="perf", step=epoch)
                log_writer.update(val_lh_acc5=test_stats['lh_acc5'], head="perf", step=epoch)
                if not args.one_stream:
                    log_writer.update(val_rh_acc1=test_stats['rh_acc1'], head="perf", step=epoch)
                    log_writer.update(val_rh_acc5=test_stats['rh_acc5'], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)

        # Log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        if lh_data_loader_val is not None:
            log_stats.update({f'val_{k}': v for k, v in test_stats.items()})
        
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Final test
    file_prefix = os.path.join(args.output_dir, str(global_rank))
    test_stats = final_test_alternating(
        lh_data_loader_test, rh_data_loader_test if not args.one_stream else None, model, device, file_prefix)
    
    torch.distributed.barrier()
    
    if global_rank == 0:
        # Merge results for both hands
        print("Merging left hand results...")
        lh_final_top1, lh_final_top5 = merge(args.output_dir + '_lh', num_tasks)
        print(f"Final Left Hand - Top-1: {lh_final_top1:.2f}%, Top-5: {lh_final_top5:.2f}%")
        if not args.one_stream:
            print("Merging right hand results...")
            rh_final_top1, rh_final_top5 = merge(args.output_dir + '_rh', num_tasks)
            print(f"Final Right Hand - Top-1: {rh_final_top1:.2f}%, Top-5: {rh_final_top5:.2f}%")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)