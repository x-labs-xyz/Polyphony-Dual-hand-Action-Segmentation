# Dual-hand video action recognition training script with semantic feature alignment
import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma

import models  # noqa: F401
import models.modeling_finetune_dual_semantic  # Import dual model with semantic alignment
import utils
from dataset.build import build_dual_hand_datasets_with_semantic
from engine_for_dual_hand_finetuning_semantic import (
    train_one_epoch_dual_with_semantic,
    validation_one_epoch_dual_with_semantic,
    final_test_dual_with_semantic,
)
from optim_factory import (
    LayerDecayValueAssigner,
    create_optimizer,
    get_parameter_groups,
)
from utils import NativeScalerWithGradNormCount as NativeScaler


def get_args():
    parser = argparse.ArgumentParser(
        'Dual-hand VideoMAE fine-tuning script with semantic alignment', add_help=False)
    
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224_dual_semantic', type=str, metavar='MODEL',
                        help='Name of dual-head model with semantic alignment to train')
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
    parser.add_argument('--lh_data_dir', required=True, type=str, help='Left hand dataset directory')
    parser.add_argument('--rh_data_dir', required=True, type=str, help='Right hand dataset directory')
    parser.add_argument('--lh_train_ann', required=True, type=str, help='Left hand train annotation')
    parser.add_argument('--rh_train_ann', required=True, type=str, help='Right hand train annotation')
    parser.add_argument('--lh_val_ann', required=True, type=str, help='Left hand val annotation')
    parser.add_argument('--rh_val_ann', required=True, type=str, help='Right hand val annotation')
    parser.add_argument('--lh_num_classes', default=75, type=int)
    parser.add_argument('--rh_num_classes', default=75, type=int)
    
    # Semantic alignment parameters
    parser.add_argument('--semantic_model_name', default='sentence-transformers/all-mpnet-base-v2', type=str,
                        help='Name of the semantic model to use')
    parser.add_argument('--semantic_embeddings_path', default=None, type=str,
                        help='Path to precomputed semantic embeddings')
    parser.add_argument('--action_mapping_path', default=None, type=str,
                        help='Path to action mapping file')
    parser.add_argument('--semantic_alignment_weight', default=0.1, type=float,
                        help='Weight for semantic alignment loss')
    parser.add_argument('--semantic_loss_type', default='adaptive', type=str,
                        choices=['adaptive', 'cosine', 'mse'],
                        help='Type of semantic alignment loss')
    parser.add_argument('--tcn_hidden_dims', default=[512, 256], type=int, nargs='+',
                        help='Hidden dimensions for TCN in semantic alignment')
    
    # Standard dataset params (kept for compatibility)
    parser.add_argument('--data_set', default='HAVID', type=str)
    parser.add_argument('--nb_classes', default=75, type=int)  # Will be overridden
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--sparse_sample', default=False, action='store_true')
    parser.add_argument('--fname_tmpl', default='img_{:05}.jpg', type=str)
    parser.add_argument('--start_idx', default=1, type=int)

    # Output
    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
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

    # Distributed training parameters
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

    # Build dual-hand datasets with semantic features
    dataset_train, _ = build_dual_hand_datasets_with_semantic(is_train=True, test_mode=False, args=args)
    
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_dual_hand_datasets_with_semantic(is_train=False, test_mode=False, args=args)
    
    dataset_test, _ = build_dual_hand_datasets_with_semantic(is_train=False, test_mode=True, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))
    
    if args.dist_eval:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if dataset_val else None
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val else None
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # Data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True
        )
    else:
        data_loader_val = None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True
    )
    
    # Mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.lh_num_classes  # Assuming same for both hands
        )

    # Create dual-head model with semantic alignment
    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        lh_num_classes=args.lh_num_classes,
        rh_num_classes=args.rh_num_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        head_drop_rate=args.head_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        with_cp=args.with_checkpoint,
        semantic_dim=768,  # MPNet embedding dimension
        tcn_hidden_dims=args.tcn_hidden_dims,
        semantic_alignment_weight=args.semantic_alignment_weight,
    )

    # Load pretrained weights if specified
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
        
        # Remove old format keys
        for old_key in list(checkpoint_model.keys()):
            if old_key.startswith('_orig_mod.'):
                new_key = old_key[10:]
                checkpoint_model[new_key] = checkpoint_model.pop(old_key)
        
        # Remove single head weights (we have dual heads)
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model:
                print(f"Removing single head key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        # Remove semantic alignment weights (they will be randomly initialized)
        semantic_keys = [k for k in checkpoint_model.keys() if 'semantic' in k or 'tcn' in k or 'visual_projector' in k or 'semantic_projector' in k]
        for k in semantic_keys:
            print(f"Removing semantic alignment key {k} from pretrained checkpoint")
            del checkpoint_model[k]
        
        # Handle backbone prefix
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
        if 'pos_embed' in checkpoint_model and args.input_size != 224:
            # Position embedding interpolation logic here (same as original)
            pass
        
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model = %s" % str(model_without_ddp))
    print('Number of params:', n_parameters)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # Optimizer
    total_batch_size = args.batch_size * args.update_freq * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay**(num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    skip_weight_decay_list = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []

    optimizer = create_optimizer(
        args,
        model_without_ddp,
        skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None
    )
    loss_scaler = NativeScaler()

    # Learning rate schedule
    # Adjust warmup steps if they exceed total training steps
    total_training_steps = args.epochs * num_training_steps_per_epoch
    # Calculate actual warmup steps (either from warmup_steps or warmup_epochs)
    if args.warmup_steps > 0:
        actual_warmup_steps = args.warmup_steps
    else:
        actual_warmup_steps = args.warmup_epochs * num_training_steps_per_epoch
    
    if actual_warmup_steps > total_training_steps:
        # Reduce warmup steps to be less than total training steps
        if args.warmup_steps > 0:
            args.warmup_steps = max(1, total_training_steps // 2)
            print(f"Adjusted warmup steps to {args.warmup_steps} (was {actual_warmup_steps})")
        else:
            # Adjust warmup epochs instead
            original_warmup_epochs = args.warmup_epochs
            args.warmup_epochs = max(1, total_training_steps // (2 * num_training_steps_per_epoch))
            print(f"Adjusted warmup epochs to {args.warmup_epochs} (was {original_warmup_epochs})")
    
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

    # Loss
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))
    print("semantic_alignment_weight = %f" % args.semantic_alignment_weight)
    print("semantic_loss_type = %s" % args.semantic_loss_type)

    # Auto resume
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # Validation only mode
    if args.validation:
        test_stats = validation_one_epoch_dual_with_semantic(
            data_loader_val, model, device, 
            semantic_weight=args.semantic_alignment_weight,
            semantic_loss_type=args.semantic_loss_type
        )
        exit(0)

    # Evaluation only mode
    if args.eval:
        file_prefix = os.path.join(args.output_dir, str(global_rank))
        test_stats = final_test_dual_with_semantic(
            data_loader_test, model, device, file_prefix,
            semantic_weight=args.semantic_alignment_weight,
            semantic_loss_type=args.semantic_loss_type
        )
        exit(0)

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_lh_accuracy = 0.0
    max_rh_accuracy = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        
        train_stats = train_one_epoch_dual_with_semantic(
            model, criterion, data_loader_train, optimizer, device, epoch,
            loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            semantic_weight=args.semantic_alignment_weight,
            semantic_loss_type=args.semantic_loss_type,
        )
        
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or (epoch + 1) == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                    model_ema=model_ema)
        
        if data_loader_val is not None:
            test_stats = validation_one_epoch_dual_with_semantic(
                data_loader_val, model, device,
                semantic_weight=args.semantic_alignment_weight,
                semantic_loss_type=args.semantic_loss_type
            )
            
            # Check if we have new best accuracy for either hand
            save_best = False
            if test_stats.get('lh_acc1', 0) > max_lh_accuracy:
                max_lh_accuracy = test_stats['lh_acc1']
                save_best = True
            if test_stats.get('rh_acc1', 0) > max_rh_accuracy:
                max_rh_accuracy = test_stats['rh_acc1']
                save_best = True
            
            if save_best and args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch="best",
                    model_ema=model_ema)
            
            print(f'Max accuracy - LH: {max_lh_accuracy:.2f}%, RH: {max_rh_accuracy:.2f}%')
            
            if log_writer is not None:
                log_writer.update(val_lh_acc1=test_stats['lh_acc1'], head="perf", step=epoch)
                log_writer.update(val_lh_acc5=test_stats['lh_acc5'], head="perf", step=epoch)
                log_writer.update(val_rh_acc1=test_stats['rh_acc1'], head="perf", step=epoch)
                log_writer.update(val_rh_acc5=test_stats['rh_acc5'], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)
                log_writer.update(val_action_loss=test_stats['action_loss'], head="perf", step=epoch)
                log_writer.update(val_semantic_loss=test_stats['semantic_loss'], head="perf", step=epoch)

        # Log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        if data_loader_val is not None:
            log_stats.update({f'val_{k}': v for k, v in test_stats.items()})
        
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Final test
    file_prefix = os.path.join(args.output_dir, str(global_rank))
    test_stats = final_test_dual_with_semantic(
        data_loader_test, model, device, file_prefix,
        semantic_weight=args.semantic_alignment_weight,
        semantic_loss_type=args.semantic_loss_type
    )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
