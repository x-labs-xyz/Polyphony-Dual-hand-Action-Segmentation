# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os

from .datasets import RawFrameClsDataset, VideoClsDataset
from .pretrain_datasets import (  # noqa: F401
    DataAugmentationForVideoMAEv2, HybridVideoMAE, VideoMAE,
)

class DualHandDataset(Dataset):
    """Wrapper dataset that combines two VideoClsDataset instances for dual-hand training"""
    
    def __init__(self, lh_dataset, rh_dataset):
        """
        Args:
            lh_dataset: Left-hand VideoClsDataset instance
            rh_dataset: Right-hand VideoClsDataset instance
        """
        self.lh_dataset = lh_dataset
        self.rh_dataset = rh_dataset
        
        # Handle datasets of different lengths by using the minimum length
        self.min_length = min(len(lh_dataset), len(rh_dataset))
        print(f"Dataset lengths: LH={len(lh_dataset)}, RH={len(rh_dataset)}, using min={self.min_length}")
    
    def __len__(self):
        return self.min_length
    
    def __getitem__(self, index):
        # Ensure index is within bounds for both datasets
        lh_index = index % len(self.lh_dataset)
        rh_index = index % len(self.rh_dataset)
        
        # Get samples from both datasets
        lh_sample = self.lh_dataset[lh_index]
        rh_sample = self.rh_dataset[rh_index]
        
        # Handle different return formats (train vs validation/test)
        if len(lh_sample) == 4:  # Training mode: (frames, label, index, {})
            lh_frames, lh_label, lh_index, lh_extra = lh_sample
            rh_frames, rh_label, rh_index, rh_extra = rh_sample
            
            # Handle repeated augmentation (num_sample > 1)
            if isinstance(lh_frames, list):  # Multiple samples
                return {
                    'lh_frames': lh_frames,
                    'rh_frames': rh_frames,
                    'lh_label': lh_label,
                    'rh_label': rh_label,
                    'lh_index': lh_index,
                    'rh_index': rh_index,
                    'multiple_samples': True
                }
            else:  # Single sample
                return {
                    'lh_frames': lh_frames,
                    'rh_frames': rh_frames,
                    'lh_label': lh_label,
                    'rh_label': rh_label,
                    'lh_index': lh_index,
                    'rh_index': rh_index,
                    'multiple_samples': False
                }
        else:  # Validation/test mode: (frames, label, video_name)
            lh_frames, lh_label, lh_name = lh_sample
            rh_frames, rh_label, rh_name = rh_sample
            
            return {
                'lh_frames': lh_frames,
                'rh_frames': rh_frames,
                'lh_label': lh_label,
                'rh_label': rh_label,
                'lh_name': lh_name,
                'rh_name': rh_name,
                'multiple_samples': False
            }

def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAEv2(args)
    dataset = HybridVideoMAE(
        root=args.data_root,
        setting=args.data_path,
        train=True,
        test_mode=False,
        name_pattern=args.fname_tmpl,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        num_segments=1,
        num_crop=1,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        lazy_init=False,
        num_sample=args.num_sample)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if is_train:
        mode = 'train'
        anno_path = os.path.join(args.data_path, 'train.csv')
    elif test_mode:
        mode = 'test'
        anno_path = os.path.join(args.data_path, 'val.csv')
    else:
        mode = 'validation'
        anno_path = os.path.join(args.data_path, 'val.csv')

    if args.data_set == 'Kinetics-400':
        if not args.sparse_sample:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=False,
                args=args)
        else:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=1,
                frame_sample_rate=1,
                num_segment=args.num_frames,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=True,
                args=args)
        nb_classes = 400
    
    elif args.data_set == 'HAVID':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            sparse_sample=False,
            args=args)
        nb_classes = 75

    elif args.data_set == 'Kinetics-600':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 600

    elif args.data_set == 'Kinetics-700':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 700

    elif args.data_set == 'Kinetics-710':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 710

    elif args.data_set == 'SSV2':
        dataset = RawFrameClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.fname_tmpl,
            start_idx=args.start_idx,
            args=args)

        nb_classes = 174

    elif args.data_set == 'UCF101':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101

    elif args.data_set == 'HMDB51':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51

    elif args.data_set == 'Diving48':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 48
    elif args.data_set == 'MIT':
        if not args.sparse_sample:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=False,
                args=args)
        else:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=1,
                frame_sample_rate=1,
                num_segment=args.num_frames,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=True,
                args=args)
        nb_classes = 339
    else:
        raise NotImplementedError('Unsupported Dataset')

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes

def build_dual_hand_datasets(is_train, test_mode, args):
    """Build dual-hand datasets using original VideoClsDataset approach"""
    if is_train:
        mode = 'train'
        lh_anno_path = args.lh_train_ann
        rh_anno_path = args.rh_train_ann
    elif test_mode:
        mode = 'test'
        lh_anno_path = args.lh_val_ann
        rh_anno_path = args.rh_val_ann
    else:
        mode = 'validation'
        lh_anno_path = args.lh_val_ann
        rh_anno_path = args.rh_val_ann
    
    # Create left-hand dataset using original VideoClsDataset
    lh_dataset = VideoClsDataset(
        anno_path=lh_anno_path,
        data_root=args.lh_data_dir,
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        sparse_sample=False,
        args=args)
    
    # Create right-hand dataset using original VideoClsDataset
    rh_dataset = VideoClsDataset(
        anno_path=rh_anno_path,
        data_root=args.rh_data_dir,
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        sparse_sample=False,
        args=args)
    
    # Wrap the two datasets into a single dual-hand dataset
    dual_dataset = DualHandDataset(lh_dataset, rh_dataset)
    
    return dual_dataset, args.lh_num_classes
