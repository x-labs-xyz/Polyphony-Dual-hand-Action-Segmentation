import argparse
import os
import glob
from typing import List, Tuple

import numpy as np


def concatenate_pair(enh_path: str, orig_path: str) -> np.ndarray:
    enh = np.load(enh_path)  # expected shape: [D_sem, T]
    orig = np.load(orig_path)  # expected shape: [D_vis, T] (VideoMAE saved as [D, T])

    # Ensure time dimension (axis=1) matches
    if enh.shape[1] != orig.shape[1]:
        raise ValueError(f"Temporal length mismatch: enhanced {enh.shape} vs original {orig.shape}")

    # Concatenate along feature dimension
    concat = np.concatenate([orig, enh], axis=0)
    return concat


def main():
    parser = argparse.ArgumentParser(description='Concatenate enhanced semantic features with original VideoMAE features along feature axis (axis=0) and save .npy')
    parser.add_argument('--enhanced_dir', type=str, default='./havid_enhanced_features/v2/BAAI/rh_v0',
                        help='Directory with enhanced .npy')
    parser.add_argument('--original_dir', type=str, default='/home/hao/Polyphony/data/havid/videomae_features/rh_v0',
                        help='Directory with original VideoMAE .npy')
    parser.add_argument('--output_dir', type=str, default='/home/hao/Polyphony/data/havid/videomae_features_concatenated_semantic/v2/BAAI/rh_v0',
                        help='Directory to write concatenated .npy files (will be created if missing)')
    parser.add_argument('--limit', type=int, default=0, help='Optional: limit number of files processed (0 = all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    orig_paths = sorted(glob.glob(os.path.join(args.original_dir, '*.npy')))
    enhanced_set = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.enhanced_dir, '*.npy'))}

    missing: List[str] = []
    processed = 0
    errors: List[Tuple[str, str]] = []

    for orig_path in orig_paths:
        base = os.path.splitext(os.path.basename(orig_path))[0]
        enh_path = os.path.join(args.enhanced_dir, base + '.npy')
        if base not in enhanced_set or not os.path.exists(enh_path):
            missing.append(base)
            continue
        try:
            concat = concatenate_pair(enh_path, orig_path)
        except Exception as e:
            errors.append((base, str(e)))
            continue

        out_path = os.path.join(args.output_dir, base + '.npy')
        np.save(out_path, concat)
        processed += 1

        if args.limit and processed >= args.limit:
            break

    print(f"Processed: {processed}")
    print(f"Missing enhanced: {len(missing)}")
    if missing:
        print("First 20 missing:", missing[:20])
    print(f"Errors: {len(errors)}")
    if errors:
        print("First 10 errors:")
        for b, msg in errors[:10]:
            print(f"  {b}: {msg}")


if __name__ == '__main__':
    main()


