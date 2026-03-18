"""
Concatenate Visual and Semantic Features for Dual-Hand Action Segmentation

This script performs a two-step concatenation:
1. Shared features + Hand-specific features → Intermediate features
2. Intermediate features + Semantic features → Final concatenated features

For each hand:
- LH: shared [768, T] + lh [num_classes, T] + semantic [D_sem, T] = [843+D_sem, T]
- RH: shared [768, T] + rh [num_classes, T] + semantic [D_sem, T] = [843+D_sem, T]

Output structure:
- {output_dir}/left_hand/{video_name}.npy - Final LH features
- {output_dir}/right_hand/{video_name}.npy - Final RH features
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np


# ============================================================================
# Feature Concatenation
# ============================================================================

def concatenate_features(
    shared_features: np.ndarray,
    hand_features: np.ndarray,
    semantic_features: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Concatenate features along the feature dimension (axis=0)
    
    Args:
        shared_features: Shared visual features [768, T]
        hand_features: Hand-specific features [75, T]
        semantic_features: Optional semantic features [D_sem, T]
        
    Returns:
        Concatenated features [768+75+D_sem, T] or [768+75, T] if no semantic
    """
    # Step 1: Concatenate shared + hand features
    intermediate = np.concatenate([shared_features, hand_features], axis=0)  # [843, T]
    
    # Step 2: Add semantic features if provided
    if semantic_features is not None:
        final = np.concatenate([intermediate, semantic_features], axis=0)
        return final
    
    return intermediate


def validate_shapes(
    shared: np.ndarray,
    hand: np.ndarray,
    semantic: Optional[np.ndarray],
    video_name: str
) -> bool:
    """
    Validate that all features have matching temporal dimensions
    
    Args:
        shared: Shared features
        hand: Hand-specific features
        semantic: Optional semantic features
        video_name: Video name for error messages
        
    Returns:
        True if shapes are valid, False otherwise
    """
    if shared.shape[1] != hand.shape[1]:
        print(f"Warning: Temporal mismatch for {video_name}")
        print(f"  Shared: {shared.shape}, Hand: {hand.shape}")
        return False
    
    if semantic is not None and shared.shape[1] != semantic.shape[1]:
        print(f"Warning: Temporal mismatch for {video_name}")
        print(f"  Shared: {shared.shape}, Semantic: {semantic.shape}")
        return False
    
    return True


# ============================================================================
# Main Processing Function
# ============================================================================

def process_dual_hand_features(
    base_dir: Path,
    semantic_lh_dir: Optional[Path],
    semantic_rh_dir: Optional[Path],
    output_dir: Path,
    skip_semantic: bool = False
) -> Dict:
    """
    Process and concatenate dual-hand features with semantic conditioning
    
    Args:
        base_dir: Base directory containing shared_features, lh_features, rh_features
        semantic_lh_dir: Directory with semantic features for left hand
        semantic_rh_dir: Directory with semantic features for right hand
        output_dir: Output directory for final concatenated features
        skip_semantic: If True, skip semantic concatenation (only do step 1)
        
    Returns:
        Processing summary dictionary
    """
    # Input directories
    shared_dir = base_dir / 'shared_features'
    lh_dir = base_dir / 'lh_features'
    rh_dir = base_dir / 'rh_features'
    
    # Output directories
    lh_output_dir = output_dir / 'lh_v0'
    rh_output_dir = output_dir / 'rh_v0'
    metadata_dir = output_dir / 'metadata'
    
    # Create output directories
    lh_output_dir.mkdir(parents=True, exist_ok=True)
    rh_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all shared feature files (exclude head and patch files)
    shared_files = [
        f for f in shared_dir.glob('*.npy')
        if not f.name.endswith('_head.npy') and not f.name.endswith('_patch.npy')
    ]
    
    if not shared_files:
        print(f"Error: No shared feature files found in {shared_dir}")
        return {}
    
    print(f"Found {len(shared_files)} shared feature files to process")
    print(f"\nInput directories:")
    print(f"  Shared features: {shared_dir}")
    print(f"  LH features: {lh_dir}")
    print(f"  RH features: {rh_dir}")
    if not skip_semantic:
        print(f"  LH semantic: {semantic_lh_dir}")
        print(f"  RH semantic: {semantic_rh_dir}")
    print(f"\nOutput directories:")
    print(f"  LH final: {lh_output_dir}")
    print(f"  RH final: {rh_output_dir}")
    print()
    
    # Statistics
    successful_lh = 0
    successful_rh = 0
    failed_files: List[str] = []
    processing_log: List[Dict] = []
    
    # Process each video
    for shared_file in tqdm(shared_files, desc="Concatenating features"):
        try:
            video_name = shared_file.stem
            
            # Load shared features
            shared_features = np.load(shared_file)  # [768, T]
            
            # Find corresponding files
            lh_file = lh_dir / f"{video_name}.npy"
            rh_file = rh_dir / f"{video_name}.npy"
            
            # Check for hand-specific features
            if not lh_file.exists():
                failed_files.append(f"{video_name} - missing LH features")
                continue
            
            if not rh_file.exists():
                failed_files.append(f"{video_name} - missing RH features")
                continue
            
            # Load hand-specific features
            lh_features = np.load(lh_file)  # [75, T]
            rh_features = np.load(rh_file)  # [75, T]
            
            # Load semantic features if available
            semantic_lh = None
            semantic_rh = None
            
            if not skip_semantic:
                if semantic_lh_dir:
                    semantic_lh_file = semantic_lh_dir / f"{video_name}.npy"
                    if semantic_lh_file.exists():
                        semantic_lh = np.load(semantic_lh_file)
                
                if semantic_rh_dir:
                    semantic_rh_file = semantic_rh_dir / f"{video_name}.npy"
                    if semantic_rh_file.exists():
                        semantic_rh = np.load(semantic_rh_file)
            
            # Validate shapes
            if not validate_shapes(shared_features, lh_features, semantic_lh, video_name):
                failed_files.append(f"{video_name} - shape mismatch (LH)")
                continue
            
            if not validate_shapes(shared_features, rh_features, semantic_rh, video_name):
                failed_files.append(f"{video_name} - shape mismatch (RH)")
                continue
            
            # Concatenate features
            lh_final = concatenate_features(shared_features, lh_features, semantic_lh)
            rh_final = concatenate_features(shared_features, rh_features, semantic_rh)
            
            # Save concatenated features
            lh_output_path = lh_output_dir / f"{video_name}.npy"
            rh_output_path = rh_output_dir / f"{video_name}.npy"
            
            np.save(lh_output_path, lh_final)
            np.save(rh_output_path, rh_final)
            
            successful_lh += 1
            successful_rh += 1
            
            # Log processing info
            log_entry = {
                'video_name': video_name,
                'shared_shape': list(shared_features.shape),
                'lh_hand_shape': list(lh_features.shape),
                'rh_hand_shape': list(rh_features.shape),
                'lh_semantic_shape': list(semantic_lh.shape) if semantic_lh is not None else None,
                'rh_semantic_shape': list(semantic_rh.shape) if semantic_rh is not None else None,
                'lh_final_shape': list(lh_final.shape),
                'rh_final_shape': list(rh_final.shape),
                'has_semantic': semantic_lh is not None or semantic_rh is not None
            }
            processing_log.append(log_entry)
            
        except Exception as e:
            print(f"Error processing {shared_file.name}: {e}")
            failed_files.append(f"{shared_file.name} - error: {e}")
            continue
    
    # Create summary
    summary = {
        'description': 'Concatenated dual-hand features with semantic conditioning',
        'concatenation_steps': {
            'step1': 'shared [768, T] + hand [75, T] = intermediate [843, T]',
            'step2': 'intermediate [843, T] + semantic [D_sem, T] = final [843+D_sem, T]'
        },
        'processing_summary': {
            'total_files_processed': len(shared_files),
            'successful_lh': successful_lh,
            'successful_rh': successful_rh,
            'failed_files': len(failed_files),
            'semantic_used': not skip_semantic
        },
        'failed_files': failed_files,
        'processing_log': processing_log[:100]  # Store first 100 for reference
    }
    
    # Save metadata
    summary_path = metadata_dir / 'concatenation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


# ============================================================================
# Verification
# ============================================================================

def verify_concatenated_features(output_dir: Path, expected_semantic_dim: Optional[int] = None):
    """
    Verify the concatenated features
    
    Args:
        output_dir: Output directory containing lh_v0 and rh_v0 folders
        expected_semantic_dim: Expected semantic feature dimension (if used)
    """
    lh_dir = output_dir / 'lh_v0'
    rh_dir = output_dir / 'rh_v0'
    
    lh_files = list(lh_dir.glob('*.npy'))
    rh_files = list(rh_dir.glob('*.npy'))
    
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    print(f"LH concatenated files: {len(lh_files)}")
    print(f"RH concatenated files: {len(rh_files)}")
    
    if lh_files and rh_files:
        # Check first file
        example_lh = np.load(lh_files[0])
        example_rh = np.load(rh_files[0])
        
        print(f"\nExample shapes:")
        print(f"  LH: {example_lh.shape}")
        print(f"  RH: {example_rh.shape}")
        
        # Verify dimensions
        expected_base = 843  # 768 + 75
        if expected_semantic_dim:
            expected_total = expected_base + expected_semantic_dim
            if example_lh.shape[0] == expected_total and example_rh.shape[0] == expected_total:
                print(f"  ✓ Feature dimensions are correct ({expected_base} base + {expected_semantic_dim} semantic = {expected_total})")
            else:
                print(f"  ✗ Unexpected feature dimensions (expected {expected_total}, got {example_lh.shape[0]})")
        else:
            if example_lh.shape[0] == expected_base and example_rh.shape[0] == expected_base:
                print(f"  ✓ Feature dimensions are correct ({expected_base} = 768 shared + 75 hand)")
            else:
                print(f"  ✗ Unexpected feature dimensions (expected {expected_base}, got {example_lh.shape[0]})")
    
    # Check matching files
    lh_names = {f.stem for f in lh_files}
    rh_names = {f.stem for f in rh_files}
    
    missing_lh = rh_names - lh_names
    missing_rh = lh_names - rh_names
    
    if missing_lh:
        print(f"\n  Missing LH files: {len(missing_lh)}")
        print(f"  Examples: {list(missing_lh)[:5]}")
    if missing_rh:
        print(f"\n  Missing RH files: {len(missing_rh)}")
        print(f"  Examples: {list(missing_rh)[:5]}")
    
    if not missing_lh and not missing_rh:
        print(f"\n  ✓ All files have matching LH and RH versions")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Concatenate shared, hand-specific, and semantic features for dual-hand action segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input directories
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory containing shared_features, lh_features, rh_features folders'
    )
    
    parser.add_argument(
        '--semantic_lh_dir',
        type=str,
        default=None,
        help='Directory with semantic features for left hand (optional)'
    )
    
    parser.add_argument(
        '--semantic_rh_dir',
        type=str,
        default=None,
        help='Directory with semantic features for right hand (optional)'
    )
    
    # Output directory
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for concatenated features'
    )
    
    # Options
    parser.add_argument(
        '--skip_semantic',
        action='store_true',
        help='Skip semantic feature concatenation (only concatenate shared + hand features)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the concatenated results after processing'
    )
    
    parser.add_argument(
        '--expected_semantic_dim',
        type=int,
        default=None,
        help='Expected semantic feature dimension (for verification)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    semantic_lh_dir = Path(args.semantic_lh_dir) if args.semantic_lh_dir else None
    semantic_rh_dir = Path(args.semantic_rh_dir) if args.semantic_rh_dir else None
    
    # Validate inputs
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return
    
    if semantic_lh_dir and not semantic_lh_dir.exists():
        print(f"Warning: LH semantic directory does not exist: {semantic_lh_dir}")
        semantic_lh_dir = None
    
    if semantic_rh_dir and not semantic_rh_dir.exists():
        print(f"Warning: RH semantic directory does not exist: {semantic_rh_dir}")
        semantic_rh_dir = None
    
    print("=" * 60)
    print("DUAL-HAND FEATURE CONCATENATION WITH SEMANTIC CONDITIONING")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    if not args.skip_semantic:
        print(f"LH semantic: {semantic_lh_dir}")
        print(f"RH semantic: {semantic_rh_dir}")
    else:
        print("Semantic features: SKIPPED")
    print()
    
    # Process features
    summary = process_dual_hand_features(
        base_dir=base_dir,
        semantic_lh_dir=semantic_lh_dir,
        semantic_rh_dir=semantic_rh_dir,
        output_dir=output_dir,
        skip_semantic=args.skip_semantic
    )
    
    # Print summary
    if summary:
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files processed: {summary['processing_summary']['total_files_processed']}")
        print(f"Successful LH concatenations: {summary['processing_summary']['successful_lh']}")
        print(f"Successful RH concatenations: {summary['processing_summary']['successful_rh']}")
        print(f"Failed files: {summary['processing_summary']['failed_files']}")
        print(f"Semantic features used: {summary['processing_summary']['semantic_used']}")
        
        if summary['failed_files']:
            print(f"\nFailed files (showing first 10):")
            for failed_file in summary['failed_files'][:10]:
                print(f"  - {failed_file}")
            if len(summary['failed_files']) > 10:
                print(f"  ... and {len(summary['failed_files']) - 10} more")
        
        # Show example
        if summary['processing_log']:
            example = summary['processing_log'][0]
            print(f"\nExample for {example['video_name']}:")
            print(f"  Shared: {example['shared_shape']}")
            print(f"  LH hand: {example['lh_hand_shape']}")
            print(f"  RH hand: {example['rh_hand_shape']}")
            if example['has_semantic']:
                print(f"  LH semantic: {example['lh_semantic_shape']}")
                print(f"  RH semantic: {example['rh_semantic_shape']}")
            print(f"  LH final: {example['lh_final_shape']}")
            print(f"  RH final: {example['rh_final_shape']}")
        
        print(f"\nOutput structure:")
        print(f"  LH features: {output_dir / 'lh_v0'}")
        print(f"  RH features: {output_dir / 'rh_v0'}")
        print(f"  Metadata: {output_dir / 'metadata' / 'concatenation_summary.json'}")
    
    # Verify if requested
    if args.verify:
        verify_concatenated_features(output_dir, args.expected_semantic_dim)
    
    print(f"\n{'='*60}")
    print("✓ Processing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

