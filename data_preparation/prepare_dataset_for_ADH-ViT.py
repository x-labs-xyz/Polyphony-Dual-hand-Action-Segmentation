"""
Generate Action Recognition Dataset from Action Segmentation Dataset

This script extracts video clips from action segmentation datasets and prepares
them for action recognition tasks. Supports two extraction methods:

1. Segment-based: Extracts entire action segments (consecutive frames with same label)
2. Random clip-based: Samples random fixed-length clips with class balancing

Output: Organized video clips in label folders with index files for training.
"""

import os
import cv2
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


# ============================================================================
# Utility Functions
# ============================================================================

def load_label_mapping(mapping_file: Path) -> Dict[str, int]:
    """
    Load label string -> numeric ID mapping from file
    
    Supports multiple formats:
    - "index label"
    - "label index"
    - "label,index"
    - "label index" (tab or space separated)
    
    Args:
        mapping_file: Path to mapping file
        
    Returns:
        Dictionary mapping label strings to numeric IDs
    """
    label_to_id: Dict[str, int] = {}
    
    if not mapping_file.exists():
        print(f"Warning: Mapping file not found: {mapping_file}")
        return label_to_id
    
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try comma first, then whitespace split
            parts = [p.strip() for p in line.replace('\t', ' ').replace(',', ' ').split(' ') if p.strip()]
            if len(parts) < 2:
                continue
            
            a, b = parts[0], parts[1]
            
            # Determine which token is the ID
            def is_int(s: str) -> bool:
                try:
                    int(s)
                    return True
                except ValueError:
                    return False
            
            if is_int(a) and not is_int(b):
                label_to_id[b] = int(a)
            elif is_int(b) and not is_int(a):
                label_to_id[a] = int(b)
            # If both or neither are ints, skip (ambiguous)
    
    return label_to_id


def read_bundle_list(bundle_path: Path) -> List[str]:
    """
    Read video names from bundle file
    
    Args:
        bundle_path: Path to bundle file
        
    Returns:
        List of video base names (without extension)
    """
    names: List[str] = []
    
    if not bundle_path.exists():
        print(f"Warning: Bundle file not found: {bundle_path}")
        return names
    
    with open(bundle_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Remove .txt extension if present
            if line.endswith('.txt'):
                base = line[:-4]
            else:
                base = Path(line).stem
            
            names.append(base)
    
    return names


def read_frame_labels(annotation_path: Path) -> List[str]:
    """
    Load frame-wise annotations from text file
    
    Handles both formats:
    - One label per line
    - Space-separated labels on multiple lines
    
    Args:
        annotation_path: Path to annotation file
        
    Returns:
        List of frame labels
    """
    labels: List[str] = []
    
    if not annotation_path.exists():
        return labels
    
    with open(annotation_path, 'r') as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            
            # Split by whitespace to handle both formats
            tokens = raw.split()
            labels.extend(tokens)
    
    return labels


def extract_segments(frame_labels: List[str]) -> List[Tuple[int, int, str]]:
    """
    Extract action segments from frame labels
    
    A segment is a consecutive sequence of frames with the same label.
    
    Args:
        frame_labels: List of frame labels
        
    Returns:
        List of (start_frame, end_frame, label) tuples
    """
    segments: List[Tuple[int, int, str]] = []
    
    if not frame_labels:
        return segments
    
    current = frame_labels[0]
    start = 0
    
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != current:
            segments.append((start, i - 1, current))
            current = frame_labels[i]
            start = i
    
    # Add final segment
    segments.append((start, len(frame_labels) - 1, current))
    
    return segments


# ============================================================================
# Video Processing Functions
# ============================================================================

def save_video_clip(
    video_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: Optional[float] = None
) -> bool:
    """
    Extract and save a video clip from a video file
    
    Args:
        video_path: Path to input video
        out_path: Path to output video clip
        start_frame: Starting frame index (inclusive)
        end_frame: Ending frame index (inclusive)
        fps: Optional FPS override (uses video FPS if None)
        
    Returns:
        True if successful, False otherwise
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use provided FPS or video FPS, with fallback
    output_fps = fps if fps is not None else (video_fps if video_fps > 0 else 25.0)
    
    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (width, height))
    
    if not writer.isOpened():
        cap.release()
        return False
    
    try:
        # Extract frames
        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    
    # Verify output file was created and has content
    return out_path.exists() and out_path.stat().st_size > 0


def save_clip_from_frames(
    frames: List,
    out_path: Path,
    fps: float,
    width: int,
    height: int
) -> bool:
    """
    Save a list of frames as a video clip
    
    Args:
        frames: List of frame arrays
        out_path: Path to output video
        fps: Frames per second
        width: Frame width
        height: Frame height
        
    Returns:
        True if successful, False otherwise
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    
    if not writer.isOpened():
        return False
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    return out_path.exists() and out_path.stat().st_size > 0


# ============================================================================
# Extraction Methods
# ============================================================================

class SegmentBasedExtractor:
    """Extract entire action segments from videos"""
    
    def __init__(
        self,
        video_dir: Path,
        annotation_dir: Path,
        output_dir: Path,
        label_mapping: Dict[str, int],
        min_segment_length: int = 1
    ):
        """
        Args:
            video_dir: Directory containing input videos
            annotation_dir: Directory containing annotation files
            output_dir: Output directory for clips
            label_mapping: Label string -> numeric ID mapping
            min_segment_length: Minimum segment length to extract (frames)
        """
        self.video_dir = video_dir
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.label_mapping = label_mapping
        self.min_segment_length = min_segment_length
        self.label_counts: Dict[str, int] = {}
    
    def extract_from_video(self, video_base: str) -> List[Tuple[str, int]]:
        """
        Extract segments from a single video
        
        Args:
            video_base: Base name of video (without extension)
            
        Returns:
            List of (relative_clip_path, label_id) tuples
        """
        video_path = self.video_dir / f"{video_base}.mp4"
        ann_path = self.annotation_dir / f"{video_base}.txt"
        
        if not video_path.exists() or not ann_path.exists():
            return []
        
        # Load annotations
        frame_labels = read_frame_labels(ann_path)
        if len(frame_labels) == 0:
            return []
        
        # Extract segments
        segments = extract_segments(frame_labels)
        
        # Process each segment
        results = []
        label_to_count: Dict[str, int] = {}
        
        for start_frame, end_frame, label in segments:
            # Skip short segments
            segment_length = end_frame - start_frame + 1
            if segment_length < self.min_segment_length:
                continue
            
            # Skip labels not in mapping
            label_id = self.label_mapping.get(label)
            if label_id is None:
                continue
            
            # Generate output filename
            label_to_count.setdefault(label, 0)
            seg_idx = label_to_count[label]
            label_to_count[label] += 1
            
            # Create output path
            label_dir = self.output_dir / label
            out_filename = f"{video_base}_{label}_{seg_idx}.mp4"
            out_path = label_dir / out_filename
            out_rel_path = f"{label}/{out_filename}"
            
            # Extract and save clip
            if save_video_clip(video_path, out_path, start_frame, end_frame):
                results.append((out_rel_path, label_id))
                self.label_counts[label] = self.label_counts.get(label, 0) + 1
        
        return results


class RandomClipExtractor:
    """Extract random fixed-length clips from videos with class balancing"""
    
    def __init__(
        self,
        video_dir: Path,
        annotation_dir: Path,
        output_dir: Path,
        label_mapping: Dict[str, int],
        clips_per_video: int = 10,
        clip_length: int = 16,
        prefer_non_null: bool = True
    ):
        """
        Args:
            video_dir: Directory containing input videos
            annotation_dir: Directory containing annotation files
            output_dir: Output directory for clips
            label_mapping: Label string -> numeric ID mapping
            clips_per_video: Number of clips to sample per video
            clip_length: Length of each clip in frames
            prefer_non_null: Prefer clips with non-null labels
        """
        self.video_dir = video_dir
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.label_mapping = label_mapping
        self.clips_per_video = clips_per_video
        self.clip_length = clip_length
        self.prefer_non_null = prefer_non_null
        self.label_counts: Dict[str, int] = {}
    
    def extract_from_video(self, video_base: str) -> List[Tuple[str, int]]:
        """
        Extract random clips from a single video
        
        Args:
            video_base: Base name of video (without extension)
            
        Returns:
            List of (relative_clip_path, label_id) tuples
        """
        video_path = self.video_dir / f"{video_base}.mp4"
        ann_path = self.annotation_dir / f"{video_base}.txt"
        
        if not video_path.exists() or not ann_path.exists():
            return []
        
        # Load annotations
        frame_labels = read_frame_labels(ann_path)
        if len(frame_labels) == 0:
            return []
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Adjust for annotation length mismatch
        if total_frames != len(frame_labels):
            total_frames = min(total_frames, len(frame_labels))
        
        # Check if video is long enough
        max_start_frame = total_frames - self.clip_length
        if max_start_frame <= 0:
            cap.release()
            return []
        
        # Find candidate clip start positions
        middle_offset = self.clip_length // 2
        preferred_candidates: List[Tuple[int, str]] = []
        fallback_candidates: List[Tuple[int, str]] = []
        
        for start in range(max_start_frame + 1):
            middle = start + middle_offset
            if middle >= total_frames:
                continue
            
            label_str = str(frame_labels[middle]).strip()
            
            if self.prefer_non_null and label_str.lower() != 'null':
                preferred_candidates.append((start, label_str))
            else:
                fallback_candidates.append((start, label_str))
        
        # Select clips with class balancing
        num_needed = min(self.clips_per_video, max_start_frame + 1)
        
        def sort_balanced(candidates: List[Tuple[int, str]]) -> List[int]:
            """Sort candidates by class frequency (ascending) for balancing"""
            random.shuffle(candidates)  # Random tie-breaking
            return [s for s, _ in sorted(candidates, key=lambda x: self.label_counts.get(x[1], 0))]
        
        chosen_starts: List[int] = []
        
        if preferred_candidates:
            balanced_pref = sort_balanced(preferred_candidates)
            chosen_starts.extend(balanced_pref[:num_needed])
        
        if len(chosen_starts) < num_needed and fallback_candidates:
            remaining = num_needed - len(chosen_starts)
            balanced_fb = sort_balanced(fallback_candidates)
            chosen_starts.extend(balanced_fb[:remaining])
        
        if not chosen_starts:
            chosen_starts = random.sample(range(max_start_frame + 1), num_needed)
        
        # Extract and save clips
        results = []
        clip_idx = 0
        
        for start_frame in chosen_starts:
            # Get label from middle frame
            middle_frame = start_frame + middle_offset
            action_label = frame_labels[middle_frame]
            label_id = self.label_mapping.get(action_label)
            
            if label_id is None:
                continue
            
            # Extract frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            
            for _ in range(self.clip_length):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if len(frames) != self.clip_length:
                continue
            
            # Save clip
            label_dir = self.output_dir / action_label
            clip_filename = f"{video_base}_clip_{action_label}_{clip_idx}.mp4"
            out_path = label_dir / clip_filename
            out_rel_path = f"{action_label}/{clip_filename}"
            
            if save_clip_from_frames(frames, out_path, fps, width, height):
                results.append((out_rel_path, label_id))
                self.label_counts[action_label] = self.label_counts.get(action_label, 0) + 1
                clip_idx += 1
        
        cap.release()
        return results


# ============================================================================
# Main Dataset Builder
# ============================================================================

class ActionRecognitionDatasetBuilder:
    """Build action recognition dataset from action segmentation dataset"""
    
    def __init__(
        self,
        video_dir: Path,
        annotation_dir: Path,
        output_dir: Path,
        extraction_method: str = 'segment',
        label_mapping: Optional[Dict[str, int]] = None,
        **extractor_kwargs
    ):
        """
        Args:
            video_dir: Directory containing input videos
            annotation_dir: Directory containing annotation files
            output_dir: Output directory for action recognition dataset
            extraction_method: 'segment' or 'random'
            label_mapping: Label string -> numeric ID mapping
            **extractor_kwargs: Additional arguments for extractor
        """
        self.video_dir = Path(video_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.label_mapping = label_mapping or {}
        
        # Initialize extractor based on method
        if extraction_method == 'segment':
            self.extractor = SegmentBasedExtractor(
                self.video_dir,
                self.annotation_dir,
                self.output_dir,
                self.label_mapping,
                min_segment_length=extractor_kwargs.get('min_segment_length', 1)
            )
        elif extraction_method == 'random':
            self.extractor = RandomClipExtractor(
                self.video_dir,
                self.annotation_dir,
                self.output_dir,
                self.label_mapping,
                clips_per_video=extractor_kwargs.get('clips_per_video', 10),
                clip_length=extractor_kwargs.get('clip_length', 16),
                prefer_non_null=extractor_kwargs.get('prefer_non_null', True)
            )
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")
        
        self.extraction_method = extraction_method
    
    def build_dataset(
        self,
        split_list: Optional[Path] = None,
        index_file: Optional[Path] = None
    ):
        """
        Build the action recognition dataset
        
        Args:
            split_list: Optional path to file listing videos to include
            index_file: Optional path to output index file
        """
        print(f"Building action recognition dataset using {self.extraction_method}-based extraction...")
        
        # Get list of videos to process
        if split_list and split_list.exists():
            video_bases = read_bundle_list(split_list)
            video_files = [self.video_dir / f"{base}.mp4" for base in video_bases]
            video_files = [vf for vf in video_files if vf.exists()]
        else:
            video_files = list(self.video_dir.glob('*.mp4'))
        
        if not video_files:
            print(f"No video files found in {self.video_dir}")
            return
        
        print(f"Found {len(video_files)} video files")
        
        # Determine index file path
        if index_file is None:
            index_file = self.output_dir.parent / 'train_list_video.txt'
        
        index_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process videos
        all_results = []
        
        with open(index_file, 'w') as f:
            for video_file in tqdm(video_files, desc="Processing videos"):
                video_base = video_file.stem
                results = self.extractor.extract_from_video(video_base)
                
                for rel_path, label_id in results:
                    f.write(f"{rel_path} {label_id}\n")
                    all_results.append((rel_path, label_id))
        
        # Print statistics
        self.print_statistics(index_file)
    
    def print_statistics(self, index_file: Path):
        """Print dataset statistics"""
        print("\n" + "=" * 70)
        print("Dataset Statistics")
        print("=" * 70)
        
        if hasattr(self.extractor, 'label_counts'):
            total_clips = sum(self.extractor.label_counts.values())
            num_classes = len(self.extractor.label_counts)
            
            print(f"Total clips: {total_clips}")
            print(f"Number of action classes: {num_classes}")
            print("\nClips per class:")
            
            for label, count in sorted(self.extractor.label_counts.items()):
                label_id = self.label_mapping.get(label, 'N/A')
                print(f"  {label:<30} (ID: {label_id:>3}): {count:>5} clips")
        
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Index file: {index_file}")
        print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate action recognition dataset from action segmentation dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Path to directory containing input videos'
    )
    parser.add_argument(
        '--annotation_dir',
        type=str,
        required=True,
        help='Path to directory containing annotation files (.txt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for action recognition dataset'
    )
    parser.add_argument(
        '--mapping_file',
        type=str,
        required=True,
        help='Path to label mapping file (label -> numeric ID)'
    )
    
    # Extraction method
    parser.add_argument(
        '--extraction_method',
        type=str,
        choices=['segment', 'random'],
        default='segment',
        help='Extraction method: segment (entire segments) or random (fixed-length clips)'
    )
    
    # Split configuration
    parser.add_argument(
        '--split_list',
        type=str,
        default=None,
        help='Path to file listing videos to include (bundle format)'
    )
    parser.add_argument(
        '--index_file',
        type=str,
        default=None,
        help='Path to output index file (default: output_dir/../train_list_video.txt)'
    )
    
    # Segment-based parameters
    parser.add_argument(
        '--min_segment_length',
        type=int,
        default=1,
        help='Minimum segment length for segment-based extraction (frames)'
    )
    
    # Random clip-based parameters
    parser.add_argument(
        '--clips_per_video',
        type=int,
        default=10,
        help='Number of clips per video for random extraction'
    )
    parser.add_argument(
        '--clip_length',
        type=int,
        default=16,
        help='Length of each clip for random extraction (frames)'
    )
    parser.add_argument(
        '--prefer_non_null',
        action='store_true',
        default=True,
        help='Prefer non-null labels for random extraction'
    )
    parser.add_argument(
        '--no_prefer_non_null',
        dest='prefer_non_null',
        action='store_false',
        help='Disable preference for non-null labels'
    )
    
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load label mapping
    label_mapping = load_label_mapping(Path(args.mapping_file))
    print(f"Loaded {len(label_mapping)} label mappings")
    
    # Create builder
    builder = ActionRecognitionDatasetBuilder(
        video_dir=Path(args.video_dir),
        annotation_dir=Path(args.annotation_dir),
        output_dir=Path(args.output_dir),
        extraction_method=args.extraction_method,
        label_mapping=label_mapping,
        min_segment_length=args.min_segment_length,
        clips_per_video=args.clips_per_video,
        clip_length=args.clip_length,
        prefer_non_null=args.prefer_non_null
    )
    
    # Build dataset
    split_list = Path(args.split_list) if args.split_list else None
    index_file = Path(args.index_file) if args.index_file else None
    
    builder.build_dataset(split_list=split_list, index_file=index_file)
    
    print("\n✓ Dataset generation complete!")


if __name__ == '__main__':
    main()

