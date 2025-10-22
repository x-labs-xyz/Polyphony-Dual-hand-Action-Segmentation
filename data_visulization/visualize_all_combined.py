#!/usr/bin/env python3
"""
Script to visualize all action segmentation results in a single combined image.
Creates a multi-row figure showing all videos and hands.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import OrderedDict
import argparse


def read_labels(file_path):
    """Read labels from a file (one label per line)."""
    with open(file_path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def get_color_map(unique_labels):
    """Generate a fixed, deterministic color map for unique labels."""
    # Define a fixed color mapping for known labels
    # This ensures consistency across different runs
    fixed_label_colors = {
        # null
        'null': '#f0f0f0',  # very light gray
        
        # Common action labels - assign specific colors
        'w': '#1f77b4',  # blue
        'ibscb': '#ff7f0e',  # orange
        'ibacb': '#2ca02c',  # green
        'ickcb': '#d62728',  # red
        'scccb': '#9467bd',  # purple
        'lck': '#8c564b',  # brown
        'pckbx': '#e377c2',  # pink
        'sshc1': '#bcbd22',  # olive
        'sshc2': '#17becf',  # cyan
        'sshc3': '#aec7e8',  # light blue
        'sshc1dh': '#ffbb78',  # light orange
        'sshc2dh': '#98df8a',  # light green
        'sshc3dh': '#ff9896',  # light red
        'sshc2dp': '#c5b0d5',  # light purple
        'sshc3dp': '#c49c94',  # light brown
        'sshc4dp': '#f7b6d2',  # light pink
        'sftg1ws': '#7f7f7f',  # gray
        'sftg2ws': '#dbdb8d',  # light olive
        'igsft': '#9edae5',  # light cyan
        'sntftwn': '#ff6b6b',  # coral
        'sspg3dp': '#4ecdc4',  # turquoise
        'rgw': '#95e1d3',  # mint
        'ishc3': '#ffd93d',  # yellow
        'ishc1': '#c7ceea',  # lavender
        'ishc2': '#ff8c94',  # salmon
        'ishc4': '#6bcf7f',  # light green
        'ipift': '#a8e6cf',  # seafoam
        'ipift2': '#dcedc1',  # pale green
        'ibacb2': '#ffd3b6',  # peach
        'igift': '#ffaaa5',  # light coral
    }
    
    # Additional colors for any labels not in the fixed mapping
    extra_colors = [
        '#90caf9', '#ce93d8', '#80cbc4', '#fff59d', '#ffab91',
        '#b39ddb', '#81c784', '#ffcc80', '#a1887f', '#90a4ae',
        '#ef9a9a', '#f48fb1', '#9fa8da', '#80deea', '#c5e1a5',
        '#e6ee9c', '#bcaaa4', '#eeeeee', '#b0bec5', '#cfd8dc'
    ]
    
    color_map = {}
    
    # First, assign fixed colors to known labels
    for label in unique_labels:
        if label in fixed_label_colors:
            color_map[label] = fixed_label_colors[label]
    
    # For any remaining labels, assign colors deterministically based on sorted order
    remaining_labels = sorted([l for l in unique_labels if l not in fixed_label_colors])
    for i, label in enumerate(remaining_labels):
        color_map[label] = extra_colors[i % len(extra_colors)]
    
    return color_map


def create_segment_data(labels):
    """
    Convert a list of labels into segments for visualization.
    Returns a list of (start, end, label) tuples.
    """
    if not labels:
        return []
    
    segments = []
    current_label = labels[0]
    start = 0
    
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append((start, i, current_label))
            current_label = labels[i]
            start = i
    
    # Add the last segment
    segments.append((start, len(labels), current_label))
    
    return segments


def plot_single_row(ax, labels, color_map, label_text, show_xlabel=False):
    """Plot a single row of segmentation."""
    segments = create_segment_data(labels)
    
    for start, end, label in segments:
        ax.barh(0, end - start, left=start, height=0.8,
               color=color_map[label], edgecolor='black', linewidth=0.5)
    
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0, len(labels))
    ax.set_ylabel(label_text, fontsize=10, fontweight='bold')
    ax.set_yticks([])
    
    if show_xlabel:
        ax.set_xlabel('Frame Index', fontsize=10)
    else:
        ax.set_xticklabels([])
    
    ax.grid(axis='x', alpha=0.3)
    
    # Remove frame/spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def visualize_all_combined(input_dir='.', output_file='all_segmentations_combined.png',
                          layout='vertical', figsize=None):
    """
    Visualize all segmentation results in a single combined figure.
    
    Args:
        input_dir: Directory containing the data files
        output_file: Output file path
        layout: 'vertical' (all rows stacked) or 'grid' (2x2 grid layout)
        figsize: Figure size (width, height). If None, auto-calculated
    """
    input_path = Path(input_dir)
    
    # Find all prediction files and organize them
    pred_files = sorted(input_path.glob('*_predict.txt'))
    
    if not pred_files:
        print("No prediction files found!")
        return
    
    # Organize files by video
    videos = {}
    for pred_file in pred_files:
        base_name = pred_file.stem.replace('_predict', '')
        parts = base_name.rsplit('_', 1)
        if len(parts) == 2:
            video_name, hand = parts
            if video_name not in videos:
                videos[video_name] = {}
            videos[video_name][hand] = base_name
    
    print(f"Found {len(videos)} videos with {sum(len(v) for v in videos.values())} hand recordings")
    
    # Read all data and collect unique labels
    all_data = []
    all_labels = set()
    
    for video_name in sorted(videos.keys()):
        for hand in ['lh', 'rh']:
            if hand in videos[video_name]:
                base_name = videos[video_name][hand]
                
                gt_file = input_path / f"{base_name}_gt.txt"
                pred_file = input_path / f"{base_name}_predict.txt"
                
                if gt_file.exists() and pred_file.exists():
                    gt_labels = read_labels(gt_file)
                    pred_labels = read_labels(pred_file)
                    
                    all_labels.update(gt_labels)
                    all_labels.update(pred_labels)
                    
                    hand_name = "Left Hand" if hand == 'lh' else "Right Hand"
                    all_data.append({
                        'video': video_name,
                        'hand': hand_name,
                        'gt': gt_labels,
                        'pred': pred_labels
                    })
    
    if not all_data:
        print("No valid data pairs found!")
        return
    
    # Get color map
    color_map = get_color_map(all_labels)
    
    # Calculate number of rows (2 per video: GT and Pred)
    num_comparisons = len(all_data)
    num_rows = num_comparisons * 2
    
    # Set figure size
    if figsize is None:
        if layout == 'vertical':
            figsize = (18, 2 * num_rows)
        else:
            figsize = (20, 12)
    
    # Create figure
    fig, axes = plt.subplots(num_rows, 1, figsize=figsize)
    
    if num_rows == 1:
        axes = [axes]
    
    # Plot all ground truths first (top rows)
    row = 0
    for i, data in enumerate(all_data):
        ax = axes[row]
        video_label = f"{data['video']} - {data['hand']} (GT)"
        plot_single_row(ax, data['gt'], color_map, video_label, show_xlabel=False)
        row += 1
    
    # Then plot all predictions (bottom rows)
    for i, data in enumerate(all_data):
        ax = axes[row]
        video_label = f"{data['video']} - {data['hand']} (Pred)"
        is_last = (row == num_rows - 1)
        plot_single_row(ax, data['pred'], color_map, video_label, show_xlabel=is_last)
        row += 1
    
    # Add title
    fig.suptitle('Action Segmentation Results - All Videos', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Create legend
    legend_labels = sorted([l for l in all_labels if l != 'null']) + \
                   (['null'] if 'null' in all_labels else [])
    patches = [mpatches.Patch(color=color_map[label], label=label) 
              for label in legend_labels]
    
    # Place legend outside the plot
    fig.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=11, frameon=True, fancybox=True, shadow=True,
              title='Action Labels', title_fontsize=12)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved combined visualization to: {output_file}")
    print(f"   - {num_comparisons} videos")
    print(f"   - {num_rows} rows (GT + Prediction for each)")
    print(f"   - {len(all_labels)} unique action labels")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize all action segmentation results in a single combined image'
    )
    parser.add_argument('--input-dir', type=str, default='.',
                       help='Input directory containing the data files')
    parser.add_argument('--output', type=str, default='all_segmentations_combined.png',
                       help='Output file path')
    parser.add_argument('--width', type=int, default=18,
                       help='Figure width in inches')
    parser.add_argument('--height', type=int, default=None,
                       help='Figure height in inches (auto if not specified)')
    
    args = parser.parse_args()
    
    figsize = None
    if args.height:
        figsize = (args.width, args.height)
    elif args.width != 18:
        # Auto-calculate height based on width
        figsize = (args.width, None)
    
    visualize_all_combined(
        input_dir=args.input_dir,
        output_file=args.output,
        figsize=figsize
    )


if __name__ == '__main__':
    main()

