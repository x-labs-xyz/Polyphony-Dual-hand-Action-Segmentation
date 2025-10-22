#!/usr/bin/env python3
"""
Script to visualize action segmentation results as horizontal bar charts.
Each action label is displayed with a different color.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
from pathlib import Path
import argparse
from collections import OrderedDict


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


def visualize_segmentation(gt_file=None, pred_file=None, output_file=None, 
                          title=None, show_legend=True, figsize=(16, 3)):
    """
    Visualize action segmentation as horizontal bar chart.
    
    Args:
        gt_file: Path to ground truth file (optional)
        pred_file: Path to prediction file (optional)
        output_file: Path to save the figure (if None, will display)
        title: Title for the plot
        show_legend: Whether to show the legend
        figsize: Figure size (width, height)
    """
    # Read labels
    gt_labels = read_labels(gt_file) if gt_file else None
    pred_labels = read_labels(pred_file) if pred_file else None
    
    if gt_labels is None and pred_labels is None:
        raise ValueError("At least one of gt_file or pred_file must be provided")
    
    # Collect all unique labels
    all_labels = set()
    if gt_labels:
        all_labels.update(gt_labels)
    if pred_labels:
        all_labels.update(pred_labels)
    
    # Get color map
    color_map = get_color_map(all_labels)
    
    # Create figure
    num_rows = sum([gt_labels is not None, pred_labels is not None])
    fig, axes = plt.subplots(num_rows, 1, figsize=figsize)
    
    if num_rows == 1:
        axes = [axes]
    
    row = 0
    
    # Plot ground truth
    if gt_labels:
        ax = axes[row]
        segments = create_segment_data(gt_labels)
        
        for start, end, label in segments:
            ax.barh(0, end - start, left=start, height=0.8, 
                   color=color_map[label], edgecolor='black', linewidth=0.5)
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, len(gt_labels))
        ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.set_xlabel('Frame Index', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        
        # Remove frame/spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        row += 1
    
    # Plot predictions
    if pred_labels:
        ax = axes[row]
        segments = create_segment_data(pred_labels)
        
        for start, end, label in segments:
            ax.barh(0, end - start, left=start, height=0.8,
                   color=color_map[label], edgecolor='black', linewidth=0.5)
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, len(pred_labels))
        ax.set_ylabel('Prediction', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.set_xlabel('Frame Index', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        
        # Remove frame/spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    # Add title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Add legend
    if show_legend:
        # Create legend patches
        legend_labels = sorted([l for l in all_labels if l != 'null']) + (['null'] if 'null' in all_labels else [])
        patches = [mpatches.Patch(color=color_map[label], label=label) 
                  for label in legend_labels]
        
        # Place legend outside the plot
        fig.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def visualize_all_files(input_dir='.', output_dir='./visualizations'):
    """
    Visualize all prediction and ground truth files in a directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all prediction files
    pred_files = list(input_path.glob('*_predict.txt'))
    
    print(f"Found {len(pred_files)} prediction files")
    
    for pred_file in pred_files:
        # Extract base name (e.g., S02A22I23S1_lh or S02A22I23S1_rh)
        base_name = pred_file.stem.replace('_predict', '')
        
        # Find corresponding ground truth file
        gt_file = input_path / f"{base_name}_gt.txt"
        
        # Create visualization
        if gt_file.exists():
            print(f"Processing {base_name} (with ground truth)...")
            output_file = output_path / f"{base_name}_comparison.png"
            visualize_segmentation(
                gt_file=gt_file,
                pred_file=pred_file,
                output_file=output_file,
                title=f"Action Segmentation: {base_name}",
                show_legend=True,
                figsize=(16, 4)
            )
        else:
            print(f"Processing {base_name} (prediction only)...")
            output_file = output_path / f"{base_name}_prediction.png"
            visualize_segmentation(
                pred_file=pred_file,
                output_file=output_file,
                title=f"Action Segmentation: {base_name}",
                show_legend=True,
                figsize=(16, 3)
            )


def main():
    parser = argparse.ArgumentParser(
        description='Visualize action segmentation results as horizontal bar charts'
    )
    parser.add_argument('--gt', type=str, help='Ground truth file path')
    parser.add_argument('--pred', type=str, help='Prediction file path')
    parser.add_argument('--output', type=str, help='Output file path (if not specified, will display)')
    parser.add_argument('--title', type=str, help='Plot title')
    parser.add_argument('--no-legend', action='store_true', help='Hide legend')
    parser.add_argument('--width', type=int, default=16, help='Figure width in inches')
    parser.add_argument('--height', type=int, default=3, help='Figure height per row in inches')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all prediction files in the current directory')
    parser.add_argument('--input-dir', type=str, default='.', 
                       help='Input directory for batch processing')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                       help='Output directory for batch processing')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch process all files
        visualize_all_files(args.input_dir, args.output_dir)
    else:
        # Single file processing
        if not args.gt and not args.pred:
            parser.error("At least one of --gt or --pred must be specified (or use --batch)")
        
        # Adjust height based on number of rows
        num_rows = sum([args.gt is not None, args.pred is not None])
        figsize = (args.width, args.height * num_rows)
        
        visualize_segmentation(
            gt_file=args.gt,
            pred_file=args.pred,
            output_file=args.output,
            title=args.title,
            show_legend=not args.no_legend,
            figsize=figsize
        )


if __name__ == '__main__':
    main()

