"""
Standalone script to evaluate existing model and generate visualizations
Usage: python evaluate_and_visualize.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import sys

# Import from the training script
from TCN_semantic_feature_alignment_v3_minilm_havid import (
    SemanticFeatureAlignmentModel,
    HAVIDDataset,
    HAVIDEvaluator,
    dynamic_collate_fn
)

class LargerTextVisualizer:
    """Visualization with larger text and smaller bar charts"""
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality style with extra large fonts
        plt.rcParams.update({
            'font.size': 28,
            'axes.labelsize': 36,
            'axes.titlesize': 42,
            'xtick.labelsize': 32,
            'ytick.labelsize': 32,
            'legend.fontsize': 32,
            'figure.titlesize': 44
        })
        sns.set_palette("husl")
        
    def plot_alignment_metrics(self, eval_results: Dict[str, float], filename: str = 'alignment_metrics.png'):
        """Plot key alignment metrics as bar chart with extra large text and shorter height"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))  # Very wide but short
        
        # Subplot 1: Overall metrics
        metrics = ['mean_cosine_similarity', 'median_cosine_similarity', 'std_cosine_similarity']
        values = [eval_results.get(m, 0) for m in metrics]
        labels = ['Mean\nCosine Sim', 'Median\nCosine Sim', 'Std\nCosine Sim']
        
        bars = axes[0].bar(labels, values, color=['#2ecc71', '#3498db', '#e74c3c'], width=0.28)  # Extra narrow bars
        axes[0].set_ylabel('Value', fontsize=38, fontweight='bold')
        axes[0].set_title('Alignment Quality', fontsize=44, fontweight='bold', pad=30)
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(axis='y', alpha=0.3, linewidth=3)
        axes[0].tick_params(axis='both', which='major', labelsize=34, width=3, length=12)
        
        # Add value labels on bars with extra large font
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=36, fontweight='bold')
        
        # Subplot 2: Action-type specific performance
        action_types = ['actions', 'null', 'wrong']
        action_labels = ['Actions', 'Null/\nTransition', 'Wrong']
        action_values = [eval_results.get(f'{at}_mean_similarity', 0) for at in action_types]
        action_counts = [eval_results.get(f'{at}_count', 0) for at in action_types]
        
        bars = axes[1].bar(action_labels, action_values, color=['#9b59b6', '#f39c12', '#e67e22'], width=0.28)  # Extra narrow bars
        axes[1].set_ylabel('Mean Cosine Similarity', fontsize=38, fontweight='bold')
        axes[1].set_title('Performance by Type', fontsize=44, fontweight='bold', pad=30)
        axes[1].set_ylim([0, 1.0])
        axes[1].grid(axis='y', alpha=0.3, linewidth=3)
        axes[1].tick_params(axis='both', which='major', labelsize=34, width=3, length=12)
        
        # Add value labels with counts - extra large font
        for bar, count in zip(bars, action_counts):
            height = bar.get_height()
            if count > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'{height:.3f}\n(n={count})',
                            ha='center', va='bottom', fontsize=34, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()
        print(f"✓ Saved alignment metrics plot to {save_path}")
    
    def plot_similarity_distribution(self, similarities: np.ndarray, 
                                    filename: str = 'similarity_distribution.png'):
        """Plot distribution of cosine similarities as histogram with extra large text and shorter height"""
        fig, ax = plt.subplots(figsize=(18, 6))  # Wider but shorter
        
        # Histogram with thicker bars
        n, bins, patches = ax.hist(similarities, bins=50, color='#3498db', alpha=0.7, 
                                     edgecolor='black', linewidth=3)
        ax.axvline(similarities.mean(), color='#e74c3c', linestyle='--', 
                   linewidth=6, label=f'Mean: {similarities.mean():.3f}')
        ax.axvline(np.median(similarities), color='#2ecc71', linestyle='--', 
                   linewidth=6, label=f'Median: {np.median(similarities):.3f}')
        
        # Add statistics text box with extra large font
        stats_text = f"Statistics:\n"
        stats_text += f"Mean:   {similarities.mean():.3f}\n"
        stats_text += f"Median: {np.median(similarities):.3f}\n"
        stats_text += f"Std:    {similarities.std():.3f}\n"
        stats_text += f"Min:    {similarities.min():.3f}\n"
        stats_text += f"Max:    {similarities.max():.3f}"
        
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=28, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                         edgecolor='black', linewidth=3.5))
        
        ax.set_xlabel('Cosine Similarity', fontsize=38, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=38, fontweight='bold')
        #ax.set_title('Similarity Distribution', fontsize=44, fontweight='bold', pad=30)
        ax.legend(fontsize=34, loc='upper right', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linewidth=3)
        ax.tick_params(axis='both', which='major', labelsize=34, width=3, length=12)
        
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()
        print(f"✓ Saved similarity distribution plot to {save_path}")

def evaluate_existing_model(config):
    """Evaluate an existing trained model and generate visualizations"""
    
    print("="*80)
    print("EVALUATING EXISTING MODEL")
    print("="*80)
    
    device = torch.device(config['device'])
    
    # Create data loaders
    print("\n1. Creating data loaders...")
    test_dataset = HAVIDDataset(
        data_root=config['data_root'],
        split_file=config['test_split'],
        feature_path=config['feature_path'],
        annotation_path=config['annotation_path'],
        semantic_embeddings_path=config['semantic_embeddings_path'],
        downsample_rate=config['downsample_rate']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=dynamic_collate_fn
    )
    print(f"   Loaded {len(test_dataset)} test samples")
    
    # Initialize model
    print("\n2. Initializing model...")
    model = SemanticFeatureAlignmentModel(
        visual_dim=config['visual_dim'],
        semantic_dim=config['semantic_dim'],
        tcn_hidden_dims=config['tcn_hidden_dims']
    )
    
    # Load checkpoint
    print(f"\n3. Loading checkpoint from {config['checkpoint_path']}...")
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"   Model loaded successfully")
    print(f"   Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Training loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    
    # Get action mapping and label indices from dataset
    action_mapping = test_dataset.action_mapping
    label_to_idx = test_dataset.label_to_idx
    
    # Initialize evaluator
    print("\n4. Evaluating model...")
    evaluator = HAVIDEvaluator(model, device, action_mapping, label_to_idx)
    eval_results = evaluator.evaluate_alignment_quality(test_loader)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for metric, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {metric:30s}: {value:.6f}")
        else:
            print(f"  {metric:30s}: {value}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS (LARGER TEXT)")
    print("="*80)
    
    visualizer = LargerTextVisualizer(save_dir=config['visualization_dir'])
    
    # 1. Generate alignment metrics
    print("\n1. Generating alignment metrics bar chart...")
    visualizer.plot_alignment_metrics(eval_results, 'alignment_metrics.png')
    
    # 2. Collect similarities for distribution
    print("\n2. Collecting frame-wise similarities...")
    all_similarities = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            visual_feat = batch['visual_features'].to(device)
            semantic_feat = batch['semantic_features'].to(device)
            
            outputs = model(visual_feat)
            predicted_semantic = outputs['aligned_features']
            
            for i in range(len(visual_feat)):
                valid_len = batch['original_length'][i].item()
                pred = predicted_semantic[i, :valid_len].cpu().numpy()
                target = semantic_feat[i, :valid_len].cpu().numpy()
                
                # Normalize and compute similarities
                pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
                target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
                sims = np.sum(pred_norm * target_norm, axis=1)
                all_similarities.append(sims)
            
            if batch_idx >= 20:  # Limit for efficiency
                break
    
    all_similarities = np.concatenate(all_similarities)
    print(f"   Collected {len(all_similarities)} frame similarities")
    
    # 3. Generate distribution plot
    print("\n3. Generating similarity distribution histogram...")
    visualizer.plot_similarity_distribution(all_similarities, 'similarity_distribution.png')
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nVisualizations saved to: {config['visualization_dir']}")
    print(f"  ✓ alignment_metrics.png")
    print(f"  ✓ similarity_distribution.png")
    print("="*80)

if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS TO MATCH YOUR SETUP
    config = {
        # Data paths
        'data_root': '/home/hao/Polyphony/data/havid',
        'test_split': 'splits/View0/rh_pt/test.split1.bundle',
        'feature_path': '/home/hao/Polyphony/data/havid/videomae_features_extend/view0/shared_features',
        'annotation_path': '/home/hao/Polyphony/data/havid/groundTruth/View0/rh_pt',
        'semantic_embeddings_path': '/home/hao/Polyphony/data/havid/semantic_embeddings/sentence-transformers_all-MiniLM-L6-v2.pt', #BAAI_bge-large-en-v1.5  sentence-transformers_all-mpnet-base-v2.pt #sentence-transformers_all-MiniLM-L6-v2_simple_sentence.pt #sentence-transformers_all-MiniLM-L6-v2
        
        # Model checkpoint path - CHANGE THIS TO YOUR BEST MODEL
        'checkpoint_path': './havid_checkpoints/dual_hand_extend/MiniLM/rh_v0/best_model.pth',
        
        # Output directory for visualizations
        'visualization_dir': './havid_checkpoints/dual_hand_extend/MiniLM/rh_v0/visualizations_large_text',
        
        # Model configuration - MUST MATCH YOUR TRAINED MODEL
        'visual_dim': 768,
        'semantic_dim': 384,
        'tcn_hidden_dims': [512, 128, 64],  # IMPORTANT: Match your trained model
        
        # Evaluation settings
        'batch_size': 4,
        'downsample_rate': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\n" + "="*80)
    print("MODEL EVALUATION AND VISUALIZATION SCRIPT")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Checkpoint: {config['checkpoint_path']}")
    print(f"  Test split: {config['test_split']}")
    print(f"  TCN dims: {config['tcn_hidden_dims']}")
    print(f"  Output dir: {config['visualization_dir']}")
    print("="*80)
    
    # Run evaluation
    evaluate_existing_model(config)

