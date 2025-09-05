#!/usr/bin/env python3

import os
import numpy as np
from collections import defaultdict, Counter

def load_action_mapping(mapping_file):
    """Load action mapping from file"""
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    mapping[parts[1]] = int(parts[0])
    return mapping

def load_predictions(pred_file):
    """Load predictions from file"""
    with open(pred_file, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 2:
            return lines[1].strip().split()
    return []

def load_ground_truth(gt_file):
    """Load ground truth from file"""
    with open(gt_file, 'r') as f:
        return [line.strip() for line in f]

def analyze_coordination_patterns(lh_pred_dir, rh_pred_dir, lh_gt_dir, rh_gt_dir):
    """Analyze coordination patterns between hands"""
    
    # Get list of test videos
    lh_pred_files = [f for f in os.listdir(lh_pred_dir) if f.endswith('.txt')]
    
    # Statistics
    coord_stats = {
        'same_action_gt': 0,      # Ground truth has same action for both hands
        'same_action_pred': 0,    # Prediction has same action for both hands  
        'same_action_correct': 0, # Both GT and prediction have same action
        'total_frames': 0,
        'transition_sync_gt': 0,   # GT transitions happen simultaneously
        'transition_sync_pred': 0, # Predicted transitions happen simultaneously
        'null_both_gt': 0,        # Both hands null in GT
        'null_both_pred': 0,      # Both hands null in prediction
        'action_pair_consistency': defaultdict(int)  # Track common action pairs
    }
    
    misprediction_patterns = defaultdict(int)
    action_confusion = defaultdict(lambda: defaultdict(int))
    
    print("🔍 Analyzing coordination patterns...")
    
    for pred_file in lh_pred_files[:10]:  # Analyze first 10 videos for detailed analysis
        video_name = pred_file.replace('.txt', '')
        
        # Load predictions
        lh_pred = load_predictions(os.path.join(lh_pred_dir, pred_file))
        rh_pred = load_predictions(os.path.join(rh_pred_dir, pred_file))
        
        # Load ground truth
        lh_gt_file = os.path.join(lh_gt_dir, pred_file)
        rh_gt_file = os.path.join(rh_gt_dir, pred_file)
        
        if not (os.path.exists(lh_gt_file) and os.path.exists(rh_gt_file)):
            continue
            
        lh_gt = load_ground_truth(lh_gt_file)
        rh_gt = load_ground_truth(rh_gt_file)
        
        # Ensure all sequences have same length
        min_len = min(len(lh_pred), len(rh_pred), len(lh_gt), len(rh_gt))
        lh_pred = lh_pred[:min_len]
        rh_pred = rh_pred[:min_len]
        lh_gt = lh_gt[:min_len]
        rh_gt = rh_gt[:min_len]
        
        print(f"\n📹 Video: {video_name} (length: {min_len})")
        
        # Analyze frame by frame
        for i in range(min_len):
            coord_stats['total_frames'] += 1
            
            # Ground truth coordination
            if lh_gt[i] == rh_gt[i]:
                coord_stats['same_action_gt'] += 1
                
            # Prediction coordination  
            if lh_pred[i] == rh_pred[i]:
                coord_stats['same_action_pred'] += 1
                
            # Both have same action correctly
            if lh_gt[i] == rh_gt[i] and lh_pred[i] == rh_pred[i]:
                coord_stats['same_action_correct'] += 1
                
            # Track null coordination
            if lh_gt[i] == 'null' and rh_gt[i] == 'null':
                coord_stats['null_both_gt'] += 1
            if lh_pred[i] == 'null' and rh_pred[i] == 'null':
                coord_stats['null_both_pred'] += 1
                
            # Track action pair patterns
            gt_pair = tuple(sorted([lh_gt[i], rh_gt[i]]))
            coord_stats['action_pair_consistency'][gt_pair] += 1
            
            # Track mispredictions
            if lh_pred[i] != lh_gt[i] or rh_pred[i] != rh_gt[i]:
                mispred_pattern = f"GT:({lh_gt[i]},{rh_gt[i]}) -> PRED:({lh_pred[i]},{rh_pred[i]})"
                misprediction_patterns[mispred_pattern] += 1
                
                # Individual action confusion
                action_confusion[f'LH_{lh_gt[i]}'][lh_pred[i]] += 1
                action_confusion[f'RH_{rh_gt[i]}'][rh_pred[i]] += 1
        
        # Analyze transitions
        for i in range(1, min_len):
            # GT transitions
            lh_gt_trans = lh_gt[i] != lh_gt[i-1]
            rh_gt_trans = rh_gt[i] != rh_gt[i-1]
            if lh_gt_trans and rh_gt_trans:
                coord_stats['transition_sync_gt'] += 1
                
            # Predicted transitions
            lh_pred_trans = lh_pred[i] != lh_pred[i-1]
            rh_pred_trans = rh_pred[i] != rh_pred[i-1]
            if lh_pred_trans and rh_pred_trans:
                coord_stats['transition_sync_pred'] += 1
    
    return coord_stats, misprediction_patterns, action_confusion

def print_analysis_results(coord_stats, misprediction_patterns, action_confusion):
    """Print detailed analysis results"""
    
    print("\n" + "="*80)
    print("📊 COORDINATION ANALYSIS RESULTS")
    print("="*80)
    
    total_frames = coord_stats['total_frames']
    
    print(f"\n🎯 **COORDINATION STATISTICS**")
    print(f"Total frames analyzed: {total_frames:,}")
    print(f"GT same-action frames: {coord_stats['same_action_gt']:,} ({100*coord_stats['same_action_gt']/total_frames:.1f}%)")
    print(f"Pred same-action frames: {coord_stats['same_action_pred']:,} ({100*coord_stats['same_action_pred']/total_frames:.1f}%)")
    print(f"Both same-action correct: {coord_stats['same_action_correct']:,} ({100*coord_stats['same_action_correct']/total_frames:.1f}%)")
    
    print(f"\n🤝 **NULL ACTION COORDINATION**")
    print(f"GT both-null frames: {coord_stats['null_both_gt']:,} ({100*coord_stats['null_both_gt']/total_frames:.1f}%)")
    print(f"Pred both-null frames: {coord_stats['null_both_pred']:,} ({100*coord_stats['null_both_pred']/total_frames:.1f}%)")
    
    print(f"\n🔄 **TRANSITION SYNCHRONIZATION**")
    print(f"GT synchronized transitions: {coord_stats['transition_sync_gt']:,}")
    print(f"Pred synchronized transitions: {coord_stats['transition_sync_pred']:,}")
    
    print(f"\n⚠️  **TOP MISPREDICTION PATTERNS**")
    for pattern, count in sorted(misprediction_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {count:4d}× {pattern}")
    
    print(f"\n📈 **MOST COMMON ACTION PAIRS IN GT**")
    for pair, count in sorted(coord_stats['action_pair_consistency'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {count:4d}× {pair[0]} + {pair[1]}")
    
    print(f"\n❌ **TOP ACTION CONFUSIONS**")
    for gt_action, predictions in action_confusion.items():
        if sum(predictions.values()) > 50:  # Only show frequently confused actions
            print(f"  {gt_action}:")
            for pred_action, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    -> {pred_action}: {count:3d}×")

def generate_improvement_suggestions(coord_stats, misprediction_patterns, action_confusion):
    """Generate specific improvement suggestions based on analysis"""
    
    print("\n" + "="*80)
    print("💡 IMPROVEMENT SUGGESTIONS")
    print("="*80)
    
    total_frames = coord_stats['total_frames']
    same_action_rate_gt = coord_stats['same_action_gt'] / total_frames
    same_action_rate_pred = coord_stats['same_action_pred'] / total_frames
    
    print(f"\n🎯 **COORDINATION INSIGHTS**")
    
    if same_action_rate_gt > 0.6:
        print(f"✅ High natural coordination: {same_action_rate_gt:.1%} of GT frames have same action")
        if same_action_rate_pred < same_action_rate_gt * 0.8:
            print(f"⚠️  Model under-predicts coordination: {same_action_rate_pred:.1%} vs {same_action_rate_gt:.1%}")
            print("   💡 SUGGESTION: Increase coordination loss weights, especially temporal_synchronization_loss")
    else:
        print(f"📋 Moderate natural coordination: {same_action_rate_gt:.1%} of GT frames have same action")
        print("   💡 SUGGESTION: Focus on action-specific coordination patterns rather than universal synchronization")
    
    # Analyze transition synchronization
    if coord_stats['transition_sync_gt'] > 0:
        sync_ratio = coord_stats['transition_sync_pred'] / coord_stats['transition_sync_gt']
        print(f"\n🔄 **TRANSITION ANALYSIS**")
        print(f"Transition sync ratio: {sync_ratio:.2f} (pred/gt)")
        if sync_ratio < 0.7:
            print("   💡 SUGGESTION: Increase boundary_synchronization_loss and action_transition_loss weights")
    
    # Null action analysis
    null_accuracy = coord_stats['null_both_pred'] / max(coord_stats['null_both_gt'], 1)
    print(f"\n🤝 **NULL ACTION COORDINATION**")
    print(f"Null coordination accuracy: {null_accuracy:.2f}")
    if null_accuracy < 0.8:
        print("   💡 SUGGESTION: Add specific loss terms for null action prediction consistency")
    
    print(f"\n🔧 **SPECIFIC MODEL IMPROVEMENTS**")
    
    # Analyze specific action confusions
    action_specific_suggestions = []
    for gt_action, predictions in action_confusion.items():
        total_confusion = sum(predictions.values())
        if total_confusion > 30:  # Significant confusion
            most_confused = max(predictions.items(), key=lambda x: x[1])
            if most_confused[1] / total_confusion > 0.3:  # >30% confusion to one action
                action_specific_suggestions.append(
                    f"   - {gt_action} often confused with {most_confused[0]} ({most_confused[1]}/{total_confusion} cases)"
                )
    
    if action_specific_suggestions:
        print("   Action-specific confusions to address:")
        for suggestion in action_specific_suggestions[:5]:
            print(suggestion)
    
    print(f"\n🚀 **RECOMMENDED NEXT STEPS**")
    print("   1. **Adjust coordination loss weights**:")
    print("      - Increase temporal_synchronization_loss: 0.1 -> 0.15")
    print("      - Increase boundary_synchronization_loss: 0.06 -> 0.10")
    print("      - Increase action_transition_loss: 0.03 -> 0.08")
    
    print("   2. **Add new loss functions**:")
    print("      - Null coordination loss (specific for null actions)")
    print("      - Action-pair consistency loss (for common coordinated pairs)")
    print("      - Confidence-based coordination loss (stronger penalty when both hands are confident)")
    
    print("   3. **Architecture improvements**:")
    print("      - Add explicit cross-hand feature sharing at multiple encoder layers")
    print("      - Use action-conditional coordination (different coordination for different action types)")
    print("      - Add temporal context window for coordination decisions")
    
    print("   4. **Data augmentation**:")
    print("      - Temporal jittering with coordination constraints")
    print("      - Cross-hand action swapping for symmetric actions")

if __name__ == "__main__":
    # Paths
    lh_pred_dir = "result/HAVID-DualHand-Trained-v0_pt/lh_prediction"
    rh_pred_dir = "result/HAVID-DualHand-Trained-v0_pt/rh_prediction"  
    lh_gt_dir = "../data/havid/groundTruth/View0/lh_pt"
    rh_gt_dir = "../data/havid/groundTruth/View0/rh_pt"
    
    # Run analysis
    coord_stats, misprediction_patterns, action_confusion = analyze_coordination_patterns(
        lh_pred_dir, rh_pred_dir, lh_gt_dir, rh_gt_dir
    )
    
    # Print results
    print_analysis_results(coord_stats, misprediction_patterns, action_confusion)
    
    # Generate improvement suggestions
    generate_improvement_suggestions(coord_stats, misprediction_patterns, action_confusion) 