#!/usr/bin/env python3
"""
Comprehensive evaluation with per-video detailed results.
"""

import os
import glob
from collections import defaultdict

# Define fine-grained action groups
FINEGRAINED_ACTION_GROUPS = {
    'insert_gear_shaft': ['iftgl', 'iglft', 'igsft'],
    'insert_placer': ['iplft', 'ipsft'],
    'screw_shaft': ['sftg1', 'sftg1ws', 'sftg2', 'sftg2ws'],
    'screw_nut': ['sntft', 'sntftwn', 'sntn5', 'sntn5wn', 'sntsb', 'sntsbwn'],
    'screw_hex_c1': ['sshc1', 'sshc1dh', 'sshc1dp'],
    'screw_hex_c2': ['sshc2', 'sshc2dh', 'sshc2dp'],
    'screw_hex_c3': ['sshc3', 'sshc3dh', 'sshc3dp'],
    'screw_hex_c4': ['sshc4', 'sshc4dh', 'sshc4dp'],
    'screw_phillips': ['sspg3', 'sspg3dp', 'sspn4', 'sspn4dp'],
    'insert_hex_screw': ['ishc1', 'ishc2', 'ishc3', 'ishc4', 'ishck'],
    'insert_cylinder': ['icbbs', 'icbck', 'icccb', 'iccck', 'ickcb', 'ickcc'],
}

def load_predictions(filepath):
    """Load predictions from file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

def compute_frame_accuracy(pred, gt, action_groups=None):
    """Compute frame-level accuracy."""
    if len(pred) != len(gt):
        min_len = min(len(pred), len(gt))
        pred = pred[:min_len]
        gt = gt[:min_len]
    
    if action_groups is None:
        correct = sum(1 for p, g in zip(pred, gt) if p == g)
        total = len(gt)
    else:
        correct = 0
        total = 0
        for p, g in zip(pred, gt):
            if g in action_groups:
                total += 1
                if p == g:
                    correct += 1
    
    return correct, total, (correct / total * 100) if total > 0 else 0.0

def evaluate_video(video_name, pred_dir_lh, pred_dir_rh, gt_dir_lh, gt_dir_rh):
    """Evaluate a single video."""
    results = {'lh': {}, 'rh': {}}
    
    pred_lh = load_predictions(os.path.join(pred_dir_lh, f"{video_name}.txt"))
    pred_rh = load_predictions(os.path.join(pred_dir_rh, f"{video_name}.txt"))
    gt_lh = load_predictions(os.path.join(gt_dir_lh, f"{video_name}.txt"))
    gt_rh = load_predictions(os.path.join(gt_dir_rh, f"{video_name}.txt"))
    
    if pred_lh is None or gt_lh is None:
        return None, None
    
    # Get all fine-grained actions
    all_finegrained_actions = []
    for actions in FINEGRAINED_ACTION_GROUPS.values():
        all_finegrained_actions.extend(actions)
    
    # Left hand
    if pred_lh and gt_lh:
        correct, total, acc = compute_frame_accuracy(pred_lh, gt_lh)
        results['lh']['overall'] = {'correct': correct, 'total': total, 'accuracy': acc}
        
        correct_fg, total_fg, acc_fg = compute_frame_accuracy(pred_lh, gt_lh, all_finegrained_actions)
        results['lh']['finegrained'] = {'correct': correct_fg, 'total': total_fg, 'accuracy': acc_fg}
    
    # Right hand
    if pred_rh and gt_rh:
        correct, total, acc = compute_frame_accuracy(pred_rh, gt_rh)
        results['rh']['overall'] = {'correct': correct, 'total': total, 'accuracy': acc}
        
        correct_fg, total_fg, acc_fg = compute_frame_accuracy(pred_rh, gt_rh, all_finegrained_actions)
        results['rh']['finegrained'] = {'correct': correct_fg, 'total': total_fg, 'accuracy': acc_fg}
    
    return results['lh'], results['rh']

def main():
    METHOD1_LH = "/Users/hz4426/projects/Polyphony/result/dual_hand_v1_pt_shared_encoder_v4_alternating_videomae_semantic_hand_feature_fusion/prediction_lh"
    METHOD1_RH = "/Users/hz4426/projects/Polyphony/result/dual_hand_v1_pt_shared_encoder_v4_alternating_videomae_semantic_hand_feature_fusion/prediction_rh"
    METHOD2_LH = "/Users/hz4426/projects/Polyphony/result/dual_hand_v1_pt_shared_encoder_v4_alternating_videomae_hand_feature_fusion_extend/prediction_lh"
    METHOD2_RH = "/Users/hz4426/projects/Polyphony/result/dual_hand_v1_pt_shared_encoder_v4_alternating_videomae_hand_feature_fusion_extend/prediction_rh"
    GT_LH = "/Users/hz4426/projects/data/havid/groundTruth/View1/lh_pt"
    GT_RH = "/Users/hz4426/projects/data/havid/groundTruth/View1/rh_pt"
    
    video_files = glob.glob(os.path.join(METHOD1_LH, "*.txt"))
    video_names = sorted([os.path.basename(f).replace('.txt', '') for f in video_files if not os.path.basename(f).startswith('.')])
    
    # Store per-video results
    video_results = []
    
    for video_name in video_names:
        lh1, rh1 = evaluate_video(video_name, METHOD1_LH, METHOD1_RH, GT_LH, GT_RH)
        lh2, rh2 = evaluate_video(video_name, METHOD2_LH, METHOD2_RH, GT_LH, GT_RH)
        
        if lh1 and lh2:
            video_results.append({
                'video': video_name,
                'm1_lh': lh1,
                'm1_rh': rh1,
                'm2_lh': lh2,
                'm2_rh': rh2
            })
    
    # Write detailed results to file
    output_file = "/Users/hz4426/projects/Polyphony/finegrained_evaluation_detailed_results.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("DETAILED FINE-GRAINED ACTION SEGMENTATION EVALUATION\n")
        f.write("="*120 + "\n")
        f.write("\nMethod 1 (Ours): semantic_hand_feature_fusion\n")
        f.write("Method 2 (Other): hand_feature_fusion_extend\n")
        f.write(f"\nTotal videos evaluated: {len(video_results)}\n")
        f.write("="*120 + "\n\n")
        
        # Per-video results table
        f.write("PER-VIDEO FINE-GRAINED ACCURACY COMPARISON\n")
        f.write("="*120 + "\n")
        f.write(f"{'Video':<20} {'M1 LH':<10} {'M2 LH':<10} {'Diff LH':<10} {'M1 RH':<10} {'M2 RH':<10} {'Diff RH':<10} {'M1 Avg':<10} {'M2 Avg':<10} {'Diff Avg':<10}\n")
        f.write("-"*120 + "\n")
        
        for vr in video_results:
            m1_lh_acc = vr['m1_lh']['finegrained']['accuracy']
            m2_lh_acc = vr['m2_lh']['finegrained']['accuracy']
            m1_rh_acc = vr['m1_rh']['finegrained']['accuracy']
            m2_rh_acc = vr['m2_rh']['finegrained']['accuracy']
            
            diff_lh = m1_lh_acc - m2_lh_acc
            diff_rh = m1_rh_acc - m2_rh_acc
            
            m1_avg = (m1_lh_acc + m1_rh_acc) / 2
            m2_avg = (m2_lh_acc + m2_rh_acc) / 2
            diff_avg = m1_avg - m2_avg
            
            f.write(f"{vr['video']:<20} {m1_lh_acc:>9.2f} {m2_lh_acc:>9.2f} {diff_lh:>+9.2f} ")
            f.write(f"{m1_rh_acc:>9.2f} {m2_rh_acc:>9.2f} {diff_rh:>+9.2f} ")
            f.write(f"{m1_avg:>9.2f} {m2_avg:>9.2f} {diff_avg:>+9.2f}\n")
        
        # Aggregate statistics
        f.write("\n" + "="*120 + "\n")
        f.write("AGGREGATE STATISTICS\n")
        f.write("="*120 + "\n\n")
        
        # Overall accuracy
        f.write("OVERALL ACCURACY:\n")
        f.write("-"*80 + "\n")
        
        for hand, hand_name in [('lh', 'Left Hand'), ('rh', 'Right Hand')]:
            m1_correct = sum(vr[f'm1_{hand}']['overall']['correct'] for vr in video_results)
            m1_total = sum(vr[f'm1_{hand}']['overall']['total'] for vr in video_results)
            m1_acc = (m1_correct / m1_total * 100) if m1_total > 0 else 0.0
            
            m2_correct = sum(vr[f'm2_{hand}']['overall']['correct'] for vr in video_results)
            m2_total = sum(vr[f'm2_{hand}']['overall']['total'] for vr in video_results)
            m2_acc = (m2_correct / m2_total * 100) if m2_total > 0 else 0.0
            
            f.write(f"\n{hand_name}:\n")
            f.write(f"  Method 1 (Ours):  {m1_acc:.2f}% ({m1_correct}/{m1_total} frames)\n")
            f.write(f"  Method 2 (Other): {m2_acc:.2f}% ({m2_correct}/{m2_total} frames)\n")
            f.write(f"  Difference:       {m1_acc - m2_acc:+.2f}%\n")
        
        # Fine-grained accuracy
        f.write("\n" + "-"*80 + "\n")
        f.write("FINE-GRAINED ACTION ACCURACY:\n")
        f.write("-"*80 + "\n")
        
        for hand, hand_name in [('lh', 'Left Hand'), ('rh', 'Right Hand')]:
            m1_correct_fg = sum(vr[f'm1_{hand}']['finegrained']['correct'] for vr in video_results)
            m1_total_fg = sum(vr[f'm1_{hand}']['finegrained']['total'] for vr in video_results)
            m1_acc_fg = (m1_correct_fg / m1_total_fg * 100) if m1_total_fg > 0 else 0.0
            
            m2_correct_fg = sum(vr[f'm2_{hand}']['finegrained']['correct'] for vr in video_results)
            m2_total_fg = sum(vr[f'm2_{hand}']['finegrained']['total'] for vr in video_results)
            m2_acc_fg = (m2_correct_fg / m2_total_fg * 100) if m2_total_fg > 0 else 0.0
            
            f.write(f"\n{hand_name}:\n")
            f.write(f"  Method 1 (Ours):  {m1_acc_fg:.2f}% ({m1_correct_fg}/{m1_total_fg} frames)\n")
            f.write(f"  Method 2 (Other): {m2_acc_fg:.2f}% ({m2_correct_fg}/{m2_total_fg} frames)\n")
            f.write(f"  Difference:       {m1_acc_fg - m2_acc_fg:+.2f}%\n")
        
        # Average across both hands
        f.write("\n" + "="*120 + "\n")
        f.write("SUMMARY: AVERAGE ACROSS BOTH HANDS\n")
        f.write("="*120 + "\n\n")
        
        # Overall
        m1_overall_avg = 0.0
        m2_overall_avg = 0.0
        for hand in ['lh', 'rh']:
            m1_correct = sum(vr[f'm1_{hand}']['overall']['correct'] for vr in video_results)
            m1_total = sum(vr[f'm1_{hand}']['overall']['total'] for vr in video_results)
            m1_overall_avg += (m1_correct / m1_total * 100) if m1_total > 0 else 0.0
            
            m2_correct = sum(vr[f'm2_{hand}']['overall']['correct'] for vr in video_results)
            m2_total = sum(vr[f'm2_{hand}']['overall']['total'] for vr in video_results)
            m2_overall_avg += (m2_correct / m2_total * 100) if m2_total > 0 else 0.0
        m1_overall_avg /= 2
        m2_overall_avg /= 2
        
        # Fine-grained
        m1_fg_avg = 0.0
        m2_fg_avg = 0.0
        for hand in ['lh', 'rh']:
            m1_correct_fg = sum(vr[f'm1_{hand}']['finegrained']['correct'] for vr in video_results)
            m1_total_fg = sum(vr[f'm1_{hand}']['finegrained']['total'] for vr in video_results)
            m1_fg_avg += (m1_correct_fg / m1_total_fg * 100) if m1_total_fg > 0 else 0.0
            
            m2_correct_fg = sum(vr[f'm2_{hand}']['finegrained']['correct'] for vr in video_results)
            m2_total_fg = sum(vr[f'm2_{hand}']['finegrained']['total'] for vr in video_results)
            m2_fg_avg += (m2_correct_fg / m2_total_fg * 100) if m2_total_fg > 0 else 0.0
        m1_fg_avg /= 2
        m2_fg_avg /= 2
        
        f.write("Overall Accuracy:\n")
        f.write(f"  Method 1 (Ours):  {m1_overall_avg:.2f}%\n")
        f.write(f"  Method 2 (Other): {m2_overall_avg:.2f}%\n")
        f.write(f"  Difference:       {m1_overall_avg - m2_overall_avg:+.2f}%\n")
        
        f.write("\nFine-Grained Action Accuracy:\n")
        f.write(f"  Method 1 (Ours):  {m1_fg_avg:.2f}%\n")
        f.write(f"  Method 2 (Other): {m2_fg_avg:.2f}%\n")
        f.write(f"  Difference:       {m1_fg_avg - m2_fg_avg:+.2f}%\n")
        
        f.write("\n" + "="*120 + "\n")
    
    print(f"Detailed results saved to: {output_file}")
    
    # Also print summary to console
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nFine-Grained Action Accuracy:")
    print(f"  Method 1 (Ours):  {m1_fg_avg:.2f}%")
    print(f"  Method 2 (Other): {m2_fg_avg:.2f}%")
    print(f"  Improvement:      {m1_fg_avg - m2_fg_avg:+.2f}%")

if __name__ == '__main__':
    main()

