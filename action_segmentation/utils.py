"""
Utility functions for dual-hand action segmentation
Includes evaluation metrics, post-processing, and helper functions
"""

import os
import json
import random
import torch
import numpy as np
from scipy import stats
from scipy.ndimage import generic_filter
from typing import List, Tuple, Dict, Optional


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config_file(config_file: str) -> Dict:
    """Load configuration from JSON file with default values"""
    all_params = json.load(open(config_file))
    
    # Set defaults
    if 'result_dir' not in all_params:
        all_params['result_dir'] = 'result'
    
    if 'log_train_results' not in all_params:
        all_params['log_train_results'] = True
    
    if 'soft_label' not in all_params:
        all_params['soft_label'] = None

    if 'postprocess' not in all_params:
        all_params['postprocess'] = {'type': None, 'value': None}

    if 'use_instance_norm' not in all_params['encoder_params']:
        all_params['encoder_params']['use_instance_norm'] = False

    if 'detach_decoder' not in all_params['diffusion_params']:
        all_params['diffusion_params']['detach_decoder'] = False

    return all_params


# ============================================================================
# Random Seed
# ============================================================================

def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================================
# Post-processing
# ============================================================================

def mode_filter(x: np.ndarray, size: int) -> np.ndarray:
    """Apply mode filter for smoothing predictions"""
    def modal(P):
        mode = stats.mode(P)
        return mode.mode[0]
    result = generic_filter(x, modal, size)
    return result


# ============================================================================
# Label Utilities
# ============================================================================

def get_labels_start_end_time(frame_wise_labels: List, bg_class: List[str] = ["background"]) -> Tuple[List, List, List]:
    """
    Extract segment-level information from frame-wise labels
    
    Args:
        frame_wise_labels: List of frame-wise action labels
        bg_class: List of background class names to ignore
        
    Returns:
        labels: List of action labels for each segment
        starts: List of start frames for each segment
        ends: List of end frames for each segment
    """
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
        
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
            
    if last_label not in bg_class:
        ends.append(i)
        
    return labels, starts, ends


# ============================================================================
# Evaluation Metrics
# ============================================================================

def levenstein(p: List, y: List, norm: bool = False) -> float:
    """
    Compute Levenshtein distance (edit distance) between two sequences
    
    Args:
        p: Predicted sequence
        y: Ground truth sequence
        norm: Whether to normalize by sequence length
        
    Returns:
        Edit distance score
    """
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
 
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
     
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
 
    return score


def edit_score(recognized: List, ground_truth: List, norm: bool = True, 
                bg_class: List[str] = ["background"]) -> float:
    """
    Compute edit score (normalized Levenshtein distance) at segment level
    
    Args:
        recognized: Predicted frame-wise labels
        ground_truth: Ground truth frame-wise labels
        norm: Whether to normalize
        bg_class: Background classes to ignore
        
    Returns:
        Edit score (0-100, higher is better)
    """
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized: List, ground_truth: List, overlap: float, 
            bg_class: List[str] = ["background"]) -> Tuple[float, float, float]:
    """
    Compute F1 score at given overlap threshold
    
    Args:
        recognized: Predicted frame-wise labels
        ground_truth: Ground truth frame-wise labels
        overlap: IoU threshold for positive detection
        bg_class: Background classes to ignore
        
    Returns:
        tp: True positives
        fp: False positives
        fn: False negatives
    """
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
 
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
 
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
            
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def func_eval(label_dir: str, pred_dir: str, video_list: List[str]) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate predictions against ground truth
    
    Args:
        label_dir: Directory containing ground truth labels
        pred_dir: Directory containing predictions
        video_list: List of video names to evaluate
        
    Returns:
        acc: Frame-wise accuracy (%)
        edit: Edit score (0-100, higher is better)
        f1s: F1 scores at [10%, 25%, 50%] overlap thresholds
    """
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    correct = 0
    total = 0
    edit = 0

    for vid in video_list:
 
        gt_file = os.path.join(label_dir, f'{vid}.txt')
        with open(gt_file, 'r') as f:
            gt_content = f.read().split('\n')[0:-1]
 
        pred_file = os.path.join(pred_dir, f'{vid}.txt')
        with open(pred_file, 'r') as f:
            pred_content = f.read().split('\n')[1].split()
 
        assert(len(gt_content) == len(pred_content)), \
            f"Length mismatch for {vid}: GT={len(gt_content)}, Pred={len(pred_content)}"

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

        edit += edit_score(pred_content, gt_content)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
     
    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(video_list)
    f1s = np.array([0, 0 ,0], dtype=float)
    
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
        f1 = 2.0 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1) * 100
        f1s[s] = f1
 
    return acc, edit, f1s


# ============================================================================
# Sequence Restoration
# ============================================================================

def restore_full_sequence(sampled_output: np.ndarray, full_len: int, 
                         left_offset: int, right_offset: int, 
                         sample_rate: int) -> np.ndarray:
    """
    Restore full sequence from temporally sampled predictions
    
    Args:
        sampled_output: Predictions on sampled frames
        full_len: Length of original full sequence
        left_offset: Left padding offset
        right_offset: Right padding offset
        sample_rate: Temporal sampling rate
        
    Returns:
        Full sequence predictions
    """
    sampled_len = len(sampled_output)
    full_output = np.zeros(full_len, dtype=np.int64)
    
    # Map sampled predictions to full sequence
    for i in range(sampled_len):
        full_idx = left_offset + i * sample_rate
        if full_idx < full_len:
            full_output[full_idx] = sampled_output[i]
    
    # Fill in gaps with nearest neighbor
    for i in range(full_len):
        if i < left_offset:
            full_output[i] = full_output[left_offset]
        elif i >= full_len - right_offset:
            full_output[i] = full_output[full_len - right_offset - 1]
        elif (i - left_offset) % sample_rate != 0:
            # Interpolate between sampled frames
            prev_idx = left_offset + ((i - left_offset) // sample_rate) * sample_rate
            next_idx = prev_idx + sample_rate
            if next_idx < full_len:
                # Use nearest neighbor
                if (i - prev_idx) < (next_idx - i):
                    full_output[i] = full_output[prev_idx]
                else:
                    full_output[i] = full_output[next_idx]
            else:
                full_output[i] = full_output[prev_idx]
    
    return full_output


# ============================================================================
# Boundary Generation
# ============================================================================

def get_boundary_seq(event_seq: np.ndarray, smooth: Optional[float] = None) -> np.ndarray:
    """
    Generate boundary sequence from event sequence
    
    Args:
        event_seq: Frame-wise event labels
        smooth: Gaussian smoothing sigma (if None, no smoothing)
        
    Returns:
        Binary boundary sequence (1 at boundaries, 0 otherwise)
    """
    from scipy.ndimage import gaussian_filter1d
    
    boundary_seq = np.zeros_like(event_seq, dtype=np.float32)
    
    # Mark boundaries (where label changes)
    for i in range(1, len(event_seq)):
        if event_seq[i] != event_seq[i-1]:
            boundary_seq[i] = 1.0
    
    # Apply Gaussian smoothing if specified
    if smooth is not None and smooth > 0:
        boundary_seq = gaussian_filter1d(boundary_seq, sigma=smooth)
        # Normalize to [0, 1]
        if boundary_seq.max() > 0:
            boundary_seq = boundary_seq / boundary_seq.max()
    
    return boundary_seq

