"""
Configuration template for dual-hand action segmentation
Modify these values according to your dataset and requirements
"""

import os

# ============================================================================
# Dataset Configuration
# ============================================================================

# Root directory containing your dataset
ROOT_DATA_DIR = "/path/to/your/data"

# Dataset name (subdirectory under ROOT_DATA_DIR)
DATASET_NAME = "your_dataset"

# Feature directories (relative to ROOT_DATA_DIR/DATASET_NAME/)
FEATURE_SUBDIR_LH = "features/left_hand"   # Left hand features (.npy files)
FEATURE_SUBDIR_RH = "features/right_hand"   # Right hand features (.npy files)

# Label directories (relative to ROOT_DATA_DIR/DATASET_NAME/)
LABEL_SUBDIR_LH = "groundTruth/left_hand"  # Left hand labels (.txt files)
LABEL_SUBDIR_RH = "groundTruth/right_hand"  # Right hand labels (.txt files)

# Split directory (relative to ROOT_DATA_DIR/DATASET_NAME/)
SPLIT_DIR = "splits"  # Contains train.split{N}.bundle and test.split{N}.bundle

# Split ID (e.g., 1, 2, 3 for different data splits)
SPLIT_ID = 1

# ============================================================================
# Model Architecture
# ============================================================================

ENCODER_PARAMS = {
    'input_dim': 768,              # Input feature dimension
    'num_f_maps': 64,              # Number of feature maps in encoder
    'num_layers': 10,              # Number of encoder layers
    'feature_layer_indices': [-1, 7, 9],  # Which layers to use for decoder
    'use_instance_norm': True      # Whether to use instance normalization
}

DECODER_PARAMS = {
    'num_f_maps': 64,              # Number of feature maps in decoder
    'num_layers': 10               # Number of decoder layers
}

DIFFUSION_PARAMS = {
    'timesteps': 1000,             # Total diffusion timesteps
    'sampling_timesteps': 5,      # DDIM sampling steps (< timesteps)
    'ddim_sampling_eta': 1.0,      # DDIM eta (0=deterministic, 1=stochastic)
    'snr_scale': 2.0,              # Signal-to-noise ratio scale
    'detach_decoder': False,       # Whether to detach encoder features
    'cond_types': ['full', 'boundary05-', 'segment=1']  # Conditioning strategies
}

# ============================================================================
# Training Configuration
# ============================================================================

# Temporal settings
SAMPLE_RATE = 4                    # Temporal sampling rate
TEMPORAL_AUG = True                # Enable temporal augmentation
BOUNDARY_SMOOTH = None             # Gaussian smoothing for boundaries (None = no smoothing)

# Training hyperparameters
NUM_EPOCHS = 50                    # Number of training epochs
BATCH_SIZE = 1                     # Batch size (simulated via gradient accumulation)
LEARNING_RATE = 0.0005             # Learning rate
WEIGHT_DECAY = 0.0001              # Weight decay (L2 regularization)

# Loss weights
LOSS_WEIGHTS = {
    'encoder_ce_loss': 0.5,        # Encoder cross-entropy weight
    'encoder_mse_loss': 0.025,     # Encoder temporal smoothness weight
    'encoder_boundary_loss': 0.0,  # Encoder boundary weight (usually 0)
    'decoder_ce_loss': 0.5,        # Decoder cross-entropy weight
    'decoder_mse_loss': 0.025,     # Decoder temporal smoothness weight
    'decoder_boundary_loss': 0.1   # Decoder boundary weight
}

# Training options
CLASS_WEIGHTING = True             # Use class weights to handle imbalance
SOFT_LABEL = None                  # Soft label smoothing sigma (None = no smoothing)
SET_SAMPLING_SEED = True           # Set deterministic seed during inference

# ============================================================================
# Evaluation Configuration
# ============================================================================

# Post-processing
POSTPROCESS = {
    'type': 'median',              # 'median', 'mode', 'purge', or None
    'value': 7                     # Filter size (for median/mode) or min duration (for purge)
}

# Logging
LOG_FREQ = 5                       # Evaluation frequency (epochs)
LOG_TRAIN_RESULTS = True           # Whether to log train set results

# ============================================================================
# Output Configuration
# ============================================================================

# Result directory
RESULT_DIR = "results"

# Experiment name
NAMING = f"{DATASET_NAME}_split{SPLIT_ID}"

# ============================================================================
# Generate Configuration Dictionary
# ============================================================================

def get_config():
    """
    Generate complete configuration dictionary
    
    Returns:
        Dictionary containing all configuration parameters
    """
    config = {
        # Dataset
        'root_data_dir': ROOT_DATA_DIR,
        'dataset_name': DATASET_NAME,
        'feature_subdir_lh': FEATURE_SUBDIR_LH,
        'feature_subdir_rh': FEATURE_SUBDIR_RH,
        'label_subdir_lh': LABEL_SUBDIR_LH,
        'label_subdir_rh': LABEL_SUBDIR_RH,
        'split_dir': SPLIT_DIR,
        'split_id': SPLIT_ID,
        
        # Model
        'encoder_params': ENCODER_PARAMS,
        'decoder_params': DECODER_PARAMS,
        'diffusion_params': DIFFUSION_PARAMS,
        
        # Training
        'sample_rate': SAMPLE_RATE,
        'temporal_aug': TEMPORAL_AUG,
        'boundary_smooth': BOUNDARY_SMOOTH,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'loss_weights': LOSS_WEIGHTS,
        'class_weighting': CLASS_WEIGHTING,
        'soft_label': SOFT_LABEL,
        'set_sampling_seed': SET_SAMPLING_SEED,
        
        # Evaluation
        'postprocess': POSTPROCESS,
        'log_freq': LOG_FREQ,
        'log_train_results': LOG_TRAIN_RESULTS,
        
        # Output
        'result_dir': RESULT_DIR,
        'naming': NAMING
    }
    
    return config


def save_config_json(config, filepath):
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        filepath: Path to save JSON file
    """
    import json
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f'Configuration saved to: {filepath}')


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    import json
    
    # Get configuration
    config = get_config()
    
    # Print configuration
    print("Current Configuration:")
    print("=" * 70)
    print(json.dumps(config, indent=2))
    print("=" * 70)
    
    # Save to JSON
    save_config_json(config, 'configs/example_config.json')

