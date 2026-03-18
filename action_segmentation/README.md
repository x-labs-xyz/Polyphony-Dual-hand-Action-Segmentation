# Diffusion-based Dual-Hand Action Segmentation

Official implementation of diffusion-based dual-hand action segmentation. This framework supports simultaneous segmentation of left and right hand actions with cross-hand feature fusion and adaptive loss weighting.

## Features

✨ **Dual-Hand Architecture**
- Shared encoder for both hands
- Hand-specific decoders with feature fusion
- Separate class weights for each hand

🔄 **Diffusion-Based Decoder**
- DDIM sampling for fast inference
- Multiple conditioning strategies
- Cosine noise schedule

⚖️ **Adaptive Loss Weighting**
- Automatic balancing between hands
- Performance-based weight adjustment
- Configurable boost factors

🎯 **Single-Stream Mode**
- Train on single-stream data (e.g., Breakfast)
- Automatic label perturbation for synthetic dual-stream
- Configurable boundary shifting parameters

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone or navigate to the repository
cd action_segmentation

# Install dependencies
pip install -r requirements.txt

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Organize your data as follows:

```
your_dataset/
├── features/
│   ├── left_hand/
│   │   ├── video1.npy  # Shape: [T, F] or [batch, T, F]
│   │   ├── video2.npy
│   │   └── ...
│   └── right_hand/
│       ├── video1.npy
│       ├── video2.npy
│       └── ...
├── groundTruth/
│   ├── left_hand/
│   │   ├── video1.txt  # One action label per line
│   │   ├── video2.txt
│   │   └── ...
│   └── right_hand/
│       ├── video1.txt
│       ├── video2.txt
│       └── ...
├── splits/
│   ├── train.split1.bundle  # List of training video names
│   └── test.split1.bundle   # List of test video names
└── mapping.txt  # Format: "0 action_name_0\n1 action_name_1\n..."
```

**Note on Features**: Each `.npy` file should contain visual features extracted from videos. Features should have shape `[T, F]` where `T` is the temporal dimension and `F` is the feature dimension.

### 2. Create Configuration

Create a configuration file `configs/my_config.json`:

```json
{
    "root_data_dir": "/path/to/your/data",
    "dataset_name": "your_dataset",
    "feature_subdir_lh": "features/left_hand",
    "feature_subdir_rh": "features/right_hand",
    "label_subdir_lh": "groundTruth/left_hand",
    "label_subdir_rh": "groundTruth/right_hand",
    "split_dir": "splits",
    "split_id": 1,
    
    "encoder_params": {
        "input_dim": 768, 
        "num_f_maps": 64,
        "num_layers": 10,
        "feature_layer_indices": [-1, 7, 9],
        "use_instance_norm": true
    },
    
    "decoder_params": {
        "num_f_maps": 64,
        "num_layers": 10
    },
    
    "diffusion_params": {
        "timesteps": 1000,
        "sampling_timesteps": 25,
        "ddim_sampling_eta": 1.0,
        "snr_scale": 2.0,
        "detach_decoder": false,
        "cond_types": ["full", "boundary05-", "segment=1"]
    },
    
    "sample_rate": 4,
    "temporal_aug": true,
    "boundary_smooth": null,
    "num_epochs": 50,
    "batch_size": 1,
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    
    "loss_weights": {
        "encoder_ce_loss": 0.5,
        "encoder_mse_loss": 0.025,
        "encoder_boundary_loss": 0.0,
        "decoder_ce_loss": 0.5,
        "decoder_mse_loss": 0.025,
        "decoder_boundary_loss": 0.1
    },
    
    "class_weighting": true,
    "soft_label": null,
    "set_sampling_seed": true,
    
    "postprocess": {
        "type": "median",
        "value": 7
    },
    
    "log_freq": 5,
    "log_train_results": true,
    "result_dir": "results",
    "naming": "my_experiment"
}
```

### 3. Train the Model

**For dual-hand datasets:**

```bash
# Using the training script
./train.sh configs/my_config.json 0  # 0 is the GPU ID

# Or directly with Python
python main.py --config configs/my_config.json --device 0
```

**For single-stream datasets (e.g., Breakfast):**

```bash
# Enable single-stream mode with label perturbation
python main.py --config configs/my_config.json --device 0 --one_stream

# With custom perturbation parameters
python main.py \
    --config configs/my_config.json \
    --device 0 \
    --one_stream \
```

In single-stream mode:
- Left-hand data is copied to right-hand
- Right-hand labels are perturbed by randomly shifting action boundaries
- This creates synthetic dual-stream data for training

### 5. Evaluate

Evaluation is performed automatically during training at intervals specified by `log_freq`. Results are saved in:

```
results/my_experiment/
├── epoch-{N}.model              # Model checkpoints
├── latest.pt                    # Latest checkpoint (includes optimizer state)
├── prediction_lh/               # Left hand predictions
│   └── video*.txt
├── prediction_rh/               # Right hand predictions
│   └── video*.txt
├── test_results_*_lh_epoch{N}.npy
├── test_results_*_rh_epoch{N}.npy
└── events.out.tfevents.*        # TensorBoard logs
```

## Architecture

### Model Overview

```
Video Features (LH & RH)
         ↓
    Shared Encoder
         ↓
   Feature Fusion ←→ (Cross-hand context)
         ↓
    ┌────────┴────────┐
    ↓                 ↓
Decoder LH        Decoder RH
(Diffusion)      (Diffusion)
    ↓                 ↓
Predictions LH   Predictions RH
```

### Components

1. **Shared Encoder**: Temporal convolutional network that processes features from both hands
2. **Feature Fusion**: Module that allows each hand to incorporate context from the other
3. **Hand-Specific Decoders**: Separate diffusion-based decoders for each hand
4. **Adaptive Loss Manager**: Dynamically adjusts loss weights based on per-hand performance

### Diffusion Process

- **Training**: Adds noise to ground truth labels at random timesteps
- **Inference**: DDIM sampling to denoise predictions in 25 steps (configurable)
- **Conditioning**: Multiple strategies (full, boundary-based, segment-based)


## Contact

For questions or issues, please open an issue on GitHub.


