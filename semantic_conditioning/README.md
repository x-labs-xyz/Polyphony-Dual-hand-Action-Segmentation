# Semantic Feature Alignment for Video Action Recognition

This module aligns visual features extracted from videos with semantic embeddings from language models using Temporal Convolutional Networks (TCN). The enhanced features can be used for downstream action segmentation tasks.

## Overview

The semantic conditioning approach:
1. Takes visual features from a video encoder (e.g., VideoMAE)
2. Processes them through a Temporal Convolutional Network (TCN)
3. Aligns them with semantic embeddings from text descriptions
4. Outputs enhanced features for action segmentation

## Key Features

✅ **Dynamic Batching** - Pads sequences to batch max length (not fixed), reducing memory usage  
✅ **Temporal Modeling** - TCN with dilated convolutions captures temporal context  
✅ **Semantic Alignment** - Aligns visual and semantic features in shared embedding space  
✅ **Flexible Loss Functions** - Supports adaptive, MSE, cosine, and smooth L1 losses  
✅ **Precomputed Embeddings** - Uses cached semantic embeddings for efficiency  
✅ **Comprehensive Evaluation** - Includes visualizations and metrics  

## Directory Structure

```
semantic_conditioning/
├── main.py              # Main training script
├── config.py            # Configuration file
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── checkpoints/         # Saved models (created during training)
└── enhanced_features/   # Enhanced visual features (created during training)
```

## Usage

### Quick Start

Simply run the main script with default configuration:

```bash
python main.py
```

### Custom Configuration

Edit `config.py` to customize:
- Data paths
- Model architecture
- Training hyperparameters
- Loss function type
- Semantic embedding model

### Training

The training process includes:
1. **Data Loading** - Loads visual features and frame annotations
2. **Model Initialization** - Creates TCN-based alignment model
3. **Training Loop** - Trains with early stopping and checkpointing
4. **Evaluation** - Computes alignment metrics
5. **Feature Saving** - Saves enhanced features for downstream tasks

Training output:
```
Epoch 1/150
Batch 0/100, Loss: 0.234567
...
Train Loss: 0.234567, Val Loss: 0.345678
Current LR: 3.00e-04
New best model saved with val_loss: 0.345678
```

### Output Files

After training, you'll have:

1. **Checkpoints** (`./checkpoints/`)
   - `best_model.pth` - Best model based on validation loss
   - `checkpoint_epoch_20.pth`, `checkpoint_epoch_40.pth`, etc.

2. **Enhanced Features** (`./enhanced_features/`)
   - `<video_id>.npy` - Enhanced features for each video (transposed format)

## Model Architecture

```
Visual Features [B, T, 768]
        ↓
Temporal Convolutional Network (TCN)
  - Layer 1: 768 → 512 (dilation=1)
  - Layer 2: 512 → 128 (dilation=2)
  - Layer 3: 128 → 64 (dilation=4)
        ↓
Visual Projector
  - Linear: 64 → 512
  - LayerNorm + ReLU + Dropout
  - Linear: 512 → 384
  - LayerNorm + Dropout
        ↓
Aligned Features [B, T, 384]
```

## Loss Functions

### Adaptive Loss (Default)
Combines cosine similarity and MSE:
```python
loss = α * (1 - cosine_sim) + (1 - α) * mse
```
- **α = 0.7**: Emphasizes directional alignment
- **(1 - α) = 0.3**: Maintains feature magnitude

### Other Options
- **MSE**: Mean squared error
- **Cosine**: Cosine similarity loss
- **Smooth L1**: Huber loss variant

## Evaluation Metrics

The evaluator computes:
- **MSE Loss** - Mean squared error between aligned and target features
- **Mean Cosine Similarity** - Average cosine similarity
- **Median/Std Cosine Similarity** - Distribution statistics
- **Per-Action-Type Performance** - Separate metrics for:
  - Regular actions
  - Null/transition states
  - Wrong actions

## Data Format

### Input Requirements

1. **Visual Features** (`.npy` files)
   - Shape: `(feature_dim, seq_len)` - will be transposed to `(seq_len, feature_dim)`
   - Typical dimension: 768 (ViT-base backbone)

2. **Frame Annotations** (`.txt` files)
   - One action label per line
   - Example:
     ```
     action_1
     action_1
     action_2
     null
     action_3
     ```

3. **Action Mapping** (`havid_description.txt`)
   - Format: `<label> "<description>"`
   - Example:
     ```
      ibacb "insert the ball into the cylinder base"
      ibscb "insert the ball seat into the cylinder base"
      ......
     ```

4. **Split Files** (e.g., `train.split1.bundle`)
   - One video ID per line
   - Example:
     ```
     video_001
     video_002
     video_003
     ```

5. **Semantic Embeddings** (`.pt` file)
   - Precomputed embeddings for all action labels (see [Precompute Semantic Embeddings](../data_preparation/README.md#2-precompute-semantic-embeddings))
   - Format: `{'embeddings': {label: tensor, ...}}`

### Output Format

Enhanced features are saved as `.npy` files:
- Shape: `(feature_dim, seq_len)` - transposed for downstream compatibility
- Typical dimension: 384 (MiniLM semantic dim)


## Concatenate Features
Concatenate the ADH-ViT features with the semantic conditioned features to form the MAS features.

The `concatenate_features.py` script performs a two-step concatenation:

1. **Step 1**: Shared visual features + Hand-specific features → Intermediate features
   - LH: `shared [768, T] + lh [num_class, T] = intermediate [768+num_class, T]`
   - RH: `shared [768, T] + rh [num_class, T] = intermediate [768+num_class, T]`

2. **Step 2**: Intermediate features + Semantic features → Final features
   - LH: `intermediate [768+num_class, T] + semantic [D_sem, T] = final [768+num_class+D_sem, T]`
   - RH: `intermediate [768+num_class, T] + semantic [D_sem, T] = final [768+num_class+D_sem, T]`

#### Usage

**Basic usage** (with semantic features):

```bash
python concatenate_features_with_semantic.py \
    --base_dir /path/to/features \
    --semantic_lh_dir /path/to/semantic/lh \
    --semantic_rh_dir /path/to/semantic/rh \
    --output_dir /path/to/output
```

#### Input Directory Structure

```
base_dir/
├── shared_features/      # Shared visual features [768, T]
│   ├── video1.npy
│   └── ...
├── lh_features/          # Left-hand specific features [num_class, T]
│   ├── video1.npy
│   └── ...
└── rh_features/          # Right-hand specific features [num_class, T]
    ├── video1.npy
    └── ...
```

#### Output Directory Structure

```
output_dir/
├── lh_v0/               # Final LH concatenated features
│   ├── video1.npy       # [768+num_class+D_sem, T]
│   └── ...
├── rh_v0/               # Final RH concatenated features
│   ├── video1.npy       # [768+num_class+D_sem, T]
│   └── ...
└── metadata/            # Processing metadata
    └── concatenation_summary.json
```

## Contact

For questions or issues, please open an issue in the repository.

