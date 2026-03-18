# Data Preparation Tools

Utilities for preparing datasets for dual-hand action segmentation.

## Overview

This folder contains scripts for preprocessing datasets, extracting features, and generating semantic embeddings for action labels.

## Dataset Preparation Guide

This section explains how to download and prepare action segmentation datasets from their original sources.

### Download datasets

We benchmark on three datasets:

**HA-VID (a human assembly video dataset)** a human assembly video dataset with dual-hand action annotations. Download HA-ViD from the official website: [HAVID Dataset](https://iai-hrc.github.io/ha-vid).

**ATTACH** annotated two-handed assembly actions for human action understanding. Download ATTACH from the official website: [ATTACH Dataset](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/attach-dataset).

**Breakfast** a single-stream action segmentation dataset for cooking activities. Download Breakfast dataset from: [ms-tcn](https://github.com/yabufarha/ms-tcn).

### Directory Structure

After downloading, organize your data as follows:

```
data/dataset/
├── videos                  # Original video files (.mp4)
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── features                    # Original I3D features (.npy)
|   ├── left_hand
|   │   ├── video1.mp4
|   │   ├── video2.mp4
|   │   └── ...
|   ├── right_hand
|   │   ├── video1.mp4
|   │   ├── video2.mp4
|   │   └── ...
├── groundTruth               # Frame-wise annotations (.txt)
|   ├── left_hand
|   │   ├── video1.txt
|   │   ├── video1.txt
|   │   └── ...
|   ├── right_hand
|   │   ├── video1.txt
|   │   ├── video1.txt
|   │   └── ...
├── mapping.txt                # Label to ID mapping
│   # Format: "0 label1" or "label1 0"
├── splits/                    # Train/test splits
|   ├── left_hand
|   │   ├── train.split1.bundle
|   │   └── test.split1.bundle
|   ├── right_hand
|   │   ├── train.split1.bundle
|   │   └── test.split1.bundle
└── description.txt      # Action descriptions for semantic embeddings (stored in folder ./action_descriptions)
```

For description files, they can be found in the folder `action_descriptions`.

### Next Steps

After preparing your dataset:
1. **Generate data for training ADH-ViT** (see [Script 1](#1-generate-action-recognition-dataset))
2. **Precompute semantic embeddings** (see [Script 2](#2-precompute-semantic-embeddings)) 

---

## Scripts

### 1. Generate data for training ADH-ViT

Extract video clips from action segmentation datasets and prepare them for training ADH-ViT.

**Script**: `prepare_dataset_for_ADH-ViT.py`

#### Features

- ✅ **Two Extraction Methods**:
  - **Segment-based**: Extracts entire action segments (consecutive frames with same label)
  - **Clip-based**: Samples random fixed-length clips with class balancing
- ✅ **Flexible Configuration**: Supports various video formats and annotation styles
- ✅ **Organized Output**: Clips organized by label folders with index files

#### Usage

**Segment-based extraction** (extract entire action segments):

```bash
python prepare_dataset_for_ADH-ViT.py \
    --video_dir /path/to/videos \
    --annotation_dir /path/to/annotations \
    --output_dir /path/to/output \
    --mapping_file /path/to/mapping.txt \
    --extraction_method segment \
    --split_list /path/to/train.split1.bundle
```

**Clip-based extraction** (sample fixed-length clips):

```bash
python prepare_dataset_for_ADH-ViT.py \
    --video_dir /path/to/videos \
    --annotation_dir /path/to/annotations \
    --output_dir /path/to/output \
    --mapping_file /path/to/mapping.txt \
    --extraction_method random \
    --clips_per_video 10 \
    --clip_length 16 \
    --prefer_non_null \
    --split_list /path/to/train.split1.bundle
```

#### Extraction Methods

**1. Segment-Based (`--extraction_method segment`)**

- Extracts entire action segments (consecutive frames with same label)
- Each segment becomes a separate video clip
- Preserves complete action boundaries
- Best for: Datasets with well-defined action segments

**Parameters:**
- `--min_segment_length`: Minimum segment length to extract (default: 1 frame)

**2. Random Clip-Based (`--extraction_method random`)**

- Samples random fixed-length clips from videos
- Balances class distribution automatically
- Prefers non-null labels (optional)
- Best for: Creating balanced training sets

**Parameters:**
- `--clips_per_video`: Number of clips per video (default: 10)
- `--clip_length`: Length of each clip in frames (default: 16)
- `--prefer_non_null`: Prefer clips with non-null labels (default: True)

#### Output Format

**Directory Structure:**

```
output_dir/
├── left_hand/
|   ├── video_train/
|   │   ├── label1/
|   │   |   ├── label1_0.mp4
|   │   |   ├── label1_1.mp4
|   │   |   └── ...
|   |   └── ...
|   ├── video_val/
|   │   ├── label1/
|   │   |   ├── label1_0.mp4
|   │   |   ├── label1_1.mp4
|   │   |   └── ...
|   |   └── ...
|   ├── train_list_video.txt
|   └── val_list_video.txt
├── right_hand/
|   ├── video_train/
|   ├── video_val/
|   ├── train_list_video.txt
|   └── val_list_video.txt
```

**Index File** (`train_list_video.txt`):
```
label1/label1_0.mp4 0
label1/label1_1.mp4 0
label2/label2_0.mp4 1
...
```

### 2. Precompute Semantic Embeddings

Generate semantic embeddings for action labels using transformer models.

**Script**: `precompute_semantic_embeddings.py`

#### Features

- ✅ Parses natural language action descriptions
- ✅ Converts to structured format (verb, object, target, tool)
- ✅ Generates embeddings using HuggingFace transformers
- ✅ Supports multiple semantic models
- ✅ Handles special labels ('null', 'w')

#### Usage

**Basic usage:**

```bash
python precompute_semantic_embeddings.py \
    --data_root /path/to/your/dataset \
    --mapping_file action_mapping.txt
```

**With specific model:**

```bash
python precompute_semantic_embeddings.py \
    --data_root /path/to/your/dataset \
    --mapping_file action_mapping.txt \
    --semantic_model_name sentence-transformers/all-MiniLM-L6-v2
```

**Custom output path:**

```bash
python precompute_semantic_embeddings.py \
    --data_root /path/to/your/dataset \
    --mapping_file action_mapping.txt \
    --output_path /path/to/output/embeddings.pt
```

#### Input Format

**Mapping File** (`./action_descriptions/*_description.txt`):
```
label1 "natural language description 1"
label2 "natural language description 2"
...
```

**Example** (HAVID dataset):
```
ibacb "insert the ball into the cylinder base"
ibscb "insert the ball seat into the cylinder base"
```

#### Output Format

**PyTorch .pt file** containing:
```python
{
    'embeddings': {
        'label1': torch.Tensor([...]),  # Shape: [embedding_dim]
        'label2': torch.Tensor([...]),
        ...
    },
    'meta': {
        'semantic_model_name': 'model_name',
        'num_labels': N,
        'embedding_dim': D,
        'data_root': '/path/to/data',
        'mapping_file': '/path/to/mapping.txt'
    }
}
```

#### Structured Description Format

The script converts natural language descriptions to structured format:

**Input**: `"insert the ball into the cylinder base"`

**Output**: 
```
The action is insert. 
The manipulated object is ball. 
The target object is cylinder base. 
The tool used is null.
```

This structured format improves semantic embedding quality by explicitly identifying action components.

### Example 1: HAVID Dataset

```bash
python precompute_semantic_embeddings.py \
    --data_root /data/havid \
    --mapping_file havid_description.txt \
    --semantic_model_name sentence-transformers/all-MiniLM-L6-v2
```

**Output**: `/data/havid/semantic_embeddings/sentence-transformers_all-MiniLM-L6-v2.pt`


## Contact

For questions or issues, please open an issue on GitHub.

