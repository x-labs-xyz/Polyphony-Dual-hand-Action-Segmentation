# Single-Stream Training with `--one_stream` Flag

ADH-ViT now supports training on datasets that only have one hand/stream of data using the `--one_stream` flag.

## Overview

The `--one_stream` flag allows you to train the dual-head model even when you only have data for one hand (typically left hand). The model will:
- Use left-hand data for both heads
- Train both heads using the same input data
- Still maintain the alternating training strategy
- Report only left-hand metrics

This is useful for:
- Datasets with only single-stream annotations
- Testing the architecture on simpler problems
- Pretraining before full dual-hand training

---

## Usage

### Basic Command

```bash
cd /home/hao/Polyphony/ADH-ViT

python run_alternating_hand_finetuning.py \
    --one_stream \
    --lh_data_path /path/to/single/stream/data \
    --lh_data_root /path/to/single/stream/data \
    --lh_num_classes 75 \
    --rh_num_classes 75 \
    --model vit_base_patch16_224_alternating \
    --finetune models/vit_b_k710_dl_from_giant.pth \
    --output_dir output/single_stream_experiment \
    --batch_size 4 \
    --epochs 50 \
    --alternation_steps 50
```

### Key Differences from Dual-Stream Training

| Aspect | Dual-Stream (Default) | Single-Stream (`--one_stream`) |
|--------|----------------------|-------------------------------|
| `--rh_data_path` | **Required** | Optional (ignored if provided) |
| `--rh_data_root` | **Required** | Optional (ignored if provided) |
| Data Loading | Separate LH & RH datasets | Only LH dataset (mirrored to RH) |
| Training | Alternates between hands | Still alternates, but same data |
| Validation | Reports LH & RH metrics | Reports only LH metrics |
| Final Test | Merges LH & RH results | Reports only LH results |

---

## How It Works

### 1. Argument Changes

```python
# These are now optional when using --one_stream
parser.add_argument('--rh_data_path', required=False, default='', type=str)
parser.add_argument('--rh_data_root', required=False, default='', type=str)

# New flag
parser.add_argument('--one_stream', action='store_true', default=False,
                    help='Train using only left-hand data. Right-hand inputs will mirror left-hand.')
```

### 2. Data Mirroring

When `--one_stream` is enabled:
```python
if args.one_stream:
    # Mirror left-hand paths to right-hand
    args.rh_data_path = args.lh_data_path
    args.rh_data_root = args.lh_data_root
    args.rh_num_classes = args.lh_num_classes
    
    # Use same dataset instance
    rh_dataset_train = lh_dataset_train
    rh_dataset_val = lh_dataset_val
    rh_dataset_test = lh_dataset_test
```

### 3. Data Loader Configuration

Right-hand data loaders are set to `None`:
```python
if args.one_stream:
    rh_sampler_train = None
    rh_data_loader_train = None
    rh_data_loader_val = None
    rh_data_loader_test = None
```

### 4. Training Loop

The alternating training engine handles `None` for right-hand loaders, so the model still trains in alternating mode but always uses left-hand data.

### 5. Validation & Testing

Metrics are only computed and reported for the left hand:
```python
if not args.one_stream:
    print(f'Max accuracy - LH: {max_lh_accuracy:.2f}%, RH: {max_rh_accuracy:.2f}%')
else:
    print(f'Max accuracy - LH: {max_lh_accuracy:.2f}%')
```

---

## Example Scripts

### Shell Script for Single-Stream Training

Create `scripts/finetune/train_single_stream.sh`:

```bash
#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='output/single_stream_training'
DATA_PATH='/path/to/single/stream/data'
MODEL_PATH='models/vit_b_k710_dl_from_giant.pth'

mkdir -p ${OUTPUT_DIR}

python run_alternating_hand_finetuning.py \
    --one_stream \
    --model vit_base_patch16_224_alternating \
    --lh_data_path ${DATA_PATH} \
    --lh_data_root ${DATA_PATH} \
    --lh_num_classes 75 \
    --rh_num_classes 75 \
    --data_set HAVID \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 2 \
    --num_workers 8 \
    --opt adamw \
    --lr 1e-3 \
    --drop_path 0.3 \
    --clip_grad 5.0 \
    --layer_decay 0.9 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.1 \
    --warmup_epochs 5 \
    --epochs 50 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --alternation_steps 50
```

---

## Dataset Structure

Your single-stream dataset should follow this structure:

```
/path/to/single/stream/data/
├── train_list_video.txt
├── val_list_video.txt
└── videos/
    ├── action1/
    │   └── *.mp4
    ├── action2/
    │   └── *.mp4
    └── ...
```

**List file format** (`train_list_video.txt`, `val_list_video.txt`):
```
action1/video1.mp4 0
action1/video2.mp4 0
action2/video3.mp4 1
...
```

---

## Output Structure

Training outputs will be saved to:

```
output/single_stream_training/
├── checkpoint-10.pth
├── checkpoint-20.pth
├── checkpoint-best.pth
├── log.txt
└── events.out.tfevents.*
```

Note: Unlike dual-stream training, there will be **no** separate `_lh` and `_rh` result directories.

---

## Common Use Cases

### 1. Single-Hand Action Recognition

Train on datasets that only capture one hand (e.g., egocentric cooking videos focusing on dominant hand).

```bash
python run_alternating_hand_finetuning.py \
    --one_stream \
    --lh_data_path /data/cooking_dominant_hand \
    [... other args ...]
```

### 2. Pretraining for Dual-Hand

Pretrain on a large single-stream dataset before fine-tuning on dual-hand data.

**Step 1: Pretrain on single-stream**
```bash
python run_alternating_hand_finetuning.py \
    --one_stream \
    --lh_data_path /data/large_single_stream \
    --output_dir output/pretrain_single \
    [... other args ...]
```

**Step 2: Fine-tune on dual-hand**
```bash
python run_alternating_hand_finetuning.py \
    --resume output/pretrain_single/checkpoint-best.pth \
    --lh_data_path /data/dual_hand/lh \
    --rh_data_path /data/dual_hand/rh \
    --output_dir output/finetune_dual \
    [... other args ...]
```

### 3. Testing Model Architecture

Test the alternating training mechanism on simpler single-stream data before scaling to dual-stream.

---

## Performance Considerations

### Advantages
- ✅ **Simpler Data Requirements**: No need for synchronized dual-hand annotations
- ✅ **Faster Data Preparation**: Only one annotation stream needed
- ✅ **Good for Debugging**: Easier to verify model is working correctly

### Limitations
- ❌ **No Cross-Hand Learning**: Model doesn't learn hand coordination
- ❌ **Redundant Computation**: Both heads process same data
- ❌ **Limited Applicability**: Not suitable for tasks requiring dual-hand understanding

---

## Differences from Original Dual-Hand Training

| Feature | Dual-Hand | Single-Stream (`--one_stream`) |
|---------|-----------|-------------------------------|
| Dataset Loading | 2x (LH + RH) | 1x (LH only) |
| Memory Usage | Higher | Lower (~50% less) |
| Training Speed | Slower | Faster (~2x) |
| Model Capability | Dual-hand coordination | Single-hand only |
| Use Case | Bimanual tasks | Unimanual tasks |

---

## Troubleshooting

### Issue: "Right-hand data path required"
**Solution**: Make sure you're using the `--one_stream` flag.

### Issue: Model still expects RH data during validation
**Solution**: This is a bug. The engine should handle `None` for RH loaders. Check that you're using the updated `engine_for_alternating_finetuning.py`.

### Issue: Both heads give identical predictions
**Solution**: This is expected in `--one_stream` mode. Both heads see the same data, so they'll converge to similar representations.

---

## Technical Details

### Memory Savings

In `--one_stream` mode:
- Only 1 dataset is loaded (vs 2 in dual-stream)
- Only 1 data loader is active during training
- Samplers for RH are set to `None`

This reduces memory usage by approximately 50% for data loading.

### Training Efficiency

The alternating strategy is still applied, but with mirrored data:
- Steps 0-49: Train LH head with LH data
- Steps 50-99: Train RH head with LH data (same data!)
- Steps 100-149: Train LH head with LH data
- ...

This maintains the training loop structure but processes each batch twice.

---

## Summary

The `--one_stream` flag enables flexible training on single-stream datasets while maintaining compatibility with the dual-head architecture. This is particularly useful for:
- Datasets with only one hand annotated
- Pretraining on larger single-stream corpora
- Testing and debugging the model architecture

Simply add `--one_stream` to your training command and specify only the left-hand data paths!

---

**Questions?** Check the main README.md or HOWTO_RUN.md for more details.

