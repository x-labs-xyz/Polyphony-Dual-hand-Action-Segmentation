# ADH-ViT: Alternating Dual-Hand Vision Transformer

This is official implementation of ADH-ViT. It serves as a feature extractor in Polyphony, but it can also operate as a standalone dual-hand action recognition model. By blocking one stream, it can also work on single-stream action recognition tasks.

---

## 📁 Project Structure

```
ADH-ViT/
├── train.sh    # Example training script
├── main.py    # Main training script
├── engine_for_alternating_finetuning.py  # Training/validation/test loops
├── engine_for_finetuning.py              # Helper functions (merge, etc.)
├── utils.py                               # Utilities (distributed training, logging)
├── optim_factory.py                       # Optimizer with layer-wise LR decay
│
├── models/
│   ├── __init__.py                        # Model registry
│   ├── modeling_finetune.py               # Base VisionTransformer
│   ├── modeling_finetune_alternating.py   # Alternating dual-head ViT
│   └── vit_b_k710_dl_from_giant.pth      # Pretrained weights (download the weights from VideoMAE V2 official website)
│
├── dataset/
│   ├── __init__.py
│   ├── build.py                           # Dataset builder
│   ├── datasets.py                        # RawFrameClsDataset, VideoClsDataset
│   ├── loader.py                          # Data loader utilities
│   └── [augmentation files]               # transforms, rand_augment, etc.
│
└── requirements.txt                       # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /path_to/ADH-ViT
pip install -r requirements.txt
```

**Key Dependencies:**
- PyTorch >= 1.8.0
- torchvision
- timm
- decord (for video loading)
- einops

### 2. Prepare Dataset

Ensure your dataset follows this structure:

```
/path_to/dataset
├── lh_data/                          # Left hand data
│   ├── train_list_video.txt
│   ├── val_list_video.txt
│   └── videos/
│       ├── action1/
│       │   └── *.mp4
│       ├── action2/
│       └── ...
└── rh_data/                          # Right hand data
    ├── train_list_video.txt
    ├── val_list_video.txt
    └── videos/
        ├── action1/
        │   └── *.mp4
        ├── action2/
        └── ...
```

**List file format** (`train_list_video.txt`, `val_list_video.txt`):
```
action1/video1.mp4 0
action1/video2.mp4 0
action2/video3.mp4 1
...
```

### 3. Run Training

**Option A: Use the provided script (recommended)**

```bash
cd /path_to/ADH-ViT
bash train.sh
```

**Option B: Run directly with custom parameters**

```bash
cd /path_to/ADH-ViT

python run_alternating_hand_finetuning.py \
    --model vit_base_patch16_224_alternating \
    --lh_data_path /path_to/lh_data \
    --rh_data_path /path_to/rh_data \
    --lh_num_classes 75 \
    --rh_num_classes 75 \
    --data_set HAVID \
    --finetune models/vit_b_k710_dl_from_giant.pth \
    --output_dir output/havid_alternating_hands \
    --log_dir output/havid_alternating_hands \
    --batch_size 4 \
    --epochs 50 \
    --alternation_steps 50 \
    --lr 1e-3 \
    --num_frames 16 \
    --sampling_rate 4
```
---

## 🎯 Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alternation_steps` | 50 | Switch between left/right hand every N steps |
| `--lh_num_classes` | 75 (on HA-ViD) | Number of left-hand action classes |
| `--rh_num_classes` | 75 (on HA-ViD) | Number of right-hand action classes |
| `--batch_size` | 4 | Batch size per GPU |
| `--epochs` | 50 | Total training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--drop_path` | 0.3 | DropPath rate for regularization |
| `--num_frames` | 16 | Number of frames per video clip |
| `--sampling_rate` | 4 | Frame sampling rate (1 = dense, 4 = sparse) |
| `--num_sample` | 2 | Number of clips per video |
| `--layer_decay` | 0.9 | Layer-wise learning rate decay |
| `--warmup_epochs` | 5 | Warmup epochs |
| `--weight_decay` | 0.1 | Weight decay |

**Alternation Strategies:**
- `--alternation_steps 10`: Very frequent switching (more balanced but slower convergence)
- `--alternation_steps 50`: Moderate switching (recommended)
- `--alternation_steps 200`: Less frequent switching (faster but may favor one hand)

**Single-steam action:**
- `--one_stream`: Enable for single-stream training (only specify lh_data_path)
---

## 🏗️ Model Architecture

**ADH-ViT** uses a **Vision Transformer (ViT) with dual classification heads**:

```
Input Video (T×H×W×3)
        ↓
ViT Backbone (Shared)
├── Patch Embedding
├── Positional Embedding  
├── Transformer Blocks (12 layers)
└── Feature Extraction
        ↓
    ┌───┴───┐
    ↓       ↓
LH Head   RH Head
```

**Key Features:**
- **Shared backbone**: Efficient feature extraction
- **Dual heads**: Separate classification for each hand
- **Alternating training**: Switch focus between hands every N steps
- **Pretrained backbone**: Strong initialization

---

## 📊 Training Strategy

The alternating training strategy works as follows:

1. **Step 0-49**: Train left-hand head (right-hand frozen)
2. **Step 50-99**: Train right-hand head (left-hand frozen)
3. **Step 100-149**: Train left-hand head
4. **Repeat...**

This approach:
- ✅ Prevents catastrophic forgetting
- ✅ Balances training between hands
- ✅ More stable than simultaneous training
- ✅ Better final accuracy

---

## 📈 Monitoring Training

Training logs and checkpoints are saved to `output/`:

```
output/
├── checkpoint-{epoch}.pth       # Model checkpoints
├── log.txt                      # Training log
└── events.out.tfevents.*       # TensorBoard logs
```

**Resume training:**
```bash
python run_alternating_hand_finetuning.py \
    --resume output/havid_alternating_hands/checkpoint-50.pth \
    [... other args ...]
```

---

## 🧪 Evaluation

**During training:** Validation is performed every epoch by default.

**After training (test set):**
```bash
python run_alternating_hand_finetuning.py \
    --eval \
    --resume /path_to/checkpoint-best.pth \
    --lh_data_path /path_to/lh_data \
    --rh_data_path /path_to/rh_data \
    [... other args ...]
```

**Metrics reported:**
- Left-hand accuracy (Top-1, Top-5)
- Right-hand accuracy (Top-1, Top-5)
- Average accuracy across hands

---

## 💡 Tips

1. **Start with default parameters** - They work well for HA-ViD
2. **Monitor both hands** - Ensure balanced performance
3. **Experiment with alternation_steps** - Try 25, 50, 100
4. **Use data augmentation** - Enabled by default, improves generalization
5. **Save checkpoints frequently** - Use `--save_ckpt_freq 5`
6. **Use TensorBoard** - Great for visualizing training curves

---

## 🙏 Acknowledgments

ADH-ViT is built on [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2). We thank the original VideoMAEv2 authors for their excellent work.