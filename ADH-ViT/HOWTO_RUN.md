# How to Run ADH-ViT Training

## 📍 Current Location
Your clean training code is now in: `/home/hao/Polyphony/ADH-ViT/`

---

## ✅ Quick Run (3 Steps)

### Step 1: Navigate to ADH-ViT directory
```bash
cd /home/hao/Polyphony/ADH-ViT
```

### Step 2: Activate your environment and install dependencies
```bash
# Activate your conda/venv environment (example):
conda activate your_env_name
# OR
source /path/to/venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt
```

### Step 3: Run training
```bash
bash scripts/finetune/train_havid_alternating.sh
```

**That's it!** The script will start training alternating dual-hand action recognition.

---

## 📂 Directory Structure

When you `cd /home/hao/Polyphony/ADH-ViT`, you'll see:

```
ADH-ViT/
├── run_alternating_hand_finetuning.py   ← Main script
├── models/
│   ├── modeling_finetune_alternating.py ← Dual-head model
│   └── vit_b_k710_dl_from_giant.pth    ← Pretrained weights ✓
├── dataset/                             ← Data loading
├── scripts/finetune/
│   └── train_havid_alternating.sh      ← Launch script ✓
└── [other support files]
```

---

## 🔧 Customizing Training

### Edit the training script:
```bash
nano scripts/finetune/train_havid_alternating.sh
```

### Key parameters to adjust:
```bash
# Data paths (UPDATE THESE if your data is elsewhere)
LH_DATA_PATH='/home/hao/Polyphony/data/havid_mmaction/lh_v0'
RH_DATA_PATH='/home/hao/Polyphony/data/havid_mmaction/rh_v0'

# Training hyperparameters
--batch_size 4              # Increase if you have more GPU memory
--epochs 50                 # Total training epochs
--alternation_steps 50      # Switch hands every N steps (try 25, 50, 100)
--lr 1e-3                   # Learning rate
--num_workers 8             # Data loading workers

# Multi-GPU training
torchrun --nproc_per_node=2  # Change to number of GPUs
```

---

## 🚀 Running Options

### Option 1: Use the provided shell script (Easiest)
```bash
cd /home/hao/Polyphony/ADH-ViT
bash scripts/finetune/train_havid_alternating.sh
```

### Option 2: Run Python directly
```bash
cd /home/hao/Polyphony/ADH-ViT

python run_alternating_hand_finetuning.py \
    --model vit_base_patch16_224_alternating \
    --lh_data_path /home/hao/Polyphony/data/havid_mmaction/lh_v0 \
    --lh_data_root /home/hao/Polyphony/data/havid_mmaction/lh_v0 \
    --rh_data_path /home/hao/Polyphony/data/havid_mmaction/rh_v0 \
    --rh_data_root /home/hao/Polyphony/data/havid_mmaction/rh_v0 \
    --lh_num_classes 75 \
    --rh_num_classes 75 \
    --data_set HAVID \
    --finetune models/vit_b_k710_dl_from_giant.pth \
    --output_dir output/havid \
    --batch_size 4 \
    --epochs 50 \
    --alternation_steps 50
```

### Option 3: Multi-GPU training
```bash
cd /home/hao/Polyphony/ADH-ViT

# Edit the script to use multiple GPUs
sed -i 's/nproc_per_node=1/nproc_per_node=2/g' scripts/finetune/train_havid_alternating.sh

# Run
bash scripts/finetune/train_havid_alternating.sh
```

---

## 📊 Outputs

Training outputs will be saved to:
```
/home/hao/Polyphony/ADH-ViT/output/havid_alternating_hands/
├── checkpoint-10.pth       # Checkpoints every 10 epochs
├── checkpoint-20.pth
├── checkpoint-best.pth     # Best model
├── log.txt                 # Training log
└── events.out.tfevents.*  # TensorBoard logs
```

### View training progress:
```bash
# In a separate terminal
cd /home/hao/Polyphony/ADH-ViT
tensorboard --logdir output/havid_alternating_hands
```

---

## 🐛 Common Issues

### Issue: "No module named 'torch'"
**Solution:** Activate your environment first
```bash
conda activate your_pytorch_env
pip install torch torchvision
```

### Issue: "No module named 'decord'"
**Solution:**
```bash
pip install decord
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in the script:
```bash
--batch_size 2  # or even 1
```

### Issue: Can't find data
**Solution:** Check your data paths in `scripts/finetune/train_havid_alternating.sh`:
```bash
ls /home/hao/Polyphony/data/havid_mmaction/lh_v0/
ls /home/hao/Polyphony/data/havid_mmaction/rh_v0/
```

---

## 💡 Pro Tips

1. **Always run from ADH-ViT directory**
   ```bash
   cd /home/hao/Polyphony/ADH-ViT  # Important!
   bash scripts/finetune/train_havid_alternating.sh
   ```

2. **Use tmux/screen for long training**
   ```bash
   tmux new -s training
   cd /home/hao/Polyphony/ADH-ViT
   bash scripts/finetune/train_havid_alternating.sh
   # Press Ctrl+B, then D to detach
   # Later: tmux attach -t training
   ```

3. **Monitor GPU usage**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Resume training if interrupted**
   ```bash
   python run_alternating_hand_finetuning.py \
       --resume output/havid_alternating_hands/checkpoint-latest.pth \
       [... other args ...]
   ```

---

## 📚 Documentation

- **README.md** - Full documentation
- **QUICK_START.txt** - Quick reference
- **CLEANUP_SUMMARY.md** - What was removed from VideoMAEv2
- **HOWTO_RUN.md** - This file!

---

## ✨ Summary

```bash
# The complete workflow in 3 lines:
cd /home/hao/Polyphony/ADH-ViT
conda activate your_env  # or source venv
bash scripts/finetune/train_havid_alternating.sh
```

**That's all you need!** 🎉

The script will handle everything else:
- ✅ Load pretrained model
- ✅ Load left & right hand datasets
- ✅ Train with alternating strategy
- ✅ Validate each epoch
- ✅ Save checkpoints
- ✅ Log metrics to TensorBoard

---

**Questions?** Check README.md for detailed documentation.

