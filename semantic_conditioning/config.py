"""
Configuration file for semantic feature alignment training
"""

# Data paths
DATA_ROOT = '/home/hao/Polyphony/data/havid'
TRAIN_SPLIT = 'splits/View2/lh_pt/train.split1.bundle'
TEST_SPLIT = 'splits/View2/lh_pt/test.split1.bundle'
FEATURE_PATH = '/home/hao/Polyphony/data/havid/videomae_features_extend/view2/shared_features'
ANNOTATION_PATH = '/home/hao/Polyphony/data/havid/groundTruth/View2/lh_pt'

# Output paths
SAVE_DIR = './checkpoints'
SAVE_FEATURE_DIR = './enhanced_features'
VISUALIZATION_DIR = './visualizations'

# Model hyperparameters
VISUAL_DIM = 768  # VideoMAE feature dimension
SEMANTIC_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2 dimension
TCN_HIDDEN_DIMS = [512, 128, 64]
DROPOUT = 0.3
KERNEL_SIZE = 3

# Training hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
NUM_EPOCHS = 150
PATIENCE = 15  # Early stopping patience
DOWNSAMPLE_RATE = 1

# Loss configuration
LOSS_TYPE = 'adaptive'  # Options: 'adaptive', 'mse', 'cosine', 'smooth_l1'
ALPHA = 0.7  # Weight for cosine vs MSE in adaptive loss

# Semantic embeddings
SEMANTIC_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
SEMANTIC_EMBEDDINGS_PATH = '/home/hao/Polyphony/data/havid/semantic_embeddings/sentence-transformers_all-MiniLM-L6-v2.pt'

# Device
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues

