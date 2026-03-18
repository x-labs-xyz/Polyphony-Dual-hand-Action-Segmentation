"""
Semantic Feature Alignment for Video Action Recognition
Aligns visual features with semantic embeddings using Temporal Convolutional Networks (TCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import os
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class Chomp1d(nn.Module):
    """Remove padding from the end of sequence to maintain causal convolution"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Individual temporal block with dilated convolution and residual connection"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for processing frame-wise features"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        layers_dims = [input_dim] + hidden_dims
        
        for i in range(len(layers_dims) - 1):
            dilation = 2 ** i  # Exponential dilation
            in_channels = layers_dims[i]
            out_channels = layers_dims[i + 1]
            
            self.layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, seq_len, hidden_dims[-1]]
        """
        # TCN expects [batch_size, channels, seq_len]
        x = x.transpose(1, 2)  # [B, D, T]
        
        for layer in self.layers:
            x = layer(x)
        
        # Back to [batch_size, seq_len, channels]
        x = x.transpose(1, 2)  # [B, T, D]
        return x


class SemanticFeatureAlignmentModel(nn.Module):
    """TCN-based model for aligning visual features with semantic features"""
    
    def __init__(
        self,
        visual_dim: int = 768,
        semantic_dim: int = 768,
        tcn_hidden_dims: List[int] = [512, 512, 256],
        kernel_size: int = 3,
        dropout: float = 0.3,
        alignment_dim: int = 256
    ):
        super().__init__()
        
        # TCN for temporal modeling
        self.tcn = TemporalConvNet(
            input_dim=visual_dim,
            hidden_dims=tcn_hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Progressive dimension mapping: TCN -> intermediate -> semantic
        intermediate_dim = min(semantic_dim, tcn_hidden_dims[-1] * 2)
        
        self.visual_projector = nn.Sequential(
            nn.Linear(tcn_hidden_dims[-1], intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # Semantic feature processor
        self.semantic_projector = nn.Sequential(
            nn.Linear(semantic_dim, alignment_dim),
            nn.LayerNorm(alignment_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(alignment_dim, semantic_dim)
        )
        
        self.alignment_dim = alignment_dim
        
    def forward(self, visual_features: torch.Tensor, return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: [batch_size, seq_len, visual_dim]
            return_intermediate: Whether to return intermediate features
        
        Returns:
            Dictionary containing processed features
        """
        # Apply TCN for temporal context
        tcn_features = self.tcn(visual_features)
        
        # Project to semantic space
        aligned_features = self.visual_projector(tcn_features)
        
        outputs = {
            'aligned_features': aligned_features,
            'tcn_features': tcn_features
        }
        
        if return_intermediate:
            outputs['visual_features'] = visual_features
            
        return outputs
    
    def process_semantic_features(self, semantic_features: torch.Tensor) -> torch.Tensor:
        """Process semantic features through projector"""
        return self.semantic_projector(semantic_features)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class AdaptiveAlignmentLoss(nn.Module):
    """Advanced loss function combining multiple alignment objectives"""
    
    def __init__(self, loss_type: str = 'adaptive', alpha: float = 0.7, temperature: float = 0.07):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self, predicted_features: torch.Tensor, target_features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predicted_features: [batch_size, seq_len, feature_dim]
            target_features: [batch_size, seq_len, feature_dim]
            mask: [batch_size, seq_len] - 1 for valid frames, 0 for padding
        """
        if self.loss_type == 'adaptive':
            # Combine cosine and MSE losses
            pred_norm = F.normalize(predicted_features, dim=-1)
            target_norm = F.normalize(target_features, dim=-1)
            
            cos_sim = torch.sum(pred_norm * target_norm, dim=-1)
            cosine_loss = 1 - cos_sim
            
            mse_loss = F.mse_loss(predicted_features, target_features, reduction='none')
            mse_loss = mse_loss.mean(dim=-1)
            
            loss = self.alpha * cosine_loss + (1 - self.alpha) * mse_loss
            
        elif self.loss_type == 'mse':
            loss = F.mse_loss(predicted_features, target_features, reduction='none')
            loss = loss.mean(dim=-1)
            
        elif self.loss_type == 'cosine':
            pred_norm = F.normalize(predicted_features, dim=-1)
            target_norm = F.normalize(target_features, dim=-1)
            cos_sim = torch.sum(pred_norm * target_norm, dim=-1)
            loss = 1 - cos_sim
            
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(predicted_features, target_features, reduction='none')
            loss = loss.mean(dim=-1)
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum() if mask.sum() > 0 else loss.mean()
        else:
            return loss.mean()


# ============================================================================
# DATASET
# ============================================================================

class SemanticAlignmentDataset(Dataset):
    """Dataset for frame-wise semantic alignment with dynamic batching"""
    
    def __init__(
        self,
        data_root: str,
        split_file: str,
        feature_path: str,
        annotation_path: str,
        semantic_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        semantic_embeddings_path: Optional[str] = None,
        downsample_rate: int = 1
    ):
        self.data_root = Path(data_root)
        self.feature_path = Path(feature_path)
        self.annotation_path = Path(annotation_path)
        self.downsample_rate = downsample_rate
        
        # Load action label to description mapping
        self.action_mapping = self._load_action_mapping()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.action_mapping.keys())}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Load video list for this split
        self.video_list = self._load_split_file(split_file)

        # Initialize semantic feature source
        self.semantic_embeddings_path = semantic_embeddings_path
        self.semantic_model_name = semantic_model_name
        self._label_to_embedding: Optional[Dict[str, torch.Tensor]] = None
        
        if semantic_embeddings_path and os.path.exists(semantic_embeddings_path):
            print(f"Loading precomputed semantic embeddings from {semantic_embeddings_path}")
            saved = torch.load(semantic_embeddings_path, map_location='cpu')
            embeddings = saved.get('embeddings', {})
            self._label_to_embedding = {
                k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) 
                for k, v in embeddings.items()
            }
            print(f"Loaded {len(self._label_to_embedding)} embeddings")
        else:
            # Fallback to online model encoding
            self.tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
            self.semantic_model = AutoModel.from_pretrained(semantic_model_name)
            self.semantic_model.eval()
        
        print(f"Loaded {len(self.video_list)} videos for split")
        print(f"Found {len(self.action_mapping)} action classes")
        
    def _load_action_mapping(self) -> Dict[str, str]:
        """Load action label to description mapping"""
        mapping_file = self.data_root / "havid_description.txt"
        action_mapping = {}
        
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '"' in line:
                    parts = line.split(' "', 1)
                    if len(parts) == 2:
                        label = parts[0].strip()
                        description = parts[1].rstrip('"')
                        action_mapping[label] = description
        
        return action_mapping
    
    def _load_split_file(self, split_file: str) -> List[str]:
        """Load video list from split file"""
        split_path = self.data_root / split_file
        video_list = []
        
        with open(split_path, 'r') as f:
            for line in f:
                video_id = line.strip()
                if video_id:
                    video_list.append(video_id)
        
        return video_list
    
    def __len__(self) -> int:
        return len(self.video_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns item WITHOUT padding - padding happens in collate_fn"""
        video_id = self.video_list[idx].split(".")[0]
        
        # Load visual features
        feature_path = self.feature_path / f"{video_id}.npy"
        visual_features_np = np.load(feature_path)
        # Transpose from (feature_dim, seq_len) to (seq_len, feature_dim)
        visual_features_np = visual_features_np.T
        visual_features = torch.from_numpy(visual_features_np).float()
        
        # Load frame-wise annotations
        annotation_path = self.annotation_path / f"{video_id}.txt"
        frame_labels = self._load_frame_annotations(annotation_path)
        
        # Ensure same length
        min_len = min(len(visual_features), len(frame_labels))
        visual_features = visual_features[:min_len]
        frame_labels = frame_labels[:min_len]
        
        # Apply downsampling if specified
        if self.downsample_rate > 1:
            indices = torch.arange(0, len(visual_features), self.downsample_rate)
            visual_features = visual_features[indices]
            frame_labels = [frame_labels[i] for i in indices]
        
        # Convert labels to indices
        action_indices = [self.label_to_idx.get(label, 0) for label in frame_labels]
        action_indices = torch.tensor(action_indices, dtype=torch.long)
        
        # Extract semantic features
        semantic_features = self._extract_semantic_features(frame_labels)
        semantic_features = torch.stack(semantic_features)
        
        return {
            'visual_features': visual_features,
            'semantic_features': semantic_features,
            'action_indices': action_indices,
            'video_id': video_id,
            'frame_labels': frame_labels
        }
    
    def _load_frame_annotations(self, annotation_path: Path) -> List[str]:
        """Load frame-wise action labels"""
        frame_labels = []
        
        with open(annotation_path, 'r') as f:
            for line in f:
                label = line.strip()
                if label:
                    frame_labels.append(label)
        
        return frame_labels
    
    def _extract_semantic_features(self, labels: List[str]) -> List[torch.Tensor]:
        """Get semantic features for labels using precomputed cache if available"""
        semantic_features: List[torch.Tensor] = []
        
        if self._label_to_embedding is not None:
            for label in labels:
                if label in self._label_to_embedding:
                    semantic_features.append(self._label_to_embedding[label])
                elif label == 'null' and 'null' in self._label_to_embedding:
                    semantic_features.append(self._label_to_embedding['null'])
                elif label == 'w' and 'w' in self._label_to_embedding:
                    semantic_features.append(self._label_to_embedding['w'])
                else:
                    # Fallback: unknowns map to zero vector
                    any_vec = next(iter(self._label_to_embedding.values()))
                    semantic_features.append(torch.zeros_like(any_vec))
            return semantic_features
        
        # Fallback to online computation
        with torch.no_grad():
            for label in labels:
                if label in self.action_mapping:
                    description = self.action_mapping[label]
                elif label == 'null':
                    description = "no action or transition state"
                elif label == 'w':
                    description = "wrong or incorrect action"
                else:
                    description = f"unknown action {label}"
                
                inputs = self.tokenizer(
                    description, return_tensors='pt',
                    max_length=64, truncation=True, padding=True
                )
                outputs = self.semantic_model(**inputs)
                semantic_feat = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                semantic_features.append(semantic_feat)
        
        return semantic_features


def dynamic_collate_fn(batch):
    """Dynamic collate function that pads to the longest sequence in the batch"""
    # Extract all components
    visual_features = [item['visual_features'] for item in batch]
    semantic_features = [item['semantic_features'] for item in batch]
    action_indices = [item['action_indices'] for item in batch]
    video_ids = [item['video_id'] for item in batch]
    frame_labels = [item['frame_labels'] for item in batch]
    
    # Store original lengths BEFORE padding
    original_lengths = torch.tensor([len(vf) for vf in visual_features], dtype=torch.long)
    
    # Pad sequences to the max length in this batch
    visual_features_padded = pad_sequence(visual_features, batch_first=True, padding_value=0)
    semantic_features_padded = pad_sequence(semantic_features, batch_first=True, padding_value=0)
    action_indices_padded = pad_sequence(action_indices, batch_first=True, padding_value=-1)
    
    # Create valid mask: 1 for real frames, 0 for padding
    batch_size = len(visual_features)
    max_len = visual_features_padded.size(1)
    valid_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    for i, length in enumerate(original_lengths):
        valid_mask[i, :length] = 1.0
    
    # Pad frame labels
    frame_labels_padded = []
    for labels in frame_labels:
        padded = labels + ['<pad>'] * (max_len - len(labels))
        frame_labels_padded.append(padded)
    
    return {
        'visual_features': visual_features_padded,
        'semantic_features': semantic_features_padded,
        'action_indices': action_indices_padded,
        'valid_mask': valid_mask,
        'frame_labels': frame_labels_padded,
        'video_id': video_ids,
        'original_length': original_lengths
    }


def create_data_loaders(
    data_root: str,
    train_split: str,
    test_split: str, 
    feature_path: str,
    annotation_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    downsample_rate: int = 1,
    semantic_embeddings_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders with dynamic batching"""
    
    train_dataset = SemanticAlignmentDataset(
        data_root=data_root,
        split_file=train_split,
        feature_path=feature_path,
        annotation_path=annotation_path,
        semantic_embeddings_path=semantic_embeddings_path,
        downsample_rate=downsample_rate
    )
    
    test_dataset = SemanticAlignmentDataset(
        data_root=data_root,
        split_file=test_split,
        feature_path=feature_path,
        annotation_path=annotation_path,
        semantic_embeddings_path=semantic_embeddings_path,
        downsample_rate=downsample_rate
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dynamic_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dynamic_collate_fn
    )
    
    return train_loader, test_loader


# ============================================================================
# TRAINER
# ============================================================================

class SemanticAlignmentTrainer:
    """Training pipeline for TCN-based semantic alignment"""
    
    def __init__(
        self,
        model: SemanticFeatureAlignmentModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        loss_type: str = 'smooth_l1'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.base_lr = learning_rate
        
        # Loss function
        self.criterion = AdaptiveAlignmentLoss(loss_type=loss_type)
        
        # Optimizer with different learning rates for TCN and projector
        tcn_params = list(self.model.tcn.parameters())
        projector_params = list(self.model.visual_projector.parameters()) + \
                          list(self.model.semantic_projector.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': tcn_params, 'lr': learning_rate * 0.5, 'weight_decay': 1e-4},
            {'params': projector_params, 'lr': learning_rate * 1.0, 'weight_decay': 1e-3}
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6, verbose=True
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            visual_feat = batch['visual_features'].to(self.device)
            semantic_feat = batch['semantic_features'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(visual_feat)
            predicted_semantic = outputs['aligned_features']
            
            # Compute loss with masking
            loss = self.criterion(predicted_semantic, semantic_feat, valid_mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                visual_feat = batch['visual_features'].to(self.device)
                semantic_feat = batch['semantic_features'].to(self.device)
                valid_mask = batch['valid_mask'].to(self.device)
                
                outputs = self.model(visual_feat)
                predicted_semantic = outputs['aligned_features']
                
                loss = self.criterion(predicted_semantic, semantic_feat, valid_mask)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, num_epochs: int, save_dir: str = './checkpoints', patience: int = 15):
        """Full training loop with early stopping"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping and best model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved with val_loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter}/{patience} epochs")
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))


# ============================================================================
# EVALUATOR
# ============================================================================

class SemanticAlignmentEvaluator:
    """Evaluator for semantic alignment model"""
    
    def __init__(self, model: SemanticFeatureAlignmentModel, device: torch.device, 
                 action_mapping: Dict[str, str], label_to_idx: Dict[str, int]):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.action_mapping = action_mapping
        self.label_to_idx = label_to_idx
        self.w_action_idx = label_to_idx.get('w', None)
        
    def evaluate_alignment_quality(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate semantic alignment quality"""
        total_mse = 0
        total_cosine_sim = 0
        num_valid_frames = 0
        
        all_predictions = []
        all_targets = []
        
        # Track performance by action type
        action_type_stats = {'null': [], 'wrong': [], 'actions': []}
        
        with torch.no_grad():
            for batch in data_loader:
                visual_feat = batch['visual_features'].to(self.device)
                semantic_feat = batch['semantic_features'].to(self.device)
                valid_mask = batch['valid_mask'].to(self.device)
                action_indices = batch['action_indices'].to(self.device)
                
                outputs = self.model(visual_feat)
                predicted_semantic = outputs['aligned_features']
                
                # Evaluate all valid frames
                valid_frames = (valid_mask == 1)
                
                if valid_frames.sum() > 0:
                    pred_valid = predicted_semantic[valid_frames]
                    target_valid = semantic_feat[valid_frames]
                    action_valid = action_indices[valid_frames]
                    
                    # MSE
                    mse = F.mse_loss(pred_valid, target_valid).item()
                    total_mse += mse * valid_frames.sum().item()
                    
                    # Cosine similarity
                    pred_norm = F.normalize(pred_valid, dim=-1)
                    target_norm = F.normalize(target_valid, dim=-1)
                    cos_sims = torch.sum(pred_norm * target_norm, dim=-1)
                    cos_sim = torch.mean(cos_sims).item()
                    total_cosine_sim += cos_sim * valid_frames.sum().item()
                    
                    num_valid_frames += valid_frames.sum().item()
                    
                    # Track by action type
                    cos_sims_np = cos_sims.cpu().numpy()
                    action_valid_np = action_valid.cpu().numpy()
                    
                    null_mask = action_valid_np == 0
                    w_idx = self.w_action_idx
                    if w_idx is not None:
                        wrong_mask = action_valid_np == w_idx
                        action_mask = ~(null_mask | wrong_mask)
                    else:
                        wrong_mask = np.zeros_like(null_mask, dtype=bool)
                        action_mask = ~null_mask
                    
                    if null_mask.sum() > 0:
                        action_type_stats['null'].extend(cos_sims_np[null_mask])
                    if wrong_mask.sum() > 0:
                        action_type_stats['wrong'].extend(cos_sims_np[wrong_mask])
                    if action_mask.sum() > 0:
                        action_type_stats['actions'].extend(cos_sims_np[action_mask])
                    
                    all_predictions.append(pred_norm.cpu().numpy())
                    all_targets.append(target_norm.cpu().numpy())
        
        # Compute overall similarity statistics
        if num_valid_frames > 0:
            all_pred = np.vstack(all_predictions)
            all_tgt = np.vstack(all_targets)
            
            similarities = np.sum(all_pred * all_tgt, axis=1)
            
            results = {
                'mse_loss': total_mse / num_valid_frames,
                'mean_cosine_similarity': total_cosine_sim / num_valid_frames,
                'median_cosine_similarity': np.median(similarities),
                'std_cosine_similarity': np.std(similarities),
                'min_cosine_similarity': np.min(similarities),
                'max_cosine_similarity': np.max(similarities),
                'num_frames_evaluated': num_valid_frames
            }
            
            # Add action-type specific statistics
            for action_type, sims in action_type_stats.items():
                if len(sims) > 0:
                    results[f'{action_type}_mean_similarity'] = np.mean(sims)
                    results[f'{action_type}_count'] = len(sims)
                else:
                    results[f'{action_type}_mean_similarity'] = 0.0
                    results[f'{action_type}_count'] = 0
            
            return results
        else:
            return {'error': 'No valid frames found for evaluation'}
    
    def save_enhanced_features(self, data_loader: DataLoader, save_dir: str):
        """Save enhanced frame-wise features as .npy for downstream action segmentation"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving enhanced features to {save_dir}...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                visual_feat = batch['visual_features'].to(self.device)
                video_ids = batch['video_id']
                original_lengths = batch['original_length']
                
                outputs = self.model(visual_feat, return_intermediate=True)
                
                # Save features for each video in the batch
                for i in range(len(video_ids)):
                    video_id = video_ids[i]
                    orig_len = original_lengths[i].item()
                    
                    # Extract valid features
                    enhanced_feats = outputs['aligned_features'][i, :orig_len].cpu()
                    
                    # Save as .npy (transposed to match expected format)
                    save_path = os.path.join(save_dir, f"{video_id}.npy")
                    np.save(save_path, enhanced_feats.numpy().T)
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        print("Enhanced features saved successfully!")

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = {
        'data_root': '/path_to/data/',
        'train_split': 'path_to/train.split1.bundle',
        'test_split': 'path_to/test.split1.bundle',
        'feature_path': 'path_to/extracted_features/shared_features',
        'annotation_path': 'path_to/groundTruth',
        'save_dir': './checkpoints',
        'save_feature_dir': './enhanced_features',
        'batch_size': 4,
        'learning_rate': 3e-4,
        'num_epochs': 150,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'visual_dim': 768,
        'semantic_dim': 384,  # MiniLM dimension
        'tcn_hidden_dims': [512, 128, 64],
        'downsample_rate': 1,
        'loss_type': 'adaptive',
        'semantic_embeddings_path': 'path_to/semantic_embeddings/sentence-transformers_all-MiniLM-L6-v2.pt'
    }
    
    print("=" * 80)
    print("SEMANTIC FEATURE ALIGNMENT TRAINING")
    print("=" * 80)
    print(f"\nKey features:")
    print(f"  ✓ Dynamic padding to batch max length")
    print(f"  ✓ Temporal Convolutional Network (TCN)")
    print(f"  ✓ Precomputed semantic embeddings")
    print(f"  ✓ Adaptive alignment loss")
    print("=" * 80)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, test_loader = create_data_loaders(
        data_root=config['data_root'],
        train_split=config['train_split'],
        test_split=config['test_split'],
        feature_path=config['feature_path'],
        annotation_path=config['annotation_path'],
        batch_size=config['batch_size'],
        downsample_rate=config['downsample_rate'],
        num_workers=0,
        semantic_embeddings_path=config['semantic_embeddings_path']
    )
    
    # Check data loading
    print("\nChecking data loading...")
    sample_batch = next(iter(train_loader))
    print(f"Visual features shape: {sample_batch['visual_features'].shape}")
    print(f"Semantic features shape: {sample_batch['semantic_features'].shape}")
    print(f"Valid mask shape: {sample_batch['valid_mask'].shape}")
    print(f"Original lengths: {sample_batch['original_length']}")
    
    # Initialize model
    print("\nInitializing model...")
    model = SemanticFeatureAlignmentModel(
        visual_dim=config['visual_dim'],
        semantic_dim=config['semantic_dim'],
        tcn_hidden_dims=config['tcn_hidden_dims']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = SemanticAlignmentTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=torch.device(config['device']),
        learning_rate=config['learning_rate'],
        loss_type=config['loss_type']
    )
    
    # Train the model
    print(f"\nStarting training on {config['device']}...")
    trainer.train(num_epochs=config['num_epochs'], save_dir=config['save_dir'])
    
    # Load best model and evaluate
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(config['save_dir'] + '/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get action mapping and label indices from dataset
    action_mapping = train_loader.dataset.action_mapping
    label_to_idx = train_loader.dataset.label_to_idx
    
    # Initialize evaluator
    evaluator = SemanticAlignmentEvaluator(
        model, torch.device(config['device']), action_mapping, label_to_idx
    )
    
    # Evaluate alignment quality
    print("\nEvaluating alignment quality...")
    eval_results = evaluator.evaluate_alignment_quality(test_loader)
    
    print("\nAlignment Quality Results:")
    for metric, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.6f}")
        else:
            print(f"  {metric}: {value}")
    
    # Save enhanced features for action segmentation
    print("\nSaving enhanced features for downstream action segmentation...")
    evaluator.save_enhanced_features(test_loader, config['save_feature_dir'])
    evaluator.save_enhanced_features(train_loader, config['save_feature_dir'])
    
    print("\nTraining and feature enhancement completed!")
    print(f"Enhanced features saved to: {config['save_feature_dir']}")

