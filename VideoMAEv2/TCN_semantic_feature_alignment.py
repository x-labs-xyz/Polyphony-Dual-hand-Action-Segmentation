import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import json
import os
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import math
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
            
            # Temporal convolutional layer with residual connection
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

class TemporalBlock(nn.Module):
    """Individual temporal block with dilated convolution and residual connection"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
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
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        return self.relu(out + residual)

class Chomp1d(nn.Module):
    """Remove padding from the end of sequence to maintain causal convolution"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class SemanticFeatureAlignmentModel(nn.Module):
    """TCN-based model for aligning visual features with semantic features"""
    
    def __init__(
        self,
        visual_dim: int = 768,
        semantic_dim: int = 768,
        tcn_hidden_dims: List[int] = [512, 512, 256],
        kernel_size: int = 3,
        dropout: float = 0.2,
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
        
        # Projection layers for semantic alignment
        self.visual_projector = nn.Sequential(
            nn.Linear(tcn_hidden_dims[-1], alignment_dim),
            nn.LayerNorm(alignment_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(alignment_dim, semantic_dim)
        )
        
        # Optional: semantic feature processor
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
        tcn_features = self.tcn(visual_features)  # [B, T, tcn_hidden_dims[-1]]
        
        # Project to semantic space
        aligned_features = self.visual_projector(tcn_features)  # [B, T, semantic_dim]
        
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

class FrameWiseAlignmentLoss(nn.Module):
    """Loss function for frame-wise semantic alignment"""
    
    def __init__(self, loss_type: str = 'mse', temperature: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
    
    def forward(self, predicted_features: torch.Tensor, target_features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predicted_features: [batch_size, seq_len, feature_dim]
            target_features: [batch_size, seq_len, feature_dim]
            mask: [batch_size, seq_len] - 1 for valid frames, 0 for padding
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_features, target_features, reduction='none')
            loss = loss.mean(dim=-1)  # [B, T]
            
        elif self.loss_type == 'cosine':
            # Normalize features
            pred_norm = F.normalize(predicted_features, dim=-1)
            target_norm = F.normalize(target_features, dim=-1)
            
            # Cosine similarity
            cos_sim = torch.sum(pred_norm * target_norm, dim=-1)  # [B, T]
            loss = 1 - cos_sim  # Convert to loss (0 = perfect, 2 = worst)
            
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(predicted_features, target_features, reduction='none')
            loss = loss.mean(dim=-1)  # [B, T]
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply mask if provided (only excludes padding, not null/wrong actions)
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum() if mask.sum() > 0 else loss.mean()
        else:
            return loss.mean()

class HAVIDDataset(Dataset):
    """Dataset for HAVID assembly actions with frame-wise semantic alignment"""
    
    def __init__(
        self,
        data_root: str = "./data/havid",
        split_file: str = "train_split1_bundle",
        semantic_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        semantic_embeddings_path: Optional[str] = None,
        max_frames: int = 1024,
        downsample_rate: int = 1  # Downsample factor for long sequences
    ):
        self.data_root = Path(data_root)
        self.max_frames = max_frames
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
        if semantic_embeddings_path is not None and os.path.exists(semantic_embeddings_path):
            print(f"Loading precomputed semantic embeddings from {semantic_embeddings_path}")
            saved = torch.load(semantic_embeddings_path, map_location='cpu')
            embeddings: Dict[str, torch.Tensor] = saved.get('embeddings', {})
            # Ensure tensors
            self._label_to_embedding = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in embeddings.items()}
            print(f"Loaded {len(self._label_to_embedding)} embeddings")
        else:
            # Fallback to online model encoding
            self.tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
            self.semantic_model = AutoModel.from_pretrained(semantic_model_name)
            self.semantic_model.eval()
        
        # # Initialize semantic feature extractor
        # self.tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
        # self.semantic_model = AutoModel.from_pretrained(semantic_model_name)
        # self.semantic_model.eval()
        
        print(f"Loaded {len(self.video_list)} videos for {split_file} split")
        print(f"Found {len(self.action_mapping)} action classes")
        
    def _load_action_mapping(self) -> Dict[str, str]:
        """Load action label to description mapping"""
        mapping_file = self.data_root / "havid_description.txt"
        action_mapping = {}
        
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '"' in line:
                    # Parse format: label "description"
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
        video_id = self.video_list[idx].split(".")[0]
        
        # Load visual features
        feature_path = self.data_root / "videomae_features" / "lh_v0" / f"{video_id}.npy"
        visual_features_np = np.load(feature_path)  # Load as numpy array
        # Transpose from (feature_dim, seq_len) to (seq_len, feature_dim)
        visual_features_np = visual_features_np.T
        visual_features = torch.from_numpy(visual_features_np).float()  # Convert to torch tensor
        
        # Load frame-wise annotations
        annotation_path = self.data_root / "groundTruth" / "View0" / "lh_pt" / f"{video_id}.txt"
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
        
        # Handle sequence length
        original_length = len(visual_features)
        
        if len(visual_features) > self.max_frames:
            # Uniformly sample frames
            indices = torch.linspace(0, len(visual_features) - 1, self.max_frames).long()
            visual_features = visual_features[indices]
            frame_labels = [frame_labels[i] for i in indices]
            original_length = self.max_frames
        
        # Create padding mask
        valid_mask = torch.ones(len(visual_features), dtype=torch.float32)
        
        # Pad if necessary
        if len(visual_features) < self.max_frames:
            pad_length = self.max_frames - len(visual_features)
            
            # Pad visual features
            feature_dim = visual_features.shape[-1]
            padding = torch.zeros(pad_length, feature_dim)
            visual_features = torch.cat([visual_features, padding], dim=0)
            
            # Pad labels (repeat last label)
            last_label = frame_labels[-1] if frame_labels else 'null'
            frame_labels.extend([last_label] * pad_length)
            
            # Update mask
            padded_mask = torch.zeros(self.max_frames, dtype=torch.float32)
            padded_mask[:original_length] = 1.0
            valid_mask = padded_mask
        
        # Convert labels to indices
        action_indices = [self.label_to_idx.get(label, self.label_to_idx['null']) for label in frame_labels]
        action_indices = torch.tensor(action_indices, dtype=torch.long)
        
        # Extract semantic features
        semantic_features = self._extract_semantic_features(frame_labels)
        
        return {
            'visual_features': visual_features,  # [max_frames, visual_dim]
            'semantic_features': torch.stack(semantic_features),  # [max_frames, semantic_dim]
            'action_indices': action_indices,  # [max_frames]
            'valid_mask': valid_mask,  # [max_frames]
            'video_id': video_id,
            'original_length': original_length,
            'frame_labels': frame_labels  # List of string labels
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
        """Get semantic features for labels using precomputed cache if available."""
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
                    # Fallback: unknowns map to a small zero vector of same dim
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
                    description,
                    return_tensors='pt',
                    max_length=64,
                    truncation=True,
                    padding=True
                )
                outputs = self.semantic_model(**inputs)
                semantic_feat = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                semantic_features.append(semantic_feat)
        return semantic_features

class HAVIDSemanticAlignmentTrainer:
    """Training pipeline for HAVID TCN-based semantic alignment"""
    
    def __init__(
        self,
        model: SemanticFeatureAlignmentModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        loss_type: str = 'cosine'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = FrameWiseAlignmentLoss(loss_type=loss_type)
        
        # Optimizer with different learning rates for TCN and projector
        tcn_params = list(self.model.tcn.parameters())
        projector_params = list(self.model.visual_projector.parameters()) + \
                          list(self.model.semantic_projector.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': tcn_params, 'lr': learning_rate},
            {'params': projector_params, 'lr': learning_rate * 2}
        ], weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
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
            action_indices = batch['action_indices'].to(self.device)
            
            # Forward pass
            outputs = self.model(visual_feat)
            predicted_semantic = outputs['aligned_features']
            
            # Compute loss with masking (includes null and wrong actions)
            loss = self.criterion(
                predicted_semantic, 
                semantic_feat, 
                valid_mask
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                action_indices = batch['action_indices'].to(self.device)
                
                outputs = self.model(visual_feat)
                predicted_semantic = outputs['aligned_features']
                
                loss = self.criterion(
                    predicted_semantic, 
                    semantic_feat, 
                    valid_mask
                )
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, num_epochs: int, save_dir: str = './havid_checkpoints'):
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved with val_loss: {val_loss:.6f}")
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

class HAVIDEvaluator:
    """Evaluator for HAVID semantic alignment model"""
    
    def __init__(self, model: SemanticFeatureAlignmentModel, device: torch.device, action_mapping: Dict[str, str], label_to_idx: Dict[str, int]):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.action_mapping = action_mapping
        self.label_to_idx = label_to_idx
        
        # Store 'w' action index for detailed analysis
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
                
                # Evaluate all valid frames (including null and wrong actions)
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
                    
                    # Track by action type for detailed analysis
                    cos_sims_np = cos_sims.cpu().numpy()
                    action_valid_np = action_valid.cpu().numpy()
                    
                    # Assuming index 0 = null, index for 'w' 
                    null_mask = action_valid_np == 0
                    # Find 'w' index from action mapping
                    w_idx = getattr(self, 'w_action_idx', None)
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
            
            # Compute pairwise similarities
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
    
    def save_enhanced_features_for_segmentation(self, data_loader: DataLoader, save_dir: str):
        """Save ONLY enhanced frame-wise features (aligned features) as .npy for downstream action segmentation"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving enhanced features to {save_dir}...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                visual_feat = batch['visual_features'].to(self.device)
                valid_mask = batch['valid_mask']
                video_ids = batch['video_id']
                original_lengths = batch['original_length']
                
                outputs = self.model(visual_feat, return_intermediate=True)
                
                # Save features for each video in the batch
                for i in range(len(video_ids)):
                    video_id = video_ids[i]
                    orig_len = original_lengths[i].item()
                    
                    # Extract valid features
                    enhanced_feats = outputs['aligned_features'][i, :orig_len].cpu()
                    tcn_feats = outputs['tcn_features'][i, :orig_len].cpu()
                    visual_feats = outputs['visual_features'][i, :orig_len].cpu()
                                        # Save features
                    # save_data = {
                    #     'enhanced_features': enhanced_feats,  # TCN + semantic alignment
                    #     'tcn_features': tcn_feats,           # Just TCN features
                    #     'original_features': visual_feats,   # Original visual features
                    #     'video_id': video_id,
                    #     'length': orig_len
                    # }
                    
                    # save_path = os.path.join(save_dir, f"{video_id}.pt")
                    # torch.save(save_data, save_path)
                    
                    # Save ONLY enhanced features as .npy (transposed)
                    save_path = os.path.join(save_dir, f"{video_id}.npy")
                    np.save(save_path, enhanced_feats.numpy().T)
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        print("Enhanced features saved successfully!")

def create_havid_data_loaders(
    data_root: str = "./data/havid",
    train_split: str = "train_split1_bundle",
    test_split: str = "test_split1_bundle", 
    batch_size: int = 4,
    num_workers: int = 4,
    max_frames: int = 1024,
    downsample_rate: int = 1,
    semantic_embeddings_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create HAVID train and test data loaders"""
    
    train_dataset = HAVIDDataset(
        data_root=data_root,
        split_file=train_split,
        semantic_embeddings_path=semantic_embeddings_path,
        max_frames=max_frames,
        downsample_rate=downsample_rate
    )
    
    test_dataset = HAVIDDataset(
        data_root=data_root,
        split_file=test_split,
        semantic_embeddings_path=semantic_embeddings_path,
        max_frames=max_frames,
        downsample_rate=downsample_rate
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, test_loader


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    visual_features = [item['visual_features'] for item in batch]
    semantic_features = [item['semantic_features'] for item in batch] 
    action_indices = [item['action_indices'] for item in batch]
    valid_masks = [item['valid_mask'] for item in batch]
    frame_labels = [item['frame_labels'] for item in batch]
    video_ids = [item['video_id'] for item in batch]
    original_lengths = [item['original_length'] for item in batch]
    
    # Pad sequences to the same length using pad_sequence
    # All feature dimensions should be consistent now after transposing
    visual_features_padded = pad_sequence(visual_features, batch_first=True, padding_value=0)
    semantic_features_padded = pad_sequence(semantic_features, batch_first=True, padding_value=0)
    action_indices_padded = pad_sequence(action_indices, batch_first=True, padding_value=-1)
    valid_masks_padded = pad_sequence(valid_masks, batch_first=True, padding_value=0)
    
    # For frame_labels, pad with '<pad>' tokens
    max_len = max(len(labels) for labels in frame_labels)
    frame_labels_padded = []
    for labels in frame_labels:
        padded = labels + ['<pad>'] * (max_len - len(labels))
        frame_labels_padded.append(padded)
    
    return {
        'visual_features': visual_features_padded,
        'semantic_features': semantic_features_padded, 
        'action_indices': action_indices_padded,
        'valid_mask': valid_masks_padded,
        'frame_labels': frame_labels_padded,
        'video_id': video_ids,
        'original_length': torch.tensor(original_lengths)
    }

# Main training script for HAVID
if __name__ == "__main__":
    # Configuration
    config = {
        'data_root': '/home/hao/Polyphony/data/havid',
        'train_split': 'splits/View0/rh_pt/train.split1.bundle',
        'test_split': 'splits/View0/rh_pt/test.split1.bundle',
        'save_dir': './havid_checkpoints/MiniLM_v2/rh_v0',
        'save_feature_dir': './havid_enhanced_features/MiniLM_v2/rh_v0',
        'batch_size': 1,  # Small batch size due to long sequences
        'learning_rate': 5e-5,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'visual_dim': 768,  # VideoMAE feature dimension
        'semantic_dim': 384,  # sentence-transformers/all-MiniLM-L6-v2 dimension
        'tcn_hidden_dims': [512, 512, 256],
        'max_frames': 3000,
        'downsample_rate': 1,  # Set to 2 or 4 if sequences are too long
        'loss_type': 'cosine',
        'semantic_embeddings_path': '/home/hao/Polyphony/data/havid/semantic_embeddings/sentence-transformers_all-MiniLM-L6-v2.pt'
    }
    
    print("Creating HAVID data loaders...")
    train_loader, test_loader = create_havid_data_loaders(
        data_root=config['data_root'],
        train_split=config['train_split'],
        test_split=config['test_split'],
        batch_size=config['batch_size'],
        max_frames=config['max_frames'],
        downsample_rate=config['downsample_rate'],
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        semantic_embeddings_path=config['semantic_embeddings_path']
    )
    
    # Check data loading
    print("\nChecking data loading...")
    sample_batch = next(iter(train_loader))
    print(f"Visual features shape: {sample_batch['visual_features'].shape}")
    print(f"Semantic features shape: {sample_batch['semantic_features'].shape}")
    print(f"Action indices shape: {sample_batch['action_indices'].shape}")
    print(f"Sample labels: {sample_batch['frame_labels'][0][:10]}")  # First 10 labels
    
    # Initialize model
    print("\nInitializing model...")
    model = SemanticFeatureAlignmentModel(
        visual_dim=config['visual_dim'],
        semantic_dim=config['semantic_dim'],
        tcn_hidden_dims=config['tcn_hidden_dims']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = HAVIDSemanticAlignmentTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test as validation
        device=torch.device(config['device']),
        learning_rate=config['learning_rate'],
        loss_type=config['loss_type']
    )
    
    # Train the model
    print(f"\nStarting training on {config['device']}...")
    trainer.train(num_epochs=config['num_epochs'], save_dir=config['save_dir'])
    
    # Load best model and evaluate
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(config['save_dir']+'/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get action mapping and label indices from dataset
    action_mapping = train_loader.dataset.action_mapping
    label_to_idx = train_loader.dataset.label_to_idx
    
    # Initialize evaluator
    evaluator = HAVIDEvaluator(model, torch.device(config['device']), action_mapping, label_to_idx)
    
    # Evaluate alignment quality
    print("\nEvaluating alignment quality...")
    eval_results = evaluator.evaluate_alignment_quality(test_loader)
    
    print("\nAlignment Quality Results (includes null and wrong actions):")
    for metric, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.6f}")
        else:
            print(f"  {metric}: {value}")
    
    # Save enhanced features for action segmentation
    print("\nSaving enhanced features for downstream action segmentation...")
    evaluator.save_enhanced_features_for_segmentation(test_loader, config['save_feature_dir'])
    evaluator.save_enhanced_features_for_segmentation(train_loader, config['save_feature_dir'])
    
    print("\nTraining and feature enhancement completed!")
    print("Enhanced features saved in './havid_enhanced_features' for downstream action segmentation.")
    print("Note: All action types (including null transitions and wrong actions) are modeled.")