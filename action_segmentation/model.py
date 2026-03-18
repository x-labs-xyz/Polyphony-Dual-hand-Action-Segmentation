"""
Dual-hand action segmentation model with diffusion-based decoder
Includes shared encoder, hand-specific decoders, and feature fusion
"""

import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Tuple, Optional, List


# ============================================================================
# Diffusion Utilities
# ============================================================================

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Build sinusoidal timestep embeddings for diffusion
    From "Denoising Diffusion Probabilistic Models"
    
    Args:
        timesteps: Batch of timestep indices [batch_size]
        embedding_dim: Embedding dimension
        
    Returns:
        Timestep embeddings [batch_size, embedding_dim]
    """
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # Zero pad if odd
        emb = F.pad(emb, (0, 1, 0, 0))
    
    return emb


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function"""
    return x * torch.sigmoid(x)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """Extract appropriate t index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule for diffusion beta values
    From "Improved Denoising Diffusion Probabilistic Models"
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def normalize(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Normalize from [0,1] to [-scale, scale]"""
    return (x * 2 - 1.) * scale


def denormalize(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Denormalize from [-scale, scale] to [0,1]"""
    return ((x / scale) + 1) / 2


# ============================================================================
# Feature Fusion Module
# ============================================================================

class HandFeatureFusion(nn.Module):
    """
    Fusion module for combining left and right hand features
    Each hand gets context from the other hand via cross-attention
    """
    
    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim: Feature dimension
        """
        super().__init__()
        
        # Left hand fusion: combines LH + RH → fused LH
        self.fusion_lh = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, 1)
        )
        
        # Right hand fusion: combines RH + LH → fused RH
        self.fusion_rh = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, 1)
        )
    
    def forward(self, feats_lh: torch.Tensor, feats_rh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse features from both hands
        
        Args:
            feats_lh: Left hand features [B, D, T]
            feats_rh: Right hand features [B, D, T]
            
        Returns:
            Fused left and right hand features
        """
        # Concatenate both hand features
        combined = torch.cat([feats_lh, feats_rh], dim=1)
        
        # Fuse with residual connections
        fused_lh = self.fusion_lh(combined) + feats_lh
        fused_rh = self.fusion_rh(combined) + feats_rh
        
        return fused_lh, fused_rh


# ============================================================================
# Encoder Model (imported from base - placeholder for reference)
# ============================================================================

class EncoderModel(nn.Module):
    """
    NOTE: This is a placeholder. In actual use, import from your base model:
    from your_base_model import EncoderModel
    
    The encoder processes video features and outputs:
    - Action predictions
    - Intermediate features for the decoder
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        # Implement your encoder architecture here
        # Typically: temporal convolutions + attention
        raise NotImplementedError("Import EncoderModel from your base implementation")
    
    def forward(self, x, get_features=False):
        raise NotImplementedError("Import EncoderModel from your base implementation")


class DecoderModel(nn.Module):
    """
    NOTE: This is a placeholder. In actual use, import from your base model:
    from your_base_model import DecoderModel
    
    The decoder refines predictions using diffusion:
    - Takes backbone features + timestep + noisy predictions
    - Outputs denoised action predictions
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        # Implement your decoder architecture here
        # Typically: U-Net style with timestep conditioning
        raise NotImplementedError("Import DecoderModel from your base implementation")
    
    def forward(self, backbone_feats, t, x):
        raise NotImplementedError("Import DecoderModel from your base implementation")


# ============================================================================
# Dual-Hand Diffusion Model
# ============================================================================

class DualHandASDiffusionModel(nn.Module):
    """
    Dual-hand action segmentation model with diffusion-based decoder
    
    Architecture:
    - Shared encoder for both hands
    - Feature fusion module
    - Separate decoders for left and right hands
    - DDIM sampling for inference
    """
    
    def __init__(self, encoder_params: Dict, decoder_params: Dict, 
                 diffusion_params: Dict, num_classes: int, device: torch.device):
        """
        Args:
            encoder_params: Encoder configuration
            decoder_params: Decoder configuration
            diffusion_params: Diffusion hyperparameters
            num_classes: Number of action classes
            device: Device (cpu or cuda)
        """
        super().__init__()
        
        self.device = device
        self.num_classes = num_classes
        
        # ===== Diffusion Parameters =====
        timesteps = diffusion_params['timesteps']
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = diffusion_params['sampling_timesteps']
        assert self.sampling_timesteps <= timesteps
        
        self.ddim_sampling_eta = diffusion_params['ddim_sampling_eta']
        self.scale = diffusion_params['snr_scale']
        
        # Register buffers (not trainable)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        # ===== Model Configuration =====
        self.detach_decoder = diffusion_params['detach_decoder']
        self.cond_types = diffusion_params['cond_types']
        self.use_instance_norm = encoder_params['use_instance_norm']
        
        if self.use_instance_norm:
            self.ins_norm = nn.InstanceNorm1d(encoder_params['input_dim'], track_running_stats=False)
        
        # ===== Calculate Decoder Input Dimension =====
        decoder_input_dim = len([i for i in encoder_params['feature_layer_indices'] if i not in [-1, -2]]) * encoder_params['num_f_maps']
        
        if -1 in encoder_params['feature_layer_indices']:  # Video features
            decoder_input_dim += encoder_params['input_dim']
        if -2 in encoder_params['feature_layer_indices']:  # Encoder predictions
            decoder_input_dim += self.num_classes
        
        # ===== Create Models =====
        # Prepare encoder parameters
        encoder_params_clean = dict(encoder_params)
        encoder_params_clean['num_classes'] = num_classes
        encoder_params_clean.pop('use_instance_norm', None)
        
        # Prepare decoder parameters
        decoder_params_clean = dict(decoder_params)
        decoder_params_clean['input_dim'] = decoder_input_dim
        decoder_params_clean['num_classes'] = num_classes
        
        # NOTE: Import these from your base model implementation
        # from your_base_model import EncoderModel, DecoderModel
        self.encoder = EncoderModel(**encoder_params_clean)
        
        # Feature fusion module
        self.hand_fusion = HandFeatureFusion(decoder_input_dim)
        
        # Separate decoders for each hand
        self.decoder_left = DecoderModel(**decoder_params_clean)
        self.decoder_right = DecoderModel(**decoder_params_clean)
    
    # ===== Diffusion Core Functions =====
    
    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Predict noise from x_t and x_0"""
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: add noise to x_start"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def model_predictions(self, backbone_feats_lh: torch.Tensor, backbone_feats_rh: torch.Tensor,
                         x_left: torch.Tensor, x_right: torch.Tensor, t: torch.Tensor) -> Tuple:
        """
        Make predictions for both hands
        
        Args:
            backbone_feats_lh: Left hand features
            backbone_feats_rh: Right hand features
            x_left: Noisy left hand predictions
            x_right: Noisy right hand predictions
            t: Timestep
            
        Returns:
            Tuple of (pred_noise_left, x_left_start, pred_noise_right, x_right_start)
        """
        # Left hand
        x_left_m = torch.clamp(x_left, min=-self.scale, max=self.scale)
        x_left_m = denormalize(x_left_m, self.scale)
        assert x_left_m.max() <= 1 and x_left_m.min() >= 0
        
        x_left_start = self.decoder_left(backbone_feats_lh, t, x_left_m.float())
        x_left_start = F.softmax(x_left_start, 1)
        x_left_start = normalize(x_left_start, self.scale)
        x_left_start = torch.clamp(x_left_start, min=-self.scale, max=self.scale)
        
        pred_noise_left = self.predict_noise_from_start(x_left, t, x_left_start)
        
        # Right hand
        x_right_m = torch.clamp(x_right, min=-self.scale, max=self.scale)
        x_right_m = denormalize(x_right_m, self.scale)
        assert x_right_m.max() <= 1 and x_right_m.min() >= 0
        
        x_right_start = self.decoder_right(backbone_feats_rh, t, x_right_m.float())
        x_right_start = F.softmax(x_right_start, 1)
        x_right_start = normalize(x_right_start, self.scale)
        x_right_start = torch.clamp(x_right_start, min=-self.scale, max=self.scale)
        
        pred_noise_right = self.predict_noise_from_start(x_right, t, x_right_start)
        
        return pred_noise_left, x_left_start, pred_noise_right, x_right_start
    
    def prepare_targets(self, event_gt_left: torch.Tensor, event_gt_right: torch.Tensor) -> Tuple:
        """
        Prepare diffusion targets for training
        
        Args:
            event_gt_left: Ground truth left hand labels [B, C, T]
            event_gt_right: Ground truth right hand labels [B, C, T]
            
        Returns:
            Tuple of (event_diffused_left, noise_left, event_diffused_right, noise_right, t)
        """
        assert event_gt_left.max() <= 1 and event_gt_left.min() >= 0
        assert event_gt_right.max() <= 1 and event_gt_right.min() >= 0
        
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        
        # Left hand
        noise_left = torch.randn(size=event_gt_left.shape, device=self.device)
        x_start_left = (event_gt_left * 2. - 1.) * self.scale
        x_left = self.q_sample(x_start=x_start_left, t=t, noise=noise_left)
        x_left = torch.clamp(x_left, min=-self.scale, max=self.scale)
        event_diffused_left = ((x_left / self.scale) + 1) / 2.
        
        # Right hand
        noise_right = torch.randn(size=event_gt_right.shape, device=self.device)
        x_start_right = (event_gt_right * 2. - 1.) * self.scale
        x_right = self.q_sample(x_start=x_start_right, t=t, noise=noise_right)
        x_right = torch.clamp(x_right, min=-self.scale, max=self.scale)
        event_diffused_right = ((x_right / self.scale) + 1) / 2.
        
        return event_diffused_left, noise_left, event_diffused_right, noise_right, t
    
    # ===== Forward Pass =====
    
    def forward(self, backbone_feats: torch.Tensor, t: torch.Tensor,
                event_diffused_left: torch.Tensor, event_diffused_right: torch.Tensor,
                event_gt_left: Optional[torch.Tensor] = None, event_gt_right: Optional[torch.Tensor] = None,
                boundary_gt_left: Optional[torch.Tensor] = None, boundary_gt_right: Optional[torch.Tensor] = None,
                hand: str = 'both') -> Dict[str, torch.Tensor]:
        """
        Forward pass with conditioning
        
        Args:
            backbone_feats: Backbone features
            t: Timestep
            event_diffused_left: Diffused left hand events
            event_diffused_right: Diffused right hand events
            event_gt_left: Ground truth left events (for conditioning)
            event_gt_right: Ground truth right events (for conditioning)
            boundary_gt_left: Ground truth left boundaries (for conditioning)
            boundary_gt_right: Ground truth right boundaries (for conditioning)
            hand: 'left', 'right', or 'both'
            
        Returns:
            Dictionary with 'left' and/or 'right' predictions
        """
        if self.detach_decoder:
            backbone_feats = backbone_feats.detach()
        
        assert event_diffused_left.max() <= 1 and event_diffused_left.min() >= 0
        assert event_diffused_right.max() <= 1 and event_diffused_right.min() >= 0
        
        # Random conditioning strategy
        cond_type = random.choice(self.cond_types)
        
        results = {}
        
        if hand in ['left', 'both']:
            backbone_feats_cond = self._apply_conditioning(
                backbone_feats, cond_type, event_gt_left, boundary_gt_left
            )
            results['left'] = self.decoder_left(backbone_feats_cond, t, event_diffused_left.float())
        
        if hand in ['right', 'both']:
            backbone_feats_cond = self._apply_conditioning(
                backbone_feats, cond_type, event_gt_right, boundary_gt_right
            )
            results['right'] = self.decoder_right(backbone_feats_cond, t, event_diffused_right.float())
        
        return results
    
    def _apply_conditioning(self, backbone_feats: torch.Tensor, cond_type: str,
                           event_gt: Optional[torch.Tensor], boundary_gt: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply conditioning strategy to features"""
        if cond_type == 'full':
            return backbone_feats
        
        elif cond_type == 'zero':
            return torch.zeros_like(backbone_feats)
        
        elif cond_type == 'boundary05-':
            feature_mask = (boundary_gt < 0.5).float()
            return feature_mask * backbone_feats
        
        elif cond_type == 'boundary03-':
            feature_mask = (boundary_gt < 0.3).float()
            return feature_mask * backbone_feats
        
        elif cond_type == 'segment=1':
            event_gt_idx = torch.argmax(event_gt, dim=1, keepdim=True).long()
            events = torch.unique(event_gt_idx)
            random_event = np.random.choice(events.cpu().numpy())
            feature_mask = (event_gt_idx != random_event).float()
            return feature_mask * backbone_feats
        
        elif cond_type == 'segment=2':
            event_gt_idx = torch.argmax(event_gt, dim=1, keepdim=True).long()
            events = torch.unique(event_gt_idx)
            random_event_1 = np.random.choice(events.cpu().numpy())
            random_event_2 = np.random.choice(events.cpu().numpy())
            feature_mask = (event_gt_idx != random_event_1).float() * (event_gt_idx != random_event_2).float()
            return feature_mask * backbone_feats
        
        else:
            raise ValueError(f'Invalid conditioning type: {cond_type}')
    
    # ===== Training Loss Computation =====
    
    def get_training_loss(self, video_feats_lh: torch.Tensor, video_feats_rh: torch.Tensor,
                         event_gt_left: torch.Tensor, boundary_gt_left: torch.Tensor,
                         event_gt_right: torch.Tensor, boundary_gt_right: torch.Tensor,
                         encoder_ce_criterion: nn.Module, encoder_mse_criterion: nn.Module,
                         encoder_boundary_criterion: nn.Module,
                         decoder_ce_criterion_left: nn.Module, decoder_ce_criterion_right: nn.Module,
                         decoder_mse_criterion: nn.Module, decoder_boundary_criterion: nn.Module,
                         soft_label: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for both hands
        
        Returns:
            Dictionary of losses for each component
        """
        # Instance normalization
        if self.use_instance_norm:
            video_feats_lh = self.ins_norm(video_feats_lh)
            video_feats_rh = self.ins_norm(video_feats_rh)
        
        # Encoder forward for both hands
        encoder_out_lh, backbone_feats_lh = self.encoder(video_feats_lh, get_features=True)
        encoder_out_rh, backbone_feats_rh = self.encoder(video_feats_rh, get_features=True)
        
        # Feature fusion
        backbone_feats_lh, backbone_feats_rh = self.hand_fusion(backbone_feats_lh, backbone_feats_rh)
        
        # Encoder losses
        encoder_loss_dict_lh = self._compute_encoder_losses(
            encoder_out_lh, event_gt_left, encoder_ce_criterion,
            encoder_mse_criterion, encoder_boundary_criterion, soft_label
        )
        encoder_loss_dict_rh = self._compute_encoder_losses(
            encoder_out_rh, event_gt_right, encoder_ce_criterion,
            encoder_mse_criterion, encoder_boundary_criterion, soft_label
        )
        
        # Decoder losses
        left_losses = self._compute_decoder_losses(
            backbone_feats_lh, event_gt_left, boundary_gt_left, 'left',
            decoder_ce_criterion_left, decoder_mse_criterion, decoder_boundary_criterion, soft_label
        )
        right_losses = self._compute_decoder_losses(
            backbone_feats_rh, event_gt_right, boundary_gt_right, 'right',
            decoder_ce_criterion_right, decoder_mse_criterion, decoder_boundary_criterion, soft_label
        )
        
        # Combine losses
        loss_dict = {}
        for k, v in encoder_loss_dict_lh.items():
            loss_dict[f'encoder_lh_{k}'] = v
        for k, v in encoder_loss_dict_rh.items():
            loss_dict[f'encoder_rh_{k}'] = v
        for k, v in left_losses.items():
            loss_dict[f'decoder_left_{k}'] = v
        for k, v in right_losses.items():
            loss_dict[f'decoder_right_{k}'] = v
        
        return loss_dict
    
    def _compute_encoder_losses(self, encoder_out: torch.Tensor, event_gt: torch.Tensor,
                                ce_criterion: nn.Module, mse_criterion: nn.Module,
                                boundary_criterion: nn.Module, soft_label: Optional[float]) -> Dict:
        """Compute encoder losses"""
        if soft_label is None:
            encoder_ce_loss = ce_criterion(
                encoder_out.transpose(2, 1).contiguous().view(-1, self.num_classes),
                torch.argmax(event_gt, dim=1).view(-1).long()
            )
        else:
            soft_event_gt = torch.clone(event_gt).float().cpu().numpy()
            for i in range(soft_event_gt.shape[1]):
                soft_event_gt[0, i] = gaussian_filter1d(soft_event_gt[0, i], soft_label)
            soft_event_gt = torch.from_numpy(soft_event_gt).to(self.device)
            encoder_ce_loss = -soft_event_gt * F.log_softmax(encoder_out, 1)
            encoder_ce_loss = encoder_ce_loss.sum(0).sum(0)
        
        encoder_mse_loss = torch.clamp(mse_criterion(
            F.log_softmax(encoder_out[:, :, 1:], dim=1),
            F.log_softmax(encoder_out.detach()[:, :, :-1], dim=1)),
            min=0, max=16)
        
        encoder_boundary_loss = torch.tensor(0).to(self.device)
        
        return {
            'ce_loss': encoder_ce_loss.mean(),
            'mse_loss': encoder_mse_loss.mean(),
            'boundary_loss': encoder_boundary_loss
        }
    
    def _compute_decoder_losses(self, backbone_feats: torch.Tensor, event_gt: torch.Tensor,
                                boundary_gt: torch.Tensor, hand: str,
                                ce_criterion: nn.Module, mse_criterion: nn.Module,
                                boundary_criterion: nn.Module, soft_label: Optional[float]) -> Dict:
        """Compute decoder losses for specific hand"""
        # Prepare diffusion targets
        if hand == 'left':
            event_diffused, noise, _, _, t = self.prepare_targets(event_gt, event_gt)
            event_out = self.forward(backbone_feats, t, event_diffused, event_diffused,
                                   event_gt, event_gt, boundary_gt, boundary_gt, hand='left')['left']
        else:  # right
            _, _, event_diffused, noise, t = self.prepare_targets(event_gt, event_gt)
            event_out = self.forward(backbone_feats, t, event_diffused, event_diffused,
                                   event_gt, event_gt, boundary_gt, boundary_gt, hand='right')['right']
        
        # Compute boundary from predictions
        decoder_boundary = 1 - torch.einsum('bicl,bcjl->bijl',
            F.softmax(event_out[:, None, :, 1:], 2),
            F.softmax(event_out[:, :, None, :-1].detach(), 1)
        ).squeeze(1)
        
        # CE loss
        if soft_label is None:
            decoder_ce_loss = ce_criterion(
                event_out.transpose(2, 1).contiguous().view(-1, self.num_classes),
                torch.argmax(event_gt, dim=1).view(-1).long()
            )
        else:
            soft_event_gt = torch.clone(event_gt).float().cpu().numpy()
            for i in range(soft_event_gt.shape[1]):
                soft_event_gt[0, i] = gaussian_filter1d(soft_event_gt[0, i], soft_label)
            soft_event_gt = torch.from_numpy(soft_event_gt).to(self.device)
            decoder_ce_loss = -soft_event_gt * F.log_softmax(event_out, 1)
            decoder_ce_loss = decoder_ce_loss.sum(0).sum(0)
        
        # MSE loss (temporal smoothness)
        decoder_mse_loss = torch.clamp(mse_criterion(
            F.log_softmax(event_out[:, :, 1:], dim=1),
            F.log_softmax(event_out.detach()[:, :, :-1], dim=1)),
            min=0, max=16)
        
        # Boundary loss
        decoder_boundary_loss = boundary_criterion(
            decoder_boundary[:, :, :-1], boundary_gt[:, :, 1:]
        )
        
        return {
            'ce_loss': decoder_ce_loss.mean(),
            'mse_loss': decoder_mse_loss.mean(),
            'boundary_loss': decoder_boundary_loss.mean()
        }
    
    # ===== DDIM Sampling for Inference =====
    
    @torch.no_grad()
    def ddim_sample(self, video_feats_lh: torch.Tensor, video_feats_rh: torch.Tensor,
                    seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM sampling for inference
        
        Args:
            video_feats_lh: Left hand video features
            video_feats_rh: Right hand video features
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (left_hand_predictions, right_hand_predictions)
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        batch, _, length = video_feats_lh.shape
        
        # Instance normalization
        if self.use_instance_norm:
            video_feats_lh = self.ins_norm(video_feats_lh)
            video_feats_rh = self.ins_norm(video_feats_rh)
        
        # Encoder forward
        _, backbone_feats_lh = self.encoder(video_feats_lh, get_features=True)
        _, backbone_feats_rh = self.encoder(video_feats_rh, get_features=True)
        
        # Feature fusion
        backbone_feats_lh, backbone_feats_rh = self.hand_fusion(backbone_feats_lh, backbone_feats_rh)
        
        # Initialize noise
        x_left = torch.randn((batch, self.num_classes, length), device=self.device)
        x_right = torch.randn((batch, self.num_classes, length), device=self.device)
        
        # DDIM timesteps
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # Denoise
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            
            _, x_left_start, _, x_right_start = self.model_predictions(
                backbone_feats_lh, backbone_feats_rh, x_left, x_right, time_cond
            )
            
            if time_next < 0:
                x_left = x_left_start
                x_right = x_right_start
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x_left)
            x_left = x_left_start * alpha_next.sqrt() + c * (x_left - alpha.sqrt() * x_left_start) / (1 - alpha).sqrt() + sigma * noise
            
            noise = torch.randn_like(x_right)
            x_right = x_right_start * alpha_next.sqrt() + c * (x_right - alpha.sqrt() * x_right_start) / (1 - alpha).sqrt() + sigma * noise
        
        # Denormalize
        x_left = torch.clamp(x_left, min=-self.scale, max=self.scale)
        x_left = denormalize(x_left, self.scale)
        
        x_right = torch.clamp(x_right, min=-self.scale, max=self.scale)
        x_right = denormalize(x_right, self.scale)
        
        return x_left, x_right

