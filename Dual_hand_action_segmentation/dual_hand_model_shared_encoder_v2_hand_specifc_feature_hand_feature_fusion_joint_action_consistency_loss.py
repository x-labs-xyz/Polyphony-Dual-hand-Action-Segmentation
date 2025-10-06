import copy
import math
import torch
import random
import numpy as np
import time as Time
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

# Import the original model components
from model import (
    get_timestep_embedding, swish, extract, cosine_beta_schedule, 
    normalize, denormalize, EncoderModel, DecoderModel
)

class HandFeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fusion_lh = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, 1)
        )
        self.fusion_rh = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, 1)
        )
        
    def forward(self, feats_lh, feats_rh):
        combined = torch.cat([feats_lh, feats_rh], dim=1)
        fused_lh = self.fusion_lh(combined) + feats_lh
        fused_rh = self.fusion_rh(combined) + feats_rh
        return fused_lh, fused_rh

class DualHandASDiffusionModel(nn.Module):
    """
    Dual-hand action segmentation model with shared encoder and separate decoders.
    Enhanced with support for hand-specific loss criteria.
    """
    def __init__(self, encoder_params, decoder_params, diffusion_params, num_classes, device):
        super(DualHandASDiffusionModel, self).__init__()

        self.device = device
        self.num_classes = num_classes

        # Diffusion parameters (same as original)
        timesteps = diffusion_params['timesteps']
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = diffusion_params['sampling_timesteps']
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = diffusion_params['ddim_sampling_eta']
        self.scale = diffusion_params['snr_scale']

        # Register diffusion buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Model configuration
        self.detach_decoder = diffusion_params['detach_decoder']
        self.cond_types = diffusion_params['cond_types']
        self.use_instance_norm = encoder_params['use_instance_norm']
        
        if self.use_instance_norm:
            self.ins_norm = nn.InstanceNorm1d(encoder_params['input_dim'], track_running_stats=False)

        # Calculate decoder input dimension
        decoder_input_dim = len([i for i in encoder_params['feature_layer_indices'] if i not in [-1, -2]]) * encoder_params['num_f_maps']
        if -1 in encoder_params['feature_layer_indices']:
            decoder_input_dim += encoder_params['input_dim']
        if -2 in encoder_params['feature_layer_indices']:
            decoder_input_dim += self.num_classes

        # Create models
        encoder_params_clean = dict(encoder_params)
        encoder_params_clean['num_classes'] = num_classes
        encoder_params_clean.pop('use_instance_norm', None)
        
        decoder_params_clean = dict(decoder_params)
        decoder_params_clean['input_dim'] = decoder_input_dim
        decoder_params_clean['num_classes'] = num_classes

        # Shared encoder for both hands
        self.encoder = EncoderModel(**encoder_params_clean)

        self.hand_fusion = HandFeatureFusion(decoder_input_dim)
        
        # Separate decoders for left and right hands
        self.decoder_left = DecoderModel(**decoder_params_clean)
        self.decoder_right = DecoderModel(**decoder_params_clean)
        
        self.action_compatibility = nn.Parameter(
                torch.eye(num_classes) + torch.randn(num_classes, num_classes) * 0.01
            )

    def compute_joint_consistency_loss(self, pred_lh, pred_rh):
        """Learnable compatibility between hand actions"""
        prob_lh = F.softmax(pred_lh, dim=1)
        prob_rh = F.softmax(pred_rh, dim=1)
    
        # Compute compatibility score
        compatibility = torch.einsum('bct,cd,bdt->bt',
                                    prob_lh, 
                                    torch.sigmoid(self.action_compatibility),
                                    prob_rh)
    
        return -compatibility.mean()

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def model_predictions(self, backbone_feats_lh, backbone_feats_rh, x_left, x_right, t):
        """Make predictions for both hands with separate backbone features"""
        # Process left hand
        x_left_m = torch.clamp(x_left, min=-1 * self.scale, max=self.scale)
        x_left_m = denormalize(x_left_m, self.scale)
        assert(x_left_m.max() <= 1 and x_left_m.min() >= 0)
        
        x_left_start = self.decoder_left(backbone_feats_lh, t, x_left_m.float())
        x_left_start = F.softmax(x_left_start, 1)
        x_left_start = normalize(x_left_start, self.scale)
        x_left_start = torch.clamp(x_left_start, min=-1 * self.scale, max=self.scale)
        
        pred_noise_left = self.predict_noise_from_start(x_left, t, x_left_start)
        
        # Process right hand
        x_right_m = torch.clamp(x_right, min=-1 * self.scale, max=self.scale)
        x_right_m = denormalize(x_right_m, self.scale)
        assert(x_right_m.max() <= 1 and x_right_m.min() >= 0)
        
        x_right_start = self.decoder_right(backbone_feats_rh, t, x_right_m.float())
        x_right_start = F.softmax(x_right_start, 1)
        x_right_start = normalize(x_right_start, self.scale)
        x_right_start = torch.clamp(x_right_start, min=-1 * self.scale, max=self.scale)
        
        pred_noise_right = self.predict_noise_from_start(x_right, t, x_right_start)
        
        return pred_noise_left, x_left_start, pred_noise_right, x_right_start

    def prepare_targets(self, event_gt_left, event_gt_right):
        """Prepare diffusion targets for both hands"""
        assert(event_gt_left.max() <= 1 and event_gt_left.min() >= 0)
        assert(event_gt_right.max() <= 1 and event_gt_right.min() >= 0)

        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()

        # Left hand
        noise_left = torch.randn(size=event_gt_left.shape, device=self.device)
        x_start_left = (event_gt_left * 2. - 1.) * self.scale
        x_left = self.q_sample(x_start=x_start_left, t=t, noise=noise_left)
        x_left = torch.clamp(x_left, min=-1 * self.scale, max=self.scale)
        event_diffused_left = ((x_left / self.scale) + 1) / 2.

        # Right hand
        noise_right = torch.randn(size=event_gt_right.shape, device=self.device)
        x_start_right = (event_gt_right * 2. - 1.) * self.scale
        x_right = self.q_sample(x_start=x_start_right, t=t, noise=noise_right)
        x_right = torch.clamp(x_right, min=-1 * self.scale, max=self.scale)
        event_diffused_right = ((x_right / self.scale) + 1) / 2.

        return (event_diffused_left, noise_left, event_diffused_right, noise_right, t)

    def forward(self, backbone_feats, t, event_diffused_left, event_diffused_right, 
                event_gt_left=None, event_gt_right=None, 
                boundary_gt_left=None, boundary_gt_right=None, hand='both'):
        """Forward pass for training with conditioning"""
        
        if self.detach_decoder:
            backbone_feats = backbone_feats.detach()

        assert(event_diffused_left.max() <= 1 and event_diffused_left.min() >= 0)
        assert(event_diffused_right.max() <= 1 and event_diffused_right.min() >= 0)
    
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

    def _apply_conditioning(self, backbone_feats, cond_type, event_gt, boundary_gt):
        """Apply conditioning strategy to backbone features"""
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
            raise Exception(f'Invalid Cond Type: {cond_type}')

    def get_training_loss(self, video_feats_lh, video_feats_rh,
                         event_gt_left, boundary_gt_left,
                         event_gt_right, boundary_gt_right,
                         encoder_ce_criterion, encoder_mse_criterion, encoder_boundary_criterion,
                         decoder_ce_criterion_left, decoder_ce_criterion_right,
                         decoder_mse_criterion, decoder_boundary_criterion, 
                         soft_label=None):
        """Compute training losses for both hands with hand-specific features and criteria"""
        
        if self.use_instance_norm:
            video_feats_lh = self.ins_norm(video_feats_lh)
            video_feats_rh = self.ins_norm(video_feats_rh)

        # Separate encoder forward passes for each hand
        encoder_out_lh, backbone_feats_lh = self.encoder(video_feats_lh, get_features=True)
        encoder_out_rh, backbone_feats_rh = self.encoder(video_feats_rh, get_features=True)

        backbone_feats_lh, backbone_feats_rh = self.hand_fusion(
            backbone_feats_lh, backbone_feats_rh
        )

        # Encoder losses for each hand separately
        encoder_loss_dict_lh = self._compute_encoder_losses(
            encoder_out_lh, event_gt_left, encoder_ce_criterion, 
            encoder_mse_criterion, encoder_boundary_criterion, soft_label
        )
        encoder_loss_dict_rh = self._compute_encoder_losses(
            encoder_out_rh, event_gt_right, encoder_ce_criterion, 
            encoder_mse_criterion, encoder_boundary_criterion, soft_label
        )

        # Decoder losses for both hands
        decoder_loss_dict = {}
        
        # Left hand decoder losses
        left_losses = self._compute_decoder_losses(
            backbone_feats_lh, event_gt_left, boundary_gt_left, 'left',
            decoder_ce_criterion_left, decoder_mse_criterion, decoder_boundary_criterion, soft_label
        )
        for k, v in left_losses.items():
            decoder_loss_dict[f'decoder_left_{k}'] = v

        # Right hand decoder losses  
        right_losses = self._compute_decoder_losses(
            backbone_feats_rh, event_gt_right, boundary_gt_right, 'right',
            decoder_ce_criterion_right, decoder_mse_criterion, decoder_boundary_criterion, soft_label
        )
        for k, v in right_losses.items():
            decoder_loss_dict[f'decoder_right_{k}'] = v

        # Joint action consistency loss
        event_diffused_left, noise_left, event_diffused_right, noise_right, t = self.prepare_targets(
            event_gt_left, event_gt_right
        )

        cond_type = random.choice(self.cond_types)
        backbone_feats_cond_lh = self._apply_conditioning(
            backbone_feats_lh, cond_type, event_gt_left, boundary_gt_left
        )
        backbone_feats_cond_rh = self._apply_conditioning(
            backbone_feats_rh, cond_type, event_gt_right, boundary_gt_right
        )

        pred_lh_for_consistency = self.decoder_left(
            backbone_feats_cond_lh, t, event_diffused_left.float()
        )
        pred_rh_for_consistency = self.decoder_right(
            backbone_feats_cond_rh, t, event_diffused_right.float()
        )

        consistency_loss = self.compute_joint_consistency_loss(
            pred_lh_for_consistency, pred_rh_for_consistency
        )
        decoder_loss_dict['consistency_loss'] = consistency_loss

        # Combine all losses
        loss_dict = {}
        
        # Add encoder losses (average of both hands)
        for k, v in encoder_loss_dict_lh.items():
            loss_dict[f'encoder_lh_{k}'] = v
        for k, v in encoder_loss_dict_rh.items():
            loss_dict[f'encoder_rh_{k}'] = v
            
        # Add decoder losses
        loss_dict.update(decoder_loss_dict)
        
        return loss_dict

    def _compute_encoder_losses(self, encoder_out, event_gt, ce_criterion, mse_criterion, boundary_criterion, soft_label):
        """Compute encoder losses"""
        if soft_label is None:
            encoder_ce_loss = ce_criterion(
                encoder_out.transpose(2, 1).contiguous().view(-1, self.num_classes), 
                torch.argmax(event_gt, dim=1).view(-1).long()
            )
        else:
            soft_event_gt = torch.clone(event_gt).float().cpu().numpy()
            for i in range(soft_event_gt.shape[1]):
                soft_event_gt[0,i] = gaussian_filter1d(soft_event_gt[0,i], soft_label)
            soft_event_gt = torch.from_numpy(soft_event_gt).to(self.device)
            encoder_ce_loss = - soft_event_gt * F.log_softmax(encoder_out, 1)
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

    def _compute_decoder_losses(self, backbone_feats, event_gt, boundary_gt, hand, 
                               ce_criterion, mse_criterion, boundary_criterion, soft_label):
        """Compute decoder losses for specific hand"""
        # Prepare diffusion targets
        if hand == 'left':
            event_diffused, noise, _, _, t = self.prepare_targets(event_gt, event_gt)  # Use same GT twice for single hand
            event_out = self.forward(backbone_feats, t, event_diffused, event_diffused, 
                                   event_gt, event_gt, boundary_gt, boundary_gt, hand='left')['left']
        else:  # right
            _, _, event_diffused, noise, t = self.prepare_targets(event_gt, event_gt)  # Use same GT twice for single hand
            event_out = self.forward(backbone_feats, t, event_diffused, event_diffused,
                                   event_gt, event_gt, boundary_gt, boundary_gt, hand='right')['right']

        # Compute boundary from predictions
        decoder_boundary = 1 - torch.einsum('bicl,bcjl->bijl', 
            F.softmax(event_out[:,None,:,1:], 2), 
            F.softmax(event_out[:,:,None,:-1].detach(), 1)
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
                soft_event_gt[0,i] = gaussian_filter1d(soft_event_gt[0,i], soft_label)
            soft_event_gt = torch.from_numpy(soft_event_gt).to(self.device)
            decoder_ce_loss = - soft_event_gt * F.log_softmax(event_out, 1)
            decoder_ce_loss = decoder_ce_loss.sum(0).sum(0)

        # MSE loss
        decoder_mse_loss = torch.clamp(mse_criterion(
            F.log_softmax(event_out[:, :, 1:], dim=1), 
            F.log_softmax(event_out.detach()[:, :, :-1], dim=1)), 
            min=0, max=16)

        # Boundary loss
        decoder_boundary_loss = boundary_criterion(decoder_boundary, boundary_gt[:,:,1:])
        
        return {
            'ce_loss': decoder_ce_loss.mean(),
            'mse_loss': decoder_mse_loss.mean(),
            'boundary_loss': decoder_boundary_loss.mean()
        }

    @torch.no_grad()
    def ddim_sample(self, video_feats_lh, video_feats_rh, seed=None):
        """DDIM sampling for both hands with separate features"""
        if self.use_instance_norm:
            video_feats_lh = self.ins_norm(video_feats_lh)
            video_feats_rh = self.ins_norm(video_feats_rh)

        encoder_out_lh, backbone_feats_lh = self.encoder(video_feats_lh, get_features=True)
        encoder_out_rh, backbone_feats_rh = self.encoder(video_feats_rh, get_features=True)

        backbone_feats_lh, backbone_feats_rh = self.hand_fusion(
            backbone_feats_lh, backbone_feats_rh
            )

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Sample shape (use LH features' shape)
        shape = (video_feats_lh.shape[0], self.num_classes, video_feats_lh.shape[2])
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        
        # Time steps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Initialize random noise for both hands
        x_left = torch.randn(shape, device=self.device)
        x_right = torch.randn(shape, device=self.device)

        x_left_start = None
        x_right_start = None
        
        for time, time_next in time_pairs:
            time_cond = torch.full((1,), time, device=self.device, dtype=torch.long)

            pred_noise_left, x_left_start, pred_noise_right, x_right_start = self.model_predictions(
                backbone_feats_lh, backbone_feats_rh, x_left, x_right, time_cond
            )
            
            if time_next < 0:
                x_left = x_left_start
                x_right = x_right_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise_left = torch.randn_like(x_left)
            noise_right = torch.randn_like(x_right)

            x_left = x_left_start * alpha_next.sqrt() + c * pred_noise_left + sigma * noise_left
            x_right = x_right_start * alpha_next.sqrt() + c * pred_noise_right + sigma * noise_right

        x_left_return = denormalize(x_left_start, self.scale)  
        x_right_return = denormalize(x_right_start, self.scale)

        if seed is not None:
            t = 1000 * Time.time()
            t = int(t) % 2**16
            random.seed(t)
            torch.manual_seed(t)
            torch.cuda.manual_seed_all(t)

        return x_left_return, x_right_return
