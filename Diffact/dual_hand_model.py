import torch
import torch.nn as nn
import copy
from model import ASDiffusionModel


class DualHandASDiffusionModel(nn.Module):
    def __init__(self, encoder_params, decoder_params, diffusion_params, num_classes, device):
        super(DualHandASDiffusionModel, self).__init__()
        
        self.device = device
        self.num_classes = num_classes
        
        # Create deep copies of parameters to avoid modification issues
        # The ASDiffusionModel constructor modifies the parameter dictionaries
        lh_encoder_params = copy.deepcopy(encoder_params)
        lh_decoder_params = copy.deepcopy(decoder_params)
        lh_diffusion_params = copy.deepcopy(diffusion_params)
        
        rh_encoder_params = copy.deepcopy(encoder_params)
        rh_decoder_params = copy.deepcopy(decoder_params)
        rh_diffusion_params = copy.deepcopy(diffusion_params)
        
        # Create two separate diffusion models for left and right hands
        #**Each `ASDiffusionModel` contains**:
        # 1. **Encoder** (`EncoderModel`):
        #    - Input: Video features [Batch, Features, Time]
        #    - Architecture: MixedConvAttModule (conv + attention layers)
        #    - Output: Action predictions [Batch, Classes, Time]
        # 2. **Decoder** (`DecoderModel`):
        #    - Input: Noisy action sequence + timestep + encoder features
        #    - Architecture: MixedConvAttModuleV2 (conv + cross-attention + time embedding)
        #    - Output: Denoised action predictions [Batch, Classes, Time]
        # 3. **Diffusion Components**:
        #    - Noise scheduler (cosine beta schedule)
        #    - DDIM sampler for inference
        #    - Multiple conditioning types
        self.left_hand_model = ASDiffusionModel(
            lh_encoder_params, lh_decoder_params, lh_diffusion_params, num_classes, device
        )
        self.right_hand_model = ASDiffusionModel(
            rh_encoder_params, rh_decoder_params, rh_diffusion_params, num_classes, device
        )
        
    def get_training_loss(self, lh_video_feats, rh_video_feats, 
                         lh_event_gt, rh_event_gt, 
                         lh_boundary_gt, rh_boundary_gt,
                         encoder_ce_criterion, encoder_mse_criterion, encoder_boundary_criterion,
                         decoder_ce_criterion, decoder_mse_criterion, decoder_boundary_criterion,
                         soft_label):
        """
        Compute training loss for both hands simultaneously
        """
        # Get loss for left hand
        lh_loss_dict = self.left_hand_model.get_training_loss(
            lh_video_feats, lh_event_gt, lh_boundary_gt,
            encoder_ce_criterion, encoder_mse_criterion, encoder_boundary_criterion,
            decoder_ce_criterion, decoder_mse_criterion, decoder_boundary_criterion,
            soft_label
        )
        
        # Get loss for right hand
        rh_loss_dict = self.right_hand_model.get_training_loss(
            rh_video_feats, rh_event_gt, rh_boundary_gt,
            encoder_ce_criterion, encoder_mse_criterion, encoder_boundary_criterion,
            decoder_ce_criterion, decoder_mse_criterion, decoder_boundary_criterion,
            soft_label
        )
        
        # Combine losses with prefixes
        combined_loss_dict = {}
        for key, value in lh_loss_dict.items():
            combined_loss_dict[f'lh_{key}'] = value
        for key, value in rh_loss_dict.items():
            combined_loss_dict[f'rh_{key}'] = value
            
        return combined_loss_dict
    
    def ddim_sample_both_hands(self, lh_video_feats, rh_video_feats, seed=None):
        """
        Sample from both hand models simultaneously
        """
        lh_output = self.left_hand_model.ddim_sample(lh_video_feats, seed)
        rh_output = self.right_hand_model.ddim_sample(rh_video_feats, seed)
        
        return lh_output, rh_output
    
    def get_encoders_output(self, lh_video_feats, rh_video_feats):
        """
        Get encoder outputs for both hands
        """
        lh_encoder_out, lh_backbone_feats = self.left_hand_model.encoder(lh_video_feats, get_features=True)
        rh_encoder_out, rh_backbone_feats = self.right_hand_model.encoder(rh_video_feats, get_features=True)
        
        return lh_encoder_out, rh_encoder_out
    
    def parameters(self):
        """
        Return parameters from both models
        """
        params = list(self.left_hand_model.parameters()) + list(self.right_hand_model.parameters())
        return params
    
    def state_dict(self):
        """
        Return state dict for both models
        """
        return {
            'left_hand_model': self.left_hand_model.state_dict(),
            'right_hand_model': self.right_hand_model.state_dict()
        }
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict for both models
        """
        if strict:
            self.left_hand_model.load_state_dict(state_dict['left_hand_model'])
            self.right_hand_model.load_state_dict(state_dict['right_hand_model'])
        else:
            # Use strict=False for flexible loading
            missing_keys_lh, unexpected_keys_lh = self.left_hand_model.load_state_dict(
                state_dict['left_hand_model'], strict=False)
            missing_keys_rh, unexpected_keys_rh = self.right_hand_model.load_state_dict(
                state_dict['right_hand_model'], strict=False)
            
            # Combine missing and unexpected keys
            missing_keys = missing_keys_lh + [f"right_hand_model.{k}" for k in missing_keys_rh] + [f"left_hand_model.{k}" for k in missing_keys_lh]
            unexpected_keys = unexpected_keys_lh + [f"right_hand_model.{k}" for k in unexpected_keys_rh] + [f"left_hand_model.{k}" for k in unexpected_keys_lh]
            
            return missing_keys, unexpected_keys
    
    def to(self, device):
        """
        Move both models to device
        """
        self.left_hand_model.to(device)
        self.right_hand_model.to(device)
        return super().to(device)
    
    def train(self, mode=True):
        """
        Set training mode for both models
        """
        self.left_hand_model.train(mode)
        self.right_hand_model.train(mode)
        return super().train(mode)
    
    def eval(self):
        """
        Set evaluation mode for both models
        """
        self.left_hand_model.eval()
        self.right_hand_model.eval()
        return super().eval() 