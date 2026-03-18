# Alternating dual-head VideoMAE model for two-hand action recognition
import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from .modeling_finetune import VisionTransformer


class AlternatingDualHeadVisionTransformer(VisionTransformer):
    """Vision Transformer with dual heads that can be trained alternately"""
    
    def __init__(self, lh_num_classes=75, rh_num_classes=75, **kwargs):
        # Remove num_classes from kwargs if present
        kwargs.pop('num_classes', None)
        # Initialize with dummy num_classes
        super().__init__(num_classes=1000, **kwargs)
        
        # Remove the single head and create two heads
        embed_dim = self.embed_dim
        del self.head
        
        # Create two classification heads
        self.lh_head = nn.Linear(embed_dim, lh_num_classes) if lh_num_classes > 0 else nn.Identity()
        self.rh_head = nn.Linear(embed_dim, rh_num_classes) if rh_num_classes > 0 else nn.Identity()
        
        # Initialize the heads
        if isinstance(self.lh_head, nn.Linear):
            trunc_normal_(self.lh_head.weight, std=.02)
            nn.init.constant_(self.lh_head.bias, 0)
        if isinstance(self.rh_head, nn.Linear):
            trunc_normal_(self.rh_head.weight, std=.02)
            nn.init.constant_(self.rh_head.bias, 0)
            
        self.lh_num_classes = lh_num_classes
        self.rh_num_classes = rh_num_classes
        
        # Training mode: 'lh', 'rh', or 'both'
        self.training_mode = 'lh'
        
    def set_training_mode(self, mode):
        """Set which hand to train: 'lh', 'rh', or 'both'"""
        assert mode in ['lh', 'rh', 'both']
        self.training_mode = mode
    
    def forward(self, x, hand_type=None):
        """
        Args:
            x: input tensor
            hand_type: 'lh', 'rh', or None (uses training_mode)
        """
        # Extract features using the shared backbone
        features = self.forward_features(x)
        
        # Determine which head to use
        if hand_type is None:
            hand_type = self.training_mode
        
        # Return predictions based on mode
        if hand_type == 'lh':
            return self.lh_head(features)
        elif hand_type == 'rh':
            return self.rh_head(features)
        elif hand_type == 'both':
            return {
                'lh_pred': self.lh_head(features),
                'rh_pred': self.rh_head(features)
            }
        else:
            raise ValueError(f"Invalid hand_type: {hand_type}")
    
    def get_classifier(self):
        return {'lh_head': self.lh_head, 'rh_head': self.rh_head}
    
    def reset_classifier(self, lh_num_classes=None, rh_num_classes=None):
        if lh_num_classes is not None:
            self.lh_num_classes = lh_num_classes
            self.lh_head = nn.Linear(self.embed_dim, lh_num_classes) if lh_num_classes > 0 else nn.Identity()
            if isinstance(self.lh_head, nn.Linear):
                trunc_normal_(self.lh_head.weight, std=.02)
                nn.init.constant_(self.lh_head.bias, 0)
        if rh_num_classes is not None:
            self.rh_num_classes = rh_num_classes
            self.rh_head = nn.Linear(self.embed_dim, rh_num_classes) if rh_num_classes > 0 else nn.Identity()
            if isinstance(self.rh_head, nn.Linear):
                trunc_normal_(self.rh_head.weight, std=.02)
                nn.init.constant_(self.rh_head.bias, 0)


@register_model
def vit_base_patch16_224_alternating(pretrained=False, **kwargs):
    model = AlternatingDualHeadVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = {
        'url': '',
        'num_classes': (75, 75),
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
    return model


@register_model
def vit_large_patch16_224_alternating(pretrained=False, **kwargs):
    model = AlternatingDualHeadVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = {
        'url': '',
        'num_classes': (75, 75),
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
    return model


@register_model
def vit_giant_patch14_224_alternating(pretrained=False, **kwargs):
    model = AlternatingDualHeadVisionTransformer(
        patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = {
        'url': '',
        'num_classes': (75, 75),
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
    return model