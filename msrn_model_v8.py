import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Optional, Literal, List
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MSRNHybridConfig:
    """Configuration for Enhanced MSRN with tile-based processing"""
    # Core Architecture
    base_features: int = 64
    num_blocks: int = 8
    use_pyramid: bool = True
    pyramid_scales: int = 2  # Maximum of 2 scales for VRAM efficiency
    
    # Multi-Scale Block Options
    use_multi_path: bool = True
    use_5x5_path: bool = True
    use_dilated_convs: bool = True
    path_weights_learnable: bool = True
    
    # Processing Options
    tile_size: int = 256  # 256 or 512
    use_alpha: bool = False  # 4-channel support
    use_attention: bool = True
    
    # Residual Options
    residual_mode: Literal["early", "late", "both"] = "late"
    residual_scale: float = 0.1  # Scale factor for residual connection
    
    # Training Parameters
    batch_size: int = 32  # Will be adjusted based on tile_size and VRAM
    learning_rate: float = 0.0001
    
    def validate(self):
        """Validate configuration parameters"""
        if self.base_features <= 0:
            raise ValueError("base_features must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.tile_size not in [256, 512]:
            raise ValueError("tile_size must be either 256 or 512")
        if self.pyramid_scales not in [1, 2]:
            raise ValueError("pyramid_scales must be 1 or 2")
            
    def estimate_vram_usage(self) -> Dict[str, float]:
        """Estimate VRAM usage in GB"""
        # Base memory per tile
        channels = 4 if self.use_alpha else 3
        tile_pixels = self.tile_size * self.tile_size
        
        # Memory for one tile (in bytes)
        bytes_per_float = 4
        tile_memory = channels * tile_pixels * bytes_per_float
        
        # Feature memory per block (including all paths)
        num_paths = sum([
            1,  # 3x3 path always present
            1 if self.use_5x5_path else 0,
            2 if self.use_dilated_convs else 0
        ])
        block_memory = self.base_features * tile_pixels * bytes_per_float * num_paths
        
        # Calculate for pyramid structure if used
        total_memory = 0
        if self.use_pyramid and self.pyramid_scales == 2:
            scale1_memory = block_memory * (self.num_blocks // 2)
            scale2_memory = block_memory * (self.num_blocks // 2) / 4  # Quarter size
            total_memory = (scale1_memory + scale2_memory) * self.batch_size
        else:
            total_memory = block_memory * self.num_blocks * self.batch_size
            
        # Add overhead for gradients, optimizer states, etc.
        total_gb = total_memory / (1024**3)  # Convert to GB
        return {
            "feature_maps": total_gb,
            "gradients": total_gb * 2,  # Approximate gradient memory
            "optimizer": total_gb * 0.5,  # Approximate optimizer state
            "total_estimated": total_gb * 3.5  # Total with overhead
        }
    
    def adjust_for_vram_limit(self, vram_limit_gb: float):
        """Adjust configuration to fit within VRAM limit"""
        while self.estimate_vram_usage()["total_estimated"] > vram_limit_gb * 0.8:  # Keep 20% buffer
            if self.batch_size > 4:
                self.batch_size = max(4, self.batch_size - 4)
                continue
            if self.base_features > 32:
                self.base_features -= 8
                continue
            if self.num_blocks > 4:
                self.num_blocks -= 2
                continue
            if self.use_dilated_convs:
                self.use_dilated_convs = False
                continue
            if self.use_5x5_path:
                self.use_5x5_path = False
                continue
            if self.pyramid_scales > 1:
                self.pyramid_scales = 1
                continue
            raise ValueError(f"Cannot adjust configuration to fit within {vram_limit_gb}GB VRAM")
            
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MSRNHybridConfig':
        return cls(**config_dict)
    

class EnhancedMSRBlock(nn.Module):

# Enhanced Multi-Scale Residual Block with configurable paths 

    
    def __init__(self, channels: int, config: MSRNHybridConfig):
        super().__init__()
        
        # Track active paths for fusion
        self.active_paths = []
        
        # Path 1: 3x3 convolutions (always present)
        self.path_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.active_paths.append('3x3')
        
        # Path 2: 5x5 convolutions (optional)
        if config.use_5x5_path:
            self.path_5x5 = nn.Sequential(
                nn.Conv2d(channels, channels, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 5, padding=2)
            )
            self.active_paths.append('5x5')
        
        # Paths 3 & 4: Dilated convolutions (optional)
        if config.use_dilated_convs:
            self.path_dilated_2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
            )
            self.path_dilated_4 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=4, dilation=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
            )
            self.active_paths.extend(['d2', 'd4'])
        
        # Path weights (if learnable)
        num_paths = len(self.active_paths)
        if config.path_weights_learnable:
            self.path_weights = nn.Parameter(torch.ones(num_paths))
        else:
            self.register_buffer('path_weights', torch.ones(num_paths) / num_paths)
        
        # Fusion of paths
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * num_paths, channels, 1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        identity = x
        
        # Collect outputs from all active paths
        path_outputs = [self.path_3x3(x)]
        
        if hasattr(self, 'path_5x5'):
            path_outputs.append(self.path_5x5(x))
            
        if hasattr(self, 'path_dilated_2'):
            path_outputs.append(self.path_dilated_2(x))
            path_outputs.append(self.path_dilated_4(x))
        
        # Apply weights
        weights = F.softmax(self.path_weights, dim=0)
        weighted_outputs = [out * w for out, w in zip(path_outputs, weights)]
        
        # Combine paths
        combined = self.fusion(torch.cat(weighted_outputs, dim=1))
        
        return combined + identity

class AttentionModule(nn.Module):
    """Enhanced attention module with both spatial and channel attention"""
    def __init__(self, channels: int):
        super().__init__()
        
        # Channel attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention with larger receptive field
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        chan_att = self.channel_gate(x)
        x = x * chan_att
        
        # Spatial attention
        spat_att = self.spatial_gate(x)
        x = x * spat_att
        
        return x
    


class MSRNHybridModel(nn.Module):
    """Enhanced MSRN with configurable multi-scale processing"""
    def __init__(self, config: MSRNHybridConfig):
        super().__init__()
        config.validate()
        self.config = config
        
        # Input/output channels
        in_channels = 4 if config.use_alpha else 3
        out_channels = 4 if config.use_alpha else 3
        
        # Initial feature extraction
        self.initial = nn.Conv2d(in_channels, config.base_features, 3, padding=1)
        
        if config.use_pyramid and config.pyramid_scales == 2:
            # Split blocks between scales
            blocks_per_scale = config.num_blocks // 2
            
            # First scale blocks
            self.scale1_blocks = nn.ModuleList([
                EnhancedMSRBlock(config.base_features, config)
                for _ in range(blocks_per_scale)
            ])
            
            # Second scale blocks
            self.scale2_blocks = nn.ModuleList([
                EnhancedMSRBlock(config.base_features, config)
                for _ in range(blocks_per_scale)
            ])
            
            # Scale transitions
            self.downsample = nn.AvgPool2d(2)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
            # Scale fusion
            self.scale_fusion = nn.Conv2d(config.base_features * 2, config.base_features, 1)
        else:
            # Single scale processing
            self.blocks = nn.ModuleList([
                EnhancedMSRBlock(config.base_features, config)
                for _ in range(config.num_blocks)
            ])
        
        # Optional attention
        self.attention = AttentionModule(config.base_features) if config.use_attention else None
        
        # Final reconstruction
        self.final = nn.Sequential(
            nn.Conv2d(config.base_features, config.base_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.base_features // 2, out_channels, 3, padding=1)
        )
        
    def forward(self, x):
        # Store input for residual
        identity = x
        
        # Initial features
        feat = self.initial(x)
        
        if hasattr(self, 'scale1_blocks'):  # Pyramid processing
            # First scale
            scale1_feat = feat
            for block in self.scale1_blocks:
                scale1_feat = block(scale1_feat)
            
            # Second scale
            scale2_feat = self.downsample(feat)
            for block in self.scale2_blocks:
                scale2_feat = block(scale2_feat)
            
            # Combine scales
            scale2_feat = self.upsample(scale2_feat)
            feat = self.scale_fusion(torch.cat([scale1_feat, scale2_feat], dim=1))
        else:  # Single scale processing
            for block in self.blocks:
                feat = block(feat)
        
        # Apply attention if enabled
        if self.attention is not None:
            feat = self.attention(feat)
        
        # Final reconstruction
        out = self.final(feat)
        
        # Apply residual connection based on config
        if self.config.residual_mode in ['late', 'both']:
            out = identity + out * self.config.residual_scale
            
        return out

def create_msrn_model(config: MSRNHybridConfig) -> MSRNHybridModel:
    """Factory function to create MSRN model with validated config"""
    return MSRNHybridModel(config)

# Example configurations for different VRAM sizes
def get_24gb_config():
    return MSRNHybridConfig(
        base_features=64,
        num_blocks=8,
        use_pyramid=True,
        pyramid_scales=2,
        use_5x5_path=True,
        use_dilated_convs=True,
        tile_size=256,
        batch_size=32
    )

def get_12gb_config():
    return MSRNHybridConfig(
        base_features=48,
        num_blocks=6,
        use_pyramid=True,
        pyramid_scales=2,
        use_5x5_path=True,
        use_dilated_convs=True,
        tile_size=256,
        batch_size=16
    )
