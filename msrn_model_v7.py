# msrn_model_v7.py

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
    # Core Architecture
    base_features: int = 64
    use_pyramid: bool = True
    pyramid_levels: int = 2  # Number of pyramid levels when use_pyramid is True
    
    # Block Distribution (per pyramid level, from lowest to highest)
    blocks_per_level: List[int] = field(default_factory=lambda: [8, 6])
    
    # Enhanced MSRB Options
    use_multi_path: bool = True
    use_dilated_convs: bool = True
    path_weights_learnable: bool = True
    
    # Residual connection options
    residual_mode: Literal["early", "late", "both"] = "late"
    residual_scaling: float = 1.0  # Scale factor for residual connection
    
    # Memory Optimization
    vram_optimization_level: Literal["none", "balanced", "aggressive"] = "balanced"
    
    # Training Parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    tile_size: int = 256
    min_overlap: Tuple[int, int] = (32, 32)
    
    def validate(self):
        if self.base_features <= 0:
            raise ValueError("base_features must be positive")
            
        if self.use_pyramid:
            if not (1 <= self.pyramid_levels <= 3):
                raise ValueError("pyramid_levels must be between 1 and 3")
            if len(self.blocks_per_level) != self.pyramid_levels:
                raise ValueError(f"blocks_per_level length ({len(self.blocks_per_level)}) "
                               f"must match pyramid_levels ({self.pyramid_levels})")
            if not all(b > 0 for b in self.blocks_per_level):
                raise ValueError("All entries in blocks_per_level must be positive")
        else:
            if len(self.blocks_per_level) != 1:
                raise ValueError("When not using pyramid, blocks_per_level should have length 1")
                
        if self.residual_mode not in ["early", "late", "both"]:
            raise ValueError("residual_mode must be 'early', 'late', or 'both'")
        if self.residual_scaling <= 0:
            raise ValueError("residual_scaling must be positive")
        
    def get_memory_requirements(self) -> Dict[str, float]:
        """Estimate VRAM requirements in GB"""
        total_params = 0
        max_activation_size = 0
        
        # Base feature calculations
        feature_sizes = [self.base_features * (2**i) for i in range(self.pyramid_levels)]
        
        for level, num_blocks in enumerate(self.blocks_per_level):
            channels = feature_sizes[level]
            # Parameters per block
            params_per_block = (
                # Basic convolutions
                2 * (channels * channels * 9 + channels) +  # 3x3 convs
                2 * (channels * channels * 25 + channels)   # 5x5 convs
            )
            if self.use_dilated_convs:
                params_per_block += (
                    2 * (channels * channels * 9 + channels) +  # Dilated 2
                    2 * (channels * channels * 9 + channels)    # Dilated 4
                )
            
            total_params += params_per_block * num_blocks
            
            # Estimate maximum activation size
            if self.tile_size > 0:
                size = self.tile_size // (2**level)
                max_activation_size = max(
                    max_activation_size,
                    self.batch_size * channels * size * size * 4  # 4 bytes per float
                )
        
        return {
            "parameters_gb": total_params * 4 / (1024**3),  # 4 bytes per parameter
            "max_activation_gb": max_activation_size / (1024**3),
            "estimated_total_gb": (total_params * 4 + max_activation_size) / (1024**3)
        }
    
    def adjust_for_vram_limit(self, vram_limit_gb: float):
        """Adjust configuration to fit within VRAM limit"""
        while self.get_memory_requirements()["estimated_total_gb"] > vram_limit_gb:
            # Try different strategies to reduce memory usage
            if self.batch_size > 1:
                self.batch_size = max(1, self.batch_size - 1)
                continue
                
            if len(self.blocks_per_level) > 1:
                # Reduce blocks in higher pyramid levels first
                for i in range(len(self.blocks_per_level) - 1, -1, -1):
                    if self.blocks_per_level[i] > 2:
                        self.blocks_per_level[i] -= 1
                        break
                continue
            
            if self.base_features > 32:
                self.base_features = max(32, self.base_features - 8)
                continue
                
            raise ValueError(f"Cannot adjust configuration to fit within {vram_limit_gb}GB VRAM")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MSRNHybridConfig':
        return cls(**config_dict)

class EnhancedMSRB(nn.Module):
    """Enhanced Multi-Scale Residual Block with configurable residual connection"""
    
    def __init__(self, channels: int, use_dilated: bool = True, 
                 learnable_weights: bool = True, use_early_residual: bool = True):
        super().__init__()
        self.use_early_residual = use_early_residual
        self.use_dilated = use_dilated
        
        # Regular 3x3 convolution path
        self.path_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        # 5x5 convolution path for broader spatial context
        self.path_5x5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 5, padding=2)
        )
        
        # Dilated convolution paths if enabled
        if use_dilated:
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
        
        # Path fusion
        num_paths = 4 if use_dilated else 2
        if learnable_weights:
            self.path_weights = nn.Parameter(torch.ones(num_paths) / num_paths)
        else:
            self.register_buffer('path_weights', torch.ones(num_paths) / num_paths)
        
        self.fusion = nn.Conv2d(channels, channels, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        # Process all paths
        path_outputs = [self.path_3x3(x), self.path_5x5(x)]
        if self.use_dilated:
            path_outputs.extend([self.path_dilated_2(x), self.path_dilated_4(x)])
        
        # Weight and combine paths
        weights = F.softmax(self.path_weights, dim=0)
        weighted_sum = sum(w * p for w, p in zip(weights, path_outputs))
        
        # Final fusion
        out = self.fusion(weighted_sum)
        
        # Apply early residual if enabled
        if self.use_early_residual:
            out = out + identity
            
        return self.relu(out)
    



    
    
    class PyramidLevel(nn.Module):
    """Single level of the pyramid structure with configurable residual connections"""
    
    def __init__(self, config: MSRNHybridConfig, level: int, use_early_residual: bool):
        super().__init__()
        channels = config.base_features * (2 ** level)
        num_blocks = config.blocks_per_level[level]
        
        logger.info(f"Creating pyramid level {level} with {num_blocks} blocks "
                   f"and {channels} channels")
        
        # Feature extraction blocks
        self.blocks = nn.ModuleList([
            EnhancedMSRB(
                channels, 
                config.use_dilated_convs, 
                config.path_weights_learnable,
                use_early_residual
            )
            if config.use_multi_path else
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_blocks)
        ])
        
        # Level-specific fusion
        self.level_fusion = nn.Conv2d(channels * num_blocks, channels, 1)
        
    def forward(self, x):
        block_outputs = []
        feat = x
        
        for block in self.blocks:
            feat = block(feat)
            block_outputs.append(feat)
            
        return self.level_fusion(torch.cat(block_outputs, dim=1))

class MSRNHybridModel(nn.Module):
    """Hybrid MSRN model with configurable residual connections"""
    
    def __init__(self, config: MSRNHybridConfig):
        super().__init__()
        self.config = config
        config.validate()
        
        # Initial feature extraction
        self.initial_conv = nn.Conv2d(3, config.base_features, 3, padding=1)
        
        # Determine early residual usage based on config
        use_early_residual = config.residual_mode in ["early", "both"]
        
        if config.use_pyramid:
            self.pyramid_levels = nn.ModuleList([
                PyramidLevel(config, level, use_early_residual)
                for level in range(config.pyramid_levels)
            ])
            
            # Downsampling and upsampling connections
            self.downsample = nn.ModuleList([
                nn.Conv2d(config.base_features * (2**i), 
                         config.base_features * (2**(i+1)), 
                         3, stride=2, padding=1)
                for i in range(config.pyramid_levels - 1)
            ])
            
            self.upsample = nn.ModuleList([
                nn.ConvTranspose2d(config.base_features * (2**(i+1)),
                                 config.base_features * (2**i),
                                 4, stride=2, padding=1)
                for i in range(config.pyramid_levels - 1)
            ])
        else:
            self.main_level = PyramidLevel(config, 0, use_early_residual)
        
        # Final reconstruction
        self.reconstruction = nn.Conv2d(config.base_features, 3, 3, padding=1)
        
    def forward(self, x):
        identity = x
        feat = self.initial_conv(x)
        
        if self.config.use_pyramid:
            # Store features at each level
            level_features = [feat]
            
            # Downsample path
            for i in range(self.config.pyramid_levels - 1):
                feat = self.downsample[i](feat)
                level_features.append(feat)
            
            # Process each level
            processed_features = []
            for i, level in enumerate(self.pyramid_levels):
                processed_features.append(level(level_features[i]))
            
            # Upsample and combine path
            feat = processed_features[-1]
            for i in range(self.config.pyramid_levels - 2, -1, -1):
                feat = self.upsample[i](feat)
                feat = feat + processed_features[i]  # Skip connection
        else:
            feat = self.main_level(feat)
        
        # Final reconstruction
        out = self.reconstruction(feat)
        
        # Apply late residual if configured
        if self.config.residual_mode in ["late", "both"]:
            out = out + (identity * self.config.residual_scaling)
            
        return out

def create_hybrid_model(config: MSRNHybridConfig) -> MSRNHybridModel:
    return MSRNHybridModel(config)

# Example usage
if __name__ == "__main__":
    # Create default configuration
    config = MSRNHybridConfig(
        base_features=64,
        blocks_per_level=[8, 6],
        use_pyramid=True,
        pyramid_levels=2,
        residual_mode="both"
    )
    
    # Create model
    model = create_hybrid_model(config)
    
    # Test with sample input
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    # Print memory requirements
    mem_req = config.get_memory_requirements()
    print(f"\nEstimated memory requirements:")
    print(f"Parameters: {mem_req['parameters_gb']:.2f}GB")
    print(f"Max activations: {mem_req['max_activation_gb']:.2f}GB")
    print(f"Total VRAM: {mem_req['estimated_total_gb']:.2f}GB")