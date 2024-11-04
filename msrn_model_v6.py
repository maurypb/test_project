# msrn_model_v6.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Optional, Literal
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config 
@dataclass
class MSRNConfigV6:
    # Architecture
    num_scales: int = 3
    num_residual_blocks: int = 5
    base_features: int = 64
    use_pyramid: bool = False
    use_attention: bool = False
    
    # New options
    residual_connection: Literal["early", "late"] = "late"
    reconstruction_type: Literal["direct", "multi_stage", "progressive"] = "progressive"
    use_multi_scale_blocks: bool = True
    progressive_steps: int = 3
    
    # Existing options (unchanged)
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"mse": 1.0, "perceptual": 0.1})
    save_interval: int = 10
    champion_improvement_factor: float = 0.1
    tile_size: int = 256
    min_overlap: Tuple[int, int] = (32, 32)
    augmentation_factor: float = 0.2
    rotation_range: Tuple[float, float] = (-5, 5)
    scale_range: Tuple[float, float] = (0.95, 1.05)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[float, float] = (-0.2, 0.2)
    hue_range: Tuple[float, float] = (-30, 30)
    noise_stddev_range: Tuple[float, float] = (0.01, 0.05)
    test_image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MSRNConfigV6':
        return cls(**config_dict)

# Building Blocks
class MSRBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 5, padding=2)
        )
        self.fusion = nn.Conv2d(channels*2, channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        b3 = self.branch_3x3(x)
        b5 = self.branch_5x5(x)
        out = self.fusion(torch.cat([b3, b5], dim=1))
        return self.relu(out + res)

class GlobalFeatureFusion(nn.Module):
    def __init__(self, num_blocks: int, channels: int):
        super().__init__()
        self.bottleneck = nn.Conv2d(channels * num_blocks, channels, 1)
        
    def forward(self, features: list):
        return self.bottleneck(torch.cat(features, dim=1))

class ProgressiveReconstruction(nn.Module):
    def __init__(self, in_channels: int, steps: int):
        super().__init__()
        self.steps = steps
        step_channels = [max(in_channels//(2**i), 32) for i in range(steps)]
        
        layers = []
        curr_channels = in_channels
        for channels in step_channels:
            layers.extend([
                nn.Conv2d(curr_channels, channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
            curr_channels = channels
        
        layers.append(nn.Conv2d(curr_channels, 3, 3, padding=1))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)


# Model

class MSRNModelV6(nn.Module):
    def __init__(self, config: MSRNConfigV6):
        super().__init__()
        config.validate()
        self.config = config
        
        # Initial feature extraction
        self.initial_conv = nn.Conv2d(3, config.base_features, 3, padding=1)
        
        # Multi-scale residual blocks
        self.blocks = nn.ModuleList([
            MSRBlock(config.base_features) if config.use_multi_scale_blocks
            else ResidualBlock(config.base_features)
            for _ in range(config.num_residual_blocks)
        ])
        
        # Global feature fusion
        self.fusion = GlobalFeatureFusion(
            config.num_residual_blocks, 
            config.base_features
        )
        
        # Reconstruction
        if config.reconstruction_type == "progressive":
            self.reconstruction = ProgressiveReconstruction(
                config.base_features,
                config.progressive_steps
            )
        else:
            self.reconstruction = nn.Conv2d(
                config.base_features, 3, 3, padding=1
            )

    def forward(self, x):
        # Store input for residual
        input_img = x
        
        # Initial features
        x = self.initial_conv(x)
        
        # Store block outputs for fusion
        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)
            
        # Global fusion
        x = self.fusion(block_outputs)
        
        # Reconstruction
        x = self.reconstruction(x)
        
        # Residual connection
        if self.config.residual_connection == "late":
            x = x + input_img
            
        return x
    
    def validate(self):
        if self.num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if self.num_residual_blocks <= 0:
            raise ValueError("num_residual_blocks must be positive")
        if self.base_features <= 0:
            raise ValueError("base_features must be positive")
        if self.progressive_steps <= 0:
            raise ValueError("progressive_steps must be positive")
        logger.info("Configuration validated successfully")

def create_msrn_model_from_config(config: MSRNConfigV6) -> MSRNModelV6:
    return MSRNModelV6(config)

if __name__ == "__main__":
    # Create configuration
    config = MSRNConfigV6(
        num_residual_blocks=8,
        base_features=64,
        use_multi_scale_blocks=True,
        reconstruction_type="progressive",
        progressive_steps=3
    )
    
    # Create model
    model = create_msrn_model_from_config(config)
    
    # Test with sample input
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(f"Output shape: {out.shape}")