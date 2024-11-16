# msrn_model_v9.py 11/14/24
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Optional, Literal, List
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model architecture only"""
    # Core Architecture Parameters
    base_features: int = 64
    num_blocks: int = 8
    tile_size: int = 256  # Part of model architecture as it affects feature processing
    
    # Processing Structure
    use_pyramid: bool = True
    pyramid_scales: int = 2
    use_multi_path: bool = True
    use_5x5_path: bool = True
    use_dilated_convs: bool = True
    path_weights_learnable: bool = True
    
    # Channel Configuration
    use_alpha: bool = False  # 4-channel support
    use_attention: bool = True
    
    # Residual Configuration
    residual_mode: Literal["early", "late", "both"] = "late"
    residual_scale: float = 0.1

    def validate(self):
        """Validate model architecture parameters"""
        if self.base_features <= 0:
            raise ValueError("base_features must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.tile_size not in [256, 512]:
            raise ValueError("tile_size must be either 256 or 512")
        if self.pyramid_scales not in [1, 2]:
            raise ValueError("pyramid_scales must be 1 or 2")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        return cls(**config_dict)

@dataclass
class TrainingConfig:
    """Configuration for training parameters that can be modified during training"""
    # Core Training Parameters
    batch_size: int = 32
    learning_rate: float = 0.0001
    num_epochs: int = 1000

    # Tile Generation Parameters
    min_overlap: Tuple[int, int] = (32, 32)  # Minimum tile overlap for inference
    sampling_density_factor: float = 1.5  # Controls density of training tile sampling


    # Loss Configuration  (what is this?)
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"mse": 1.0, "perceptual": 0.1}
    )

    # Checkpointing Parameters
    save_interval: int = 100
    champion_improvement_threshold: float = 0.1
    
    # Augmentation Parameters (moved from v5)
    augmentation_factor: float = 0.2
    rotation_range: Tuple[float, float] = (-5, 5)
    scale_range: Tuple[float, float] = (0.95, 1.05)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[float, float] = (-0.2, 0.2)
    hue_range: Tuple[float, float] = (-30, 30)
    noise_stddev_range: Tuple[float, float] = (0.01, 0.05)

    # Memory Management
    vram_limit_gb: Optional[float] = None  # If set, will be used to adjust batch size



    # Testing
    test_image: Optional[str] = None  # Moved from v5



    def validate(self):
        """Validate training parameters"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.champion_improvement_threshold <= 0:
            raise ValueError("champion_improvement_threshold must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.sampling_density_factor < 1.0:
            raise ValueError("sampling_density_factor must be >= 1.0")
        if min(self.min_overlap) < 0:
            raise ValueError("min_overlap must be non-negative")
        if self.augmentation_factor < 0 or self.augmentation_factor > 1:
            raise ValueError("augmentation_factor must be between 0 and 1")
        
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**config_dict)

@dataclass
class TrainingState:
    """State tracking for training progress history and operational status"""

    # training progress metrics
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = float('inf')
    best_loss: float = float('inf')
    losses: List[float] = field(default_factory=list)
    champion_epochs: List[int] = field(default_factory=list)

    # Training operational flags
    is_training: bool = False  # Currently actively training
    is_paused: bool = False    # Training is paused
    model_ready: bool = False  # Model is loaded and ready

    # Dataset state
    dataset_signature: Optional[str] = None
    image_set_validated: bool = False

    # Visualization state
    visualization_images_opened: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingState':
        return cls(**config_dict)
    
    def pause_training(self):
        """Pause training and update flags"""
        self.is_training = False
        self.is_paused = True
    
    def resume_training(self):
        """Resume training and update flags"""
        self.is_training = True
        self.is_paused = False
    
    def mark_model_ready(self):
        """Mark model as ready for inference"""
        self.model_ready = True
    

# part 2 model components


class EnhancedMSRBlock(nn.Module):
    """Enhanced Multi-Scale Residual Block with configurable paths"""
    def __init__(self, channels: int, config: ModelConfig):
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
    
# part 3 model architecture
class MSRNHybridModel(nn.Module):
    """Enhanced MSRN with configurable multi-scale processing"""
    def __init__(self, config: ModelConfig):
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
    

# part 3b vram estimation

class VRAMEstimator:
    """Utility class for VRAM estimation and management"""
    @staticmethod
    def estimate_vram_usage(model_config: ModelConfig, training_config: TrainingConfig) -> Dict[str, float]:
        """Estimate VRAM usage in GB based on model and training configuration"""
        # Base memory per tile
        channels = 4 if model_config.use_alpha else 3
        tile_pixels = model_config.tile_size * model_config.tile_size
        
        # Memory for one tile (in bytes)
        bytes_per_float = 4
        tile_memory = channels * tile_pixels * bytes_per_float
        
        # Feature memory per block
        num_paths = sum([
            1,  # 3x3 path always present
            1 if model_config.use_5x5_path else 0,
            2 if model_config.use_dilated_convs else 0
        ])
        block_memory = model_config.base_features * tile_pixels * bytes_per_float * num_paths
        
        # Calculate total memory including pyramid if used
        if model_config.use_pyramid and model_config.pyramid_scales == 2:
            scale1_memory = block_memory * (model_config.num_blocks // 2)
            scale2_memory = block_memory * (model_config.num_blocks // 2) / 4  # Quarter size
            feature_memory = (scale1_memory + scale2_memory) * training_config.batch_size
        else:
            feature_memory = block_memory * model_config.num_blocks * training_config.batch_size
            
        # Convert to GB
        total_gb = feature_memory / (1024**3)
        
        return {
            "feature_maps": total_gb,
            "gradients": total_gb * 2,  # Approximate gradient memory
            "optimizer": total_gb * 0.5,  # Approximate optimizer state
            "total_estimated": total_gb * 3.5  # Total with overhead
        }
    
    @staticmethod
    def adjust_batch_size_for_vram(model_config: ModelConfig, 
                                 training_config: TrainingConfig, 
                                 vram_limit_gb: float) -> int:
        """Calculate maximum batch size for given VRAM limit"""
        test_config = TrainingConfig(batch_size=training_config.batch_size)
        while True:
            vram_usage = VRAMEstimator.estimate_vram_usage(model_config, test_config)
            if vram_usage["total_estimated"] <= vram_limit_gb * 0.8:  # Keep 20% buffer
                return test_config.batch_size
            test_config.batch_size = max(1, test_config.batch_size - 4)
            if test_config.batch_size == 1:
                raise ValueError(f"Cannot fit model within {vram_limit_gb}GB VRAM even with batch size 1")
            
# part 4 utilities

def create_msrn_model(config: ModelConfig) -> MSRNHybridModel:
    """Factory function to create MSRN model with validated config"""
    return MSRNHybridModel(config)

# Example configurations for different VRAM sizes
def get_24gb_config() -> Tuple[ModelConfig, TrainingConfig]:
    """Get recommended configurations for 24GB VRAM"""
    model_config = ModelConfig(
        base_features=64,
        num_blocks=8,
        use_pyramid=True,
        pyramid_scales=2,
        use_5x5_path=True,
        use_dilated_convs=True,
        tile_size=256
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=0.0001,
        vram_limit_gb=24.0
    )
    
    return model_config, training_config

def get_12gb_config() -> Tuple[ModelConfig, TrainingConfig]:
    """Get recommended configurations for 12GB VRAM"""
    model_config = ModelConfig(
        base_features=48,
        num_blocks=6,
        use_pyramid=True,
        pyramid_scales=2,
        use_5x5_path=True,
        use_dilated_convs=True,
        tile_size=256
    )
    
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=0.0001,
        vram_limit_gb=12.0
    )
    
    return model_config, training_config

def get_8gb_config() -> Tuple[ModelConfig, TrainingConfig]:
    """Get recommended configurations for 8GB VRAM"""
    model_config = ModelConfig(
        base_features=32,
        num_blocks=4,
        use_pyramid=True,
        pyramid_scales=2,
        use_5x5_path=False,  # Reduce complexity for memory savings
        use_dilated_convs=False,
        tile_size=256
    )
    
    training_config = TrainingConfig(
        batch_size=8,
        learning_rate=0.0001,
        vram_limit_gb=8.0
    )
    
    return model_config, training_config

if __name__ == "__main__":
    # Example usage
    model_config, training_config = get_24gb_config()
    
    # Optionally adjust batch size based on available VRAM
    if training_config.vram_limit_gb:
        adjusted_batch_size = VRAMEstimator.adjust_batch_size_for_vram(
            model_config, training_config, training_config.vram_limit_gb
        )
        if adjusted_batch_size != training_config.batch_size:
            logger.warning(f"Adjusted batch size from {training_config.batch_size} to {adjusted_batch_size}")
            training_config.batch_size = adjusted_batch_size
    
    # Create model
    model = create_msrn_model(model_config)
    
    # Estimate VRAM usage
    vram_usage = VRAMEstimator.estimate_vram_usage(model_config, training_config)
    logger.info(f"Estimated VRAM usage: {vram_usage['total_estimated']:.2f}GB")
    logger.info(f"Memory breakdown:")
    for key, value in vram_usage.items():
        logger.info(f"  {key}: {value:.2f}GB")
    
    # Test with sample input
    sample_input = torch.randn(
        training_config.batch_size, 
        4 if model_config.use_alpha else 3, 
        model_config.tile_size, 
        model_config.tile_size
    )
    output = model(sample_input)
    logger.info(f"Output shape: {output.shape}")