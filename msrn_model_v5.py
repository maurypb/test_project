import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Optional
import json
import os
#from image_set_analyzer import ImageSetAnalyzer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
@dataclass
class MSRNConfig:
    num_scales: int = 3
    num_residual_blocks: int = 5
    base_features: int = 64
    use_pyramid: bool = False
    use_attention: bool = False
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

    # def __post_init__(self):
    #     if self.image_dir:
    #         self.analyze_image_set()

    # def analyze_image_set(self):
    #     logger.info(f"Analyzing image set in directory: {self.image_dir}")
    #     analyzer = ImageSetAnalyzer(self.image_dir)
    #     analyzer.analyze()
    #     results = analyzer.get_analysis_results()
    #     self.median_image_size = (results['median_width'], results['median_height'])
    #     self.optimal_overlap = analyzer.calculate_optimal_overlap(self.tile_size, self.base_overlap)
    #     logger.info(f"Median image size: {self.median_image_size}")
    #     logger.info(f"Optimal overlap: {self.optimal_overlap}")




    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MSRNConfig':
        return cls(**config_dict)



    def validate(self):
        if self.num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if self.num_residual_blocks <= 0:
            raise ValueError("num_residual_blocks must be positive")
        if self.base_features <= 0:
            raise ValueError("base_features must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if not all(weight >= 0 for weight in self.loss_weights.values()):
            raise ValueError("All loss weights must be non-negative")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if not 0 < self.champion_improvement_factor < 1:
            raise ValueError("champion_improvement_factor must be between 0 and 1")
        if self.test_image and not os.path.isfile(self.test_image):
            raise ValueError(f"test_image file does not exist: {self.test_image}")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if min(self.min_overlap) < 0 or max(self.min_overlap) >= self.tile_size:
            raise ValueError("base_overlap must be non-negative and less than tile_size")
        #print(f"***augmentation_factor: {self.augmentation_factor}")
        if not 0 <= self.augmentation_factor <= 1:
            raise ValueError("augmentation_factor must be between 0 and 1")

        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not all(0 <= overlap < self.tile_size for overlap in self.min_overlap):
            raise ValueError("min_overlap must be non-negative and less than tile_size")
        # ... (add more validation as needed) ...

        logger.info("Configuration validated successfully")

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    

    def __eq__(self, other):
        if not isinstance(other, MSRNConfig):
            return NotImplemented
        
        # Compare all attributes that could affect dataset initialization, AND ONLY THOSE.
        # In other words, only attributes that affect the creation of the tiled dataset should be compared.
        return (
            self.tile_size == other.tile_size and
            self.min_overlap == other.min_overlap 
            # Add any other attributes that might affect dataset initialization
            )

class AttentionModule(nn.Module):
    """Attention module for feature refinement."""
    
    def __init__(self, num_features: int):
        """
        Initialize the AttentionModule.

        Args:
            num_features (int): Number of input feature channels.
        """
        super(AttentionModule, self).__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_features, max(num_features // 16, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(num_features // 16, 8), num_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionModule.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Attention-refined output tensor of shape (B, C, H, W).
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual block for feature extraction."""
    
    def __init__(self, num_features: int):
        """
        Initialize the ResidualBlock.

        Args:
            num_features (int): Number of input and output feature channels.
        """
        super(ResidualBlock, self).__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class MSRNModel(nn.Module):
    """Multi-Scale Recurrent Network (MSRN) model."""
    
    def __init__(self, config: MSRNConfig):
        """
        Initialize the MSRNModel.

        Args:
            config (MSRNConfig): Configuration object for the MSRN model.
        """
        super(MSRNModel, self).__init__()
        config.validate()
        self.config = config
        
        # Add initial convolution to convert 3-channel input to base_features
        self.initial_conv = nn.Conv2d(3, config.base_features, kernel_size=3, padding=1)
        
        if config.use_pyramid:
            self._init_pyramid(config.num_scales, config.num_residual_blocks, config.base_features)
        else:
            self._init_flat(config.num_scales, config.num_residual_blocks, config.base_features)
        
        if config.use_attention:
            self._init_attention()
        
        # Final convolution layer is now defined separately for flat and pyramid structures
        
        logger.info(f"Initialized MSRNModel with configuration: {config.to_json()}")


    def _init_flat(self, num_scales: int, num_residual_blocks: int, num_features: int):
        self.scales = nn.ModuleList([self._make_scale(num_features, num_residual_blocks) for _ in range(num_scales)])
        self.feature_sizes = [num_features] * num_scales
        self.final_conv = nn.Conv2d(num_features * num_scales, 3, kernel_size=3, padding=1)

    def _init_pyramid(self, num_scales: int, num_residual_blocks: int, base_features: int):
        self.encoder_scales = nn.ModuleList()
        self.decoder_scales = nn.ModuleList()
        self.feature_sizes = []
        
        # Encoder
        for i in range(num_scales):
            features = base_features * (2**i)
            self.feature_sizes.append(features)
            logger.debug(f"Encoder scale {i}: {features} features")
            self.encoder_scales.append(self._make_scale(features, num_residual_blocks))
            if i < num_scales - 1:
                self.encoder_scales.append(nn.Conv2d(features, features*2, kernel_size=3, stride=2, padding=1))
        
        # Decoder
        for i in range(num_scales - 1, -1, -1):
            features = base_features * (2**i)
            logger.debug(f"Decoder scale {i}: {features} features")
            if i < num_scales - 1:
                self.decoder_scales.append(nn.ConvTranspose2d(features*2, features, kernel_size=4, stride=2, padding=1))
                self.decoder_scales.append(nn.Conv2d(features*3, features, kernel_size=1))
            self.decoder_scales.append(self._make_scale(features, num_residual_blocks))
        
        self.final_conv = nn.Conv2d(base_features, 3, kernel_size=3, padding=1)

    def _init_attention(self):
        """Initialize attention modules."""
        if self.config.use_pyramid:
            self.attention = nn.ModuleList([AttentionModule(features) for features in self.feature_sizes])
        else:
            self.attention = nn.ModuleList([AttentionModule(self.config.base_features) for _ in range(self.config.num_scales)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MSRNModel.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C', H, W), where C' depends on the model configuration.
        """
        x = self.initial_conv(x)
        if self.config.use_pyramid:
            return self._forward_pyramid(x)
        else:
            return self._forward_flat(x)

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for flat structure."""
        outputs = []
        for i, scale in enumerate(self.scales):
            x = scale(x)
            if self.config.use_attention:
                x = self.attention[i](x)
            outputs.append(x)
        return self.final_conv(torch.cat(outputs, dim=1))

    def _forward_pyramid(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for pyramid structure."""
        logger.debug(f"Input shape: {x.shape}")
        encoder_outputs = []
        attention_index = 0
        
        # Encoder
        for i, layer in enumerate(self.encoder_scales):
            x = layer(x)
            logger.debug(f"After encoder layer {i}, shape: {x.shape}")
            if isinstance(layer, nn.Sequential):
                if self.config.use_attention:
                    x = self.attention[attention_index](x)
                    logger.debug(f"After attention {attention_index}, shape: {x.shape}")
                    attention_index += 1
                encoder_outputs.append(x)
        
        # Decoder
        for i, layer in enumerate(self.decoder_scales):
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                logger.debug(f"After ConvTranspose2d {i}, shape: {x.shape}")
                if encoder_outputs:
                    encoder_output = encoder_outputs.pop()
                    logger.debug(f"Encoder output shape: {encoder_output.shape}")
                    if x.size(2) != encoder_output.size(2) or x.size(3) != encoder_output.size(3):
                        encoder_output = F.interpolate(encoder_output, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
                        logger.debug(f"After interpolation, encoder output shape: {encoder_output.shape}")
                    x = torch.cat([x, encoder_output], dim=1)
                    logger.debug(f"After concatenation, shape: {x.shape}")
            elif isinstance(layer, nn.Conv2d):
                x = layer(x)
                logger.debug(f"After Conv2d {i}, shape: {x.shape}")
            else:  # ResidualBlock
                x = layer(x)
                logger.debug(f"After ResidualBlock {i}, shape: {x.shape}")
                if self.config.use_attention and attention_index < len(self.attention):
                    x = self.attention[attention_index](x)
                    logger.debug(f"After attention {attention_index}, shape: {x.shape}")
                    attention_index += 1
        
        x = self.final_conv(x)
        logger.debug(f"Final output shape: {x.shape}")
        return x

    def _make_scale(self, num_features: int, num_blocks: int) -> nn.Sequential:
        """Create a scale with the specified number of residual blocks."""
        layers = [ResidualBlock(num_features) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

def create_msrn_model_from_config(config: MSRNConfig) -> MSRNModel:
    return MSRNModel(config)

# Example usage:
if __name__ == "__main__":
    # Create a configuration with default values
    default_config = MSRNConfig()
    print("Default configuration:")
    print(default_config.to_json())

    # Create a custom configuration
    custom_config = MSRNConfig(
        num_scales=4,
        num_residual_blocks=7,
        base_features=128,
        use_pyramid=True,
        use_attention=True,
        learning_rate=0.0005,
        batch_size=16,
        num_epochs=200,
        loss_weights={"mse": 0.7, "perceptual": 0.3}
    )
    print("\nCustom configuration:")
    print(custom_config.to_json())

    # Create an MSRN model from the custom configuration
    model = create_msrn_model_from_config(custom_config)
    print("\nCreated MSRN model with custom configuration")

    # Test the model with a sample input
    sample_input = torch.randn(1, custom_config.base_features, 64, 64)
    output = model(sample_input)
    print(f"\nModel output shape: {output.shape}")