import os
import sys
import torch
import logging
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple

from tile_generator_v8 import TileGenerator
from msrn_model_v9 import (
    ModelConfig,
    TrainingConfig,
    create_msrn_model,
    VRAMEstimator
)
from Image_load_and_save_methods_v01 import load_image

class InferenceManager:
    """
    Manages inference operations for the MSRN model with new configuration structure
    and VRAM optimization features.
    """
    def __init__(self, 
                 training_root: str,
                 source_sequence_folder: str,
                 transformed_sequence_folder: str,
                 specific_model: Optional[str] = None,
                 min_overlap: Optional[Tuple[int, int]] = None,
                 output_format: Optional[str] = None,
                 vram_limit_gb: Optional[float] = None):
        """
        Initialize the inference manager.
        
        Args:
            training_root: Root directory containing model and data
            source_sequence_folder: Directory containing source images
            transformed_sequence_folder: Directory for output images
            specific_model: Path to specific model checkpoint
            min_overlap: Optional tile overlap settings
            output_format: Output image format
            vram_limit_gb: Optional VRAM limit in GB
        """
        self.training_root = Path(training_root)
        self.source_sequence_folder = Path(source_sequence_folder)
        self.transformed_sequence_folder = Path(transformed_sequence_folder)
        self.specific_model = specific_model
        self.min_overlap = min_overlap
        self.output_format = output_format or '.exr'
        self.vram_limit_gb = vram_limit_gb
        
        self.model_dir = self.training_root / "models"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up logging
        self._setup_logging()
        
        # Load model and initialize components
        self._load_model()
        self._initialize_tile_generator()
        
        # Memory tracking
        self.peak_memory_gb = 0.0
        
    def _setup_logging(self):
        """Configure logging for the inference manager"""
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        """Load and initialize the model with configuration"""
        if self.specific_model:
            model_path = self.model_dir / self.specific_model
        else:
            model_path = self.model_dir / self._find_best_model()
        
        self.logger.info(f"Loading model from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract configurations from checkpoint
            self.model_config = ModelConfig.from_dict(checkpoint['model_config'])
            self.training_config = TrainingConfig.from_dict(checkpoint['training_config'])
            
            # Create and load model
            self.model = create_msrn_model(self.model_config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Check VRAM constraints if specified
            if self.vram_limit_gb:
                self._check_vram_requirements()
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def _check_vram_requirements(self):
        """Verify VRAM requirements and adjust if needed"""
        try:
            vram_usage = VRAMEstimator.estimate_vram_usage(
                self.model_config,
                self.training_config
            )
            
            self.logger.info(f"Estimated VRAM usage: {vram_usage['total_estimated']:.2f}GB")
            
            if vram_usage['total_estimated'] > self.vram_limit_gb:
                self.logger.warning(
                    f"Model may exceed VRAM limit of {self.vram_limit_gb}GB. "
                    "Processing will be done in smaller batches."
                )
                
            self.batch_size = VRAMEstimator.adjust_batch_size_for_vram(
                self.model_config,
                self.training_config,
                self.vram_limit_gb
            )
            
        except Exception as e:
            self.logger.error(f"VRAM estimation failed: {str(e)}")
            raise

    def _initialize_tile_generator(self):
        """Initialize the tile generator with model configuration"""
        tile_size = self.model_config.tile_size
        min_overlap = self.min_overlap or (16, 16)  # Default overlap if not specified
        self.tile_generator = TileGenerator(tile_size, min_overlap)

    def _find_best_model(self) -> str:
        """Find the model with the lowest loss in the model directory"""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
        if not model_files:
            raise ValueError("No model files found in the specified directory.")
        return min(model_files, key=lambda x: float(x.split('_loss_')[1].split('.pth')[0]))

    def process_sequence(self, prefix: str = "", postfix: str = ""):
        """Process a sequence of images"""
        self.transformed_sequence_folder.mkdir(parents=True, exist_ok=True)
        image_files = sorted(list(self.source_sequence_folder.glob('*')))
        
        try:
            for image_file in tqdm(image_files, desc="Processing images"):
                try:
                    self.process_single_image(image_file, prefix, postfix)
                except Exception as e:
                    self.logger.error(f"Error processing {image_file}: {str(e)}")
                    
                # Track memory usage
                if torch.cuda.is_available():
                    current_memory = torch.cuda.max_memory_allocated() / 1e9
                    self.peak_memory_gb = max(self.peak_memory_gb, current_memory)
                    
            self.logger.info(f"Peak VRAM usage: {self.peak_memory_gb:.2f}GB")
            
        except Exception as e:
            self.logger.error(f"Sequence processing failed: {str(e)}")
            raise

    def process_single_image(self, 
                           image_file: Path, 
                           prefix: str = "", 
                           postfix: str = "", 
                           save: bool = True) -> torch.Tensor:
        """
        Process a single image through the model.
        
        Args:
            image_file: Path to input image
            prefix: Prefix for output filename
            postfix: Postfix for output filename
            save: Whether to save the output image
            
        Returns:
            Processed image as torch tensor
        """
        # Load image
        image = load_image(image_file)
        
        # Generate tiles
        tiles = self.tile_generator.generate_tiles(image)
        
        with torch.no_grad():
            # Process tiles
            processed_tiles = []
            for tile, position in tiles:
                # Move to device and process
                processed_tile = self.model(
                    tile.unsqueeze(0).to(self.device)
                ).squeeze(0)
                processed_tiles.append((processed_tile.cpu(), position))
            
            # Reconstruct image
            reconstructed = self.tile_generator.reconstruct_image_v2(
                processed_tiles, 
                image.shape[1:]
            )
            reconstructed = torch.clamp(reconstructed, 0, 1)

        # Save if requested
        if save:
            output_filename = prefix + image_file.name + postfix
            output_path = self.transformed_sequence_folder / output_filename
            self._save_image(reconstructed, output_path)
            self._save_metadata(image_file, output_path)
            
        return reconstructed

    def _save_image(self, image: torch.Tensor, output_path: Path):
        """Save processed image with appropriate format"""
        self.logger.info(f"Saving image to: {output_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy and adjust format
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))

        # Add output format extension
        output_path = output_path.with_suffix(self.output_format)

        try:
            if self.output_format == '.exr':
                cv2.imwrite(str(output_path), 
                          cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            else:
                if self.output_format in ['.png', '.tiff', '.tif']:
                    image_np = (image_np * 65535).astype(np.uint16)
                else:  # JPEG format
                    image_np = (image_np * 255).astype(np.uint8)
                    
                cv2.imwrite(str(output_path), 
                          cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                
        except Exception as e:
            self.logger.error(f"Failed to save image: {str(e)}")
            raise

    def _save_metadata(self, input_path: Path, output_path: Path):
        """Save processing metadata"""
        metadata = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model_config": self.model_config.to_dict(),
            "processing_device": str(self.device),
            "peak_memory_gb": self.peak_memory_gb
        }
        
        metadata_path = output_path.with_suffix('.meta.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save metadata: {str(e)}")