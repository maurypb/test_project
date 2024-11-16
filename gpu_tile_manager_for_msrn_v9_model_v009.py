import torch
import numpy as np
import logging
from typing import List, Tuple, Optional
from msrn_model_v9 import ModelConfig, TrainingConfig

class GPUTileManager:
    """
    Manages GPU-based tile generation and batch processing for MSRN training.
    Updated for MSRNv9 with new configuration structure while maintaining original sampling behavior.
    """
    def __init__(self, 
                 source_images: List[torch.Tensor],
                 target_images: List[torch.Tensor],
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 track_coverage: bool = False,
                 device: str = 'cuda'):
        """
        Initialize the GPU Tile Manager.
        
        Args:
            source_images: List of source image tensors
            target_images: List of target image tensors
            model_config: Model architecture configuration
            training_config: Training parameters configuration
            device: Computing device ('cuda' or 'cpu')
        """
        if len(source_images) != len(target_images):
            raise ValueError("Number of source and target images must match")
            
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.model_config = model_config
        self.training_config = training_config
        
        # Store images
        self.source_images = source_images
        self.target_images = target_images
        
        if track_coverage:
            # Initialize coverage tracking tensors
            self.coverage_tensors = [
                torch.zeros((img.shape[1], img.shape[2]), device=device) 
                for img in source_images
            ]
        
        # Initialize state
        self.current_tiles = []
        self.tile_index = 0
        self.first_tile_of_epoch = True
        
        # Calculate area-based parameters
        self.tile_size = model_config.tile_size
        self.tile_area = self.tile_size * self.tile_size
        self.sum_of_image_areas = sum(img.shape[1] * img.shape[2] for img in source_images)
        
        # Calculate total tiles using sampling density factor
        self.total_tiles_per_epoch = int((self.sum_of_image_areas // self.tile_area) * 
                                       self.training_config.sampling_density_factor)

    def generate_epoch_tiles(self) -> int:
        """
        Generate tiles for an epoch using grid-based approach with controlled jiggling.
        Returns number of tiles generated.
        """
        self.current_tiles = []
        
        for img_idx, img in enumerate(self.source_images):
            height, width = img.shape[1:]
            
            # Calculate grid parameters with jiggling allowances
            n_tiles_h, n_tiles_v, overlap_h, overlap_v, max_offset_h, max_offset_v = \
                self._calculate_grid_parameters(width, height)
            
            # Generate tiles with position jiggling based on edge constraints
            for i in range(-1, n_tiles_v + 1):  # Extra tiles at top/bottom
                for j in range(-1, n_tiles_h + 1):  # Extra tiles at left/right
                    # Determine edge conditions
                    is_left_edge = (j <= 0)
                    is_right_edge = (j >= n_tiles_h - 1)
                    is_top_edge = (i <= 0)
                    is_bottom_edge = (i >= n_tiles_v - 1)
                    
                    # Calculate base position
                    base_y = int(i * (self.tile_size - overlap_v))
                    base_x = int(j * (self.tile_size - overlap_h))
                    
                    # Apply constrained jiggling
                    if is_left_edge or is_right_edge:
                        offset_x = 0  # No horizontal jiggling on vertical edges
                    else:
                        offset_x = torch.randint(-int(max_offset_h), int(max_offset_h) + 1, (1,)).item()
                    
                    if is_top_edge or is_bottom_edge:
                        offset_y = 0  # No vertical jiggling on horizontal edges
                    else:
                        offset_y = torch.randint(-int(max_offset_v), int(max_offset_v) + 1, (1,)).item()
                    
                    # Adjust base positions for edges
                    if is_left_edge:
                        base_x = 0
                    elif is_right_edge:
                        base_x = width - self.tile_size
                    
                    if is_top_edge:
                        base_y = 0
                    elif is_bottom_edge:
                        base_y = height - self.tile_size
                    
                    # Calculate final position with bounds checking
                    y = max(0, min(int(base_y + offset_y), height - self.tile_size))
                    x = max(0, min(int(base_x + offset_x), width - self.tile_size))
                    
                    self.current_tiles.append((img_idx, y, x))
                    if track_coverage:
                        self._update_coverage(img_idx, y, x)
        
        # Randomize tile order
        np.random.shuffle(self.current_tiles)
        self.tile_index = 0
        return len(self.current_tiles)

    def _calculate_grid_parameters(self, width: int, height: int) -> Tuple[int, int, float, float, float, float]:
        """
        Calculate grid parameters including overlap and maximum allowed offsets.
        Scales jiggling with sampling density while ensuring no coverage gaps.
        
        Returns:
            Tuple containing:
            - n_tiles_h: Number of horizontal tiles
            - n_tiles_v: Number of vertical tiles
            - overlap_h: Horizontal overlap between tiles
            - overlap_v: Vertical overlap between tiles
            - max_offset_h: Maximum horizontal jiggle
            - max_offset_v: Maximum vertical jiggle
        """
        # Calculate number of tiles needed
        n_tiles_h = int(np.ceil((width - self.tile_size/2) / (self.tile_size/2))) + 1
        n_tiles_v = int(np.ceil((height - self.tile_size/2) / (self.tile_size/2))) + 1
        
        # Calculate actual overlaps
        overlap_h = ((n_tiles_h * self.tile_size) - width) / (n_tiles_h - 1)
        overlap_v = ((n_tiles_v * self.tile_size) - height) / (n_tiles_v - 1)
        
        # Calculate maximum offsets scaled by density factor
        jiggle_factor = min(self.training_config.sampling_density_factor, 2.0)
        # Ensure no gaps by limiting to half of overlap
        max_offset_h = min(overlap_h/2, self.tile_size * 0.25 * jiggle_factor)
        max_offset_v = min(overlap_v/2, self.tile_size * 0.25 * jiggle_factor)
        
        return n_tiles_h, n_tiles_v, overlap_h, overlap_v, max_offset_h, max_offset_v

    def get_next_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of tiles for training"""
        if self.tile_index + batch_size > len(self.current_tiles):
            self.generate_epoch_tiles()

        source_batch_tiles = []
        target_batch_tiles = []
        
        for _ in range(batch_size):
            img_idx, y, x = self.current_tiles[self.tile_index]
            source_tile = self.source_images[img_idx][:, y:y+self.tile_size, x:x+self.tile_size]
            target_tile = self.target_images[img_idx][:, y:y+self.tile_size, x:x+self.tile_size]
            source_batch_tiles.append(source_tile)
            target_batch_tiles.append(target_tile)
            self.tile_index += 1

        source_batch = torch.stack(source_batch_tiles).to(self.device, dtype=torch.float32)
        target_batch = torch.stack(target_batch_tiles).to(self.device, dtype=torch.float32)
        
        return source_batch, target_batch

    def _update_coverage(self, img_idx: int, y: int, x: int):
        """Update coverage tracking for generated tile"""
        self.coverage_tensors[img_idx][y:y+self.tile_size, x:x+self.tile_size] += 1

    def get_coverage_stats(self) -> List[Tuple[float, int, float, float]]:
        """Get coverage statistics for monitoring"""
        return [(c.min().item(), (c == c.min()).sum().item(), 
                 c.max().item(), c.mean().item()) 
                for c in self.coverage_tensors]

    def reset_coverage(self):
        """Reset coverage tracking tensors"""
        for c in self.coverage_tensors:
            c.zero_()