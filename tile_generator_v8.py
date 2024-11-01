import logging
from typing import List, Tuple
import torch
import math
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

class TileGenerator:
    def __init__(self, tile_size: int, min_overlap: Tuple[int,int]):
        self.tile_size = tile_size
        self.min_overlap = min_overlap
        #logger.info(f"TileGenerator initialized with tile size {tile_size} and optimal overlap {optimal_overlap}")


    def calculate_tiling_parameters(self, image_size: int, tile_size: int, min_overlap: int) -> Tuple[int, float, int]:
        num_tiles = math.ceil((image_size - min_overlap) / (tile_size - min_overlap))
        actual_overlap = (num_tiles * tile_size - image_size) / (num_tiles - 1)
        last_tile_offset = image_size - tile_size
        return num_tiles, actual_overlap, last_tile_offset

    def generate_tiles(self, image: torch.Tensor) -> List[Tuple[torch.Tensor, Tuple[int, int]]]:
        _, height, width = image.shape
        tiles = []

        y_params = self.calculate_tiling_parameters(height, self.tile_size, self.min_overlap[1])
        x_params = self.calculate_tiling_parameters(width, self.tile_size, self.min_overlap[0])

        num_tiles_y, overlap_v, last_tile_offset_y = y_params
        num_tiles_x, overlap_h, last_tile_offset_x = x_params

        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                if x == num_tiles_x - 1:  # Last tile in x-direction
                    x_pos = last_tile_offset_x
                else:
                    x_pos = round(x * (self.tile_size - overlap_h))
                
                if y == num_tiles_y - 1:  # Last tile in y-direction
                    y_pos = last_tile_offset_y
                else:
                    y_pos = round(y * (self.tile_size - overlap_v))
                
                tile = image[:, y_pos:y_pos+self.tile_size, x_pos:x_pos+self.tile_size]
                tiles.append((tile, (x_pos, y_pos)))

        return tiles
    
    def reconstruct_image_v2(self, tiles: List[Tuple[torch.Tensor, Tuple[int, int]]], original_size: Tuple[int, int]) -> torch.Tensor:
        channels, height, width = tiles[0][0].shape[0], original_size[0], original_size[1]
        reconstructed = torch.zeros((channels, height, width))
        weight = torch.zeros((height, width))

        y_params = self.calculate_tiling_parameters(height, self.tile_size, self.min_overlap[1])
        x_params = self.calculate_tiling_parameters(width, self.tile_size, self.min_overlap[0])

        num_tiles_y, overlap_v, last_tile_offset_y = y_params
        num_tiles_x, overlap_h, last_tile_offset_x = x_params

        # Calculate actual overlaps for last tiles
        last_overlap_h = self.tile_size - (last_tile_offset_x - (num_tiles_x - 2) * (self.tile_size - overlap_h))
        last_overlap_v = self.tile_size - (last_tile_offset_y - (num_tiles_y - 2) * (self.tile_size - overlap_v))

        # Precalculate feather patterns
        def get_feather(overlap):
            feather_size = min(int(overlap), self.tile_size // 2)
            return torch.linspace(0, 1, feather_size)

        h_feather = get_feather(overlap_h)
        v_feather = get_feather(overlap_v)
        last_h_feather = get_feather(last_overlap_h)
        last_v_feather = get_feather(last_overlap_v)

        for tile, (x, y) in tiles:
            tile_weight = torch.ones_like(tile[0])
            
            # Determine if this is the last or second-to-last tile in each dimension
            is_last_x = x == last_tile_offset_x
            is_second_last_x = x == (num_tiles_x - 2) * (self.tile_size - overlap_h)
            is_last_y = y == last_tile_offset_y
            is_second_last_y = y == (num_tiles_y - 2) * (self.tile_size - overlap_v)

            # Apply horizontal feathering
            if x > 0:
                tile_weight[:, :len(h_feather)] *= h_feather[None, :]
            if not is_last_x:
                if is_second_last_x:
                    tile_weight[:, -len(last_h_feather):] *= last_h_feather.flip(0)[None, :]
                else:
                    tile_weight[:, -len(h_feather):] *= h_feather.flip(0)[None, :]

            # Apply vertical feathering
            if y > 0:
                tile_weight[:len(v_feather), :] *= v_feather[:, None]
            if not is_last_y:
                if is_second_last_y:
                    tile_weight[-len(last_v_feather):, :] *= last_v_feather.flip(0)[:, None]
                else:
                    tile_weight[-len(v_feather):, :] *= v_feather.flip(0)[:, None]

            # Ensure we don't go out of bounds for the last tiles
            y_end = min(y + self.tile_size, height)
            x_end = min(x + self.tile_size, width)

            reconstructed[:, y:y_end, x:x_end] += tile[:, :y_end-y, :x_end-x] * tile_weight[:y_end-y, :x_end-x]
            weight[y:y_end, x:x_end] += tile_weight[:y_end-y, :x_end-x]

        # Normalization
        reconstructed /= weight.clamp(min=1)
        #return reconstructed, weight #for debugging purposes
        return reconstructed

    def reconstruct_image_naive(self, tiles: List[Tuple[torch.Tensor, Tuple[int, int]]], original_size: Tuple[int, int]) -> torch.Tensor:
        channels, height, width = tiles[0][0].shape[0], original_size[0], original_size[1]
        reconstructed = torch.zeros((channels, height, width))
        weight = torch.zeros((height, width))

        y_params = self.calculate_tiling_parameters(height, self.tile_size, self.min_overlap[1])
        x_params = self.calculate_tiling_parameters(width, self.tile_size, self.min_overlap[0])

        overlap_h, overlap_v = x_params[1], y_params[1]  # actual overlaps
        feather_h = min(int(overlap_h), self.tile_size // 2)
        feather_v = min(int(overlap_v), self.tile_size // 2)

        h_feather = torch.linspace(0, 1, feather_h)
        v_feather = torch.linspace(0, 1, feather_v)

        for tile, (x, y) in tiles:
            # tile_weight = torch.ones_like(tile[0])
            
            # # Apply feathering
            # if x > 0:
            #     tile_weight[:feather_h, :] *= h_feather[:, None]
            # if x < x_params[2]:  # Not the last tile in x-direction
            #     tile_weight[-feather_h:, :] *= h_feather.flip(0)[:, None]
            # if y > 0:
            #     tile_weight[:, :feather_v] *= v_feather[None, :]
            # if y < y_params[2]:  # Not the last tile in y-direction
            #     tile_weight[:, -feather_v:] *= v_feather.flip(0)[None, :]

            reconstructed[:, y:y+self.tile_size, x:x+self.tile_size] = tile 
            #weight[y:y+self.tile_size, x:x+self.tile_size] += tile_weight

        #reconstructed = torch.where(weight > 0, reconstructed / weight, reconstructed)
        return reconstructed




# Example usage
if __name__ == "__main__":
    # Create a sample image
    image = torch.rand(3, 512, 768)
    
    # Initialize TileGenerator
    tile_generator = TileGenerator(tile_size=256, optimal_overlap=(32, 32))
    
    # Generate tiles
    tiles = tile_generator.generate_tiles(image)
    
    # Reconstruct image
    reconstructed = tile_generator.reconstruct_image(tiles, (512, 768))
    
    print(f"Original image shape: {image.shape}")
    print(f"Reconstructed image shape: {reconstructed.shape}")
    print(f"Reconstruction error: {torch.mean((image - reconstructed)**2)}")
