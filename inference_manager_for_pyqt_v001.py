import os
import sys

#********************
#note: currently, the msrn_model_v5.py file is in the parent directory of the tiling_system directory.
relative_parent_directory_path = os.path.join(os.path.dirname(__file__), '..')

# Convert to an absolute path
parent_directory_path = os.path.abspath(relative_parent_directory_path)

# Add the parent directory to sys.path
sys.path.append(parent_directory_path)
#********************

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" #enable exr support in opencv


import torch
import logging
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
from tile_generator_v8 import TileGenerator
from msrn_model_v5 import MSRNConfig, create_msrn_model_from_config

class InferenceManager:
    def __init__(self, training_root: str, source_sequence_folder: str, 
                 transformed_sequence_folder: str, specific_model: str = None, 
                 min_overlap: tuple = None, output_format: str = None):
        self.training_root = Path(training_root)
        self.source_sequence_folder = Path(source_sequence_folder)
        self.transformed_sequence_folder = Path(transformed_sequence_folder)
        self.specific_model = specific_model
        self.min_overlap = min_overlap
        self.output_format = output_format
        
        self.model_dir = self.training_root / "models"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_logging()
        self._load_model()
        self._initialize_tile_generator()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        if self.specific_model:
            model_path = self.model_dir / self.specific_model
        else:
            model_path = self.model_dir / self._find_best_model()
        
        self.logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = MSRNConfig.from_dict(checkpoint['config'])
        self.model = create_msrn_model_from_config(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def _find_best_model(self):
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
        if not model_files:
            raise ValueError("No model files found in the specified directory.")
        return min(model_files, key=lambda x: float(x.split('_loss_')[1].split('.pth')[0]))

    def _initialize_tile_generator(self):
        tile_size = self.config.tile_size
        min_overlap = self.min_overlap or self.config.min_overlap
        self.tile_generator = TileGenerator(tile_size, min_overlap)


    def _load_image(self, path: str) -> torch.Tensor:
        file_extension = os.path.splitext(path)[1].lower()
        supported_formats = ['.exr', '.png', '.jpg', '.jpeg', '.tiff', '.tif']
        
        if file_extension not in supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are: {', '.join(supported_formats)}")
        
        if file_extension == '.exr':
            # Load EXR file
            image = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if image is None:
                raise IOError(f"Failed to load EXR image: {path}")
            # OpenEXR files are typically float32, so we don't need to normalize
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Load other image formats
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise IOError(f"Failed to load image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize based on bit depth
            if file_extension in ['.jpg', '.jpeg'] or image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            else:
                raise ValueError(f"Unsupported image bit depth for file: {path}")
        
        # Convert to PyTorch tensor and change to channel-first format
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        self.logger.info(f"Loaded image {path}")
        self.logger.info(f"Image shape: {image_tensor.shape}")
        self.logger.info(f"Image range: min={image_tensor.min().item():.4f}, max={image_tensor.max().item():.4f}")
        
        return image_tensor



    def _save_image(self, image: torch.Tensor, output_path: Path, output_format: str):
        # clamp the image to [0, 1] range
        #image = torch.clamp(image, 0, 1)
        
        self.logger.info(f"Saving image to: {output_path}")
        self.logger.info(f"Normalized image range: min={image.min().item():.4f}, max={image.max().item():.4f}")

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert from PyTorch tensor to numpy array
        image_np = image.cpu().numpy()

        # Convert from channel-first to channel-last format
        image_np = np.transpose(image_np, (1, 2, 0))

        # Use the provided output_format
        output_path = output_path.with_suffix(output_format)

        if output_format == '.exr':
            # Save as EXR (assuming float32 data)
            cv2.imwrite(str(output_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            self.logger.info(f"Saved EXR image range: min={image_np.min():.4f}, max={image_np.max():.4f}")
        elif output_format in ['.png', '.tiff', '.tif']:
            # For 16-bit formats, scale to 0-65535
            image_np = (image_np * 65535).astype(np.uint16)
            cv2.imwrite(str(output_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            self.logger.info(f"Saved 16-bit image range: min={image_np.min()}, max={image_np.max()}")
        elif output_format in ['.jpg', '.jpeg']:
            # For JPEG, scale to 0-255 (8-bit)
            image_np = (image_np * 255).astype(np.uint8)
            cv2.imwrite(str(output_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            self.logger.info(f"Saved 8-bit image range: min={image_np.min()}, max={image_np.max()}")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        self.logger.info(f"Saved image to {output_path}")



    def _save_metadata(self, input_path: Path, output_path: Path):
        # Implement metadata saving logic here
        # For now, we'll just log some basic information
        self.logger.info(f"Processed {input_path} to {output_path}")
        self.logger.info(f"Model used: {self.specific_model or 'Best model'}")
        self.logger.info(f"Tile size: {self.config.tile_size}")
        self.logger.info(f"Minimum overlap: {self.min_overlap or self.config.min_overlap}")

    def process_sequence(self, prefix: str = "", postfix: str = ""):
        self.transformed_sequence_folder.mkdir(parents=True, exist_ok=True)
        image_files = list(self.source_sequence_folder.glob('*'))
        image_files = sorted(image_files, key=lambda x: x.name)  # Sort by filename
        
        with torch.no_grad():
            for image_file in tqdm(image_files, desc="Processing images"):
                try:
                    self.process_single_image(image_file, prefix, postfix)
                except Exception as e:
                    self.logger.error(f"Error processing {image_file}: {str(e)}")

    def process_single_image(self, image_file: Path, prefix: str = "", postfix: str = "", save: bool = True):
        # Load image
        image = self._load_image(image_file)
        
        # Generate tiles
        tiles = self.tile_generator.generate_tiles(image)
        #print(f"tiles generated; Number of tiles: {len(tiles)}")
        with torch.no_grad():
                
            # Process tiles
            processed_tiles = []
            for tile, position in tiles:
                
                processed_tile = self.model(tile.unsqueeze(0).to(self.device)).squeeze(0)
                #print(f"processed a tile; position: {position}")
                processed_tiles.append((processed_tile.cpu(), position))
            
        # Clear the original tiles to potentially free some memory
        #del tiles
        
        # Reconstruct image
        reconstructed = self.tile_generator.reconstruct_image_v2(processed_tiles, image.shape[1:])
        print(f"reconstructed image")
        #clamp the image to [0, 1] range if needed  
        reconstructed = torch.clamp(reconstructed, 0, 1)


        # Clear processed_tiles to potentially free some memory
        #del processed_tiles
        if save: 
            # Save transformed image
            #add prefix and postfix to the image file name
            image_file = Path(prefix + image_file.name + postfix)
            output_path = self.transformed_sequence_folder / image_file.name
            self._save_image(reconstructed, output_path, self.output_format)
            # Generate and save metadata (not implemented yet)
            #self._save_metadata(image_file, output_path)
        return reconstructed

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on a sequence of images")
    parser.add_argument("training_root", help="Path to the training root directory")
    parser.add_argument("source_sequence", help="Path to the source sequence folder")
    parser.add_argument("transformed_sequence", help="Path to save the transformed sequence")
    parser.add_argument("--model", help="Specific model to use (optional)")
    parser.add_argument("--overlap", nargs=2, type=int, help="Minimum overlap (optional)")
    parser.add_argument("--output_format", help="Output format for saved images (optional)")
    
    args = parser.parse_args()
    
    inference_manager = InferenceManager(
        args.training_root,
        args.source_sequence,
        args.transformed_sequence,
        args.model,
        tuple(args.overlap) if args.overlap else None,
        args.output_format
    )
    
    inference_manager.process_sequence()
