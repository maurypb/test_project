import torch
from pathlib import Path
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" #enable exr support in opencv
import sys
import cv2
import numpy as np

def load_image(path: str) -> torch.Tensor:
    #supports 16-bit PNG and OpenEXR
    file_extension = os.path.splitext(path)[1].lower()
    
    if file_extension == '.exr':
        # Load EXR file
        image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if image is None:
            raise IOError(f"Failed to load EXR image: {path}")
        # OpenEXR files are typically float32, so we don't need to normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Load other image formats (including 16-bit PNG)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"Failed to load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize based on bit depth
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        else:
            raise ValueError(f"Unsupported image bit depth for file: {path}")
    
    # Convert to PyTorch tensor and change to channel-first format
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def save_image(image: torch.Tensor, output_path: Path, output_format: str):
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
