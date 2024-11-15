import os
import logging
import re
import torch
from typing import Optional
from dataclasses import asdict
from msrn_model_v9 import ModelConfig, TrainingConfig, TrainingState


class ModelSaver:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)

    def save_model(self, 
                    model:torch.nn.Module, 
                    optimizer:torch.optim.Optimizer, 
                    model_config:ModelConfig,
                    training_config:TrainingConfig,
                    training_state:TrainingState,

                    #epoch, loss, dataset_signature, 
                    #all_champion_epochs, current_champion_models, losses, 
                    is_champion:Bool=False, 
                    filename_prefix:Optional[str]=None)->Optional[str]:
        
        """
        Save model checkpoint with all configurations and states.
        
        Args:
            model: The PyTorch model
            optimizer: The optimizer
            model_config: Model architecture configuration
            training_config: Training parameters configuration
            training_state: Current training state
            is_champion: Whether this is a champion model
            filename_prefix: Optional prefix for the filename
            
        Returns:
            str: Filename of saved champion model, or None if not a champion
        """
        
        # Generate filename based on epoch and loss
        if is_champion:
            filename = f'champion_epoch_{training_state.current_epoch:06d}_loss_{training_state.current_loss:.6f}.pth'
        else:
            filename = f'msrn_epoch_{training_state.current_epoch:06d}_loss_{training_state.current_loss:.6f}.pth'
            
        if filename_prefix:
            filename = f"{filename_prefix}_{filename}"
            
        filepath = os.path.join(self.model_dir, filename)
        
        # Prepare checkpoint dictionary
        checkpoint = {
            'model_config': model_config.to_dict(),
            'training_config': training_config.to_dict(),
            'training_state': training_state.to_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint, filepath)
            self.logger.info(f"Saved checkpoint to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
        
        return filename if is_champion else None

    # def load_model(self, checkpoint_path, model, optimizer, device):
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     return checkpoint

    def load_model(self, 
                  checkpoint_path: str,
                  model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device) -> dict:
        """
        Load model checkpoint and configurations.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            device: Device to load model onto
            
        Returns:
            dict: Complete checkpoint including configs and states
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Convert configs back to their proper classes
            checkpoint['model_config'] = ModelConfig.from_dict(checkpoint['model_config'])
            checkpoint['training_config'] = TrainingConfig.from_dict(checkpoint['training_config'])
            checkpoint['training_state'] = TrainingState.from_dict(checkpoint['training_state'])
            
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
            return checkpoint
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise


    def find_latest_champion(self) -> Optional[str]:
        """Find the most recent champion model in the model directory."""
        champion_files = [f for f in os.listdir(self.model_dir) if f.startswith('champion_') and f.endswith('.pth')]
        if not champion_files:
            return None
            
        # Sort by epoch number and loss
        latest = max(champion_files, key=lambda x: (
            int(re.search(r'epoch_(\d+)', x).group(1)),
            -float(re.search(r'loss_([\d.]+)', x).group(1))
        ))
        
        return os.path.join(self.model_dir, latest)
    

    def find_best_model(self) -> Optional[str]:
        """Find the model with lowest loss in the model directory."""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
        if not model_files:
            return None
            
        # Sort by loss value
        best = min(model_files, key=lambda x: float(re.search(r'loss_([\d.]+)', x).group(1)))
        return os.path.join(self.model_dir, best)