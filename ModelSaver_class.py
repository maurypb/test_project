import os
import logging
import re
import torch



class ModelSaver:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def save_model(self, model, optimizer, config, epoch, loss, dataset_signature, 
                   all_champion_epochs, current_champion_models, losses, is_champion=False, filename_prefix=None):
        
        filename = f'champion_epoch_{epoch:06d}_loss_{loss:.6f}.pth' if is_champion else f'msrn_epoch_{epoch:06d}_loss_{loss:.6f}.pth'
        if filename_prefix:
            filename = f"{filename_prefix}_{filename}"
        filepath = os.path.join(self.model_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.to_dict(),
            'loss': loss,
            'dataset_signature': dataset_signature,
            'all_champion_epochs': all_champion_epochs,
            'current_champion_models': current_champion_models,
            'losses': losses
        }, filepath)
        
        return filename if is_champion else None

    def load_model(self, checkpoint_path, model, optimizer, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

