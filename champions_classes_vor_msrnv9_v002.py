import os
import logging
import re
from typing import List, Set, Optional
from dataclasses import dataclass
from msrn_model_v9 import ModelConfig, TrainingConfig, TrainingState

@dataclass
class ChampionModel:
    """Represents a champion model with epoch and loss information"""
    epoch: int
    loss: float
    filename: str
    model_config: ModelConfig
    training_config: TrainingConfig
    training_state: TrainingState

    def to_dict(self):
        """Convert champion model to dictionary"""
        return {
            'epoch': self.epoch,
            'loss': self.loss,
            'filename': self.filename,
            'model_config': self.model_config.to_dict(),
            'training_config': self.training_config.to_dict(),
            'training_state': self.training_state.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChampionModel':
        """Create ChampionModel from dictionary"""
        return cls(
            epoch=data['epoch'],
            loss=data['loss'],
            filename=data['filename'],
            model_config=ModelConfig.from_dict(data['model_config']),
            training_config=TrainingConfig.from_dict(data['training_config']),
            training_state=TrainingState.from_dict(data['training_state'])
        )

class ChampionManager:
    """Manages champion models with new configuration structure support"""
    def __init__(self, model_dir: str, max_champions: int = 10):
        self.model_dir = model_dir
        self.max_champions = max_champions
        self.current_champions: List[ChampionModel] = []# List of ChampionModel objects, kept sorted
        self.all_champion_epochs: Set[int] = set()# set of all epochs that were ever champions - a set, so when loading existing models, we prevent duplication.
        self.logger = logging.getLogger(__name__)
        
        # Initialize from filesystem
        self.sync_with_filesystem()

    def add_champion(self, epoch: int, loss: float, filename: str,
                    model_config: ModelConfig, training_config: TrainingConfig,
                    training_state: TrainingState):
        """Add a new champion model with configuration data"""
        new_champion = ChampionModel(
            epoch=epoch,
            loss=loss,
            filename=filename,
            model_config=model_config,
            training_config=training_config,
            training_state=training_state
        )
        
        self.current_champions.append(new_champion)
        self.current_champions.sort(key=lambda x: x.loss)
        self.all_champion_epochs.add(epoch)
        
        # Remove excess champions
        if len(self.current_champions) > self.max_champions:
            removed_champion = self.current_champions.pop()
            self.logger.info(f"Removed excess champion model: {removed_champion.filename}")
            filepath = os.path.join(self.model_dir, removed_champion.filename)
            if os.path.exists(filepath):
                os.remove(filepath)

    def get_current_champions(self) -> List[ChampionModel]:
        """Get list of current champion models"""
        return self.current_champions

    def get_all_champion_epochs(self) -> List[int]:
        """Get sorted list of all champion epochs"""
        return sorted(list(self.all_champion_epochs))

    def sync_with_filesystem(self):
        """Synchronize champions with filesystem, handling new configuration structure"""
        champion_files = [f for f in os.listdir(self.model_dir) 
                         if f.startswith('champion_')]
        filesystem_champions = []
        
        for file in champion_files:
            try:
                # Extract basic information from filename
                match = re.search(r'champion_epoch_(\d+)_loss_([\d.]+)\.pth', file)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    
                    # Load checkpoint to get configurations
                    checkpoint = torch.load(
                        os.path.join(self.model_dir, file),
                        map_location='cpu'
                    )
                    
                    # Create champion model with configurations
                    champion = ChampionModel(
                        epoch=epoch,
                        loss=loss,
                        filename=file,
                        model_config=ModelConfig.from_dict(checkpoint['model_config']),
                        training_config=TrainingConfig.from_dict(checkpoint['training_config']),
                        training_state=TrainingState.from_dict(checkpoint['training_state'])
                    )
                    
                    filesystem_champions.append(champion)
                    self.all_champion_epochs.add(epoch)
                    
            except Exception as e:
                self.logger.error(f"Error loading champion {file}: {str(e)}")
        
        # Sort and limit champions
        filesystem_champions.sort(key=lambda x: x.loss)
        self.current_champions = filesystem_champions[:self.max_champions]
        
        # Remove any excess files
        for file in champion_files:
            if file not in [c.filename for c in self.current_champions]:
                try:
                    os.remove(os.path.join(self.model_dir, file))
                    self.logger.info(f"Removed excess champion file: {file}")
                except Exception as e:
                    self.logger.error(f"Error removing {file}: {str(e)}")

    def get_best_champion(self) -> Optional[ChampionModel]:
        """Get the champion model with lowest loss"""
        return self.current_champions[0] if self.current_champions else None