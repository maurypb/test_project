import os
import logging
import re

# a meaningless comment here.
# a third meaningless comment here..

class ChampionModel:
    def __init__(self, epoch, loss, filename):
        self.epoch = epoch
        self.loss = loss
        self.filename = filename

class ChampionManager:
    def __init__(self, model_dir, max_champions=10):
        self.model_dir = model_dir
        self.max_champions = max_champions
        self.current_champions = []  # List of ChampionModel objects, kept sorted
        self.all_champion_epochs = set()  # set of all epochs that were ever champions - a set, so when loading existing models, we prevent duplication.
        self.sync_with_filesystem()

    def add_champion(self, epoch, loss, filename):
        new_champion = ChampionModel(epoch, loss, filename)
        self.current_champions.append(new_champion)
        self.current_champions.sort(key=lambda x: x.loss)
        self.all_champion_epochs.add(epoch)
        
        if len(self.current_champions) > self.max_champions:
            removed_champion = self.current_champions.pop()
            logging.info(f"Removed excess champion model: {removed_champion.filename}")
            os.remove(os.path.join(self.model_dir, removed_champion.filename))

    def get_current_champions(self):
        return self.current_champions

    def get_all_champion_epochs(self):
        return sorted(list(self.all_champion_epochs))  # Return a sorted list for consistency

    def sync_with_filesystem(self):
        champion_files = [f for f in os.listdir(self.model_dir) if f.startswith('champion_')]
        filesystem_champions = []
        
        for file in champion_files:
            match = re.search(r'champion_epoch_(\d+)_loss_([\d.]+)\.pth', file)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                filesystem_champions.append(ChampionModel(epoch, loss, file))
                self.all_champion_epochs.add(epoch)
        
        # Sort champions by loss and limit to max_champions
        filesystem_champions.sort(key=lambda x: x.loss)
        self.current_champions = filesystem_champions[:self.max_champions]
        
        # Ensure all_champion_epochs includes epochs from filesystem champions
        self.all_champion_epochs.update(c.epoch for c in self.current_champions)
        
        # Remove any champion files that are no longer in the top max_champions
        for file in champion_files:
            if file not in [c.filename for c in self.current_champions]:
                os.remove(os.path.join(self.model_dir, file))