import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" #enable exr support in opencv
import sys
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
#from tiled_dataset_class_v5 import TiledDataset
from PIL import Image,ImageDraw, ImageFont
import cv2
from io import BytesIO


#integrate the threaded training manager class into the tiled system
import threading
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage
from msrn_model_v5 import MSRNConfig, create_msrn_model_from_config
import logging
import time
from datetime import timedelta

import subprocess
import webbrowser
import atexit
import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import hashlib
from typing import Dict, Tuple   
import shutil
from pathlib import Path
import subprocess
import platform
import re
import numpy as np
from champions_classes import ChampionModel, ChampionManager
from ModelSaver_class import ModelSaver
from gpu_tile_manager_v8 import GPUTileManager
from Image_load_and_save_methods_v01 import load_image, save_image
from PyQt6.QtGui import QImage

class TiledTrainingManager(QObject): #now threaded
    #signals from the training manager to the UI main thread
    training_progress = pyqtSignal(int, float)  # Epoch, Loss
    training_complete = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_progress = pyqtSignal(int, int)  # Epoch, Batch Index
    update_loss_graph = pyqtSignal(QImage)  # Loss graph update
    update_sample_tiles = pyqtSignal(QImage)   # Sample tiles update
    update_loss_graph_interactive = pyqtSignal(list, int, int, float, float, list, list)  # losses, current_epoch, total_epochs, current_loss, min_loss, all_champions, current_champions

    def __init__(self, config: MSRNConfig, training_root: str):
        """
        Initialize the TiledTrainingManager.
        
        Args:
            config (MSRNConfig): Configuration for the MSRN model and training
            train_dataset (TiledDataset): The dataset for training
            model_dir (str): Directory to save models and logs
        """
        super().__init__() #initialize the QObject class
        self.training_thread=None
        self.stop_requested=False
        signal.signal(signal.SIGINT, self.handle_sigint)  #this is what handles ctrl+c interrupts
        self.config = config
        self.training_root = Path(training_root)
        self.dataset_signature = None
        # Define standard subdirectories
        self.model_dir = self.training_root / "models"
        self.source_images_dir = self.training_root / "source_images"
        self.target_images_dir = self.training_root / "target_images"

        self.global_step = 0 # Global step counter for TensorBoard

        # Ensure all directories exist
        for dir_path in [self.model_dir, self.source_images_dir, self.target_images_dir]:
                         #self.source_tiles_dir, self.target_tiles_dir, self.tb_log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize other components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_msrn_model_from_config(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        

        self.model_saver = ModelSaver(str(self.model_dir))
        
        # Initialize training dataset
        #should we delete all of the image tiles first? added that functionality to the TiledDataset class.
        #self.train_dataset = self._initialize_dataset()
        self.gpu_tile_manager=self._initialize_gpu_tile_manager()

        # Initialize other training state variables
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.avg_loss = float('inf')
        self.champion_manager = ChampionManager(str(self.model_dir), max_champions=6)  # Adjust max_champions as needed
        self.losses = []
        #self.champion_epochs = []

        # Set up logging and visualization
        self.tb_log_dir = os.path.join(self.model_dir, 'tensorboard_logs')
        #self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        self.model_saver = ModelSaver(self.model_dir)
        #self.model_saver.synchronize_champion_models() # Ensure the champion models are up-to-date and synchronozed at start
        #self.jupyter_notebook_path = os.path.join(self.model_dir, 'training_visualization.ipynb')
        self.images_opened=False #this is a flag to keep track of whether the visualisation images have been opened yet.
        self.visualize_interval=5 #how often to visualize the training progress; eg 10 means every 10 epochs.







    def _initialize_gpu_tile_manager(self) -> GPUTileManager:
        
        source_images, target_images,success,error_message = self.validate_images(self.source_images_dir, self.target_images_dir)
        if not success:
            logging.error(error_message)
            raise ValueError(error_message)

        return GPUTileManager(source_images, target_images, self.config.tile_size, device=self.device,use_grid_method=True)
       
    @staticmethod 
    def validate_images(source_images_dir, target_images_dir):
        """Validate source and target images and return a list of paired image paths."""
        #made into a separate static method so that it can be used by other classes.
        source_filenames = set(os.listdir(source_images_dir))
        target_filenames = set(os.listdir(target_images_dir))
        success=True
        error_message="" 
        if source_filenames != target_filenames:
            #raise ValueError("Source and target image sets do not match.")
            success=False
            error_message+="Source and target image sets do not match.\n"
            return None, None, success, error_message
        source_images = []
        target_images = []
        
        for image_name in sorted(source_filenames):
             

            source_path = str(source_images_dir / image_name)
            target_path = str(target_images_dir / image_name)

            # Load images
            source_image = load_image(source_path)
            target_image = load_image(target_path)

            if source_image.shape != target_image.shape:
                #raise ValueError(f"Size mismatch for {image_name}: {source_image.shape} vs {target_image.shape}")
                success=False
                error_message+=f"Size mismatch for {image_name}: {source_image.shape} vs {target_image.shape}\n"
            source_images.append(source_image)
            target_images.append(target_image)

        return source_images, target_images, success, error_message

    def train(self):
        if self.training_thread is None or not self.training_thread.is_alive():
            self.stop_requested=False
            self.training_thread = threading.Thread(target=self._train_thread)
            self.training_thread.start()

    def _train_thread(self):
        """Main training loop."""
        try:
            start_time = time.time()
            for epoch in range(self.current_epoch, self.config.num_epochs):
                if self.stop_requested:
                    logging.info("Training stopped.")
                    #self.save_checkpoint(epoch, self.model, self.optimizer, self.avg_loss, is_champion=False,filename_prefix="user_stopped_training")
                    break
                epoch_start_time = time.perf_counter()
                self.train_epoch(epoch)
                epoch_time = time.perf_counter()-epoch_start_time
                if (epoch) % self.config.save_interval == 0:
                    if self.dataset_signature is None:
                        self.dataset_signature = self._generate_dataset_signature()

                    self.save_checkpoint(epoch, self.model, self.optimizer, self.avg_loss, is_champion=False)
                    


                if epoch % self.visualize_interval == 0:
                    #sample_tiles_start_time = time.perf_counter()
                    #sample_tiles = self.get_sample_tiles()
                    #sample_tiles_time=time.perf_counter()-sample_tiles_start_time
                    plots_start_time = time.perf_counter()
                    self.update_loss_graph_plot(epoch + 1)
                    #self.update_loss_graph_plot_interactive(epoch + 1)  # Update the interactive plot
                    self.update_representative_tiles_image()
                    plots_time=time.perf_counter()-plots_start_time
                    logging.info(f"epoch time={epoch_time}, plots time={plots_time}")
                    #self.open_plot_images()
                    self.log_progress(epoch, start_time)
                    self.current_epoch = epoch
                    total_epoch_time = time.perf_counter()-epoch_start_time
                    #logging.info(f"tiles pct: {sample_tiles_time/total_epoch_time}, plots pct: {plots_time/total_epoch_time}")
                #send progress to the UI thread
                self.training_progress.emit(epoch, self.avg_loss)#send progress to the UI thread...(epoch, loss) types are (int, float)
                self.update_loss_graph_plot_interactive(epoch + 1)  # Update the interactive plot
            #training is complete        
            self.training_complete.emit()
        except Exception as e:
            self.training_error.emit(str(e))
            logging.error(str(e))   

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        """
        epoch_start_time = time.perf_counter()
      
        self.model.train()
        total_loss = 0
        batch_count = self.gpu_tile_manager.total_tiles_per_epoch // self.config.batch_size 
        
        progress_bar = tqdm(range(batch_count), desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx in progress_bar:
            source_batch, target_batch = self.gpu_tile_manager.get_next_batch(self.config.batch_size)
            if self.stop_requested:
                break   
            
            self.optimizer.zero_grad()
            outputs = self.model(source_batch)
            loss = self.criterion(outputs, target_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            #self.writer.add_scalar('Loss/batch', loss.item(), epoch * len(train_loader) + batch_idx)
            
            progress_bar.set_postfix({'loss': loss.item()})
        
            self.epoch_progress.emit(batch_idx + 1, batch_count) #send progress to the UI thread...(batch index, total number of batches in this epoch)
            # use (batch_idx+1)/batch_count * 100 to get the percentage of completion for the progress bar, back in the pyqt GUI.

        
        self.avg_loss = total_loss / batch_count
        self.losses.append(self.avg_loss)
        #self.writer.add_scalar('Loss/epoch', self.avg_loss, self.global_step)
        self.global_step += 1
        
        if self.avg_loss < self.best_loss * (1 - self.config.champion_improvement_factor):  # New champion
            if self.dataset_signature is None:
                self.dataset_signature = self._generate_dataset_signature()

            filename = f'champion_epoch_{epoch:06d}_loss_{self.avg_loss:.6f}.pth'
            #self.champion_manager.add_champion(epoch, self.avg_loss, filename)
            self.save_checkpoint(epoch, self.model, self.optimizer, self.avg_loss, is_champion=True)

            self.best_loss = self.avg_loss  # note: this is not really the best loss, but the best loss of saved champions
        
        epoch_time = time.perf_counter()-epoch_start_time
        
        logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        #logging.info(f"statistics: augmentation percent={aug_percent}, dataloader percent={dataloader_percent}, compute percent={compute_percent}")

    def stop_training(self, save_checkpoint=False, filename_prefix=None):
        if save_checkpoint:
            if filename_prefix:
                self.save_checkpoint(self.current_epoch, self.model, self.optimizer, self.avg_loss, is_champion=False, filename_prefix=filename_prefix)
            else:
                self.save_checkpoint(self.current_epoch, self.model, self.optimizer, self.avg_loss, is_champion=False)
        self.stop_requested=True
    

    def update_loss_graph_plot_interactive(self, current_epoch):
        all_champion_epochs = self.champion_manager.get_all_champion_epochs()
        current_champions = self.champion_manager.get_current_champions()
        
        all_champions_data = [{'epoch': e, 'loss': self.losses[e-1]} for e in all_champion_epochs if e <= len(self.losses)]
        current_champions_data = [{'epoch': c.epoch, 'loss': c.loss} for c in current_champions]

        current_loss = self.losses[-1] if self.losses else float('inf')
        min_loss = min(self.losses) if self.losses else float('inf')

        self.update_loss_graph_interactive.emit(self.losses, current_epoch, self.config.num_epochs, current_loss, min_loss, all_champions_data, current_champions_data)


    def update_loss_graph_plot(self, current_epoch):
        #self.model_saver.synchronize_champion_models()
        current_champion_epochs = [c.epoch for c in self.champion_manager.get_current_champions()]
        all_champion_epochs = self.champion_manager.get_all_champion_epochs()

        # Create a dual-axis plot with linear and log scales
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot the losses on the linear scale (left y-axis)
        ax1.plot(self.losses, label='Training Loss (Linear)', color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (Linear)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(f'Training Loss (Epoch {current_epoch}/{self.config.num_epochs}, Current Loss: {self.losses[-1]:.6f}, Min Loss: {min(self.losses):.6f})')

        # Create a second y-axis for the log scale
        ax2 = ax1.twinx()
        ax2.plot(self.losses, label='Training Loss (Log)', color='green')
        ax2.set_yscale('log')
        ax2.set_ylabel('Loss (Log)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Plot vertical lines for champion models on both axes
        for e in all_champion_epochs:
            if e <= len(self.losses):
                color = 'red' if e in current_champion_epochs else 'black'
                linestyle = '--' if e in current_champion_epochs else ':'
                linewidth = 1.0 if e in current_champion_epochs else 0.5  # Adjust thickness
                ax1.axvline(x=e-1, color=color, linestyle=linestyle, linewidth=linewidth,
                            label='Current Champion' if e == all_champion_epochs[0] and e in current_champion_epochs else 
                                'Past Champion' if e == all_champion_epochs[0] else "")
                # Add small labels aligned with the log plot
                ax2.annotate(f'{self.losses[e-1]:.6f}', (e-1, self.losses[e-1]), 
                            textcoords="offset points", xytext=(5,0), ha='left', 
                            fontsize=5.5, color="black")

        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # # Save the plot as an image
        # loss_plot_path2 = os.path.join(self.model_dir, 'loss_plot_dual_axis.png')
        # plt.savefig(loss_plot_path2)

        # Convert dual-axis plot to QImage to pass to gui
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        dual_axis_plot_image = QImage.fromData(buf.getvalue())
        self.update_loss_graph.emit(dual_axis_plot_image)





        # Close the current figure
        plt.close()
        
        # Create representative tiles image
        #self.create_representative_tiles_image(sample_tiles)

    def open_plot_images(self):
        """Open plot images using the system's default image viewer."""
        if not self.images_opened:
            images = [
                #os.path.join(self.model_dir, 'loss_plot.png'),
                os.path.join(self.model_dir, 'loss_plot_dual_axis.png'),
                os.path.join(self.model_dir, 'representative_tiles.png')  # We'll create this in visualize_data
            ]

            for image_path in images:
                if os.path.exists(image_path):
                    if platform.system() == 'Darwin':  # macOS
                        subprocess.call(('open', image_path))
                    elif platform.system() == 'Windows':
                        os.startfile(image_path)
                    else:  # Linux and other Unix-like
                        subprocess.call(('xdg-open', image_path))

            self.images_opened = True

    def update_representative_tiles_image(self):
        sample_tiles=self.get_sample_tiles()
        num_tiles = len(sample_tiles)
        tile_size = 128
        padding = 10
        font_size = 12
        header_height = 30

        # Calculate grid dimensions
        grid_width = (tile_size * 5 + padding * 6)
        grid_height = header_height + (tile_size + padding) * num_tiles + padding
        
        # Create base image using PIL
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        draw = ImageDraw.Draw(grid_image)
        font = ImageFont.load_default().font_variant(size=font_size)
        
        # Add column headers
        headers = ['Source', 'Target', 'Transformed', 'Difference', 'Gamma Up Diff']
        for i, header in enumerate(headers):
            x = i * (tile_size + padding) + padding + tile_size // 2
            y = header_height // 2
            draw.text((x, y), header, fill='black', font=font, anchor='mm')

        # Convert PIL Image to numpy array for tile placement
        grid_array = np.array(grid_image)

        for row, tile_set in enumerate(sample_tiles):
            y_offset = header_height + row * (tile_size + padding) + padding
            
            for col, (key, tensor) in enumerate([('source', tile_set['source']), 
                                                ('target', tile_set['target']), 
                                                ('output', tile_set['output'])]):
                # Convert tensor to numpy array and resize
                np_image = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                np_image = cv2.resize(np_image, (tile_size, tile_size))
                
                # Place the image in the grid
                x_offset = col * (tile_size + padding) + padding
                grid_array[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size] = np_image

            # Calculate and add difference image
            diff = torch.abs(tile_set['target'] - tile_set['output'])
            diff_np = (diff.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            diff_np = cv2.resize(diff_np, (tile_size, tile_size))
            grid_array[y_offset:y_offset+tile_size, 3*(tile_size+padding)+padding:4*(tile_size+padding)] = diff_np

            # Calculate and add normalized difference image
            norm_diff = (diff / diff.max())**0.25  # Apply gamma correction for better visualization
            norm_diff_np = (norm_diff.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            norm_diff_np = cv2.resize(norm_diff_np, (tile_size, tile_size))
            grid_array[y_offset:y_offset+tile_size, 4*(tile_size+padding)+padding:5*(tile_size+padding)] = norm_diff_np

        # # Save the grid image using cv2.imwrite
        # grid_image_path = os.path.join(self.model_dir, 'representative_tiles.png')
        # cv2.imwrite(grid_image_path, cv2.cvtColor(grid_array, cv2.COLOR_RGB2BGR))

        # Convert the numpy array to a QImage
        height, width, channel = grid_array.shape
        bytes_per_line = 3 * width
        self.update_sample_tiles.emit(QImage(grid_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888))

        return QImage(grid_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)





        #cv2.imwrite(grid_image_path, grid_array,)
        print(f"Saved grid image to {grid_image_path}")

        return grid_image_path

    def log_progress(self, epoch, start_time):
        """
        Log training progress.
        
        Args:
            epoch (int): Current epoch number
            start_time (float): Time when training started
        """
        elapsed_time = time.time() - start_time
        epochs_remaining = self.config.num_epochs - (epoch + 1)
        estimated_time_remaining = (elapsed_time / (epoch + 1 - self.start_epoch)) * epochs_remaining

        logging.info(f"Epoch [{epoch+1}/{self.config.num_epochs}], Average Loss: {self.avg_loss:.6f}")
        logging.info(f"Elapsed time: {timedelta(seconds=int(elapsed_time))}")
        logging.info(f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}")

    def get_sample_tiles(self, num_samples=3):
        """
        Get sample tiles for visualization using the GPUTileManager.
        
        Args:
            num_samples (int): Number of sample tile sets to return
        Returns:
            list: List of dictionaries containing sample source, target, and output tiles
        """
        self.model.eval()
        sample_tiles = []
        with torch.no_grad():
            # Get a batch of tiles from the GPUTileManager
            source_batch, target_batch = self.gpu_tile_manager.get_next_batch(num_samples)
            
            # Process the batch through the model
            output_batch = self.model(source_batch)
            
            # Create sample tile dictionaries
            for i in range(num_samples):
                sample_tiles.append({
                    'source': source_batch[i],
                    'target': target_batch[i],
                    'output': output_batch[i]
                })
        
        return sample_tiles



    def load_checkpoint(self, checkpoint_path: str):
        old_config = self.config

        checkpoint = self.model_saver.load_model(checkpoint_path, self.model, self.optimizer, self.device)
        self.config = MSRNConfig.from_dict(checkpoint['config'])
        self.current_epoch = checkpoint['epoch']+1
        self.start_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        self.dataset_signature = checkpoint['dataset_signature']
        self.losses = checkpoint['losses']
        self.global_step=len(self.losses) # Update global step counter for TensorBoard
        
        # Update all_champion_epochs with historical data from the checkpoint
        self.champion_manager.all_champion_epochs.update(checkpoint['all_champion_epochs'])
        

        self.champion_manager.sync_with_filesystem()
        
        current_signature = self.get_dataset_signature()
        if self.dataset_signature != current_signature or old_config.min_overlap != self.config.min_overlap or old_config.tile_size != self.config.tile_size:
            logging.warning("Training dataset or parameters have changed. Reinitializing dataset.")
            self.train_dataset = self._initialize_dataset()
            self.dataset_signature = current_signature


        logging.info(f"Resuming from epoch {self.current_epoch + 1}")

    def save_checkpoint(self, epoch, model, optimizer, loss, is_champion=False, filename_prefix=None):
        if self.dataset_signature is None:
            self.dataset_signature = self._generate_dataset_signature()
        
        champion_filename = self.model_saver.save_model(
            model, optimizer, self.config, epoch, loss, self.dataset_signature,
            self.champion_manager.get_all_champion_epochs(),
            [c.__dict__ for c in self.champion_manager.get_current_champions()],
            self.losses, is_champion, filename_prefix
        )
        
        if is_champion and champion_filename:
            self.champion_manager.add_champion(epoch, loss, champion_filename)

    def get_dataset_signature(self):
        if self.dataset_signature is None:
            self.dataset_signature = self._generate_dataset_signature()
        return self.dataset_signature

    def _generate_dataset_signature(self) -> Dict[str, Tuple[str, str]]:
        signature = {}
        source_files = set(os.listdir(self.source_images_dir))
        target_files = set(os.listdir(self.target_images_dir))

        if source_files != target_files:
            raise ValueError("Mismatch between source and target image files")

        total_files = len(source_files)
        logging.info(f"Generating signature for {total_files} image pairs")

        for i, filename in enumerate(sorted(source_files), 1):
            source_path = os.path.join(self.source_images_dir, filename)
            target_path = os.path.join(self.target_images_dir, filename)
            
            source_hash = self._calculate_file_hash(source_path)
            target_hash = self._calculate_file_hash(target_path)
            
            signature[filename] = (source_hash, target_hash)
            
            if i % 5 == 0 or i == total_files:
                logging.info(f"Processed {i}/{total_files} image pairs")

        return signature

    def _calculate_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def handle_sigint(self, signal, frame):  #this handles the ctrl+c interrupt signal.  The event is registered in the __init__ method.
        logging.info("Training interrupted. Saving model and exiting.")
        self.save_checkpoint(self.current_epoch, self.model, self.optimizer, self.avg_loss, is_champion=False,filename_prefix="interrupted")
        #self.stop_tensorboard()
        #self.stop_jupyter()
        sys.exit(0)

    

if __name__ == "__main__":


    # Example usage
    config = MSRNConfig(
        num_scales=4, 
        num_residual_blocks=28, #increased
        base_features=32, 
        use_pyramid=True,
        use_attention=True,
        learning_rate=0.0001,
        batch_size=8,
        num_epochs=24000,
        save_interval=100,
        champion_improvement_factor=0.1,
        tile_size=256,
        min_overlap=(16, 16),
        augmentation_factor=0.,
        rotation_range= (-2, 2),
        scale_range=(0.9, 1.1),
        contrast_range=(1.0, 1.0),
        brightness_range= (0.0, 0.0),
        hue_range = (-2, 2),
        noise_stddev_range = (0.0, 0.001),
        test_image= None,
        
        # ... other parameters ...
    )   

    # also works well with the following config
    # # Example usage
    # config = MSRNConfig(
    #     num_scales=4,
    #     num_residual_blocks=7,
    #     base_features=64,
    #     use_pyramid=True,
    #     use_attention=True,
    #     learning_rate=0.0001,
    #     batch_size=8,
    #     num_epochs=24000,
    #     save_interval=100,
    #     champion_improvement_factor=0.1,
    #     tile_size=256,
    #     min_overlap=(16, 16),
    #     augmentation_factor=0.,
    #     rotation_range= (-2, 2),
    #     scale_range=(0.9, 1.1),
    #     contrast_range=(1.0, 1.0),
    #     brightness_range= (0.0, 0.0),
    #     hue_range = (-2, 2),
    #     noise_stddev_range = (0.0, 0.001),
    #     test_image= None,
        
    #     # ... other parameters ...
    # )   










    ##WORKS PRETTY WELL...SEE /mnt/tank/PROJECTS/SOFTWARE_PROJECTS/duplidog/test_images/tucson_180/transformed_sequence_241002_1035
    # config = MSRNConfig(
    #     num_scales=4,
    #     num_residual_blocks=7,
    #     base_features=64,
    #     use_pyramid=False,
    #     use_attention=True,
    #     learning_rate=0.0001,
    #     batch_size=8,
    #     num_epochs=24000,
    #     save_interval=100,
    #     champion_improvement_factor=0.1,
    #     tile_size=256,
    #     min_overlap=(16, 16),
    #     augmentation_factor=0.,
    #     rotation_range= (-2, 2),
    #     scale_range=(0.9, 1.1),
    #     contrast_range=(1.0, 1.0),
    #     brightness_range= (0.0, 0.0),
    #     hue_range = (-2, 2),
    #     noise_stddev_range = (0.0, 0.001),
    #     test_image= None,
        
    #     # ... other parameters ...
    #)   




    #training_root = "/path/to/training/root"
    #training_root = "/mnt/tank/PROJECTS/SOFTWARE_PROJECTS/duplidog/test_images/lexus_testing_data_v2"
    training_root = "/mnt/tank/PROJECTS/SOFTWARE_PROJECTS/duplidog/test_images/tucson_180"
    trainer = TiledTrainingManager(config, training_root)

    #automatically load the latest champion model if available
    latest_champion = (trainer.model_dir / trainer.champion_manager.current_champions[0].filename 
                    if trainer.champion_manager.current_champions else None)


    if latest_champion:
        print(f"Loading latest champion model: {latest_champion}",flush=True)
        trainer.load_checkpoint(latest_champion)
    else:
        logging.info("No valid champion model found. Starting from scratch.")



    ## Optionally load a specific checkpoint
    # resume_from_this_checkpoint="/mnt/tank/PROJECTS/SOFTWARE_PROJECTS/duplidog/test_images/lexus_testing_data_v2/models/msrn_epoch_016200_loss_0.000033.pth"
    # trainer.load_checkpoint(resume_from_this_checkpoint)
    # #override some of the loaded model's config settings
    #trainer.config.learning_rate=0.001
    # trainer.config.num_epochs=24000 #override the number of epochs, so we keep training longer
    # trainer.config.augmentation_factor=0.5
    # trainer.config.rotation_range= (-2, 2)
    # trainer.config.scale_range=(0.9, 1.1)
    # trainer.config.contrast_range=(1.0, 1.0)
    # trainer.config.brightness_range= (0.0, 0.0)
    # trainer.config.hue_range = (0, 0)
    # trainer.config.noise_stddev_range = (0.0, 0.001)


    # Start training
    trainer.train()




