import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
import logging
import time
from datetime import timedelta
import threading
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from io import BytesIO

from msrn_model_v9 import (
    ModelConfig, 
    TrainingConfig, 
    TrainingState,
    MSRNHybridModel,
    VRAMEstimator,
    create_msrn_model
)
from champions_classes import ChampionModel, ChampionManager
from ModelSaver_class_v02 import ModelSaver
from gpu_tile_manager_v8 import GPUTileManager
from Image_load_and_save_methods_v01 import load_image

class TiledTrainingManager(QObject):
    # Qt Signals
    training_progress = pyqtSignal(int, float)  # Epoch, Loss
    training_complete = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_progress = pyqtSignal(int, int)  # Current batch, Total batches
    update_loss_graph = pyqtSignal(QImage)
    update_sample_tiles = pyqtSignal(QImage)
    update_loss_graph_interactive = pyqtSignal(list, int, int, float, float, list, list)
    vram_warning = pyqtSignal(str)  # New signal for VRAM-related warnings

    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig, training_root: str):
        """
        Initialize the TiledTrainingManager with separated configurations.
        
        Args:
            model_config: Configuration for model architecture
            training_config: Configuration for training parameters
            training_root: Root directory for training data and outputs
        """
        super().__init__()
        signal.signal(signal.SIGINT, self.handle_sigint)  #this is what handles ctrl+c interrupts
        self.training_thread = None
        self.stop_requested = False
     
        
        # Store configurations
        self.model_config = model_config
        self.training_config = training_config
        self.training_state = TrainingState(
            total_epochs=training_config.num_epochs  # Initialize with total epochs
        )

       # Set up directory structure
        self.training_root = Path(training_root)
        self.model_dir = self.training_root / "models"
        self.source_images_dir = self.training_root / "source_images"
        self.target_images_dir = self.training_root / "target_images"
        
        # Ensure directories exist
        for dir_path in [self.model_dir, self.source_images_dir, self.target_images_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


        # Initialize model related components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_model_and_optimizer()
        self.model_saver = ModelSaver(str(self.model_dir))

        # Initialize training components
        self.champion_manager = ChampionManager(str(self.model_dir), max_champions=6)
        self.gpu_tile_manager = self._initialize_gpu_tile_manager()
        self.criterion = nn.MSELoss()
        
        # Check VRAM constraints and adjust if needed
        if training_config.vram_limit_gb:
            self.check_vram_constraints()
        
        # Visualization settings
        self.visualize_interval = 5

        # # Training state flags - these are all tracked in the TrainingState object now.
        # self.training = False  # Tracks if training is actively running
        # self.training_paused = False  # Tracks if training is paused
        # self.model_ready = False  # Tracks if model is ready for inference
        # self.image_set_validated = False  # Tracks if image set has been validated
        # self.images_opened = False  # Tracks if visualization images have been opened
        
        # # Dataset tracking
        # self.dataset_signature = None  # Will be generated when needed
        

    def initialize_model_and_optimizer(self):
        """Initialize or reinitialize model and optimizer with current configurations"""
        self.model = create_msrn_model(self.model_config).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.training_config.learning_rate
        )

    def check_vram_constraints(self):
        """Check and adjust batch size based on VRAM constraints"""
        try:
            adjusted_batch_size = VRAMEstimator.adjust_batch_size_for_vram(
                self.model_config,
                self.training_config,
                self.training_config.vram_limit_gb
            )
            
            if adjusted_batch_size != self.training_config.batch_size:
                message = (f"Adjusted batch size from {self.training_config.batch_size} "
                          f"to {adjusted_batch_size} based on VRAM constraints")
                self.vram_warning.emit(message)
                self.training_config.batch_size = adjusted_batch_size
                
        except ValueError as e:
            self.training_error.emit(str(e))
            raise

    def _initialize_gpu_tile_manager(self) -> GPUTileManager:
        """Initialize GPU tile manager with validated images"""
        source_images, target_images, success, error_message = self.validate_images(
            self.source_images_dir, 
            self.target_images_dir
        )
        
        if not success:
            logging.error(error_message)
            raise ValueError(error_message)

        return GPUTileManager(
            source_images, 
            target_images, 
            self.model_config.tile_size,
            device=self.device,
            use_grid_method=True
        )

    @staticmethod
    def validate_images(source_images_dir, target_images_dir):
        """Validate and load training images"""
        source_filenames = set(os.listdir(source_images_dir))
        target_filenames = set(os.listdir(target_images_dir))
        
        if source_filenames != target_filenames:
            return None, None, False, "Source and target image sets do not match."
        
        source_images = []
        target_images = []
        success = True
        error_message = ""
        
        for image_name in sorted(source_filenames):
            source_path = str(source_images_dir / image_name)
            target_path = str(target_images_dir / image_name)
            
            source_image = load_image(source_path)
            target_image = load_image(target_path)
            
            if source_image.shape != target_image.shape:
                success = False
                error_message += f"Size mismatch for {image_name}: {source_image.shape} vs {target_image.shape}\n"
                
            source_images.append(source_image)
            target_images.append(target_image)

        return source_images, target_images, success, error_message



    def train(self):
        """Start training in a separate thread"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.stop_requested = False
            self.training_state.resume_training()  # Sets is_training=True, is_paused=False
            self.training_thread = threading.Thread(target=self._train_thread)
            self.training_thread.start()

    def stop_training(self, save_checkpoint=False, filename_prefix=None):
        """Stop training gracefully with optional checkpoint save"""
        if save_checkpoint:
            self.save_checkpoint(
                is_champion=False,
                filename_prefix=filename_prefix
            )
        #update operational flags
        self.stop_requested = True
        self.training_state.pause_training()  # Sets is_training=False, is_paused=True

    def _train_thread(self):
        """Main training loop running in separate thread"""
        try:
            start_time = time.time()
            
            while (self.training_state.current_epoch < self.training_config.num_epochs 
                   and not self.stop_requested):
                
                if not self.training_state.is_training:
                    break
                
                epoch_start_time = time.perf_counter()
                self.train_epoch()
                epoch_time = time.perf_counter() - epoch_start_time
                
                # Save checkpoint if interval reached
                if (self.training_state.current_epoch % self.training_config.save_interval == 0):
                    if self.training_state.dataset_signature is None:
                        self.training_state.dataset_signature = self._generate_dataset_signature()
                    self.save_checkpoint(is_champion=False)
                
                # Visualization updates
                if self.training_state.current_epoch % self.visualize_interval == 0:
                    self.update_visualizations()
                    self.log_progress(start_time)

                # Update UI
                self.training_progress.emit(
                    self.training_state.current_epoch, 
                    self.training_state.current_loss
                )

                # Update interactive loss plot
                self.update_loss_graph_interactive.emit(
                    self.training_state.losses,
                    self.training_state.current_epoch,
                    self.training_state.total_epochs,
                    self.training_state.current_loss,
                    self.training_state.best_loss,
                    [{'epoch': e, 'loss': l} for e, l in zip(
                        self.training_state.champion_epochs,
                        [self.training_state.losses[e-1] for e in self.training_state.champion_epochs 
                         if e <= len(self.training_state.losses)]
                    )],
                    self.champion_manager.get_current_champions()
                )
                


                
                self.training_state.current_epoch += 1
            
            self.training_complete.emit()
            
        except Exception as e:
            self.training_error.emit(str(e))
            logging.error(str(e))

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_count = self.gpu_tile_manager.total_tiles_per_epoch // self.training_config.batch_size

        for batch_idx in range(batch_count):
            if self.stop_requested:
                break

            # Get batch and train
            source_batch, target_batch = self.gpu_tile_manager.get_next_batch(
                self.training_config.batch_size
            )
            
            self.optimizer.zero_grad()
            outputs = self.model(source_batch)
            loss = self.criterion(outputs, target_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress
            self.epoch_progress.emit(batch_idx + 1, batch_count)

        # Update training state
        self.training_state.current_loss = total_loss / batch_count
        self.training_state.losses.append(self.training_state.current_loss)
        
        # Check for new champion
        if self.training_state.current_loss < self.training_state.best_loss * (
            1 - self.training_config.champion_improvement_threshold
        ):
            self.save_checkpoint(is_champion=True)
            self.training_state.best_loss = self.training_state.current_loss

    def update_visualizations(self):
        """Update all visualization elements"""
        self.update_loss_graph_plot(self.training_state.current_epoch + 1)
        self.update_loss_graph_plot_interactive(self.training_state.current_epoch + 1)   
        self.update_representative_tiles_image()

    def save_checkpoint(self, is_champion: bool = False, filename_prefix: str = None):
        """Save a checkpoint or champion model"""
        return self.model_saver.save_model(
            model=self.model,
            optimizer=self.optimizer,
            model_config=self.model_config,
            training_config=self.training_config,
            training_state=self.training_state,
            is_champion=is_champion,
            filename_prefix=filename_prefix
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint with all configurations"""
        checkpoint = self.model_saver.load_model(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.device
        )

        
        # Update configurations and state
        self.model_config = checkpoint['model_config']
        self.training_config = checkpoint['training_config']
        self.training_state = checkpoint['training_state']

        # Update operational flags
        self.model_ready = True
        self.training_paused = True
        self.training = False
        
        # Reinitialize components if needed
        self.initialize_model_and_optimizer()
        if self.training_config.vram_limit_gb:
            self.check_vram_constraints()
            
        self.champion_manager.sync_with_filesystem()
        
        logging.info(f"Resuming from epoch {self.training_state.current_epoch + 1}")

    def handle_sigint(self, signal, frame):
        """Handle interrupt signal (Ctrl+C) by saving state and exiting gracefully"""
        logging.info("Training interrupted. Saving model and exiting.")
        self.stop_training(
            save_checkpoint=True,
            filename_prefix="interrupted"
        )
        sys.exit(0)

    def update_loss_graph_plot(self, current_epoch):
        """Create and emit the loss plot visualization"""
        # Create a dual-axis plot with linear and log scales
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot the losses on the linear scale (left y-axis)
        ax1.plot(self.training_state.losses, label='Training Loss (Linear)', color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (Linear)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(
            f'Training Loss (Epoch {current_epoch}/{self.training_state.total_epochs}, '
            f'Current Loss: {self.training_state.current_loss:.6f}, '
            f'Min Loss: {self.training_state.best_loss:.6f})'
        )

        # Create a second y-axis for the log scale
        ax2 = ax1.twinx()
        ax2.plot(self.training_state.losses, label='Training Loss (Log)', color='green')
        ax2.set_yscale('log')
        ax2.set_ylabel('Loss (Log)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Plot champion indicators
        for epoch in self.training_state.champion_epochs:
            if epoch <= len(self.training_state.losses):
                is_current = any(c.epoch == epoch for c in self.champion_manager.get_current_champions())
                color = 'red' if is_current else 'black'
                linestyle = '--' if is_current else ':'
                linewidth = 1.0 if is_current else 0.5
                ax1.axvline(x=epoch-1, color=color, linestyle=linestyle, linewidth=linewidth,
                          label='Current Champion' if is_current and epoch == self.training_state.champion_epochs[0] 
                          else 'Past Champion' if epoch == self.training_state.champion_epochs[0] else "")
                
                # Add loss value annotations
                ax2.annotate(
                    f'{self.training_state.losses[epoch-1]:.6f}',
                    (epoch-1, self.training_state.losses[epoch-1]),
                    textcoords="offset points",
                    xytext=(5,0),
                    ha='left',
                    fontsize=5.5,
                    color="black"
                )

        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Convert to QImage and emit
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.update_loss_graph.emit(QImage.fromData(buf.getvalue()))
        plt.close()

    def update_loss_graph_plot_interactive(self, current_epoch):
        """
        Update the interactive loss plot with current training state.
        Now uses consolidated TrainingState for all metrics.
        """
        current_champions = self.champion_manager.get_current_champions()
        
        # Convert champion data into the format expected by the UI
        all_champions_data = [
            {'epoch': e, 'loss': self.training_state.losses[e-1]} 
            for e in self.training_state.champion_epochs 
            if e <= len(self.training_state.losses)
        ]
        current_champions_data = [
            {'epoch': c.epoch, 'loss': c.loss} 
            for c in current_champions
        ]

        # Emit signal with current state
        self.update_loss_graph_interactive.emit(
            self.training_state.losses,
            current_epoch,
            self.training_state.total_epochs,
            self.training_state.current_loss,
            self.training_state.best_loss,
            all_champions_data,
            current_champions_data
        )

    def update_representative_tiles_image(self):
        """Create and emit the sample tiles visualization"""
        sample_tiles = self.get_sample_tiles(num_samples=3)
        
        # Image layout parameters
        tile_size = 128
        padding = 10
        font_size = 12
        header_height = 30
        num_tiles = len(sample_tiles)

        # Calculate grid dimensions
        grid_width = (tile_size * 5 + padding * 6)
        grid_height = header_height + (tile_size + padding) * num_tiles + padding
        
        # Create base image using PIL
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        draw = ImageDraw.Draw(grid_image)
        font = ImageFont.load_default().font_variant(size=font_size)
        
        # Add headers
        headers = ['Source', 'Target', 'Transformed', 'Difference', 'Gamma Up Diff']
        for i, header in enumerate(headers):
            x = i * (tile_size + padding) + padding + tile_size // 2
            y = header_height // 2
            draw.text((x, y), header, fill='black', font=font, anchor='mm')

        # Convert from PIL to numpy array for processing
        grid_array = np.array(grid_image)

        # Process each tile set
        for row, tile_set in enumerate(sample_tiles):
            y_offset = header_height + row * (tile_size + padding) + padding
            
            # Place source, target, and output tiles
            for col, (key, tensor) in enumerate([
                ('source', tile_set['source']),
                ('target', tile_set['target']),
                ('output', tile_set['output'])
            ]):
                 # Convert tensor to numpy array and resize
                np_image = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                np_image = cv2.resize(np_image, (tile_size, tile_size))

                # Place in grid array
                x_offset = col * (tile_size + padding) + padding
                grid_array[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size] = np_image

            # Calculate and add difference visualization
            diff = torch.abs(tile_set['target'] - tile_set['output'])
            diff_np = (diff.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            diff_np = cv2.resize(diff_np, (tile_size, tile_size))
            grid_array[
                y_offset:y_offset+tile_size,
                3*(tile_size+padding)+padding:4*(tile_size+padding)
            ] = diff_np

            # Add gamma-corrected difference visualization
            norm_diff = (diff / diff.max())**0.25
            norm_diff_np = (norm_diff.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            norm_diff_np = cv2.resize(norm_diff_np, (tile_size, tile_size))
            grid_array[
                y_offset:y_offset+tile_size,
                4*(tile_size+padding)+padding:5*(tile_size+padding)
            ] = norm_diff_np

        # Convert to QImage and emit
        height, width, channel = grid_array.shape
        bytes_per_line = 3 * width
        self.update_sample_tiles.emit(
            QImage(grid_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        )

        return QImage(grid_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

    def get_sample_tiles(self, num_samples=3):
        """Get sample tiles for visualization"""
        self.model.eval()
        sample_tiles = []
        
        with torch.no_grad():
            source_batch, target_batch = self.gpu_tile_manager.get_next_batch(num_samples)
            output_batch = self.model(source_batch)
            
            for i in range(num_samples):
                sample_tiles.append({
                    'source': source_batch[i],
                    'target': target_batch[i],
                    'output': output_batch[i]
                })
        
        return sample_tiles


    def _generate_dataset_signature(self) -> str:
        """
        Generate a unique signature for the current dataset.
        Updates the signature in TrainingState.
        """
        source_files = set(os.listdir(self.source_images_dir))
        target_files = set(os.listdir(self.target_images_dir))

        if source_files != target_files:
            raise ValueError("Mismatch between source and target image files")

        total_files = len(source_files)
        logging.info(f"Generating signature for {total_files} image pairs")

        signature = {}
        for i, filename in enumerate(sorted(source_files), 1):
            source_path = os.path.join(self.source_images_dir, filename)
            target_path = os.path.join(self.target_images_dir, filename)
            
            source_hash = self._calculate_file_hash(source_path)
            target_hash = self._calculate_file_hash(target_path)
            signature[filename] = (source_hash, target_hash)
            
            if i % 5 == 0 or i == total_files:
                logging.info(f"Processed {i}/{total_files} image pairs")

        # Update the signature in training state
        self.training_state.dataset_signature = signature
        return signature


    def get_dataset_signature(self) -> str:
        """
        Get current dataset signature, generating if needed.
        """
        if self.training_state.dataset_signature is None:
            self.training_state.dataset_signature = self._generate_dataset_signature()
        return self.training_state.dataset_signature


    def _calculate_file_hash(self, file_path: str) -> str:
        """Helper method for generating dataset signature"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def log_progress(self, start_time):
        """
        Log training progress using consolidated state.
        """
        elapsed_time = time.time() - start_time
        epochs_remaining = (self.training_state.total_epochs - 
                          (self.training_state.current_epoch + 1))
        estimated_time_remaining = (elapsed_time / 
                                  (self.training_state.current_epoch + 1) * 
                                  epochs_remaining)

        logging.info(
            f"Epoch [{self.training_state.current_epoch + 1}/"
            f"{self.training_state.total_epochs}], "
            f"Average Loss: {self.training_state.current_loss:.6f}"
        )
        logging.info(f"Elapsed time: {timedelta(seconds=int(elapsed_time))}")
        logging.info(
            f"Estimated time remaining: "
            f"{timedelta(seconds=int(estimated_time_remaining))}"
        )


if __name__ == "__main__":
    import logging
    from pathlib import Path

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 1. Set up configurations
    model_config = ModelConfig(
        base_features=64,
        num_blocks=8,
        tile_size=256,
        use_pyramid=True,
        pyramid_scales=2,
        use_multi_path=True,
        use_attention=True,
        path_weights_learnable=True
    )

    training_config = TrainingConfig(
        batch_size=8,
        learning_rate=0.0001,
        num_epochs=1000,
        save_interval=10,
        champion_improvement_threshold=0.1,
        vram_limit_gb=24.0  # For automatic VRAM management
    )

    # 2. Set up training directory structure
    training_root = Path("/path/to/training/root")
    
    try:
        # 3. Initialize the training manager
        trainer = TiledTrainingManager(
            model_config=model_config,
            training_config=training_config,
            training_root=str(training_root)
        )

        # 4. Optional: Connect to Qt signals if using in GUI
        def handle_progress(epoch, loss):
            logger.info(f"Training progress - Epoch: {epoch}, Loss: {loss:.6f}")

        def handle_training_complete():
            logger.info("Training completed successfully")

        def handle_training_error(error_msg):
            logger.error(f"Training error: {error_msg}")

        def handle_vram_warning(warning_msg):
            logger.warning(f"VRAM warning: {warning_msg}")

        trainer.training_progress.connect(handle_progress)
        trainer.training_complete.connect(handle_training_complete)
        trainer.training_error.connect(handle_training_error)
        trainer.vram_warning.connect(handle_vram_warning)

        # 5. Load existing checkpoint (optional)
        latest_champion = trainer.model_dir / "champion_latest.pth"
        if latest_champion.exists():
            logger.info(f"Loading checkpoint: {latest_champion}")
            trainer.load_checkpoint(str(latest_champion))
            logger.info(
                f"Resumed from epoch {trainer.training_state.current_epoch}, "
                f"best loss: {trainer.training_state.best_loss:.6f}"
            )

        # 6. Start training
        logger.info("Starting training...")
        trainer.train()

        # 7. Training can be stopped gracefully
        def handle_interrupt():
            logger.info("Interrupting training...")
            trainer.stop_training(
                save_checkpoint=True,
                filename_prefix="interrupted"
            )

        # Example of accessing training state
        def print_training_status():
            state = trainer.training_state
            logger.info(f"""
                Training Status:
                - Current Epoch: {state.current_epoch}/{state.total_epochs}
                - Current Loss: {state.current_loss:.6f}
                - Best Loss: {state.best_loss:.6f}
                - Is Training: {state.is_training}
                - Is Paused: {state.is_paused}
                - Model Ready: {state.model_ready}
                - Number of Champion Models: {len(state.champion_epochs)}
            """)

        # Example of using trained model for inference
        def run_inference(input_image_path):
            if trainer.training_state.model_ready:
                trainer.stop_training(save_checkpoint=True)
                
                # Load and process image
                input_image = load_image(input_image_path)
                with torch.no_grad():
                    trainer.model.eval()
                    # Generate tiles
                    tiles = trainer.gpu_tile_manager.generate_tiles(input_image)
                    processed_tiles = []
                    
                    for tile, position in tiles:
                        processed_tile = trainer.model(
                            tile.unsqueeze(0).to(trainer.device)
                        ).squeeze(0)
                        processed_tiles.append((processed_tile.cpu(), position))
                    
                    # Reconstruct image
                    result = trainer.gpu_tile_manager.reconstruct_image_v2(
                        processed_tiles, 
                        input_image.shape[1:]
                    )
                    
                    return torch.clamp(result, 0, 1)
            else:
                logger.error("Model not ready for inference")
                return None

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

    finally:
        # Clean up
        logger.info("Cleaning up...")
        if trainer.training_state.is_training:
            trainer.stop_training(save_checkpoint=True)