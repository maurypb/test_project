# part 1 core initialization


"""
TODO UI Updates Required in Qt Designer:
1. Add QLabel 'label_vram_usage' to the training parameters frame
   - Suggested text: "VRAM Usage: --GB / --GB"
   - Place near the top of training parameters
2. Add QSpinBox 'spinBox_pyramid_scales' to model parameters frame
   - Default value: 2
   - Range: 1-2
   - Label: "Pyramid Scales"
3. Remove unused legacy controls:
   - Remove any v5-specific model parameters
   - Remove compatibility-related options
4. Update terminology in labels to match v9:
   - "Residual Blocks" → "Processing Blocks"
   - "Use Pyramid" → "Multi-Scale Processing"
"""

import sys
import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QTabWidget, 
    QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox
)
from PyQt6.uic import loadUi    
from PyQt6.QtCore import pyqtSlot, Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage

from threaded_training_manager_for_msrn_v9 import TiledTrainingManager
from msrn_model_v9 import ModelConfig, TrainingConfig, VRAMEstimator
from inference_manager_for_pyqt_v002 import InferenceManager
from pyside_settings_manager_class_v001 import SettingsManager
from Image_load_and_save_methods_v01 import load_image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("setup_training_v014.ui", self)
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.init_device()
        self.init_paths()
        self.init_configurations()
        self.init_ui_state()
        
        # Set up VRAM monitoring if available
        self.setup_vram_monitoring()
        
        # Load settings and connect signals
        self.settings_manager = SettingsManager(self)
        self.settings_manager.load_settings()
        self.connect_base_signals()

    def init_device(self):
        """Initialize compute device and VRAM tracking"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.vram_limit_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU detected with {self.vram_limit_gb:.1f}GB VRAM")
        else:
            self.vram_limit_gb = None
            self.logger.warning("No GPU detected - running on CPU")

    def init_paths(self):
        """Initialize path variables"""
        self.training_root = None
        self.source_dir = None
        self.target_dir = None
        self.model_dir = None
        self.selected_model = None
        
        # Supported image formats
        self.image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".exr"}

    def init_configurations(self):
        """Initialize default configurations"""
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        
        # Training state
        self.trainer = None
        self.training_ready = False
        self.training_active = False

    def init_ui_state(self):
        """Initialize UI element states"""
        # Disable frames until needed
        self.frame_model_params.setEnabled(False)
        self.frame_training_params.setEnabled(True)
        self.frame_select_model.setEnabled(False)
        
        # Initialize buttons
        self.btn_start_training.setEnabled(False)
        self.btn_pause_training.setEnabled(False)
        
        # Initialize tabs
        self.tab_widget = self.findChild(QTabWidget, "tabWidget")
        if self.tab_widget:
            self.tab_widget.setCurrentIndex(0)
            self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Initialize viewers
        self._setup_viewers()

        # Set up VRAM monitoring if available
        if torch.cuda.is_available():
            self.setup_vram_monitoring()
        
        # Initialize training states
        self.init_training_state()
        
        # Update UI state
        self.update_training_ui_state()




    def _setup_viewers(self):
        """Initialize visualization components"""
        self.image_viewer = self.findChild(ImageViewer, 'imageViewer')
        self.preview_viewer = self.findChild(ImageViewer, 'preview_viewer')
        self.loss_graph_widget = self.findChild(LossGraphWidget, 'lossGraphWidget')
        
        if not all([self.image_viewer, self.preview_viewer, self.loss_graph_widget]):
            self.logger.error("Failed to initialize one or more viewers")

    def setup_vram_monitoring(self):
        """Set up VRAM monitoring if GPU is available"""
        if torch.cuda.is_available():
            self.vram_timer = QTimer(self)
            self.vram_timer.timeout.connect(self.update_vram_display)
            self.vram_timer.start(1000)  # Update every second

    def update_vram_display(self):
        """Update VRAM usage display"""
        if torch.cuda.is_available():
            used_vram = torch.cuda.memory_allocated() / 1e9
            self.label_vram_usage.setText(
                f"VRAM Usage: {used_vram:.2f}GB / {self.vram_limit_gb:.2f}GB"
            )

    def connect_base_signals(self):
        """Connect basic UI signals"""
        # Menu actions
        if hasattr(self, 'actionSave_Config'):
            self.actionSave_Config.triggered.connect(self.on_actionSave_Settings_triggered)
        if hasattr(self, 'actionSave_As'):
            self.actionSave_As.triggered.connect(self.on_actionSave_As_triggered)
        if hasattr(self, 'actionLoad_Config'):
            self.actionLoad_Config.triggered.connect(self.on_actionLoad_Config_triggered)




    def on_tab_changed(self, index):
        """Handle tab changes"""
        tab_name = self.tab_widget.tabText(index)
        if tab_name == "Batch Inference":
            #this automatically loads the best model when you change to the inference tab... I'm not sure if that's the best behavior.
            if hasattr(self, 'trainer') and self.trainer and \
            self.trainer.training_state.model_ready:
                best_model = self.trainer.model_saver.find_best_model()
                if best_model:
                    self.lineEdit_inference_model.setText(str(best_model))


    def closeEvent(self, event):
        """Handle application close"""
        if self.trainer and self.training_active:
            self.trainer.stop_training(
                save_checkpoint=True,
                filename_prefix="training_stopped_by_user"
            )
            
        self.settings_manager.save_settings()
        event.accept()
        QTimer.singleShot(0, QApplication.instance().quit)



# part 2 configuration handling

    """
    TODO UI Updates Required in Qt Designer:
    1. Update spinBox_residual_blocks label to "Processing Blocks"
    2. Update checkBox_use_pyramid label to "Multi-Scale Processing"
    3. Add spinBox_pyramid_scales:
    - Label: "Scale Levels"
    - Default: 2
    - Range: 1-2
    - Tooltip: "Number of resolution scales for multi-scale processing"
    4. Remove any v5-specific options that are now fixed in v9
    """

    def create_model_config(self) -> ModelConfig:
        """Create ModelConfig from UI settings"""
        try:
            config = ModelConfig(
                # Core architecture parameters
                base_features=self.spinBox_base_features.value(),
                num_blocks=self.spinBox_residual_blocks.value(), #UI needs updated label to "Processing Blocks"
                tile_size=512 if self.rdo_tile_512.isChecked() else 256,
                
                # Multi-scale processing
                use_pyramid=self.checkBox_use_pyramid.isChecked(),
                pyramid_scales=2 if self.checkBox_use_pyramid.isChecked() else 1,  # Forced to 2 when pyramid enabled
                #pyramid_scales=self.spinBox_pyramid_scales.value(),
                
                # Fixed v9 features
                use_multi_path=True,
                use_5x5_path=True,
                use_dilated_convs=True, #maybe this can be a user option?
                path_weights_learnable=True,
                
                # Optional features
                use_attention=self.checkBox_use_attention.isChecked(),
                residual_mode="late", #maybe this can be a user option for early, late or both?  I don't think the other choices are implemented in the model.
                residual_scale=0.1 #maybe add a spinner for this?
            )
            
            config.validate()
            return config
            
        except ValueError as e:
            self.logger.error(f"Error creating model configuration: {str(e)}")
            raise

    def create_training_config(self) -> TrainingConfig:
        """Create TrainingConfig from UI settings"""
        try:
            config = TrainingConfig(
                # Core training parameters
                batch_size=self.spinBox_batch_size.value(),
                learning_rate=self.doubleSpinBox_learning_rate.value(),
                num_epochs=self.spinBox_num_epochs.value(),
                
                # Checkpointing
                save_interval=self.spinBox_save_interval.value(),
                champion_improvement_threshold=self.spinBox_champion_improvement_pct.value() / 100.0,
                
                # VRAM management
                vram_limit_gb=self.vram_limit_gb if torch.cuda.is_available() else None,
                
                # Fixed parameters
                min_overlap=(16, 16)
            )
            
            config.validate()
            return config
            
        except ValueError as e:
            self.logger.error(f"Error creating training configuration: {str(e)}")
            raise

    def validate_vram_requirements(self, model_config: ModelConfig, training_config: TrainingConfig) -> bool:
        """Validate and adjust configurations based on VRAM requirements"""
        if not torch.cuda.is_available():
            return True
            
        try:
            vram_usage = VRAMEstimator.estimate_vram_usage(model_config, training_config)
            self.logger.info(f"Estimated VRAM usage: {vram_usage['total_estimated']:.2f}GB")
            
            if vram_usage['total_estimated'] > self.vram_limit_gb:
                adjusted_batch_size = VRAMEstimator.adjust_batch_size_for_vram(
                    model_config,
                    training_config,
                    self.vram_limit_gb
                )
                
                if adjusted_batch_size != training_config.batch_size:
                    msg = (f"Batch size needs to be adjusted from {training_config.batch_size} "
                        f"to {adjusted_batch_size} due to VRAM constraints. Continue?")
                    response = QMessageBox.question(self, "VRAM Management", msg)
                    
                    if response == QMessageBox.StandardButton.Yes:
                        self.spinBox_batch_size.setValue(adjusted_batch_size)
                        return True
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"VRAM validation error: {str(e)}")
            QMessageBox.critical(self, "Error", f"VRAM validation failed: {str(e)}")
            return False

    def load_model_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint and update configurations"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load configurations
            self.model_config = ModelConfig.from_dict(checkpoint['model_config'])
            self.training_config = TrainingConfig.from_dict(checkpoint['training_config'])


            # Initialize trainer with loaded configs
            self.trainer = TiledTrainingManager(
                self.model_config,
                self.training_config,
                str(self.training_root)
            )

            # Load model state
            self.trainer.model.load_state_dict(checkpoint['model_state_dict'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
            
            # Update UI with loaded configurations
            self.update_ui_from_configs()
            
            self.logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load checkpoint: {str(e)}")
            return False

    def update_ui_from_configs(self):
        """Update UI elements to reflect current configurations"""
        # Model parameters
        self.spinBox_base_features.setValue(self.model_config.base_features)
        self.spinBox_residual_blocks.setValue(self.model_config.num_blocks)
        self.rdo_tile_512.setChecked(self.model_config.tile_size == 512)
        self.rdo_tile_256.setChecked(self.model_config.tile_size == 256)
        self.checkBox_use_pyramid.setChecked(self.model_config.use_pyramid)
        self.spinBox_pyramid_scales.setValue(self.model_config.pyramid_scales)
        self.checkBox_use_attention.setChecked(self.model_config.use_attention)
        
        # Training parameters
        self.spinBox_batch_size.setValue(self.training_config.batch_size)
        self.doubleSpinBox_learning_rate.setValue(self.training_config.learning_rate)
        self.spinBox_num_epochs.setValue(self.training_config.num_epochs)
        self.spinBox_save_interval.setValue(self.training_config.save_interval)
        self.spinBox_champion_improvement_pct.setValue(
            int(self.training_config.champion_improvement_threshold * 100)
        )

    def get_current_configurations(self) -> Tuple[ModelConfig, TrainingConfig]:
        """Get current configurations for saving/loading"""
        return self.model_config, self.training_config

    def validate_configurations(self) -> bool:
        """Validate all configurations before training"""
        try:
            model_config = self.create_model_config()
            training_config = self.create_training_config()
            
            if not self.validate_vram_requirements(model_config, training_config):
                return False
                
            # Store validated configurations
            self.model_config = model_config
            self.training_config = training_config
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"Configuration validation failed: {str(e)}")
            return False
        
# part 3 training management

    """
    TODO UI Updates Required in Qt Designer:
    1. Update progressBar_current_epoch tooltip to clarify batch progress
    2. Add label for estimated time remaining (optional enhancement)
    3. Consider adding a QLabel for current training phase status
    """

    def connect_trainer_signals(self):
        """Connect all training-related signals"""
        if not self.trainer:
            return
            
        try:
            # Disconnect any existing connections to avoid duplicates
            self.trainer.training_progress.disconnect()
            self.trainer.epoch_progress.disconnect()
            self.trainer.update_loss_graph.disconnect()
            self.trainer.update_sample_tiles.disconnect()
            self.trainer.update_loss_graph_interactive.disconnect()
            self.trainer.vram_warning.disconnect()
        except:
            pass  # Ignore if signals weren't connected
            
        # Connect training signals
        self.trainer.training_progress.connect(self.update_training_progress)
        self.trainer.epoch_progress.connect(self.update_epoch_progress)
        self.trainer.update_loss_graph.connect(self.update_loss_graph_display)
        self.trainer.update_sample_tiles.connect(self.update_sample_tiles_display)
        self.trainer.update_loss_graph_interactive.connect(self.update_loss_graph_interactive)
        if hasattr(self.trainer, 'vram_warning'):
            self.trainer.vram_warning.connect(self.handle_vram_warning)

    @pyqtSlot()
    def on_btn_start_training_clicked(self):
        """Handle start training button click"""
        if not self.validate_training_requirements():
            return
            
        try:


            # Check VRAM before proceeding
            if not self.check_vram_requirements():
                return

            if not self.trainer or self.rdo_make_new_model.isChecked():
                if self.trainer and self.trainer.training_state.model_ready:
                    if not self.confirm_new_model_start():
                        return
                    
                # Initialize new training session
                self.init_training_state()
                self.initialize_new_training_session()

            else:
                self.resume_training()
                
            self.start_training()
            self.update_training_ui_state()
            
        except Exception as e:
            self.logger.error(f"Failed to start training: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")

    def init_training_state(self):
        """Initialize training state tracking"""
        if self.trainer:
            self.trainer.training_state.model_ready = False
            self.trainer.training_state.is_training = False
            self.trainer.training_state.is_paused = False

    def update_training_ui_state(self):
        """Update UI based on training state"""
        if not self.trainer:
            return
            
        state = self.trainer.training_state
        self.btn_start_training.setEnabled(not state.is_training)
        self.btn_pause_training.setEnabled(state.is_training)
        self.frame_model_params.setEnabled(not state.is_training)
        
        # Update progress display
        if state.current_epoch > 0:
            progress = (state.current_epoch / state.total_epochs) * 100
            self.progressBar_overall_progress.setValue(int(progress))
            self.label_current_loss.setText(f"Current Loss: {state.current_loss:.6f}")

    def check_vram_requirements(self) -> bool:
        """Validate VRAM requirements before training"""
        if not torch.cuda.is_available():
            return True
            
        try:
            vram_usage = VRAMEstimator.estimate_vram_usage(
                self.model_config,
                self.training_config
            )
            
            if vram_usage['total_estimated'] > self.vram_limit_gb:
                adjusted_batch_size = VRAMEstimator.adjust_batch_size_for_vram(
                    self.model_config,
                    self.training_config,
                    self.vram_limit_gb
                )
                
                if adjusted_batch_size != self.training_config.batch_size:
                    response = QMessageBox.question(
                        self,
                        "VRAM Management",
                        f"Batch size must be reduced from {self.training_config.batch_size} "
                        f"to {adjusted_batch_size} for VRAM constraints. Continue?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if response == QMessageBox.StandardButton.Yes:
                        self.spinBox_batch_size.setValue(adjusted_batch_size)
                        return True
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"VRAM validation failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"VRAM validation failed: {str(e)}")
            return False

    def initialize_new_training_session(self):
        """Initialize new training session"""
        try:
            self.trainer = TiledTrainingManager(
                model_config=self.model_config,
                training_config=self.training_config,
                training_root=str(self.training_root)
            )
            
            self.connect_trainer_signals()
            self.init_training_state()  # Initialize the training state
            self.update_training_ui_state()  # Update UI based on state
            
        except Exception as e:
            self.logger.error(f"Failed to initialize training: {str(e)}")
            raise

    def resume_training(self):
        """Resume training with current configuration"""
        if not self.validate_configurations():
            return
            
        self.trainer.update_training_config(self.training_config)
        self.update_ui_for_training_start()

    def start_training(self):
        """Start or resume the training process"""
        self.btn_start_training.setEnabled(False)
        self.btn_pause_training.setEnabled(True)
        self.frame_model_params.setEnabled(False)
        self.training_active = True
        self.trainer.train()

    @pyqtSlot()
    def on_btn_pause_training_clicked(self):
        """Handle pause training button click"""
        if self.trainer:
            self.trainer.stop_training(save_checkpoint=True)
            self.training_active = False
            self.update_ui_for_training_pause()

    @pyqtSlot()
    def on_rdo_use_previous_best_model_clicked(self):
        """Handle loading previous best model"""
        if not self.lowest_loss_model:
            return
            
        if self.load_model_checkpoint(self.lowest_loss_model):
            self.lineEdit_selected_model.setText(self.lowest_loss_model)
            self.selected_model = self.lowest_loss_model
            self.update_training_ui_state()
            self.btn_start_training.setEnabled(True)

    # @pyqtSlot()
    # def on_btn_select_existing_model_clicked(self):
    #     """Handle selecting existing model"""
    #     model_file, _ = QFileDialog.getOpenFileName(
    #         self, 
    #         "Select Model File", 
    #         str(self.model_dir), 
    #         "Model Files (*.pth)"
    #     )
        
    #     if model_file and self.load_model_checkpoint(model_file):
    #         self.lineEdit_selected_model.setText(model_file)
    #         self.selected_model = model_file
    #         self.update_training_ui_state()
    #         self.btn_start_training.setEnabled(True)


    @pyqtSlot()
    def on_btn_select_existing_model_clicked(self):
        """Handle selecting existing model for training"""
        try:
            model_file, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Model File", 
                str(self.model_dir) if self.model_dir else "", 
                "Model Files (*.pth)"
            )
            
            if model_file:
                if self.load_model_checkpoint(model_file):
                    self.lineEdit_selected_model.setText(model_file)
                    self.selected_model = model_file
                    
                    # Update UI state for selected model
                    self.btn_start_training.setEnabled(True)
                    self.frame_model_params.setEnabled(False)  # Disable model params for existing model
                    self.update_training_ui_state()
                    
                    self.logger.info(f"Selected existing model: {model_file}")
                else:
                    self.lineEdit_selected_model.setText("")
                    self.selected_model = None
                    self.btn_start_training.setEnabled(False)
                    
        except Exception as e:
            self.logger.error(f"Error selecting existing model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to select model: {str(e)}")
            self.lineEdit_selected_model.setText("")
            self.selected_model = None
            self.btn_start_training.setEnabled(False)






    def validate_training_requirements(self) -> bool:
        """Validate all requirements before training"""
        if not self.image_set_validated:
            QMessageBox.warning(self, "Warning", "Please validate image set first.")
            return False
            
        if not torch.cuda.is_available():
            response = QMessageBox.question(
                self, 
                "CPU Training",
                "No GPU detected. Training on CPU will be very slow. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            return response == QMessageBox.StandardButton.Yes
            
        return True

    def confirm_new_model_start(self) -> bool:
        """Confirm starting new model when existing model exists"""
        response = QMessageBox.question(
            self,
            "New Model",
            "Starting a new model will lose current training progress. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        return response == QMessageBox.StandardButton.Yes

    def update_ui_for_training_start(self):
        """Update UI elements when training starts"""
        self.btn_start_training.setEnabled(False)
        self.btn_pause_training.setEnabled(True)
        self.frame_model_params.setEnabled(False)
        self.frame_select_model.setEnabled(False)
        self.rdo_make_new_model.setChecked(False)

    def update_ui_for_training_pause(self):
        """Update UI elements when training is paused"""
        self.btn_start_training.setEnabled(True)
        self.btn_pause_training.setEnabled(False)
        self.frame_training_params.setEnabled(True)

    # Signal handlers
    @pyqtSlot(int, float)
    def update_training_progress(self, current_epoch: int, loss: float):
        """Handle training progress updates"""
        self.update_training_ui_state()  # Update UI based on current state
        
        # Additional progress updates as needed
        if hasattr(self, 'label_current_loss'):
            self.label_current_loss.setText(f"Current Loss: {loss:.6f}")

    @pyqtSlot(int, int)
    def update_epoch_progress(self, current_batch: int, total_batches: int):
        """Update current epoch progress"""
        progress = int((current_batch / total_batches) * 100)
        self.progressBar_current_epoch.setValue(progress)

    @pyqtSlot(str)
    def handle_vram_warning(self, message: str):
        """Handle VRAM-related warnings"""
        QMessageBox.warning(self, "VRAM Warning", message)

    @pyqtSlot(list, int, int, float, float, list, list)
    def update_loss_graph_interactive(self, losses, current_epoch, total_epochs,
                                    current_loss, min_loss, all_champions, current_champions):
        """Update interactive loss graph display"""
        if self.loss_graph_widget:
            self.loss_graph_widget.update_plot(
                losses, current_epoch, total_epochs,
                current_loss, min_loss, all_champions, current_champions
            )

    @pyqtSlot(QImage)
    def update_loss_graph_display(self, image: QImage):
        """Update loss graph image"""
        if hasattr(self, 'lbl_loss_graph'):
            pixmap = QPixmap.fromImage(image)
            self.lbl_loss_graph.setPixmap(
                pixmap.scaled(
                    self.lbl_loss_graph.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

    @pyqtSlot(QImage)
    def update_sample_tiles_display(self, image: QImage):
        """Update sample tiles display"""
        if hasattr(self, 'lbl_tile_samples'):
            pixmap = QPixmap.fromImage(image)
            self.lbl_tile_samples.setPixmap(
                pixmap.scaled(
                    self.lbl_tile_samples.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

# part 4 inference management
    """
    TODO UI Updates Required in Qt Designer:
    1. Add QComboBox for tile overlap settings (optional enhancement)
    2. Add QProgressBar for batch inference progress
    3. Consider adding a QCheckBox for VRAM optimization during inference
    """

    def setup_inference_handling(self):
        """Initialize inference-related components"""
        self.inference_manager = None
        self.test_image = None
        self.output_format = ".exr"
        self.preview_processing = False
        
        # Connect inference-related signals
        self.btn_preview_inference.clicked.connect(self.on_btn_preview_inference_clicked)
        self.btn_test_inference.clicked.connect(self.on_btn_test_inference_clicked)
        self.btn_start_inference.clicked.connect(self.on_btn_start_inference_clicked)
        self.comboBox_inference_output_format.currentIndexChanged.connect(
            self.on_inference_format_changed
        )

    @pyqtSlot()
    def on_btn_preview_inference_clicked(self):
        """Handle preview inference button click"""
        if not self.validate_model_ready():
            return
            
        if not self.preview_viewer.file:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
            
        try:
            self.preview_processing = True
            self.update_ui_for_inference_start()
            
            # Pause training if active
            was_training = self.training_active
            if was_training:
                self.on_btn_pause_training_clicked()
            
            # Process image
            image = load_image(self.preview_viewer.file)
            processed_image = self.process_single_image(image)
            
            # Display result
            self.preview_viewer.load_image(processed_image.cpu().numpy())
            
            # Resume training if it was active
            if was_training:
                self.on_btn_start_training_clicked()
                
        except Exception as e:
            self.logger.error(f"Preview inference failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"Preview inference failed: {str(e)}")
        finally:
            self.preview_processing = False
            self.update_ui_for_inference_end()

    @pyqtSlot()
    def on_btn_test_inference_clicked(self):
        """Handle test inference button click"""
        if not self.validate_inference_settings():
            return
            
        if not self.image_viewer.file:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
            
        try:
            self.update_ui_for_inference_start()
            
            if not self.inference_manager:
                self.initialize_inference_manager()
                
            # Process and display the test image
            processed_image = self.inference_manager.process_single_image(
                self.image_viewer.file,
                save=False
            )
            
            self.image_viewer.load_image(processed_image.cpu().numpy())
            
        except Exception as e:
            self.logger.error(f"Test inference failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"Test inference failed: {str(e)}")
        finally:
            self.update_ui_for_inference_end()

    @pyqtSlot()
    def on_btn_start_inference_clicked(self):
        """Handle batch inference button click"""
        if not self.validate_inference_settings():
            return
            
        try:
            self.update_ui_for_inference_start()
            
            if not self.inference_manager:
                self.initialize_inference_manager()
                
            # Process sequence
            prefix = self.lineEdit_inference_output_prefix.text().strip()
            self.inference_manager.process_sequence(prefix=prefix)
            
            QMessageBox.information(self, "Success", "Batch processing completed successfully.")
            
        except Exception as e:
            self.logger.error(f"Batch inference failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"Batch inference failed: {str(e)}")
        finally:
            self.update_ui_for_inference_end()

    @pyqtSlot()
    def on_btn_select_inference_model_clicked(self):
        """Handle inference model selection"""
        try:
            model_file, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Model File", 
                str(self.model_dir) if self.model_dir else "", 
                "Model Files (*.pth)"
            )
            
            if model_file:
                self.lineEdit_inference_model.setText(model_file)
                # Reset inference manager to use new model
                self.inference_manager = None
                self.logger.info(f"Selected inference model: {model_file}")
        except Exception as e:
            self.logger.error(f"Error selecting inference model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to select model: {str(e)}")

    @pyqtSlot()
    def on_btn_select_inference_source_dir_clicked(self):
        """Handle inference source directory selection"""
        try:
            current_dir = self.lineEdit_inference_source_dir.text()
            directory = QFileDialog.getExistingDirectory(
                self, 
                "Select Source Image Sequence Directory",
                current_dir if os.path.exists(current_dir) else ""
            )
            
            if directory:
                self.lineEdit_inference_source_dir.setText(directory)
                # Reset inference manager for new source directory
                self.inference_manager = None
                self.logger.info(f"Selected source directory: {directory}")
        except Exception as e:
            self.logger.error(f"Error selecting source directory: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to select directory: {str(e)}")

    @pyqtSlot()
    def on_btn_select_inference_output_dir_clicked(self):
        """Handle inference output directory selection"""
        try:
            current_dir = self.lineEdit_inference_output_dir.text()
            directory = QFileDialog.getExistingDirectory(
                self, 
                "Select Output Directory",
                current_dir if os.path.exists(current_dir) else ""
            )
            
            if directory:
                self.lineEdit_inference_output_dir.setText(directory)
                # Reset inference manager for new output directory
                self.inference_manager = None
                self.logger.info(f"Selected output directory: {directory}")
        except Exception as e:
            self.logger.error(f"Error selecting output directory: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to select directory: {str(e)}")


    def initialize_inference_manager(self):
        """Initialize or update the inference manager"""
        self.inference_manager = InferenceManager(
            training_root=str(self.training_root),
            source_sequence_folder=self.lineEdit_inference_source_dir.text(),
            transformed_sequence_folder=self.lineEdit_inference_output_dir.text(),
            specific_model=self.lineEdit_inference_model.text(),
            output_format=self.output_format,
            vram_limit_gb=self.vram_limit_gb if torch.cuda.is_available() else None
        )

    def process_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """Process a single image using current model"""
        self.model.eval()
        with torch.no_grad():
            # Generate tiles
            tiles = self.tile_generator.generate_tiles(image)
            
            # Process tiles
            processed_tiles = []
            for tile, position in tiles:
                processed_tile = self.model(
                    tile.unsqueeze(0).to(self.device)
                ).squeeze(0)
                processed_tiles.append((processed_tile.cpu(), position))
            
            # Reconstruct image
            reconstructed = self.tile_generator.reconstruct_image_v2(
                processed_tiles,
                image.shape[1:]
            )
            
            return torch.clamp(reconstructed, 0, 1)

    def validate_inference_settings(self) -> bool:
        """Validate all inference settings"""
        if not self.validate_model_ready():
            return False
            
        if not self.lineEdit_inference_model.text():
            QMessageBox.warning(self, "Warning", "Please select a model file.")
            return False
            
        if not os.path.exists(self.lineEdit_inference_source_dir.text()):
            QMessageBox.warning(self, "Warning", "Source directory does not exist.")
            return False
            
        # Create output directory if it doesn't exist
        os.makedirs(self.lineEdit_inference_output_dir.text(), exist_ok=True)
        
        return True

    def validate_model_ready(self) -> bool:
        """Check if model is ready for inference"""
        if not self.trainer or not self.trainer.training_state.model_ready:
            QMessageBox.warning(self, "Warning", "Model is not ready. Please load or train a model first.")
            return False
        return True

    @pyqtSlot(int)
    def on_inference_format_changed(self, index: int):
        """Handle output format selection change"""
        self.output_format = self.comboBox_inference_output_format.currentText()
        self.logger.info(f"Output format changed to: {self.output_format}")

    def update_ui_for_inference_start(self):
        """Update UI elements when inference starts"""
        self.btn_preview_inference.setEnabled(False)
        self.btn_test_inference.setEnabled(False)
        self.btn_start_inference.setEnabled(False)
        QApplication.processEvents()

    def update_ui_for_inference_end(self):
        """Update UI elements when inference ends"""
        self.btn_preview_inference.setEnabled(True)
        self.btn_test_inference.setEnabled(True)
        self.btn_start_inference.setEnabled(True)

#part 5 remaining sections
    """
    TODO UI Updates Required in Qt Designer:
    1. Update validation message display area tooltip
    2. Consider adding a "Revalidate" button for image sets
    3. Add tooltips for directory selection buttons
    """

    # Image Set Validation Methods
    def validate_image_set(self) -> str:
        """
        Validate source and target image sets.
        Returns error message string (empty if validation successful).
        """
        if not all([self.training_root, self.source_dir, self.target_dir]):
            return "Please select the training root directory"

        if not os.path.exists(self.source_dir):
            return "Source directory does not exist"
        if not os.path.exists(self.target_dir):
            return "Target directory does not exist"

        try:
            success, error_message = self.validate_image_pairs()
            if success:
                self.image_set_validated = True
                return ""
            else:
                self.image_set_validated = False
                self.btn_start_training.setEnabled(False)
                return f"Validation failed\n{error_message}"
        except Exception as e:
            self.logger.error(f"Image validation error: {str(e)}")
            return f"Validation error: {str(e)}"

    def validate_image_pairs(self) -> Tuple[bool, str]:
        """Validate matching pairs of source and target images"""
        source_images_dir = Path(self.source_dir)
        target_images_dir = Path(self.target_dir)
        
        # Get file lists
        source_filenames = set(os.listdir(source_images_dir))
        target_filenames = set(os.listdir(target_images_dir))
        
        # Basic validation
        if len(source_filenames) != len(target_filenames):
            return False, "Source and target directories contain different numbers of images"
        
        if source_filenames != target_filenames:
            return False, "Source and target filenames do not match"
        
        # Validate each pair
        error_message = ""
        for image_name in sorted(source_filenames):
            if not self._validate_image_pair(source_images_dir / image_name, 
                                        target_images_dir / image_name):
                error_message += f"Size mismatch for {image_name}\n"
        
        return error_message == "", error_message

    def _validate_image_pair(self, source_path: Path, target_path: Path) -> bool:
        """Validate a single source/target image pair"""
        try:
            source_image = load_image(str(source_path))
            target_image = load_image(str(target_path))
            return source_image.shape == target_image.shape
        except Exception as e:
            self.logger.error(f"Error validating {source_path}: {str(e)}")
            return False

    # Settings Management Methods
    @pyqtSlot()
    def on_actionSave_Settings_triggered(self):
        """Handle save settings menu action"""
        try:
            if not self.current_settings_file:
                self.current_settings_file, _ = QFileDialog.getSaveFileName(
                    self, "Save Settings", "", "Settings Files (*.json)"
                )
                if not self.current_settings_file:
                    return
                    
                if not self.current_settings_file.lower().endswith('.json'):
                    self.current_settings_file += '.json'
            
            self.current_settings_file = self.settings_manager.save_settings(
                self.current_settings_file
            )
            self.logger.info(f"Settings saved to: {self.current_settings_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save settings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

    @pyqtSlot()
    def on_actionSave_As_triggered(self):
        """Handle save settings as menu action"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Settings As", "", "Settings Files (*.json)"
            )
            if file_path:
                if not file_path.lower().endswith('.json'):
                    file_path += '.json'
                    
                self.current_settings_file = self.settings_manager.save_settings(file_path)
                self.logger.info(f"Settings saved to: {self.current_settings_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to save settings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

    @pyqtSlot()
    def on_actionLoad_Config_triggered(self):
        """Handle load settings menu action"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Settings", "", "Settings Files (*.json)"
            )
            if file_path:
                self.settings_manager.load_settings(file_path)
                self.current_settings_file = file_path
                self.logger.info(f"Settings loaded from: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load settings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load settings: {str(e)}")

    # Directory Selection Handlers
    @pyqtSlot()
    def on_btn_training_root_browse_clicked(self):
        """Handle training root directory selection"""
        try:
            initial_dir = self.lineEdit_training_root_dir.text()
            if not os.path.exists(initial_dir):
                initial_dir = os.path.expanduser("~")
                
            directory = QFileDialog.getExistingDirectory(
                self, "Select Training Root Directory", initial_dir
            )
            
            if directory:
                self.training_root = Path(directory)
                self.source_dir = self.training_root / "source_images"
                self.target_dir = self.training_root / "target_images"
                self.model_dir = self.training_root / "models"
                
                self.lineEdit_training_root_dir.setText(str(directory))
                
                # Validate directory structure
                error = self.validate_image_set()
                if not error:
                    self.handle_valid_training_root()
                else:
                    self.handle_invalid_training_root(error)
                    
        except Exception as e:
            self.logger.error(f"Error selecting training root: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to set training root: {str(e)}")

    def handle_valid_training_root(self):
        """Handle successful training root validation"""
        self.model_dir.mkdir(exist_ok=True)
        models_list = [f for f in os.listdir(self.model_dir) if f.endswith(".pth")]
        
        if models_list:
            self.find_best_model(models_list)
            self.enable_model_selection(True)
        else:
            self.textValidation_message.setText("No existing models found. Please create a new model.")
            self.enable_model_selection(True)
            self.rdo_use_previous_best_model.setEnabled(False)
        
        self.btn_start_training.setEnabled(True)

    def handle_invalid_training_root(self, error_message: str):
        """Handle failed training root validation"""
        self.frame_select_model.setEnabled(False)
        self.frame_model_params.setEnabled(False)
        self.frame_training_params.setEnabled(False)
        self.btn_start_training.setEnabled(False)
        
        self.image_set_validated = False
        self.training_ready = False
        self.textValidation_message.setText(error_message)
        
        self.trainer = None

    def find_best_model(self, models_list: List[str]):
        """Find the model with the lowest loss"""
        try:
            self.lowest_loss_model = os.path.join(
                self.model_dir,
                min(models_list, key=lambda x: float(re.search(r'loss_([\d.]+)', x).group(1)))
            )
            self.textValidation_message.setText(
                f"Model directory exists, best model found: {self.lowest_loss_model}"
            )
        except Exception as e:
            self.logger.error(f"Error finding best model: {str(e)}")
            self.lowest_loss_model = None
            self.textValidation_message.setText(
                "Found models but couldn't determine best model. Please select manually."
            )




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
