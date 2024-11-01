from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog,QTabWidget,  \
                             QMessageBox
from PyQt6.uic import loadUi    
from PyQt6.QtCore import pyqtSlot, Qt,  QTimer
from PyQt6.QtGui import QPixmap,   QImage
import sys
import os
import torch
import torch.optim as optim
from pathlib import Path
from Image_load_and_save_methods_v01 import load_image
from threaded_training_manager_class_v01 import TiledTrainingManager   
from msrn_model_v5 import MSRNConfig   
import re
from pyside_settings_manager_class_v001 import SettingsManager
from inference_manager_for_pyqt_v001 import InferenceManager
from pyqt_image_viewer_class_v002 import ImageViewer
import torch
import numpy as np
from tile_generator_v8 import TileGenerator
from LossGraphWidget_class_v001 import LossGraphWidget
import logging

# Clear any existing handlers from the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure the root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Test messages
logging.debug("Test Debug message")
logging.info("Test Info message")
logging.warning("Test Warning message")

# Print out current logging configuration
print("Current logging configuration:")
for name in logging.root.manager.loggerDict:
    logger = logging.getLogger(name)
    print(f"Logger: {name}")
    print(f"  Level: {logging.getLevelName(logger.level)}")
    print(f"  Propagate: {logger.propagate}")
    for handler in logger.handlers:
        print(f"  Handler: {type(handler).__name__}")
        print(f"    Level: {logging.getLevelName(handler.level)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("setup_training_v014.ui", self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_root=None
        self.source_dir = None
        self.target_dir = None
        self.model_dir=None
        self.lowest_loss_model=None
        self.selected_model=None
        self.image_set_validated=False
        self.training_ready=False
        self.training_paused=False
        self.model_ready=False
        self.training=False

        self.tile_generator=None
        self.trainer=None

        self.frame_model_params.setEnabled(False)
        self.frame_training_params.setEnabled(True)
        self.frame_select_model.setEnabled(False)
        self.rdo_select_existing_model.setEnabled(False)
        self.rdo_use_previous_best_model.setEnabled(False)
        self.btn_start_training.setEnabled(False)
        self.btn_pause_training.setEnabled(False)
        self.rdo_use_previous_best_model.setAutoExclusive(False)
        self.rdo_use_previous_best_model.setChecked(False)
        self.rdo_use_previous_best_model.setAutoExclusive(True)

        self.config=MSRNConfig() #a default configuration
        self.enable_training_params(True)
        self.settings_manager = SettingsManager(self)
        self.settings_manager.load_settings()  # Use default filename to load default settings.
        self.current_settings_file=None
    # INFERENCE TAB ITEMS
        self.test_image=None
        self.output_format=".exr"
        #self.btn_start_inference.setEnabled(False)
        self.image_extensions={".jpg", ".jpeg", ".png", ".tiff", ".tif", ".exr"}


        # Find the QTabWidget in the UI
        # Find the QTabWidget in the UI
        self.tab_widget: QTabWidget = self.findChild(QTabWidget, "tabWidget")

        if self.tab_widget:
            self.tab_widget.setCurrentIndex(0)
            self.tab_widget.currentChanged.connect(self.on_tab_changed)

        self.image_viewer = self.findChild(ImageViewer, 'imageViewer')  # Replace 'imageViewer' with your actual object name
        
        if self.image_viewer:
            print("ImageViewer found in the UI")
        else:
            print("Error: Could not find ImageViewer widget in the UI file")



        #find the preview viewer on the train page.  This will be used to preview the current model's output
        self.preview_viewer = self.findChild(ImageViewer, 'preview_viewer') 
        if self.preview_viewer:
            print("Preview Viewer found in the UI") 
        else:   
            print("Error: Could not find Preview Viewer widget in the UI file")

        self.loss_graph_widget = self.findChild(LossGraphWidget, 'lossGraphWidget')
        if self.loss_graph_widget:
            print("LossGraphWidget found in the UI")    
        else:
            print("Error: Could not find LossGraphWidget widget in the UI file")
        

    #progress bar update slot
    @pyqtSlot(int, int)
    def update_epoch_progress(self, current_batch, total_batches):
        progress_percentage = (current_batch / total_batches) * 100
        self.progressBar_current_epoch.setValue(int(progress_percentage))

    @pyqtSlot(int,float)
    def update_training_progress(self, current_epoch, loss):
        print(f"updating training progress for epoch {current_epoch} of {self.trainer.config.num_epochs} ({int(current_epoch/self.trainer.config.num_epochs * 100)})  with loss {loss}")
        self.progressBar_overall_progress.setValue(int(current_epoch/self.trainer.config.num_epochs * 100))
        self.label_current_loss.setText(f"Current Loss: {loss:.6f}")

    # @pyqtSlot(QImage)
    # def update_loss_graph_display(self, image):
    #     pixmap = QPixmap.fromImage(image)
    #     self.lbl_loss_graph.setPixmap(pixmap.scaled(self.lbl_loss_graph.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    # @pyqtSlot(QImage)
    # def update_sample_tiles_display(self, image):
    #     pixmap = QPixmap.fromImage(image)
    #     self.lbl_tile_samples.setPixmap(pixmap.scaled(self.lbl_tile_samples.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
   
    @pyqtSlot(QImage)
    def update_loss_graph_display(self, image):
        pixmap = QPixmap.fromImage(image)
        self.lbl_loss_graph.setPixmap(pixmap.scaled(
            self.lbl_loss_graph.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))
        self.lbl_loss_graph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        print("Updated loss graph display")  # Debug print


    @pyqtSlot(list, int, int, float, float, list, list)
    def update_loss_graph_interactive(self, losses, current_epoch, total_epochs, current_loss, min_loss, all_champions, current_champions):
        if self.loss_graph_widget:
            self.loss_graph_widget.update_plot(losses, current_epoch, total_epochs, current_loss, min_loss, all_champions, current_champions)
            print("Updated loss graph interactive")  # Debug print
        else:
            print("Error: Could not find LossGraphWidget widget in update_loss_graph_interactive")

    # @pyqtSlot(list, int, list, list)
    # def update_loss_graph_interactive(self, losses, current_epoch, all_champions, current_champions):
    #     self.loss_graph_widget.update_plot(losses, current_epoch, all_champions, current_champions)
    #     print("Updated loss graph interactive")  # Debug print

    @pyqtSlot(QImage)
    def update_sample_tiles_display(self, image):
        pixmap = QPixmap.fromImage(image)
        self.lbl_tile_samples.setPixmap(pixmap.scaled(
            self.lbl_tile_samples.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))
        self.lbl_tile_samples.setAlignment(Qt.AlignmentFlag.AlignCenter)
        print("Updated sample tiles display")  # Debug print

    # def open_file(self):
    ## replaced with drag and drop
    #     filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
    #     if filename:
    #         self.image_viewer.load_image(filename)


    # Handle tab changes
    def on_tab_changed(self, index):
        tab_name = self.tab_widget.tabText(index)
        if tab_name == "tab_train":
            print("Train tab selected")
            # Add logic for the Train tab here
        elif tab_name == "tab_batch":
            print("Inference tab selected")
            if self.lowest_loss_model is not None:
                self.lineEdit_inference_model.setText(self.lowest_loss_model)


    # handle menu items (eg config save and load)
    @pyqtSlot()
    def on_actionSave_Settings_triggered(self):
        print("Save settings clicked")
        if self.current_settings_file is None:
            self.current_settings_file, _ = QFileDialog.getSaveFileName(self, "Save Settings", "", "Settings Files (*.json)")
            # Ensure the filename has a .json extension
            if not self.current_settings_file.lower().endswith('.json'):
                self.current_settings_file += '.json'

        self.current_settings_file=self.settings_manager.save_settings(self.current_settings_file)

    @pyqtSlot()
    def on_actionSave_As_triggered(self):
        print("Save As clicked")
        self.current_settings_file, _ = QFileDialog.getSaveFileName(self, "Save Settings", "", "Settings Files (*.json)")
        # Ensure the filename has a .json extension
        if not self.current_settings_file.lower().endswith('.json'):
            self.current_settings_file += '.json'
 
        self.current_settings_file=self.settings_manager.save_settings(self.current_settings_file)


    @pyqtSlot()
    def on_btn_preview_inference_clicked(self):
        print("Preview inference clicked")
        #Make sure model is ready
        if self.model_ready==False:
            print("Model is not ready")
            return
        #Make sure an image has been loaded in the viewer:
        if self.preview_viewer.file is not None:
            self.preview_image=self.preview_viewer.file
            print("Preview image loaded, ready to perform inference")
            if self.training:
                print("Training is in progress, pausing training")
                was_training=True
            else:
                was_training=False
            #pause the training
            self.on_btn_pause_training_clicked()
            #load the image
            image = load_image(self.preview_image) # load_image is a function from the Image_load_and_save_methods_v01.py file
            #get the tiles using the tile generator

            self.tile_generator=TileGenerator(self.trainer.config.tile_size, [16,16])
            tiles=self.tile_generator.generate_tiles(image)

            #print(f"tiles generated; Number of tiles: {len(tiles)}")


            #perform inference on each tile

            #set torch to eval mode
            self.trainer.model.eval()
            with torch.no_grad():
                    
                # Process tiles
                processed_tiles = []
                for tile, position in tiles:
                    
                    processed_tile = self.trainer.model(tile.unsqueeze(0).to(self.device)).squeeze(0)
                    #print(f"processed a tile; position: {position}")
                    processed_tiles.append((processed_tile.cpu(), position)) #why do we need to call cpu() here?

                # Reconstruct image
                reconstructed = self.tile_generator.reconstruct_image_v2(processed_tiles, image.shape[1:])
                print(f"reconstructed image")
                #clamp the image to [0, 1] range if needed  
                reconstructed = torch.clamp(reconstructed, 0, 1)


            #set torch back to training mode
            self.trainer.model.train()
            #display the result

            #convert from torch tensor to numpy array
            reconstructed=reconstructed.cpu().numpy()
            reconstructed = np.transpose(reconstructed, (1, 2, 0))



            self.preview_viewer.load_image(reconstructed)


        if was_training: #resume training if it was originally running
            self.on_btn_start_training_clicked()







# inference page items:


    @pyqtSlot()
    def on_btn_test_inference_clicked(self):
        # make sure a model has been selected
        if self.lineEdit_inference_model.text()=="": 
            print("No model file selected")
            return
        #make sure an image has been loaded in the viewer:
        if self.image_viewer.file is not None:
            self.test_image=self.image_viewer.file
            print("Test image loaded, ready to perform inference")
            if self.training_root is None:
                self.training_root=Path(self.lineEdit_inference_model.text()).parent

            self.inference_manager=InferenceManager(self.training_root, self.lineEdit_inference_source_dir.text(), 
                                                    self.lineEdit_inference_output_dir.text(), 
                                                    specific_model=self.lineEdit_inference_model.text(), 
                                                    output_format=self.output_format)
            #def process_single_image(self, image_file: Path, prefix: str = "", postfix: str = "", save: bool = True):
            image=self.inference_manager.process_single_image(self.test_image, prefix="test_", save=False)
            print("Inference completed")

            #convert from torch tensor to numpy array
            image=image.cpu().numpy()
            image = np.transpose(image, (1, 2, 0)) # Change from [C, H, W] to [H, W, C]
            image=(image*255).clip(0,255).astype(np.uint8) # 
            print(f"Image shape being passed to viewer: {image.shape}")

            self.image_viewer.load_image(image)


    @pyqtSlot()
    def on_comboBox_inference_output_format_currentIndexChanged(self):
        selected_format = self.comboBox_inference_output_format.currentText()
        print(f"Selected output format: {selected_format}")
        self.output_format=selected_format

        
    @pyqtSlot()
    def on_btn_select_inference_model_clicked(self):
        print("Select inference model clicked")
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", self.model_dir, "Model Files (*.pth)")
        if model_file:
            print("Selected model file:", model_file)
            self.lineEdit_inference_model.setText(model_file)
        else:
            print("No model file selected")
            self.lineEdit_selected_model.setText("")

    @pyqtSlot()
    def on_btn_select_inference_source_dir_clicked(self):    
        print("Select inference source directory clicked")
        directory= QFileDialog.getExistingDirectory(self, "Select Source Image Sequence Directory")
        if directory:
            print("Selected directory:", directory)
            self.lineEdit_inference_source_dir.setText(directory)
        else:
            print("No directory selected")
            self.lineEdit_inference_source_dir.setText("")       


    @pyqtSlot()
    def on_btn_select_inference_output_dir_clicked(self):    
        print("Select inference output directory clicked")
        directory= QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            print("Selected directory:", directory)
            self.lineEdit_inference_output_dir.setText(directory)
        else:
            print("No directory selected")
            self.lineEdit_inference_output_dir.setText("")


    @pyqtSlot()
    def on_btn_start_inference_clicked(self):
        print("Start inference clicked")
        if self.lineEdit_inference_model.text()=="":
            print("No model file selected")
            return
        if self.lineEdit_inference_source_dir.text()=="":
            print("No source directory selected")
            return
        if self.lineEdit_inference_output_dir.text()=="":
            print("No output directory selected")
            return
        self.inference_prefix=self.lineEdit_inference_output_prefix.text().strip()
        
        #validate the parameters:
        #check if the model file exists, and is a valid model file
        #check if the source directory exists and contains image files
        #check if the output directory exists, and is writable. if it doesn't exist, create it.
        if not os.path.exists(self.lineEdit_inference_model.text()):
            print("Model file does not exist")
            return
        else:
            print("Model file exists")

        if not os.path.exists(self.lineEdit_inference_source_dir.text()):
            print("Source directory does not exist")
            return
        else:
            print("Source directory exists")
            #check if the source directory contains image files
            source_files=os.listdir(self.lineEdit_inference_source_dir.text())
            if len(source_files)==0:
                print("Source directory is empty")
                return
            else:
                print("Source directory contains files")
                has_image_files=any(Path(file).suffix.lower() in self.image_extensions for file in source_files)
                if not has_image_files:
                    print("Source directory does not contain valid image files of accepted types")
                    return
                else:
                    print("Source directory contains valid image files")
        #if we get here, the model exists, the source directory exists and contains image files
        #check if the output directory exists, if not, create it
        if not os.path.exists(self.lineEdit_inference_output_dir.text()):
            print("Output directory does not exist, creating it")
            os.makedirs(self.lineEdit_inference_output_dir.text())
        else:
            print("Output directory exists")
        


        #perform inference
        # we don't really need the training_root directory, let's assume that it's the parent of the model.
        if self.training_root is None:
            self.training_root=Path(self.lineEdit_inference_model.text()).parent

        self.inference_manager=InferenceManager(self.training_root, self.lineEdit_inference_source_dir.text(), self.lineEdit_inference_output_dir.text(), 
                                                specific_model=self.lineEdit_inference_model.text(), output_format=self.output_format)

        print("Inference started")
        self.inference_manager.process_sequence(prefix=self.inference_prefix)


    @pyqtSlot()
    def on_actionLoad_Config_triggered(self):
        print("Load settings clicked")
        settings_file, _ = QFileDialog.getOpenFileName(self, "Load Settings", "", "Settings Files (*.json)")
        if settings_file:
            self.settings_manager.load_settings(settings_file)
            self.current_settings_file=settings_file

    def closeEvent(self, event):
        # This method is called when the window is about to be closed
        self.settings_manager.save_settings()
        # Optionally, you can also do other cleanup here
        super().closeEvent(event)


    @pyqtSlot()
    def on_btn_training_root_browse_clicked(self):
        print("Training Root Button clicked")
        self.enable_model_params(False)
        self.enable_model_selection(False)
        if self.lineEdit_training_root_dir.text()!="" and os.path.exists(self.lineEdit_training_root_dir.text()):
            directory= QFileDialog.getExistingDirectory(self, "Select Directory", self.lineEdit_training_root_dir.text())
        else:
            directory= QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            print("Selected directory:", directory)
            self.training_root=directory
            self.lineEdit_training_root_dir.setText(directory)
            self.source_dir = os.path.join(self.training_root, "source_images")
            self.target_dir = os.path.join(self.training_root, "target_images")
            error=self.on_btn_validate_clicked()    #validate the image set
            if error=="":
                #the image set is valid, the training root directory is valid
                self.image_set_validated=True
                model_dir=os.path.join(self.training_root, "models")
                if os.path.exists(model_dir):
                    self.model_dir=model_dir
                    models_list = [file for file in os.listdir(self.model_dir) if file.endswith(".pth")]
                    if len(models_list)==0:
                        self.textValidation_message.setText("No models found in the model directory")
                        #self.rdo_make_new_model.setChecked(True)
                        self.enable_model_params(True)
                        self.rdo_select_existing_model.setEnabled(True)
                        self.enable_model_selection(True)
                        self.rdo_use_previous_best_model.setEnabled(False)
                        self.btn_start_training.setEnabled(True)
                    else:
                        # Find the model with the lowest loss value in the filename
                        self.lowest_loss_model =os.path.join(self.model_dir, min(models_list, key=self.extract_loss_from_filename))
                        lowest_loss=self.extract_loss_from_filename(self.lowest_loss_model)
                        self.enable_model_selection(True)
                        if lowest_loss==float('inf'):
                            #None of the models have a loss value in their title.
                            self.rdo_use_previous_best_model.setEnabled(False)
                            self.textValidation_message.setText("Model directory has models, but we cant determine the best model, please select a model or make a new one")                        
                        self.textValidation_message.setText(f"Model directory exists, the best model found is {self.lowest_loss_model}")
                else: #no models directory, so we can only make a new model
                    self.textValidation_message.setText("No models directory found, please make a new model or select an existing one.")
                    #self.rdo_make_new_model.setChecked(True)
                    self.enable_model_params(True)
                    self.rdo_select_existing_model.setEnabled(True)
                    self.rdo_use_previous_best_model.setEnabled(False)
                    self.enable_model_selection(True)
                    self.btn_start_training.setEnabled(False)


            else:
                self.frame_select_model.setEnabled(False)
                self.frame_select_existing_model.setEnabled(False)
                self.frame_model_params.setEnabled(False)
                self.frame_training_params.setEnabled(False)
                self.btn_start_training.setEnabled(False)
                self.image_set_validated=False
                self.training_ready=False
                self.textValidation_message.setText(error)
                self.trainer=None
        else:
            print("No directory selected")
            self.training_root=None
            self.trainer=None
            self.training_ready=False
            self.lineEdit_training_root_dir.setText("")
            self.textValidation_message.setText("No directory selected")
            self.image_set_validated=False


    def enable_model_params(self, enable=True):
        self.frame_model_params.setEnabled(enable)
        self.rdo_tile_256.setEnabled(enable)
        self.rdo_tile_512.setEnabled(enable)
        self.spinBox_scales.setEnabled(enable)
        self.spinBox_residual_blocks.setEnabled(enable)
        self.spinBox_base_features.setEnabled(enable)
        self.checkBox_use_pyramid.setEnabled(enable)
        self.checkBox_use_attention.setEnabled(enable)

    def enable_training_params(self, enable=True):
        self.frame_training_params.setEnabled(enable)
        self.doubleSpinBox_learning_rate.setEnabled(enable)
        self.spinBox_batch_size.setEnabled(enable)
        self.spinBox_num_epochs.setEnabled(enable)
        self.spinBox_save_interval.setEnabled(enable)
        self.spinBox_champion_improvement_pct.setEnabled(enable)


    def enable_model_selection(self, enable=True):
        self.frame_select_model.setEnabled(enable)
        self.rdo_select_existing_model.setEnabled(enable)
        self.rdo_use_previous_best_model.setEnabled(enable)
        self.btn_select_existing_model.setEnabled(enable)
        self.rdo_make_new_model.setEnabled(enable)
        if enable==False:
            self.rdo_select_existing_model.setChecked(False)
            self.rdo_use_previous_best_model.setChecked(False)
            self.rdo_make_new_model.setChecked(False)

    # Function to extract the floating-point loss from the filename
    def extract_loss_from_filename(self,filename):
        # Regular expression to match "loss_" followed by a floating-point number
        #print(f"Extracting loss from filename: {filename}")
        match = re.search(r"loss_([0-9]+\.[0-9]+)", filename)
        #match = re.search(r"loss_([0-9.e-]+)", filename)
        #match = re.search(r"loss_([0-9.e-]+)(?!\.)", filename)
        if match:
            #print(f"Extracted loss: {match.group(1)}")
            return float(match.group(1))
        return float('inf')  # If no match, return a very high number to ignore this file


    @pyqtSlot()
    def on_rdo_use_previous_best_model_clicked(self):
        print("Use previous best model clicked")
        self.lineEdit_selected_model.setText(self.lowest_loss_model)
        self.selected_model=self.lowest_loss_model
        self.btn_start_training.setEnabled(True)
        self.btn_select_existing_model.setEnabled(False)
        self.enable_model_params(False)
        self.load_model(self.lowest_loss_model)
        self.training_paused=True
        self.training_ready=True
        self.model_ready=True

    @pyqtSlot()
    def on_rdo_select_existing_model_clicked(self):
        print("Select existing model clicked")
        self.btn_select_existing_model.setEnabled(True)
        self.btn_start_training.setEnabled(False)
        self.enable_model_params(False)

    @pyqtSlot()
    def on_btn_select_existing_model_clicked(self):
        print("Select existing model clicked")
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", self.model_dir, "Model Files (*.pth)")
        if model_file:
            print("Selected model file:", model_file)
            self.lineEdit_selected_model.setText(model_file)
            self.btn_start_training.setEnabled(True)
            self.selected_model=os.path.join(self.model_dir,model_file)
            self.enable_model_params(False)
            self.load_model(self.selected_model)    
            self.training_paused=True
            self.training_ready=True
            self.model_ready=True

        else:
            print("No model file selected")
            self.lineEdit_selected_model.setText("")
            self.btn_start_training.setEnabled(False)
            self.training_ready=False
            self.model_ready=False


    @pyqtSlot()
    def on_rdo_make_new_model_clicked(self):
        print("Make new model clicked")
        self.btn_start_training.setEnabled(True)
        self.btn_select_existing_model.setEnabled(False)
        self.enable_model_params(True)
        self.selected_model=None
        self.training_ready=False
        #self.model_ready=False
        self.training_paused=False


    @pyqtSlot()
    def on_btn_pause_training_clicked(self):
        print("Break training clicked")
        self.trainer.training=False
        self.training_paused=True
        self.trainer.stop_requested=True
        self.textValidation_message.setText("Training stopped")
        self.btn_start_training.setEnabled(True)


    def connect_trainer_signals(self):
        if self.trainer:
            # Disconnect any existing connections to avoid duplicates
            try:
                self.trainer.epoch_progress.disconnect()
                self.trainer.training_progress.disconnect()
                self.trainer.update_loss_graph.disconnect()
                self.trainer.update_sample_tiles.disconnect()
                self.trainer.update_loss_graph_interactive.disconnect()
            except TypeError:
                # This exception will be raised if the signal was not connected
                pass

            # Connect the signals
            self.trainer.epoch_progress.connect(self.update_epoch_progress)
            self.trainer.training_progress.connect(self.update_training_progress)
            self.trainer.update_loss_graph.connect(self.update_loss_graph_display)
            self.trainer.update_sample_tiles.connect(self.update_sample_tiles_display)
            self.trainer.update_loss_graph_interactive.connect(self.update_loss_graph_interactive)


    def load_model(self, model_file):
        self.trainer=None
        checkpoint=torch.load(model_file) #load the model, which contains it's config.  This is wasteful.  Maybe config should be a separate file.
        config=MSRNConfig.from_dict(checkpoint['config'])
        self.config=config
        self.trainer=TiledTrainingManager(self.config,self.training_root)  #initialize the trainer with the config 
        self.connect_trainer_signals()
        self.trainer.load_checkpoint(model_file) #get the rest of the model data
        checkpoint=None
        self.update_model_params_display()
        self.update_training_params_display()
        self.training_paused=True
        self.training_ready=True
        self.model_ready=True

        #update visualizations
        self.trainer.update_loss_graph_plot_interactive(self.trainer.current_epoch) 
        #self.trainer.update_loss_graph_plot(self.trainer.current_epoch)
        self.trainer.update_representative_tiles_image()

    def create_new_model(self):
        self.config=self.create_msrn_config() #this will create a new config object with the parameters set in the GUI
        self.trainer=TiledTrainingManager(self.config, self.training_root)
        self.connect_trainer_signals()
        self.training_paused=True
        self.training_ready=True
        self.model_ready=True

        #update visualizations
        self.trainer.update_loss_graph_plot_interactive(self.trainer.current_epoch) 
        #self.trainer.update_loss_graph_plot(self.trainer.current_epoch)
        self.trainer.update_representative_tiles_image()



    def update_model_params_display(self):
        #update the model parameters in the GUI, from the trainer config.  For instance, if you load a model, the model parameters should be displayed in the GUI
        config=self.trainer.config
        self.spinBox_scales.setValue(config.num_scales)
        self.spinBox_residual_blocks.setValue(config.num_residual_blocks)
        self.spinBox_base_features.setValue(config.base_features)
        self.checkBox_use_pyramid.setChecked(config.use_pyramid)
        self.checkBox_use_attention.setChecked(config.use_attention)
        if config.tile_size==256:
            self.rdo_tile_256.setChecked(True)
        elif config.tile_size==512:
            self.rdo_tile_512.setChecked(True)

    def update_training_params_display(self):
        #update the training parameters in the GUI, from the trainer config.  For instance, if you load a model, the training parameters should be displayed in the GUI
        config=self.trainer.config
        self.doubleSpinBox_learning_rate.setValue(config.learning_rate)
        self.spinBox_batch_size.setValue(config.batch_size)
        self.spinBox_num_epochs.setValue(config.num_epochs)
        self.spinBox_save_interval.setValue(config.save_interval)
        self.spinBox_champion_improvement_pct.setValue(int(config.champion_improvement_factor*100))
        self.progressBar_overall_progress.setValue(int(self.trainer.current_epoch/self.trainer.config.num_epochs * 100))
        self.label_current_loss.setText(f"Current Loss: {self.trainer.best_loss:.6f}")

    def update_trainer_config(self):
        ##we can't change the model architecture parameters after the model has been created
        #we can change the training parameters though.
        #update optimizer learning rate
        self.trainer.config.learning_rate=self.doubleSpinBox_learning_rate.value()
        for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = self.trainer.config.learning_rate
        self.trainer.optimizer = optim.Adam(self.trainer.model.parameters(), lr=self.trainer.config.learning_rate)
        self.trainer.config.batch_size=self.spinBox_batch_size.value()
        self.trainer.config.num_epochs=self.spinBox_num_epochs.value()
        self.trainer.config.save_interval=self.spinBox_save_interval.value()
        self.trainer.config.champion_improvement_factor=float(self.spinBox_champion_improvement_pct.value()/100.0)
        self.progressBar_overall_progress.setValue(int(self.trainer.current_epoch/self.trainer.config.num_epochs * 100))
        self.label_current_loss.setText(f"Current Loss: {self.trainer.best_loss:.6f}")
        


    @pyqtSlot()
    def on_btn_validate_clicked(self):
        error_message=""
        print("validate Button clicked")

        if self.training_root is None or self.training_root=="" or self.source_dir is None or \
            self.source_dir=="" or self.target_dir is None or self.target_dir=="":
            error_message+="Please select the training root directory\n"
            print("Please select the training root directory")
            self.textValidation_message.setText(error_message)
            return error_message

        if not os.path.exists(self.source_dir):
            error_message+="Source directory does not exist\n"
            print("Source directory does not exist")
        if not os.path.exists(self.target_dir):
            error_message+="Target directory does not exist\n"
            print("Target directory does not exist")

        if error_message!="":
            self.textValidation_message.setText(error_message)
            print("Validation failed")
            print(error_message)
            self.textValidation_message.setText(error_message)
            return error_message
        #perform image set validation
        success,validate_images_error_message=self.validate_images_without_keeping_image_data(self.source_dir, self.target_dir)
        if success:
            self.textValidation_message.setText("Validation successful")
            self.image_set_validated=True
            print("Validation successful")
            error_message=""
            return error_message
        else:
            print("Validation failed")
            
            self.image_set_validated=False
            self.btn_start_training.setEnabled(False)
            self.textValidation_message.setText(error_message)
            error_message="Validation failed\n"+error_message+validate_images_error_message
            return error_message

    def validate_images_without_keeping_image_data(self, source_images_dir, target_images_dir):
        """Validate source and target images and return a list of paired image paths."""
        #made into a separate static method so that it can be used by other classes.
        success=True
        error_message="" 

        source_images_dir = Path(source_images_dir)
        target_images_dir = Path(target_images_dir)
        if not source_images_dir.is_dir():
            success=False
            error_message+="source_images directory does not exist\n"
        if not target_images_dir.is_dir():
            success=False
            error_message+="target_images directory does not exist\n"
        if not success:
            self.textValidation_message.setText(error_message)
            return success, error_message

        source_filenames = set(os.listdir(source_images_dir))
        target_filenames = set(os.listdir(target_images_dir))
        if len(source_filenames) != len(target_filenames):
            #raise ValueError("Source and target image sets do not match.")
            success=False
            error_message+="Source and target contain a different number of images.\n"
            self.textValidation_message.setText(error_message)
            return success, error_message

        if source_filenames != target_filenames:
            #raise ValueError("Source and target image sets do not match.")
            success=False
            error_message+="Source and target filenames do not match.\n"
            self.textValidation_message.setText(error_message)
            return success, error_message
        #source_images = []
        #target_images = []
        
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
            #source_images.append(source_image)
            #target_images.append(target_image)
        source_image=None #free up memory
        target_image=None
        self.textValidation_message.setText(error_message)
        return success, error_message

    @pyqtSlot()
    def on_btn_start_training_clicked(self):
        print("start training Button clicked")
        if self.image_set_validated:
            if self.rdo_make_new_model.isChecked(): #make a new model was chosen, so the model is not loaded, and the trainer is not initialized
                print(f"start training, with new model... model_ready={self.model_ready}")
                #we have to use the model parameters, to define the model architecture.
                if self.model_ready:
                    print("about to issue warning message...")
                    #we have a model loaded. We should warn the user that they'll lose their progress if they start a new model
                    dialog=QMessageBox()
                    dialog.setWindowTitle("Warning")
                    dialog.setText("You're about to start a new model from scratch.   Are you sure you want to continue?")
                    dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    dialog.setDefaultButton(QMessageBox.StandardButton.No)
                    dialog.setIcon(QMessageBox.Icon.Warning)
                    response=dialog.exec()
                    if response==QMessageBox.StandardButton.No:
                        self.rdo_make_new_model.setAutoExclusive(False)
                        self.rdo_make_new_model.setChecked(False) #reset the radio button so when we resume, we don't make a new model again, erasing our progress.
                        self.rdo_make_new_model.setAutoExclusive(True)
                        return

                    pass
                #create a new model and initialize the trainer
                self.create_new_model()

                #the next 3 lines turn off the radio button, so that when we pause training, we don't accidentally start a new model again.
                #we toggle autoexclusive off, so that we can uncheck the radio button, then turn it back on. This allows us to have NO radio buttons selected.
                #If we don't do this, the radio button will always have one selected. (the default behavior of radio buttons)
                
                self.rdo_make_new_model.setAutoExclusive(False)
                self.rdo_make_new_model.setChecked(False) #reset the radio button so when we resume, we don't make a new model again, erasing our progress.
                self.rdo_make_new_model.setAutoExclusive(True)

            else: #either use the previous best model or select an existing model was chosen, so the model is already loaded, and the trainer is already initialized
                self.update_trainer_config() #use the new training parameters

            print("Starting training, enabling break training button")
            self.btn_pause_training.setEnabled(True)   
            self.btn_start_training.setEnabled(False) 
            print(f"start training button is enabled = {self.btn_start_training.isEnabled()}; break training button is enabled = {self.btn_pause_training.isEnabled()}")
            print("about to train...")
            self.training=True
            self.training_paused=False
            self.training_ready=True
            self.trainer.train()
            self.textValidation_message.setText("Starting training")


    def create_msrn_config(self):
        # Create the MSRN configuration
        config = MSRNConfig()
        config.num_scales= self.spinBox_scales.value()
        config.num_residual_blocks= self.spinBox_residual_blocks.value()
        config.base_features= self.spinBox_base_features.value()
        config.use_pyramid=self.checkBox_use_pyramid.isChecked()
        config.use_attention=self.checkBox_use_attention.isChecked()
        config.learning_rate=self.doubleSpinBox_learning_rate.value()
        config.batch_size=self.spinBox_batch_size.value()
        config.num_epochs=self.spinBox_num_epochs.value()
        config.save_interval=self.spinBox_save_interval.value()
        config.champion_improvement_factor=float(self.spinBox_champion_improvement_pct.value()/100.0)
        if self.rdo_tile_256.isChecked():
            config.tile_size=256
        elif self.rdo_tile_512.isChecked():
            config.tile_size=512
        
        return config


    def closeEvent(self, event):
        """Handle the window close event."""
        print("Close event triggered")
        
        # Stop the training if it's running
        if hasattr(self, 'trainer') and self.trainer:
            print("Stopping trainer")
            self.trainer.stop_training(save_checkpoint=True,filename_prefix="training_stopped_by_user")

        # Save settings if needed
        if hasattr(self, 'settings_manager'):
            print("Saving settings")
            self.settings_manager.save_settings()

        # Accept the close event
        event.accept()

        # Schedule the application to quit
        QTimer.singleShot(0, QApplication.instance().quit)





if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
