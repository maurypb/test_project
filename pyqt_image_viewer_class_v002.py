import sys
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsWidget 
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QImage, QKeySequence
from PyQt6.QtCore import Qt, QRectF
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" #enable exr support in opencv
import sys
import cv2
import numpy as np
from pathlib import Path




class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.GlobalColor.gray)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.pixmap_item = self.scene.addPixmap(QPixmap())
        self.setAcceptDrops(True)
        self.file=None
        self.image_extensions={".jpg", ".jpeg", ".png", ".tiff", ".tif", ".exr"}
        #self.current_view_range = None  # To store the current pan/zoom state
        self.current_view_transform = None  # To store the current pan/zoom state
        print("ImageViewer initialized")




        # Add a keyPressEvent to reset zoom (press 'R' key)
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_R:
            self.reset_zoom()
        else:
            super().keyPressEvent(event)


    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
        else:
            factor = 0.8
        self.scale(factor, factor)

    def dragEnterEvent(self, event: QDragEnterEvent):
        print("Drag enter event")
        if event.mimeData().hasUrls():
            print("  Has URLs")
            event.acceptProposedAction()
        else:
            print("  No URLs")
            event.ignore()

    def dragMoveEvent(self, event):
        print("Drag move event")
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        print("Drop event")
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            files = [u.toLocalFile() for u in event.mimeData().urls()]
            print(f"  Dropped files: {files}")
            if files:
    
                if Path(files[0]).suffix.lower() in self.image_extensions :
                    self.file=files[0]
                    self.load_image(files[0])
                else:
                    print("  Unsupported file format")
                    self.file=None
        else:
            print("  No URLs in drop event")
            event.ignore()

    def resizeEvent(self, event):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event)


    def fancy_image_loader(self, path: str)->np.ndarray:
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
        return image

    def load_image(self, image_data): #image_data can be a filename or a numpy array
        print(f"Loading image")

        # # Save the current view range (pan and zoom)
        # if self.sceneRect().isValid():
        #     self.current_view_range = self.mapToScene(self.viewport().rect()).boundingRect()

        # Save the current transformation (pan and zoom)
        self.current_view_transform = self.transform()

        if isinstance(image_data, str):  # It's a filename
            # Use custom image loading class here
            # For now, let's assume it returns a numpy array
            print(f"Loading image from file: {image_data}")
            try:
                image = self.fancy_image_loader(image_data)
            except Exception as e:
                print(f"Failed to load image: {e}")
                return
        elif isinstance(image_data, np.ndarray):  # It's a numpy array
            print("Loading image from numpy array")
            image = image_data
        else:
            print("Unsupported image data type")
            return

        # Ensure the image is in the correct shape (H, W, C)
        if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
            image = np.transpose(image, (1, 2, 0))
        elif len(image.shape) == 2:
            image = image[:, :, np.newaxis]  # Add channel dimension for grayscale


        pixmap = self.numpy_to_qpixmap(image)
        if not pixmap.isNull():
            self.pixmap_item.setPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))

            # # Restore the previous view range if available
            # if self.current_view_range is not None:
            #     self.fitInView(self.current_view_range, Qt.AspectRatioMode.KeepAspectRatio)
            # else:
            #     self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

             # Restore the previous view transform if available
            if self.current_view_transform is not None:
                self.setTransform(self.current_view_transform)
            else:
                self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)           

            #self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            print("Image loaded successfully")
        else:
            print("Failed to convert image to pixmap")


    def numpy_to_qpixmap(self, img):
        if img.dtype == np.float32:
            # Normalize float32 data to 0-255 range
            print("Normalizing float32 data")
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = (img * 255).astype(np.uint8)
        
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if len(img.shape) == 2:  # Grayscale
            h, w = img.shape
            qimg = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB
            h, w, _ = img.shape
            qimg = QImage(img.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            h, w, _ = img.shape
            qimg = QImage(img.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        
        return QPixmap.fromImage(qimg)


    def reset_zoom(self):
        """Resets the zoom to fit the entire image in view."""
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)