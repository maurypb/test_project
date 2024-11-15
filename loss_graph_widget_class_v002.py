import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import pyqtSlot, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QCursor
import logging
import math
from typing import List, Dict, Any

class LossGraphWidget(QWidget):
    """
    Interactive widget for displaying training loss graphs with champion model indicators.
    Updated to work with the new TrainingState structure.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing LossGraphWidget")

        # Set up layout
        self.layout = QVBoxLayout(self)
        
        # Initialize graph widget with white background and log scale
        self.graph_widget = pg.PlotWidget()
        self.layout.addWidget(self.graph_widget)
        self.graph_widget.setBackground('w')
        self.graph_widget.setLogMode(y=True)
        self.graph_widget.setLabel('left', 'Loss')
        self.graph_widget.setLabel('bottom', 'Epoch')

        # Initialize plot curves and visual elements
        self.loss_curve = self.graph_widget.plot(pen='b')
        self.champion_lines = []
        self.champion_labels = []
        
        # Store champion data for tooltip access
        self.all_champions: List[Dict[str, Any]] = []
        self.champion_losses: List[float] = []

        # Create info label for champion details
        self.info_label = QLabel(self)
        self.info_label.setStyleSheet(
            "background-color: white; border: 1px solid black; padding: 4px;"
        )
        self.info_label.setVisible(False)
        self.info_label.raise_()
        
        # Connect mouse interaction events
        self.graph_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.graph_widget.scene().sigMouseClicked.connect(self._on_mouse_interaction)
        self.graph_widget.getViewBox().sigStateChanged.connect(self._on_mouse_interaction)

        self.first_time = True
        self.view_range_timer = QTimer()
        self.view_range_timer.timeout.connect(self._update_view_range)
        self.view_range_timer.setInterval(100)  # 100ms interval

    @pyqtSlot(list, int, int, float, float, list, list)
    def update_plot(self, losses: List[float], current_epoch: int, total_epochs: int,
                   current_loss: float, min_loss: float, all_champions: List[Dict],
                   current_champions: List[Dict]):
        """
        Update the plot with new training state data.
        
        Args:
            losses: List of loss values
            current_epoch: Current training epoch
            total_epochs: Total number of epochs
            current_loss: Current loss value
            min_loss: Best loss achieved
            all_champions: List of all champion models
            current_champions: List of current champion models
        """
        self.logger.debug(f"Updating plot: current_epoch={current_epoch}, current_loss={current_loss}")

        # Save current view state
        view_box = self.graph_widget.getViewBox()
        current_view_range = view_box.viewRange() if not self.first_time else None

        # Update plot data
        self.loss_curve.setData(list(range(1, len(losses) + 1)), losses)
        self.graph_widget.setTitle(
            f"Training Loss (Log Scale) - Epoch {current_epoch}/{total_epochs}\n"
            f"Current Loss: {current_loss:.6f}, Min Loss: {min_loss:.6f}"
        )

        # Store champion data for tooltip access
        self.all_champions = all_champions
        self.champion_losses = [
            losses[c['epoch'] - 1] if c['epoch'] <= len(losses) else c['loss']
            for c in all_champions
        ]

        # Update champion markers
        self._update_champion_lines(all_champions, current_champions, losses)

        # Handle view range
        if self.first_time:
            self.first_time = False
            self.graph_widget.enableAutoRange()
        else:
            view_box.setRange(xRange=current_view_range[0], yRange=current_view_range[1], padding=0)

    def _on_mouse_moved(self, pos):
        """Handle mouse movement for tooltips"""
        mouse_point = self.graph_widget.plotItem.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()

        # Calculate dynamic threshold based on view range
        view_box = self.graph_widget.getViewBox()
        current_view_range = view_box.viewRange()
        viewport_width = current_view_range[0][1] - current_view_range[0][0]
        threshold = viewport_width / 50

        # Find nearest champion line
        nearest_distance = float('inf')
        nearest_index = -1

        for i, line in enumerate(self.champion_lines):
            line_epoch = line.pos().x()
            distance = abs(x - line_epoch)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = i

        # Show tooltip if close enough to a champion line
        if nearest_index != -1 and nearest_distance < threshold:
            champion = self.all_champions[nearest_index]
            loss = self.champion_losses[nearest_index]
            
            # Enhanced tooltip content
            tooltip_text = (
                f"Champion Model\n"
                f"Epoch: {int(champion['epoch'])}\n"
                f"Loss: {loss:.6f}"
            )
            
            self.info_label.setText(tooltip_text)
            self.info_label.adjustSize()
            
            # Position tooltip near cursor
            cursor_pos = QCursor.pos()
            widget_pos = self.graph_widget.mapToGlobal(self.graph_widget.pos())
            self.info_label.move(
                cursor_pos.x() - widget_pos.x() - self.info_label.width(),
                cursor_pos.y() - widget_pos.y() + 20
            )
            self.info_label.setVisible(True)
        else:
            self.info_label.setVisible(False)

    def _on_mouse_interaction(self, *args):
        """Handle mouse clicks and drags"""
        self.info_label.setVisible(False)
        if not self.view_range_timer.isActive():
            self.view_range_timer.start()

    def _update_view_range(self):
        """Update view range after user interaction"""
        self.view_range_timer.stop()
        view_box = self.graph_widget.getViewBox()
        view_box.enableAutoRange(enable=False)

    def _update_champion_lines(self, all_champions: List[Dict], 
                             current_champions: List[Dict], losses: List[float]):
        """Update champion marker lines on the plot"""
        # Clear existing lines
        for line in self.champion_lines:
            self.graph_widget.removeItem(line)
        self.champion_lines.clear()

        # Get current champion epochs
        current_champion_epochs = [c['epoch'] for c in current_champions]

        # Add lines for each champion
        for champion in all_champions:
            is_current = champion['epoch'] in current_champion_epochs
            
            # Set line style based on champion status
            color = 'r' if is_current else QColor(100, 100, 100, 100)
            style = Qt.PenStyle.SolidLine if is_current else Qt.PenStyle.DotLine
            width = 2 if is_current else 1

            # Create and add line
            line = pg.InfiniteLine(
                pos=champion['epoch'],
                angle=90,
                pen=pg.mkPen(color=color, style=style, width=width)
            )
            self.graph_widget.addItem(line)
            self.champion_lines.append(line)