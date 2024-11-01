import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QToolTip, QLabel
from PyQt6.QtCore import pyqtSlot, Qt, QPointF, QTimer
from PyQt6.QtGui import QColor, QFont, QCursor
import numpy as np
import logging
import math

class LossGraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing LossGraphWidget")

        self.layout = QVBoxLayout(self)
        
        self.graph_widget = pg.PlotWidget()
        self.layout.addWidget(self.graph_widget)

        self.graph_widget.setBackground('w')
        self.graph_widget.setLogMode(y=True)
        self.graph_widget.setLabel('left', 'Loss')
        self.graph_widget.setLabel('bottom', 'Epoch')

        self.loss_curve = self.graph_widget.plot(pen='b')

        self.champion_lines = []
        self.champion_labels = []
        self.all_champions = []  # Store all champions for tooltip access
        self.champion_losses = []  # Store all champion losses for tooltip access

        # Create QLabel for displaying champion information
        self.info_label = QLabel(self)
        self.info_label.setStyleSheet("background-color: white; border: 1px solid black; padding: 2px;")
        self.info_label.setVisible(False)
        self.info_label.raise_()  # Bring label to front
        #self.info_label.setMinimumWidth(250)  # Set minimum width to accommodate text

        # Connect mouse move event to custom handler
        self.graph_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        # Connect mouse press event to hide the label
        self.graph_widget.scene().sigMouseClicked.connect(self._on_mouse_interaction)
        # Connect mouse drag event to hide the label (handled by mouse press)
        self.graph_widget.getViewBox().sigStateChanged.connect(self._on_mouse_interaction)

        self.first_time=True 

    @pyqtSlot(list, int, int, float, float, list, list)
    def update_plot(self, losses, current_epoch, total_epochs, current_loss, min_loss, all_champions, current_champions):
        self.logger.debug(f"Updating plot: current_epoch={current_epoch}, current_loss={current_loss}")


        # Save the current view range (pan and zoom)
        view_box = self.graph_widget.getViewBox()
        current_view_range = view_box.viewRange()

        # Update the plot
        self.loss_curve.setData(list(range(1, len(losses) + 1)), losses)
        self.graph_widget.setTitle(f"Training Loss (Log Scale) - Epoch {current_epoch}/{total_epochs}, Current Loss: {current_loss:.6f}, Min Loss: {min_loss:.6f}")

        self.all_champions = all_champions  # Store champions for later reference
        self.champion_losses = [losses[c['epoch'] - 1] if c['epoch'] <= len(losses) else c['loss'] for c in all_champions]  # Store losses for later reference

        self._update_champion_lines(all_champions, current_champions, losses)
        #self._update_champion_labels(all_champions, current_champions, losses, current_view_range)

        if self.first_time:
            self.first_time=False
            self.graph_widget.enableAutoRange()
        else:
            #self.graph_widget.enableAutoRange()
            view_box.setRange(xRange=current_view_range[0], yRange=current_view_range[1], padding=0)




    def _on_mouse_moved(self, pos):
        # Translate mouse position to plot coordinates
        mouse_point = self.graph_widget.plotItem.vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()


        # Get the current view range to determine viewport width
        view_box = self.graph_widget.getViewBox()
        current_view_range = view_box.viewRange()
        viewport_width = current_view_range[0][1] - current_view_range[0][0]

        # Dynamic threshold based on viewport width
        threshold = viewport_width / 50


        # Find the nearest champion line
        nearest_distance = float('inf')
        nearest_index = -1

        for i, line in enumerate(self.champion_lines):
            line_epoch = line.pos().x()
            distance = abs(x - line_epoch)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = i

        # Display label if the nearest line is within the threshold
        #threshold = 2
        if nearest_index != -1 and nearest_distance < threshold:
            champion = self.all_champions[nearest_index]
            loss = self.champion_losses[nearest_index]
            self.info_label.setText(f"Champion Epoch {int(champion['epoch'])}, Loss: {loss:.6f}")
            self.info_label.adjustSize()
            # Move label relative to the graph widget, ensuring it stays visible
            graph_widget_pos = self.graph_widget.mapToGlobal(self.graph_widget.pos())
            cursor_offset = QCursor.pos() - graph_widget_pos
            self.info_label.move(cursor_offset.x()-self.info_label.width(), cursor_offset.y() + 30)  # Offset to avoid overlapping with the cursor
            self.info_label.setVisible(True)
        else:
            self.info_label.setVisible(False)

    def _on_mouse_interaction(self, *args):
        # Hide the label when mouse is clicked or dragged
        self.info_label.setVisible(False)


    def _update_champion_labels(self, all_champions, current_champions, losses, current_view_range):
        # Remove existing labels
        for label in self.champion_labels:
            self.graph_widget.removeItem(label)
        self.champion_labels.clear()

        current_champion_epochs = [c['epoch'] for c in current_champions]

        for champion in all_champions:
            is_current = champion['epoch'] in current_champion_epochs
            epoch = champion['epoch']
            loss = losses[epoch - 1] if epoch <= len(losses) else champion['loss']

            # Adjust loss for log scale visualization
            loss = max(loss, 1e-6)  # Ensure loss is above the minimum for log scale

            # Map log value to view coordinates
            y_min, y_max = current_view_range[1]
            log_loss = math.log10(loss)
            log_y_min, log_y_max = math.log10(max(y_min, 1e-6)), math.log10(max(y_max, 1e-6))
            if log_y_min == log_y_max:
                continue  # Avoid division by zero if the view is extremely zoomed in
            mapped_loss = (log_loss - log_y_min) / (log_y_max - log_y_min) * (y_max - y_min) + y_min

            # Create text item for the label
            text = pg.TextItem(
                text=f"{loss:.6f}",
                color='r' if is_current else 'gray',
                anchor=(0.5, 1.0),  # Anchor at the top-center of the label to position it better
                border=pg.mkPen(color='r' if is_current else 'gray', width=1),
                fill=pg.mkBrush('w')
            )
            
            # Position the label at the mapped position
            text.setPos(epoch, mapped_loss)
            text.setFont(QFont('Arial', 10))

            self.graph_widget.addItem(text)
            self.champion_labels.append(text)

        # Ensure labels are visible in log scale
        self.graph_widget.getViewBox().updateAutoRange()

    def _update_champion_lines(self, all_champions, current_champions, losses):
        for line in self.champion_lines:
            self.graph_widget.removeItem(line)
        self.champion_lines.clear()

        current_champion_epochs = [c['epoch'] for c in current_champions]

        for champion in all_champions:
            is_current = champion['epoch'] in current_champion_epochs
            color = 'r' if is_current else QColor(100, 100, 100, 100)
            style = Qt.PenStyle.SolidLine if is_current else Qt.PenStyle.DotLine
            width = 2 if is_current else 2

            line = pg.InfiniteLine(pos=champion['epoch'], angle=90, pen=pg.mkPen(color=color, style=style, width=width))
            self.graph_widget.addItem(line)
            self.champion_lines.append(line)
