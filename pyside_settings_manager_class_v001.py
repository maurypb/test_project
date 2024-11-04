import json
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.QtWidgets import  QWidget, QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox
from PyQt6.QtWidgets import    QCheckBox, QRadioButton, QComboBox, QSlider, QDial, QDateEdit, QTimeEdit
from PyQt6.QtWidgets import     QDateTimeEdit, QListWidget, QTreeWidget

from PyQt6.QtCore import QDate, QTime, QDateTime

import sys

class SettingsManager:
    def __init__(self, main_window):
        self.main_window = main_window
        self.default_settings_file = "settings.json"

    def save_settings(self, filename=None):
        settings_file = filename or self.default_settings_file
        # Ensure the filename has a .json extension
        if not settings_file.lower().endswith('.json'):
            settings_file += '.json'


        settings = {}
        for widget in self.main_window.findChildren(QWidget):
            name = widget.objectName()
            if not name or name.startswith("qt_"):
                continue


            if isinstance(widget, QLineEdit):
                settings[name] = widget.text()
            if isinstance(widget, ( QTextEdit, QPlainTextEdit)):
                settings[name] = widget.toPlainText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                settings[name] = widget.value()
            elif isinstance(widget, (QCheckBox, QRadioButton)):
                settings[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                settings[name] = {"text": widget.currentText(), "index": widget.currentIndex()}
            elif isinstance(widget, (QSlider, QDial)):
                settings[name] = widget.value()
            elif isinstance(widget, QDateEdit):
                settings[name] = widget.date().toString(Qt.ISODate)
            elif isinstance(widget, QTimeEdit):
                settings[name] = widget.time().toString(Qt.ISODate)
            elif isinstance(widget, QDateTimeEdit):
                settings[name] = widget.dateTime().toString(Qt.ISODate)
            elif isinstance(widget, QListWidget):
                settings[name] = [widget.item(i).text() for i in range(widget.count())]
            elif isinstance(widget, QTreeWidget):
                settings[name] = self._save_tree_widget_items(widget.invisibleRootItem())

        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
        
        return settings_file 

    def load_settings(self, filename=None):
        settings_file = filename or self.default_settings_file
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)

            for widget in self.main_window.findChildren(QWidget):
                name = widget.objectName()
                # Skip widgets with empty names or those starting with "qt_"
                if not name or name.startswith("qt_") or name not in settings:
                    continue
                if isinstance(widget, QLineEdit):
                    widget.setText(settings[name])
                if isinstance(widget, (QTextEdit, QPlainTextEdit)):
                    widget.setPlainText(settings[name])
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(settings[name])
                elif isinstance(widget, (QCheckBox, QRadioButton)):
                    widget.setChecked(settings[name])
                elif isinstance(widget, QComboBox):
                    index = widget.findText(settings[name]["text"])
                    if index >= 0:
                        widget.setCurrentIndex(index)
                    else:
                        widget.setCurrentIndex(settings[name]["index"])
                elif isinstance(widget, (QSlider, QDial)):
                    widget.setValue(settings[name])
                elif isinstance(widget, QDateEdit):
                    widget.setDate(QDate.fromString(settings[name], Qt.ISODate))
                elif isinstance(widget, QTimeEdit):
                    widget.setTime(QTime.fromString(settings[name], Qt.ISODate))
                elif isinstance(widget, QDateTimeEdit):
                    widget.setDateTime(QDateTime.fromString(settings[name], Qt.ISODate))
                elif isinstance(widget, QListWidget):
                    widget.clear()
                    widget.addItems(settings[name])
                elif isinstance(widget, QTreeWidget):
                    widget.clear()
                    self._load_tree_widget_items(widget, settings[name])

        except FileNotFoundError:
            print(f"Settings file '{settings_file}' not found. Using default settings.")

    def _save_tree_widget_items(self, item):
        result = []
        for i in range(item.childCount()):
            child = item.child(i)
            child_data = {
                "text": [child.text(j) for j in range(child.columnCount())],
                "children": self._save_tree_widget_items(child)
            }
            result.append(child_data)
        return result

    def _load_tree_widget_items(self, parent, items):
        for item_data in items:
            item = QTreeWidgetItem(parent)
            for i, text in enumerate(item_data["text"]):
                item.setText(i, text)
            self._load_tree_widget_items(item, item_data["children"])

# Usage example:
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # ... initialize your UI here ...
        self.settings_manager = SettingsManager(self)

    def closeEvent(self, event):
        self.settings_manager.save_settings()  # Use default filename
        # Or specify a custom filename:
        # self.settings_manager.save_settings("custom_settings.json")
        super().closeEvent(event)

# In your main application:
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.settings_manager.load_settings()  # Use default filename
    # Or specify a custom filename:
    # main_window.settings_manager.load_settings("custom_settings.json")
    main_window.show()
    sys.exit(app.exec_())