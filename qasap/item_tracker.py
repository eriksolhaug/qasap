"""
ItemTracker - Window to track and manage plotted spectrum features
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal


class ItemTracker(QtWidgets.QWidget):
    """Window for tracking and managing plotted spectrum features"""
    
    item_deleted = pyqtSignal(str)  # Emits item_id when deleted
    
    def __init__(self):
        super().__init__()
        self.items = {}  # {item_id: {'type': 'gaussian', 'name': 'Gaussian 1', 'line_obj': ...}}
        self.item_list_widget = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Item Tracker")
        self.setGeometry(100, 700, 400, 300)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        layout.addWidget(QtWidgets.QLabel("Plotted Features:"))
        
        # List widget with multi-select
        self.item_list_widget = QtWidgets.QListWidget()
        self.item_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.item_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.item_list_widget.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.item_list_widget)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_delete = QtWidgets.QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_clear_all = QtWidgets.QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.clear_all)
        button_layout.addWidget(self.btn_delete)
        button_layout.addWidget(self.btn_clear_all)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def add_item(self, item_id, item_type, name, line_obj=None):
        """Add an item to tracker"""
        self.items[item_id] = {
            'type': item_type,
            'name': name,
            'line_obj': line_obj
        }
        self.refresh_list()
    
    def remove_item(self, item_id):
        """Remove item from tracker"""
        if item_id in self.items:
            del self.items[item_id]
            self.refresh_list()
    
    def refresh_list(self):
        """Refresh the displayed list"""
        self.item_list_widget.clear()
        for item_id, item_info in self.items.items():
            display_text = f"{item_info['name']} [{item_info['type']}]"
            list_item = QtWidgets.QListWidgetItem(display_text)
            list_item.setData(Qt.UserRole, item_id)
            self.item_list_widget.addItem(list_item)
    
    def show_context_menu(self, position):
        """Show right-click context menu"""
        menu = QtWidgets.QMenu()
        delete_action = menu.addAction("Delete")
        
        action = menu.exec_(self.item_list_widget.mapToGlobal(position))
        if action == delete_action:
            self.delete_selected()
    
    def delete_selected(self):
        """Delete selected items"""
        selected_items = self.item_list_widget.selectedItems()
        for item in selected_items:
            item_id = item.data(Qt.UserRole)
            self.item_deleted.emit(item_id)
            self.remove_item(item_id)
    
    def clear_all(self):
        """Clear all items"""
        item_ids = list(self.items.keys())
        for item_id in item_ids:
            self.item_deleted.emit(item_id)
            self.remove_item(item_id)
