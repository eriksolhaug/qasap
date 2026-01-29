"""
ItemTracker - Window to track and manage plotted spectrum features
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal


class ColorBoxDelegate(QtWidgets.QStyledItemDelegate):
    """Custom delegate to draw color boxes in table"""
    def paint(self, painter, option, index):
        color_text = index.data()
        if color_text:
            color = QtGui.QColor(color_text)
            painter.fillRect(option.rect, color)
            painter.drawRect(option.rect)


class ItemTracker(QtWidgets.QWidget):
    """Window for tracking and managing plotted spectrum features"""
    
    item_deleted = pyqtSignal(str)  # Emits item_id when deleted
    
    def __init__(self):
        super().__init__()
        self.items = {}  # {item_id: {'type': 'gaussian', 'name': 'Gaussian 1', 'position': 'bounds or value', 'color': 'red', ...}}
        self.item_table = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Item Tracker")
        self.setGeometry(100, 700, 600, 350)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        layout.addWidget(QtWidgets.QLabel("Plotted Features:"))
        
        # Table widget with columns
        self.item_table = QtWidgets.QTableWidget()
        self.item_table.setColumnCount(4)
        self.item_table.setHorizontalHeaderLabels(['Name', 'Type', 'Position', 'Color'])
        self.item_table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.item_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.item_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.item_table.customContextMenuRequested.connect(self.show_context_menu)
        self.item_table.horizontalHeader().setStretchLastSection(False)
        self.item_table.setColumnWidth(0, 150)
        self.item_table.setColumnWidth(1, 100)
        self.item_table.setColumnWidth(2, 150)
        self.item_table.setColumnWidth(3, 80)
        layout.addWidget(self.item_table)
        
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
    
    def add_item(self, item_id, item_type, name, position='', color='gray', line_obj=None):
        """Add an item to tracker"""
        self.items[item_id] = {
            'type': item_type,
            'name': name,
            'position': position,
            'color': color,
            'line_obj': line_obj
        }
        self.refresh_table()
    
    def remove_item(self, item_id):
        """Remove item from tracker"""
        if item_id in self.items:
            del self.items[item_id]
            self.refresh_table()
    
    def refresh_table(self):
        """Refresh the displayed table"""
        self.item_table.setRowCount(0)
        for item_id, item_info in self.items.items():
            row = self.item_table.rowCount()
            self.item_table.insertRow(row)
            
            # Name column
            name_item = QtWidgets.QTableWidgetItem(item_info['name'])
            name_item.setData(Qt.UserRole, item_id)
            self.item_table.setItem(row, 0, name_item)
            
            # Type column
            type_item = QtWidgets.QTableWidgetItem(item_info['type'])
            self.item_table.setItem(row, 1, type_item)
            
            # Position column
            pos_item = QtWidgets.QTableWidgetItem(str(item_info['position']))
            self.item_table.setItem(row, 2, pos_item)
            
            # Color column
            color_item = QtWidgets.QTableWidgetItem(item_info['color'])
            color_item.setBackground(QtGui.QColor(item_info['color']))
            self.item_table.setItem(row, 3, color_item)
    
    def show_context_menu(self, position):
        """Show right-click context menu"""
        menu = QtWidgets.QMenu()
        delete_action = menu.addAction("Delete")
        
        action = menu.exec_(self.item_table.mapToGlobal(position))
        if action == delete_action:
            self.delete_selected()
    
    def delete_selected(self):
        """Delete selected items"""
        selected_rows = set(index.row() for index in self.item_table.selectedIndexes())
        for row in sorted(selected_rows, reverse=True):
            item = self.item_table.item(row, 0)
            item_id = item.data(Qt.UserRole)
            self.item_deleted.emit(item_id)
            self.remove_item(item_id)
    
    def clear_all(self):
        """Clear all items"""
        item_ids = list(self.items.keys())
        for item_id in item_ids:
            self.item_deleted.emit(item_id)
            self.remove_item(item_id)

