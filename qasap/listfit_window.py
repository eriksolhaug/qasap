"""
ListfitWindow - Multi-component spectrum fitting dialog
"""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIntValidator, QIcon
import numpy as np
from scipy.optimize import curve_fit
from qasap.ui_utils import get_qasap_icon


class ListfitWindow(QtWidgets.QWidget):
    """Dialog for defining and fitting multiple spectrum components"""
    
    fit_requested = pyqtSignal(list)  # Emits list of components to fit
    
    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds
        self.components = []  # List of {'type': 'gaussian'|'voigt'|'polynomial'|'mask_feature', 'order': N, ...}
        self.gaussian_count = 0
        self.voigt_count = 0
        self.polynomial_count = 0
        self.mask_feature_count = 0
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("QASAP - List Fit")
        # Load and set window icon
        self.setWindowIcon(get_qasap_icon())
        self.setGeometry(500, 100, 700, 500)
        
        layout = QtWidgets.QHBoxLayout()
        
        # Left side: Component controls
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(QtWidgets.QLabel("Add Components:"))
        
        # Gaussian controls
        gaussian_layout = QtWidgets.QHBoxLayout()
        self.gaussian_label = QtWidgets.QLabel("Gaussian:")
        self.btn_gaussian_add = QtWidgets.QPushButton("+")
        self.btn_gaussian_add.setMaximumWidth(40)
        self.btn_gaussian_add.clicked.connect(lambda: self.add_component('gaussian'))
        self.btn_gaussian_remove = QtWidgets.QPushButton("-")
        self.btn_gaussian_remove.setMaximumWidth(40)
        self.btn_gaussian_remove.clicked.connect(lambda: self.remove_component('gaussian'))
        gaussian_layout.addWidget(self.gaussian_label)
        gaussian_layout.addWidget(self.btn_gaussian_add)
        gaussian_layout.addWidget(self.btn_gaussian_remove)
        gaussian_layout.addStretch()
        left_layout.addLayout(gaussian_layout)
        
        # Voigt controls
        voigt_layout = QtWidgets.QHBoxLayout()
        self.voigt_label = QtWidgets.QLabel("Voigt:")
        self.btn_voigt_add = QtWidgets.QPushButton("+")
        self.btn_voigt_add.setMaximumWidth(40)
        self.btn_voigt_add.clicked.connect(lambda: self.add_component('voigt'))
        self.btn_voigt_remove = QtWidgets.QPushButton("-")
        self.btn_voigt_remove.setMaximumWidth(40)
        self.btn_voigt_remove.clicked.connect(lambda: self.remove_component('voigt'))
        voigt_layout.addWidget(self.voigt_label)
        voigt_layout.addWidget(self.btn_voigt_add)
        voigt_layout.addWidget(self.btn_voigt_remove)
        voigt_layout.addStretch()
        left_layout.addLayout(voigt_layout)
        
        # Polynomial controls
        poly_layout = QtWidgets.QVBoxLayout()
        poly_header = QtWidgets.QHBoxLayout()
        self.poly_label = QtWidgets.QLabel("Polynomial:")
        poly_header.addWidget(self.poly_label)
        poly_header.addStretch()
        poly_layout.addLayout(poly_header)
        
        poly_order_layout = QtWidgets.QHBoxLayout()
        self.poly_order_label = QtWidgets.QLabel("Order:")
        self.poly_order_input = QtWidgets.QLineEdit("1")
        self.poly_order_input.setMaximumWidth(80)
        self.poly_order_input.setValidator(QIntValidator(0, 10))
        poly_order_layout.addWidget(self.poly_order_label)
        poly_order_layout.addWidget(self.poly_order_input)
        poly_order_layout.addStretch()
        poly_layout.addLayout(poly_order_layout)
        
        poly_button_layout = QtWidgets.QHBoxLayout()
        self.btn_poly_add = QtWidgets.QPushButton("+")
        self.btn_poly_add.setMaximumWidth(40)
        self.btn_poly_add.clicked.connect(self.add_polynomial)
        self.btn_poly_remove = QtWidgets.QPushButton("-")
        self.btn_poly_remove.setMaximumWidth(40)
        self.btn_poly_remove.clicked.connect(lambda: self.remove_component('polynomial'))
        poly_button_layout.addWidget(QtWidgets.QLabel(""))
        poly_button_layout.addWidget(self.btn_poly_add)
        poly_button_layout.addWidget(self.btn_poly_remove)
        poly_button_layout.addStretch()
        poly_layout.addLayout(poly_button_layout)
        
        left_layout.addLayout(poly_layout)
        
        # Mask Feature controls
        mask_layout = QtWidgets.QVBoxLayout()
        mask_header = QtWidgets.QHBoxLayout()
        self.mask_label = QtWidgets.QLabel("Mask Feature:")
        mask_header.addWidget(self.mask_label)
        mask_header.addStretch()
        mask_layout.addLayout(mask_header)
        
        mask_description = QtWidgets.QLabel("Used for Masking Non-Continuum Features from the Initial Polynomial Guess")
        mask_description.setStyleSheet("font-size: 9px; color: gray; font-style: italic;")
        mask_layout.addWidget(mask_description)
        
        mask_range_layout = QtWidgets.QHBoxLayout()
        self.mask_min_label = QtWidgets.QLabel("Min λ:")
        self.mask_min_input = QtWidgets.QLineEdit()
        self.mask_min_input.setMaximumWidth(100)
        self.mask_min_input.setPlaceholderText("e.g., 5500")
        mask_range_layout.addWidget(self.mask_min_label)
        mask_range_layout.addWidget(self.mask_min_input)
        
        self.mask_max_label = QtWidgets.QLabel("Max λ:")
        self.mask_max_input = QtWidgets.QLineEdit()
        self.mask_max_input.setMaximumWidth(100)
        self.mask_max_input.setPlaceholderText("e.g., 5550")
        mask_range_layout.addWidget(self.mask_max_label)
        mask_range_layout.addWidget(self.mask_max_input)
        mask_range_layout.addStretch()
        mask_layout.addLayout(mask_range_layout)
        
        mask_button_layout = QtWidgets.QHBoxLayout()
        self.btn_mask_add = QtWidgets.QPushButton("+")
        self.btn_mask_add.setMaximumWidth(40)
        self.btn_mask_add.clicked.connect(self.add_mask_feature)
        self.btn_mask_remove = QtWidgets.QPushButton("-")
        self.btn_mask_remove.setMaximumWidth(40)
        self.btn_mask_remove.clicked.connect(lambda: self.remove_component('mask_feature'))
        mask_button_layout.addWidget(QtWidgets.QLabel(""))
        mask_button_layout.addWidget(self.btn_mask_add)
        mask_button_layout.addWidget(self.btn_mask_remove)
        mask_button_layout.addStretch()
        mask_layout.addLayout(mask_button_layout)
        
        left_layout.addLayout(mask_layout)
        left_layout.addStretch()
        
        # Right side: Component list
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(QtWidgets.QLabel("Components to Fit:"))
        self.component_list = QtWidgets.QListWidget()
        self.component_list.itemClicked.connect(self.on_component_selected)
        right_layout.addWidget(self.component_list)
        
        # Buttons at bottom
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton("Calculate Fit")
        self.btn_fit.setMinimumHeight(40)
        self.btn_fit.clicked.connect(self.on_fit_requested)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.clicked.connect(self.close)
        button_layout.addWidget(self.btn_fit)
        button_layout.addWidget(self.btn_cancel)
        right_layout.addLayout(button_layout)
        
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)
        self.setLayout(layout)
    
    def add_component(self, comp_type):
        """Add a component of given type"""
        if comp_type == 'gaussian':
            component = {'type': 'gaussian', 'id': len(self.components)}
            label = f"Gaussian #{self.gaussian_count + 1}"
            self.gaussian_count += 1
        elif comp_type == 'voigt':
            component = {'type': 'voigt', 'id': len(self.components)}
            label = f"Voigt #{self.voigt_count + 1}"
            self.voigt_count += 1
        else:
            return
        
        self.components.append(component)
        self.component_list.addItem(label)
    
    def add_polynomial(self):
        """Add polynomial with specified order"""
        try:
            order = int(self.poly_order_input.text())
            if order < 0 or order > 10:
                order = 1
        except ValueError:
            order = 1
        
        component = {'type': 'polynomial', 'order': order, 'id': len(self.components)}
        self.components.append(component)
        label = f"Polynomial (order={order}) #{self.polynomial_count + 1}"
        self.polynomial_count += 1
        self.component_list.addItem(label)
    
    def add_mask_feature(self):
        """Add mask feature with specified wavelength range"""
        try:
            min_lambda = float(self.mask_min_input.text())
            max_lambda = float(self.mask_max_input.text())
            
            if min_lambda >= max_lambda:
                QtWidgets.QMessageBox.warning(self, "Invalid Range", "Min wavelength must be less than Max wavelength")
                return
            
            component = {'type': 'mask_feature', 'min_lambda': min_lambda, 'max_lambda': max_lambda, 'id': len(self.components)}
            self.components.append(component)
            label = f"Mask Feature ({min_lambda:.2f}-{max_lambda:.2f} Å) #{self.mask_feature_count + 1}"
            self.mask_feature_count += 1
            self.component_list.addItem(label)
            
            # Clear input fields for next mask
            self.mask_min_input.clear()
            self.mask_max_input.clear()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid wavelength values")
    
    def remove_component(self, comp_type):
        """Remove last component of given type"""
        for i in range(len(self.components) - 1, -1, -1):
            if self.components[i]['type'] == comp_type:
                self.components.pop(i)
                self.component_list.takeItem(i)
                if comp_type == 'gaussian':
                    self.gaussian_count = max(0, self.gaussian_count - 1)
                elif comp_type == 'voigt':
                    self.voigt_count = max(0, self.voigt_count - 1)
                elif comp_type == 'polynomial':
                    self.polynomial_count = max(0, self.polynomial_count - 1)
                elif comp_type == 'mask_feature':
                    self.mask_feature_count = max(0, self.mask_feature_count - 1)
                break
    
    def on_component_selected(self, item):
        """Handle component selection (for future use)"""
        pass
    
    def on_fit_requested(self):
        """Emit signal to perform fitting"""
        if not self.components:
            QtWidgets.QMessageBox.warning(self, "No Components", "Please add at least one component to fit")
            return
        self.fit_requested.emit(self.components)
        self.close()

