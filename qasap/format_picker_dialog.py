"""
Format Picker Dialog - PyQt5 based format selection UI

Displays detected spectrum formats with confidence scores and allows user
to select the format before loading the spectrum.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from typing import Optional, Dict, Any, List, Tuple


class FormatPickerDialog(QtWidgets.QDialog):
    """
    Dialog for selecting spectrum file format from detected candidates.
    
    Shows a list of auto-detected formats with confidence scores, descriptions,
    and allows user to select which one to use.
    """
    
    def __init__(self, filepath: str, candidates: List[Dict[str, Any]], parent=None):
        """
        Initialize the format picker dialog.
        
        Parameters
        ----------
        filepath : str
            Path to the spectrum file
        candidates : list of dict
            List of detected format candidates from detect_spectrum_format()
            Each dict has: {"key": "...", "score": int, "notes": str, "options": dict}
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.filepath = filepath
        self.candidates = candidates
        self.selected_format = None
        self.selected_options = None
        
        self.setWindowTitle("QASAP - Format Selection")
        self.setGeometry(100, 100, 700, 500)
        self.setModal(True)
        
        # Sort candidates by score (highest first)
        self.candidates.sort(key=lambda c: c["score"], reverse=True)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()
        
        # Header with filename
        header = QtWidgets.QLabel()
        header.setText(f"<b>Select Format for:</b> {self.filepath}")
        header.setStyleSheet("font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Info label
        info = QtWidgets.QLabel()
        info.setText(f"Detected {len(self.candidates)} possible format(s). Select one and click Load.")
        info.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 10px;")
        layout.addWidget(info)
        
        # Separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep)
        
        # Initialize details_label FIRST (before any callbacks that use it)
        self.details_label = QtWidgets.QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("background-color: #f5f5f5; padding: 10px; border-radius: 4px; font-family: monospace;")
        
        # Format list with scroll
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout()
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)
        
        self.format_buttons = {}
        self.button_group = QtWidgets.QButtonGroup()
        
        for i, candidate in enumerate(self.candidates):
            btn = self._create_format_button(candidate, i)
            scroll_layout.addWidget(btn)
            self.format_buttons[i] = btn
            self.button_group.addButton(btn)
            
            # Select first (highest confidence) by default
            if i == 0:
                btn.setChecked(True)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        
        # Separator
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep2)
        
        # Add details panel
        layout.addWidget(self.details_label)
        
        # Button bar
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        load_btn = QtWidgets.QPushButton("Load")
        load_btn.setFixedWidth(100)
        load_btn.setDefault(True)
        load_btn.clicked.connect(self.accept)
        button_layout.addWidget(load_btn)
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setFixedWidth(100)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Connect button group signal
        self.button_group.buttonClicked.connect(self._on_button_clicked)
        
        # NOW update details for the first selected item
        self._on_format_selected(0)
        
        self.setLayout(layout)
    
    def _create_format_button(self, candidate: Dict[str, Any], index: int) -> QtWidgets.QRadioButton:
        """Create a format selection button with confidence score."""
        key = candidate["key"]
        score = candidate["score"]
        notes = candidate.get("notes", "")
        
        # Color code by confidence
        if score >= 90:
            color = "#4CAF50"  # Green
            confidence = "High"
        elif score >= 75:
            color = "#FF9800"  # Orange
            confidence = "Medium"
        else:
            color = "#f44336"  # Red
            confidence = "Low"
        
        # Create button with rich text
        btn = QtWidgets.QRadioButton()
        btn.setStyleSheet(f"""
            QRadioButton {{
                spacing: 10px;
                font-size: 11px;
            }}
        """)
        
        # Create the label text
        label_text = f"{key} ({confidence}: {score}%)"
        btn.setText(label_text)
        
        # Store data
        btn.candidate_index = index
        btn.candidate_data = candidate
        
        # Tooltip with full notes
        btn.setToolTip(notes)
        
        return btn
    
    def _on_button_clicked(self, button: QtWidgets.QRadioButton):
        """Handle format button selection."""
        index = button.candidate_index
        self._on_format_selected(index)
    
    def _on_format_selected(self, index: int):
        """Update details panel when format is selected."""
        candidate = self.candidates[index]
        
        key = candidate["key"]
        score = candidate["score"]
        notes = candidate.get("notes", "")
        options = candidate.get("options", {})
        
        # Build details text
        details = f"<b>Format:</b> {key}<br>"
        details += f"<b>Confidence:</b> {score}%<br>"
        details += f"<b>Description:</b> {notes}<br>"
        
        if options:
            details += f"<b>Options:</b><br>"
            for k, v in options.items():
                details += f"&nbsp;&nbsp;{k}: {v}<br>"
        
        self.details_label.setText(details)
        self.selected_format = key
        self.selected_options = options
    
    def get_selection(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the selected format and options.
        
        Returns
        -------
        tuple or None
            (format_key, options_dict) if user clicked Load, None if cancelled
        """
        if self.selected_format:
            return (self.selected_format, self.selected_options or {})
        return None
