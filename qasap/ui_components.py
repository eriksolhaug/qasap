"""
UI Components module - PyQt5 GUI components
This is a placeholder that imports from the original qasap_v0.5 for now.
Future work: refactor the GUI components into logical pieces.
"""

import sys
from PyQt5 import QtWidgets

# Import the original classes from qasap_v0.5
# This allows gradual refactoring while maintaining functionality
try:
    from qasap_v0.5.qasap_v0.5 import SpectrumPlotter, SpectrumPlotterApp, LineListWindow
except ImportError:
    # Fallback if direct import fails
    print("Warning: Could not import UI components from qasap_v0.5")
    # Define placeholder classes
    class SpectrumPlotter(QtWidgets.QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("SpectrumPlotter placeholder - import qasap_v0.5 for full functionality")
    
    class SpectrumPlotterApp(QtWidgets.QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("SpectrumPlotterApp placeholder - import qasap_v0.5 for full functionality")
    
    class LineListWindow(QtWidgets.QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("LineListWindow placeholder - import qasap_v0.5 for full functionality")

__all__ = ['SpectrumPlotter', 'SpectrumPlotterApp', 'LineListWindow']
