"""
QASAP - Quick & Advanced Spectrum Analysis Package
Version: 0.10
"""

__version__ = "0.10"
__author__ = "Erik Solhaug"

from .spectrum_io import SpectrumIO
from .spectrum_analysis import SpectrumAnalysis
from .ui_components import SpectrumPlotter, SpectrumPlotterApp, LineListWindow

# Import main function from root qasap module for entry point
import sys
from pathlib import Path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from qasap import main

__all__ = [
    'SpectrumIO',
    'SpectrumAnalysis',
    'SpectrumPlotter',
    'SpectrumPlotterApp',
    'LineListWindow',
    'main',
]
