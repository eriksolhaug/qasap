#!/usr/bin/env python
"""
QASAP - Quick & Advanced Spectrum Analysis Package
v0.6 (Refactored)

Interactive spectral analysis tool with support for:
- Multiple input file formats (11+ FITS/ASCII formats)
- Interactive Gaussian and Voigt profile fitting
- Continuum modeling and equivalent width calculations
- Velocity mode analysis
- MCMC Bayesian fitting
- Publication-quality visualization

Author: Erik Solhaug
License: MIT
"""

import argparse
import sys
import os
from PyQt5 import QtWidgets

# Import modular components
from qasap.spectrum_io import SpectrumIO
from qasap.spectrum_analysis import SpectrumAnalysis

# Import UI components (for now, from qasap_v0.5)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qasap_v0.5'))
    from qasap_v0_5 import SpectrumPlotter, SpectrumPlotterApp
except ImportError:
    print("Error: Could not import UI components from qasap_v0.5")
    print("Please ensure qasap_v0.5/qasap_v0.5.py exists")
    sys.exit(1)


def main():
    """Main entry point for QASAP"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='QASAP v0.6 - Quick & Advanced Spectrum Analysis Package',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qasap.py spectrum.fits --file_flag 2 --redshift 0.5
  python qasap.py spectrum.txt --file_flag 3 --line_center 5007
  python qasap.py spectrum.fits --file_flag 2 --gui
        """
    )
    
    parser.add_argument('fits_file', type=str, nargs='?',
                        help='Path to spectrum file (FITS or ASCII)')
    parser.add_argument('--file_flag', type=int, default=0,
                        help='File format identifier (0-10, default: 0)')
    parser.add_argument('--redshift', type=float, default=0.0,
                        help='Initial redshift value (default: 0.0)')
    parser.add_argument('--zoom_factor', type=float, default=0.1,
                        help='Initial zoom level (default: 0.1)')
    parser.add_argument('--lsf', type=str, default="10",
                        help='Line spread function: width in km/s or path to LSF file')
    parser.add_argument('--gui', action='store_true',
                        help='Launch GUI for interactive input')
    parser.add_argument('--alfosc', action='store_true',
                        help='Extract 1D spectrum from 2D ALFOSC data')
    parser.add_argument('--alfosc_bin', type=int,
                        help='ALFOSC 1D extraction bin width in Angstroms')
    parser.add_argument('--alfosc_left', type=int,
                        help='Left x-bound for ALFOSC extraction')
    parser.add_argument('--alfosc_right', type=int,
                        help='Right x-bound for ALFOSC extraction')
    parser.add_argument('--alfosc_output', type=str,
                        help='Output file for extracted 1D spectrum')
    parser.add_argument('--version', action='version',
                        version='QASAP v0.6')
    
    args = parser.parse_args()
    
    # Initialize QApplication
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    
    # Launch GUI mode if requested
    if args.gui:
        if args.fits_file:
            plotter = SpectrumPlotter(
                args.fits_file, args.redshift, args.zoom_factor, args.file_flag,
                args.alfosc, args.alfosc_bin, args.alfosc_left, 
                args.alfosc_right, args.alfosc_output, args.lsf
            )
            opener_window = SpectrumPlotterApp(plotter)
            opener_window.show()
        else:
            print("Error: --gui mode requires a spectrum file")
            sys.exit(1)
    
    # Handle command-line mode
    elif args.fits_file:
        print(f"Loading spectrum: {args.fits_file}")
        plotter = SpectrumPlotter(
            args.fits_file, args.redshift, args.zoom_factor, args.file_flag,
            args.alfosc, args.alfosc_bin, args.alfosc_left,
            args.alfosc_right, args.alfosc_output, args.lsf
        )
        plotter.plot_spectrum()
    
    else:
        parser.print_help()
        sys.exit(1)
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
