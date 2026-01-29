#!/usr/bin/env python
"""
QASAP - Quick & Advanced Spectrum Analysis Package
v0.6 (Refactored)

Interactive spectral analysis tool with support for:
- Multiple input file formats with intelligent auto-detection (11+ FITS/ASCII formats)
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
  python qasap.py spectrum.fits                          # Auto-detect format
  python qasap.py spectrum.fits --redshift 0.5
  python qasap.py spectrum.txt --fmt ascii:3col
  python qasap.py spectrum.txt --gui
  
Format auto-detection:
  Automatic format detection with confidence scoring.
  For manual override, use: --fmt ascii:2col | ascii:3col | fits:image1d | etc.
        """
    )
    
    parser.add_argument('fits_file', type=str, nargs='?',
                        help='Path to spectrum file (FITS or ASCII)')
    parser.add_argument('--fmt', type=str, default=None,
                        help='Force format: ascii:2col | ascii:3col | ascii:flex | fits:image1d | fits:table:vector | fits:table:columns | fits:ext:spectrum')
    parser.add_argument('--detect', action='store_true',
                        help='Show detected formats and exit (useful for debugging)')
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
    
    # Handle format detection mode
    if args.detect:
        if not args.fits_file:
            print("Error: --detect requires a spectrum file")
            sys.exit(1)
        candidates = SpectrumIO.detect_spectrum_format(args.fits_file)
        print(f"\nFormat detection results for: {args.fits_file}\n")
        for i, c in enumerate(candidates, 1):
            print(f"{i}. {c['key']:<25} Score: {c['score']:>3}  {c['notes']}")
        print()
        sys.exit(0)
    
    # Initialize QApplication
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    
    # Launch GUI mode if requested
    if args.gui:
        if args.fits_file:
            plotter = SpectrumPlotter(
                args.fits_file, args.redshift, args.zoom_factor, args.fmt or 0,
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
        
        # Try to load with auto-detection or specified format
        try:
            if args.fmt:
                print(f"Using format: {args.fmt}")
                wav, spec, err, meta = SpectrumIO.read_spectrum(args.fits_file, fmt=args.fmt)
            else:
                # Auto-detect
                candidates = SpectrumIO.detect_spectrum_format(args.fits_file)
                if candidates:
                    candidates.sort(key=lambda c: c["score"], reverse=True)
                    best = candidates[0]
                    print(f"Auto-detected format: {best['key']} (confidence: {best['score']}/100)")
                    print(f"  {best['notes']}")
                    wav, spec, err, meta = SpectrumIO.read_spectrum(args.fits_file, fmt=best['key'])
                else:
                    print("Error: Could not auto-detect format")
                    sys.exit(1)
            
            print(f"Successfully loaded spectrum with {len(wav)} wavelength points")
            print(f"Wavelength range: {wav[0]:.2f} - {wav[-1]:.2f} Å")
            print(f"Flux range: {np.min(spec):.2e} - {np.max(spec):.2e}")
            
        except Exception as e:
            print(f"Error loading spectrum: {e}")
            sys.exit(1)
        
        # Launch the plotter
        plotter = SpectrumPlotter(
            args.fits_file, args.redshift, args.zoom_factor, args.fmt or 0,
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
    import numpy as np
    main()



if __name__ == '__main__':
    main()
