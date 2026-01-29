# QASAP — Quick & Advanced Spectrum Analysis Package
## Version 0.7

Interactive Python tool for comprehensive 1D spectral analysis with intelligent file format detection. QASAP provides both quick-look functionality and advanced analysis features including multi-component line fitting, continuum modeling, and Bayesian MCMC fitting.

## Features

- **Intelligent Format Auto-Detection**: Automatically detects 7+ ASCII and FITS spectrum formats with confidence scoring
- **Interactive Plotting**: Real-time spectrum visualization with zoom, pan, and smooth controls
- **Profile Fitting**: Gaussian and Voigt profile fitting (single and multi-component)
- **Line Analysis**: Equivalent width, redshift estimation, velocity mode analysis
- **Advanced Fitting**: Bayesian MCMC with posterior distributions
- **Utilities**: LSF convolution, line identification, filter overlays

## Installation

### Requirements
- Python 3.7+
- numpy, scipy, matplotlib, astropy, pandas, lmfit, PyQt5, emcee, corner

### Quick Start

```bash
git clone https://github.com/eriksolhaug/qasap.git
cd qasap
pip install -r requirements.txt
```

## Usage

### Auto-Detection (Recommended)

```bash
# Automatic format detection
python qasap.py spectrum.fits --redshift 0.5

# Preview detected formats
python qasap.py spectrum.fits --detect

# Force specific format if needed
python qasap.py spectrum.fits --fmt fits:image1d
```

### Supported Formats

| Format | Description |
|--------|-------------|
| `ascii:2col` | 2-column ASCII (wavelength, flux) |
| `ascii:3col` | 3-column ASCII (wavelength, flux, error) |
| `ascii:flex` | Flexible ASCII with custom column mapping |
| `fits:image1d` | 1D FITS image with wavelength in header |
| `fits:table:vector` | FITS table with wave/flux as vector arrays in rows |
| `fits:table:columns` | FITS table with per-pixel columns |
| `fits:ext:spectrum` | FITS SPECTRUM extension |

### Command-Line Options

```
--fmt              Force format (auto-detects if omitted)
--detect           Show detected formats and exit
--redshift         Initial redshift (default: 0.0)
--zoom_factor      Y-axis zoom (default: 0.1)
--lsf              LSF width in km/s or path to file (default: "10")
--gui              Launch GUI for interactive input
```

## Project Structure

```
qasap/
  ├── __init__.py           # Package initialization
  ├── spectrum_io.py        # File I/O with auto-detection
  ├── spectrum_analysis.py  # Fitting and analysis functions
  └── ui_components.py      # PyQt5 GUI interface
qasap_v0.5/                # v0.5 stable release
  ├── qasap_v0.5.py
  ├── README.md
  └── data files
qasap.py                   # Main entry point (v0.6)
```

## Data Files

- `emlines.txt`: Emission line catalog
- `emlines_osc.txt`: Lines with oscillator strengths
- `instrument_bands.txt`: Filter definitions

## Interactive Controls

See `qasap_v0.5/README.md` for comprehensive keyboard shortcut documentation.

## Versions

- **v0.7** (current): Refactored with intelligent format auto-detection, modular architecture
- **v0.5** (stable): Available in `qasap_v0.5/` with full Voigt/Gaussian fitting, MCMC, velocity mode

## Citation

```
Solhaug, E. (2025). QASAP v0.7: Quick & Advanced Spectrum Analysis Package.
https://github.com/eriksolhaug/qasap
```

## License

MIT License - See LICENSE file

## Author

Erik Solhaug
