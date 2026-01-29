# QASAP — Quick Analysis of Spectra and Profiles Package
## Version 0.8

Interactive Python tool for comprehensive 1D spectral analysis with intelligent file format detection. QASAP provides both quick-look functionality and advanced analysis features including multi-component line fitting, continuum modeling, and Bayesian MCMC fitting.

## Features

- **Intelligent Format Auto-Detection**: Automatically detects 7+ ASCII and FITS spectrum formats with confidence scoring
- **Interactive Plotting**: Real-time spectrum visualization with zoom, pan, and smooth controls
- **Multi-Component Fitting**: Gaussian and Voigt profile fitting (single and multi-component) with Listfit mode for simultaneous fitting
- **Line Analysis**: Equivalent width, redshift estimation, velocity mode analysis
- **Advanced Fitting**: Bayesian MCMC with posterior distributions (work in progress still - works well for single profiles)
- **Item Tracker**: Centralized management of all plotted components (Gaussians, Voigts, polynomials, continuum) with multi-select and deletion capabilities
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

### Conda Environment Setup

```bash
# Create a new conda environment
conda create -n qasap python=3.8

# Activate the environment
conda activate qasap

# Install dependencies
pip install -r requirements.txt

# Or install with conda
conda install numpy scipy matplotlib astropy pandas lmfit pyqt emcee corner

# Run QASAP
python qasap.py spectrum.fits
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
qasap.py                   # Main entry point (v0.8)
```

## Data Files

- `emlines.txt`: Emission line catalog
- `emlines_osc.txt`: Lines with oscillator strengths
- `instrument_bands.txt`: Filter definitions


## User Interface Windows

### 1. Spectrum Plotter (Main Window)
The central interactive spectrum visualization with the following controls:

**Key Features:**
- Real-time spectrum display with wavelength calibration
- Zoom and pan capabilities
- Automated continuum detection and normalization
- Multi-component profile fitting visualization
- Redshift tracking with selected line highlighting

**Keyboard Shortcuts:**
- `e`: Toggle emission line display
- `z`: Enter redshift mode (select already fitted emission line to estimate redshift)
- `g`: Single Gaussian fitting mode
- `v`: Single Voigt profile fitting mode
- `h`: Listfit mode (multi-component simultaneous fitting)
- `m`: Continuum modeling mode
- `:`: Bayesian MCMC fitting mode
- `l`: Toggle log scale
- `f`: Apply smoothing to spectrum
- `c`: Clear currently selected component
- `w`: Remove Gaussian/Voigt from plot
- `M`: Remove continuum model
- `*`: Show/hide Item Tracker window
- `q`: Quit application

**Interactions:**
- **Click on spectrum**: Select wavelength for analysis or component placement
- **Spacebar**: Confirm actions in fitting modes or confirm bounds in listfit mode

### 2. Control Panel (Sidebar)
Displays real-time analysis information and fitting parameters:

**Sections:**
- **Quick Stats**: Line center, FWHM, equivalent width, signal-to-noise
- **Redshift**: Set redshift for displayed emission lines (toggled with key `e`)
- **Polynomial Order**: Set the order for the polynomial for Continuum mode (with key `m`)

### 3. Linelist Window
Database browser for emission and absorption lines:

**Features:**
- Searchable catalog of common emission lines (H-alpha, H-beta, [OIII], [OII], Ly-alpha, etc.)
- Rest wavelength reference
- Line identification at current redshift
- Click to jump to line in spectrum
- Filter by line type or wavelength range

**Usage:**
- Select a line to overlay on spectrum
- Double-click to center spectrum on line at current redshift

### 4. Listfit Window (Multi-Component Fitting)
Dialog for simultaneous fitting of multiple spectral components:

**Workflow:**
1. Press `h` on spectrum to activate listfit mode
2. Define fitting bounds by clicking two wavelengths (spacebar confirms each)
3. Listfit Window opens with component selection
4. Choose component types and quantities:
   - **Gaussian**: Simple Gaussian profiles
   - **Voigt**: Gaussian + Lorentzian (more realistic for emission lines)
   - **Polynomial**: Continuum background (specify order: 1-4)
5. Click "Calculate Fit" to perform simultaneous fitting
6. Fitted components displayed with individual colors:
   - Red: Gaussian components
   - Orange: Voigt components
   - Magenta: Polynomial continuum
   - Dark Blue: Combined total fit

**Parameters:**
- Each component shows fitted parameters (amplitude, center wavelength, width/sigma)
- Errors computed from covariance matrix
- Quality metrics: χ² and reduced χ²

### 5. Item Tracker Window
Centralized feature management panel accessed with `*` key:

**Display Columns:**
- **Name**: Feature identifier (e.g., "Gaussian 1", "Voigt 2")
- **Type**: Component type (gaussian, voigt, polynomial, continuum)
- **Position**: Wavelength bounds or center position
- **Color**: Visual indicator with color box showing plot color

**Operations:**
- Multi-select items with Control/Shift+Click
- Right-click context menu: "Delete" to remove from plot
- "Delete Selected" button: Remove multiple items at once
- "Clear All" button: Remove all features

## Versions

- **v0.8** (current): Listfit mode for simultaneous multi-component fitting, ItemTracker for centralized feature management with multi-select and deletion, auto-fit registration, redshift mode improvements
- **v0.7** (stable): Available as tagged release on GitHub. Refactored with intelligent format auto-detection, modular architecture, and comprehensive UI
- **v0.5** (legacy): Available in `qasap_v0.5/` directory with full Voigt/Gaussian fitting, MCMC, velocity mode

## Citation

```
Solhaug, E. (2025). QASAP: Quick Analysis of Spectra and Profiles.
https://github.com/eriksolhaug/qasap
```

## License

MIT License - See LICENSE file

## Author

Erik Solhaug
