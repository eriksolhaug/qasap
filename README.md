# <img src="logo/qasap_logo.png" alt="QASAP Logo" width="80">&nbsp;&nbsp;QASAP — Quick Analysis of Spectra and Profiles
## Version 0.9

![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-blue?logo=pyqt&logoColor=white)
![scipy](https://img.shields.io/badge/scipy-1.5%2B-brightgreen?logo=python&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.1%2B-orange?logo=python&logoColor=white)
![astropy](https://img.shields.io/badge/astropy-4.0%2B-red?logo=python&logoColor=white)
![lmfit](https://img.shields.io/badge/lmfit-0.9%2B-purple?logo=python&logoColor=white)
![emcee](https://img.shields.io/badge/emcee-3.0%2B-yellowgreen?logo=python&logoColor=white)

Interactive Python tool for 1D spectral analysis with file format detection. QASAP provides both quick-look functionality and some more advanced analysis features including multi-component line fitting, continuum modeling, and Bayesian MCMC fitting.

## Features

- **File Format Auto-Detection**: Automatically detects 7+ ASCII and FITS spectrum formats with confidence scoring
- **Interactive Plotting**: Real-time spectrum visualization with zoom, pan, and smooth controls
- **Multi-Component Fitting**: Gaussian and Voigt profile fitting (single and multi-component) with Listfit mode for simultaneous fitting
- **Line Analysis**: Equivalent width, redshift estimation, velocity mode analysis
- **Other Fitting**: Bayesian MCMC with posterior distributions (work in progress still - works well for single profiles)
- **Item Tracker**: Centralized management of all plotted components (Gaussians, Voigts, polynomials, continuum) with multi-select and deletion capabilities
- **Utilities**: LSF convolution, line identification, filter overlays

## Installation

### Requirements
- Python 3.7+
- numpy, scipy, matplotlib, astropy, pandas, lmfit, PyQt5, emcee, corner

### Download

```bash
git clone https://github.com/eriksolhaug/qasap.git
```

To **download a specific tagged version** (e.g., v0.9):

```bash
# Clone only the tag (shallow clone, fastest)
git clone --depth 1 --branch v0.9 https://github.com/eriksolhaug/qasap.git

# Or clone the whole repo and checkout the tag
git clone https://github.com/eriksolhaug/qasap.git
cd qasap
git checkout v0.9
```

### Conda Environment Setup

Then install the required packages (regardless of version):

```bash
# Create a new conda environment
conda create -n qasap python=3.8

# Activate the environment
conda activate qasap

# Enter qasap directory
cd qasap # This is the repo you cloned from github

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
python qasap.py spectrum.fits

# Preview detected formats
python qasap.py spectrum.fits --detect

# Force specific format if needed
python qasap.py spectrum.fits --fmt fits:image1d
```

### Supported Formats

| Format | Description |
|--------|-------------|
| `ascii:2col` | 2-column ASCII (wavelength, flux/transmission with optional # comments) |
| `ascii:3col` | 3-column ASCII (wavelength, flux, error with optional # comments) |
| `ascii:flex` | Flexible ASCII with custom column mapping |
| `fits:image1d` | 1D FITS image with wavelength in header |
| `fits:table:vector` | FITS table with wave/flux as vector arrays in rows |
| `fits:table:columns` | FITS table with per-pixel columns |
| `fits:ext:spectrum` | FITS SPECTRUM extension |

### Command-Line Options

```
--fmt              Force format (auto-detects if omitted)
--detect           Show detected formats and exit
--redshift         Initial redshift for displaying line lists (default: 0.0)
--zoom_factor      Y-axis zoom for `y` key (default: 0.1)
--lsf              LSF width in km/s or path to file (default: "10", in development)
--gui              Launch GUI for interactive input
```

### Making `qasap` Executable

To run QASAP from anywhere as a simple `qasap` command:

1. **Make the script executable:** (run from inside the `qasap/` directory where the `qasap.py` script is)
   ```bash
   chmod +x qasap.py
   ```

2. **Ensure the shebang line is present** (should already be at the top of qasap.py):
   ```python
   #!/usr/bin/env python
   ```

3. **Create a symlink or add to PATH:**

   **Option A: Create a symlink in a directory on your PATH**
   ```bash
   # Find your qasap installation path
   QASAP_PATH=$(pwd)/qasap.py
   
   # Link to a bin directory in your PATH (example: /usr/local/bin)
   sudo ln -s $QASAP_PATH /usr/local/bin/qasap
   ```

   **Option B: Add qasap directory to PATH** (recommended for development)
   ```bash
   # Add this line to your shell profile (~/.bash_profile, ~/.zshrc, etc.)
   export PATH="/path/to/qasap/qasap:$PATH"
   
   # Then create a shell script wrapper at qasap/qasap:
   #!/bin/bash
   exec python /path/to/qasap/qasap/qasap.py "$@"
   ```

4. **Test:**
   ```bash
   qasap ~/path/to/spectrum.fits
   ```

## Package Structure

```
.
├── qasap.py                 # Main entry point (root level)
├── setup.py                 # Package setup and installation
├── requirements.txt         # Python dependencies
├── LICENSE
├── README.md
├── __init__.py              # Package initialization
├── qasap/                   # Main package directory
│   ├── __init__.py
│   ├── spectrum_io.py           # File I/O with auto-detection
│   ├── spectrum_analysis.py     # Fitting and analysis functions
│   ├── spectrum_plotter.py      # Main visualization widget
│   ├── spectrum_plotter_app.py  # Application wrapper
│   ├── ui_components.py         # UI component exports
│   ├── linelist_window.py       # Line identification window
│   ├── linelist_selector_window.py # Line list management UI
│   ├── linelist.py              # Line list data structures
│   ├── listfit_window.py        # Multi-component fitting dialog
│   ├── item_tracker.py          # Component tracking and management
│   └── format_picker_dialog.py  # Format selection dialog
└── resources/
    ├── linelist/                # Line list catalogs
    │   ├── emlines.txt
    │   ├── emlines_osc.txt
    │   ├── sdss.txt
    │   ├── sdss_emission.txt
    │   ├── sdss_absorption.txt
    │   └── sdss_sky.txt
    └── bands/                   # Instrument band definitions
        └── instrument_bands.txt
```

## Data Files

**Line Lists** (in `resources/linelist/`):
- `emlines.txt`: Emission line catalog
- `emlines_osc.txt`: Lines with oscillator strengths
- `sdss_emission.txt`: SDSS emission line catalog
- `sdss_absorption.txt`: SDSS stellar absorption features
- `sdss_sky.txt`: Telluric sky emission lines

**Instrument Bands** (in `resources/bands/`):
- `instrument_bands.txt`: Filter and instrument bandpass definitions


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

**Navigation & View Controls:**
- `[` / `]` - Pan left/right through spectrum
- `\` (backslash) - Reset spectrum view to starting bounds
- `x` - Center on wavelength position under cursor
- `u` / `i` - Set lower/upper x-bounds (wavelength bounds)
- `T` / `t` - Zoom in/out horizontally (narrow/widen x-range)
- `Y` / `y` - Zoom in/out vertically (narrow/widen y-range)
- `O` / `P` - Set lower/upper y-bounds (flux bounds)
- `l` - Toggle log y-axis
- `L` - Toggle log x-axis
- `f` - Enter fullscreen mode
- `1`-`9` - Apply Gaussian smoothing with different kernel sizes
- `0` - Remove smoothing (restore original spectrum)
- `~` (tilde) - Toggle between step plot and line plot
- `` ` `` (backtick) - Save screenshot of plot

**Fitting Modes:**
- `m` - Enter continuum fitting mode (define regions with `SPACE`, then hit `ENTER` to perform continuum fit -- note that you can change the polynomial order in the Control Panel)
- `M` - Remove a continuum region
- `d` - Enter Single Mode Gaussian fit (SPACE to select bounds)
- `|` (pipe) - Enter Multi-Gaussian fit mode (fit multiple Gaussians simultaneously, define bounds with `SPACE` and hit `ENTER` to perform fit)
- `n` - Single mode Voigt profile fitting
- `H` - Enter Listfit window (a continuum does not need to have been fitted, escape with `ESC`)
- `r` - Toggle residual panel

Note for `d`, `|`, and `n`:  A continuum must have already been fitted. Define bounds with `SPACE`.

**Line List:**
- `e` - Open Line List window

Note: You will need to select the line list and Toggle Display to view lines. Set the redshift applied to the line list in the Control Panel.

**Line Profiles:**
- `w` - Remove fitted profile under cursor
- `,` (comma) - Add a line tag to fitted profile under cursor
- `<` (less than) - Remove tag from fitted profile under cursor
- `a` - Save Gaussian fit info to file
- `A` - Save Voigt fit info to file
- `S` - Save continuum fit info to file
- `;` (semicolon) - Show/toggle total line for Single Mode fitted lines
- `v` - Calculate equivalent width of fitted line. In progress. Use with caution.

**Redshift & Velocity:**
- `z` - Enter redshift mode (select already fitted line under cursor with `SPACE`, escape with `ESC`)
- `b` - Activate velocity mode (set rest-frame wavelength). In progress.

**Instrument Filters & Bands:**
- `!` through `)` (Shift+1-0) - Toggle instrument bandpass overlays (press Shift+number)
- `-` / `_` / `=` / `+` - Show filter bandpasses (requires downloaded filter files)

**Item Management:**
- `j` - Toggle Item Tracker window visibility

**Help:**
- `?` - Show keyboard shortcuts help window

**File Storage:**
All saved screenshots, redshifts, and profile info are stored in the directory where QASAP was launched from.

### Fitting Engines by Mode

QASAP employs different fitting algorithms optimized for each analysis task:

| Mode | Key | Fitting Engine | Method | Use Case |
|------|-----|----------------|--------|----------|
| Single Gaussian (`d`) | `d` | **scipy.optimize.curve_fit** | Least-squares optimization | Quick individual profile fitting |
| Single Voigt (`n`) | `n` | **scipy.optimize.curve_fit** | Least-squares optimization | Quick individual profile fitting with natural broadening |
| Multi-Gaussian (`\|`) | `\|` | **scipy.optimize.curve_fit** | Least-squares optimization (sequential) | Multiple Gaussian fitting in same region |
| Listfit (`H`) | `H` | **lmfit (leastsq)** | Composite Model with simultaneous parameter fitting | Multi-component simultaneous fitting with full covariance estimation |
| Bayesian MCMC (`:`) | `:` | **emcee** | Posterior sampling (work in progress) | Posterior probability distributions for parameters |

**Key Differences:**

- **Single Mode (`d`/`n`) and Multi-Gaussian Mode (`|`)**: Uses `scipy.optimize.curve_fit` for individual component fitting. Fast but limited uncertainty estimation. Suitable for isolated, well-separated lines.

- **Listfit Mode (`H`)**: Uses `lmfit`'s composite Model system with `leastsq` minimization. Allows simultaneous fitting of multiple Gaussians, Voigts, and polynomial backgrounds with full parameter correlation tracking. Provides robust error estimation through covariance matrix analysis. **Recommended for crowded spectral regions or blended profiles.**

- **MCMC Mode (`:`)**: Uses `emcee` for Bayesian posterior sampling (currently in development). Provides posterior distributions and uncertainty quantification beyond frequentist approach. (Currently only works for a single line first estimated using Single Mode)


### 2. Control Panel
Displays real-time analysis information and fitting parameters:

**Sections:**
- **Quick Stats**: Line center, FWHM, equivalent width, signal-to-noise
- **Redshift**: Set redshift for displayed emission lines (toggled with key `e`)
- **Polynomial Order**: Set the order for the polynomial for Continuum mode (with key `m`)

### 3. LineList Window
Interactive line identification and management interface for viewing and selecting spectral lines:

**Access:**
- Press `e` on the spectrum to open the LineList Window
- Provides access to multiple line list catalogs

**Line Lists:**
- **emlines.txt**: Common emission lines (Hα, Hβ, [OIII], [OII], Ly-α, etc.)
- **emlines_osc.txt**: Emission lines with oscillator strength data
- **instrument_bands.txt**: Instrument and filter bandpass definitions
- **sdss_emission.txt**: SDSS emission line catalog
- **sdss_absorption.txt**: SDSS stellar absorption features
- **sdss_sky.txt**: Telluric sky emission lines

**Interface:**
The LineList Window uses a dual-panel design:

**Left Panl - Line List Selection:**
- Displays all available line list catalogs with line counts
- Format: `{ListName} ({count} lines)`
- Scrollable
- Click to select a line list and view its contents

**Right Panel - Lines Display:**
- Dynamically populates when you select a line list from the left panel
- Shows all lines in the format: `{LineName}: {Wavelength} Å`
- Scrollable
- Double-click a line to select it for redshift estimation

**Workflow:**
1. Press `e` to open LineList Window
2. Select a line list from the left panel (e.g., "sdss_emission.txt")
3. Right panel populates with all lines from that list
4. Double-click a line (e.g., "H alpha: 6564.61 Å")
5. Window closes and redshift is estimated from the selected line (see terminal for output)

**Redshift Integration:**
Lines in the spectrum automatically update when redshift is changed via:
- Manual entry in Control Panel
- Arrow buttons in the LineList Window
- Automatic redshift estimation from fitted components (`z` key)

### 4. Listfit Window (Multi-Component Fitting)
Dialog for simultaneous fitting of multiple spectral components:

**Workflow:**
1. Press `H` on spectrum to activate listfit mode
2. Define fitting bounds by clicking two wavelengths (spacebar confirms each)
3. Listfit Window opens with component selection
4. Choose component types and quantities:
   - **Gaussian**: Simple Gaussian profiles
   - **Voigt**: Gaussian + Lorentzian (more realistic for emission lines)
   - **Polynomial**: Continuum background (specify order)
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

- **v0.9** (current): Fixed Listfit parameter error extraction, PolynomialModel integration for proper coefficient variation, Item Tracker synchronization with internal storage removal
- **v0.8**: Listfit mode for simultaneous multi-component fitting, ItemTracker for centralized feature management with multi-select and deletion, auto-fit registration, redshift mode improvements
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
