# QASAP: Quick Analysis of Spectra and Profiles

A comprehensive interactive Python tool for spectral analysis, fitting, and visualization. QASAP provides an intuitive GUI for analyzing 1D spectra with support for Gaussian and Voigt profile fitting, equivalent width calculations, and Bayesian MCMC fitting.

## Features

- **Multiple File Format Support**: Read FITS and ASCII spectra with flexible file flags for different formats
- **Interactive Plotting**: Real-time spectrum visualization with zoom, pan, and smooth controls
- **Profile Fitting**: 
  - Single and multi-component Gaussian fits
  - Single and multi-component Voigt profile fits
  - Linear continuum fitting with sigma-clipping
- **Line Analysis**:
  - Equivalent width (EW) calculation
  - Column density estimation
  - Redshift estimation from line centers
- **Advanced Fitting**:
  - Bayesian MCMC fitting with corner plots
  - Confidence interval estimation
  - Fit parameter export to CSV
- **Velocity Mode**: Analyze spectra in velocity space relative to a rest-frame wavelength
- **Utilities**:
  - Line spread function (LSF) convolution
  - Instrument band visualization
  - Filter throughput overlays

## Installation

### Requirements
- Python 3.7+
- pip

### Quick Start

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run QASAP with a FITS file:

```bash
python qasap.py <path_to_fits_file> --file_flag <flag>
```

### File Flags

Choose the appropriate file flag based on your FITS file structure:

- `--file_flag 0`: Default FITS format (HDU 0, 1, 2 for wav, spec, err)
- `--file_flag 1`: ASCII text file with 3 columns (wavelength, flux, error)
- `--file_flag 2`: 1D FITS spectrum with wavelength in header (CRPIX1, CRVAL1, CDELT1)
- `--file_flag 3`: ASCII file with 2 columns (wavelength, flux)
- `--file_flag 4`: ASCII file with wavelength, ?, flux, error
- `--file_flag 5`: FITS with HDU[1] containing 'wave' and 'flux' columns
- `--file_flag 6`: FITS with HDU[1] containing nested array data
- `--file_flag 7`: FITS with 'SPECTRUM' extension containing 'wave', 'flux', 'ivar' columns
- `--file_flag 8`: FITS with 'SPECTRUM' extension containing 'WAVE', 'FLUX', 'ERR' columns
- `--file_flag 9`: FITS with HDU[1] containing 'WAVE', 'FLUX', 'ERR' columns
- `--file_flag 10`: 2-column ASCII/SED format with comment lines

### Example

```bash
# Analyze a 1D FITS spectrum with wavelength in header
python qasap.py ~/data/spectrum.fits --file_flag 2 --redshift 0.5

# Analyze an ASCII spectrum
python qasap.py ~/data/spectrum.txt --file_flag 1 --redshift 0.3
```

### Command-Line Options

```
--redshift         Initial redshift value (default: 0.0)
--zoom_factor      Zoom factor for y-axis expansion (default: 0.1)
--file_flag        File format flag (default: 0)
--lsf              Line spread function (FWHM in km/s or path to file, default: "10")
--gui              Launch interactive GUI for file selection
--alfosc           Load ALFOSC 2D spectrum and wavelength solution
--alfosc_bin       ALFOSC 1D binning in Angstroms
--alfosc_left      Left x-bound for ALFOSC extraction (inclusive)
--alfosc_right     Right x-bound for ALFOSC extraction (exclusive)
--alfosc_output    Path to save extracted 1D spectrum
```

## Quick Start Guide

### First Time Use

1. **Determine your file format** - See [File Flags](#file-flags) section above for options
2. **Open your spectrum** - Provide the file path and correct file flag
3. **Interactive analysis** - Use keyboard controls to analyze

### Common Tasks

#### Fit a Gaussian Profile
1. Press `d` to enter Gaussian fit mode
2. Press `space` at the left edge of the feature
3. Press `space` again at the right edge
4. Select a line from the line list if desired

#### Calculate Equivalent Width
1. Fit a Gaussian or Voigt profile (press `d` or `n`)
2. Press `v` and click on the fitted profile
3. Equivalent width is printed to console and visualized

#### Switch to Velocity Space
1. Press `b` to enter velocity mode
2. Select a rest-frame wavelength from the emission line list
3. X-axis now shows velocity relative to that wavelength

#### Save Your Work
1. Save fits as CSV: Press `a` (Gaussian) or `A` (Voigt) or `S` (continuum)
2. Export plot as PDF: Press `` ` `` (backtick)
3. Load previous fits: Press `K` and select a CSV file

### Tips & Tricks

- Use smoothing (`1`-`9` keys) to reduce noise before fitting
- Set continuum regions (`m` key) before fitting line profiles
- Use Bayesian fitting (`:` key) for robust error estimation
- Multi-component fitting (`D` and `N` keys) constrains component separations
- Zoom in with `t`/`y` for better feature resolution

## Interactive Controls

### Navigation
- `t` / `T`: Zoom out / in on x-axis
- `y` / `Y`: Zoom out / in on y-axis
- `[` / `]`: Pan left / right
- `x`: Center x-axis on cursor
- `u` / `i`: Set x-axis lower / upper bound
- `p` / `o`: Set y-axis lower / upper bound
- `\\`: Reset to original bounds

### Plotting
- `~`: Toggle between step and line plot
- `1`-`9`: Apply Gaussian smoothing (kernel width 1-9)
- `0`: Reset to original unsmoothed spectrum
- `r`: Toggle residual panel

### Spectral Features
- `e`: Toggle emission line list display
- `b`: Toggle velocity mode (requires selecting rest-frame wavelength)
- `z`: Redshift estimation mode

### Fitting
- `d`: Enter Gaussian fit mode (press space to set bounds)
- `D`: Enter multi-Gaussian mode (press space for multiple bounds, enter to fit)
- `n`: Enter Voigt fit mode
- `N`: Enter multi-Voigt mode
- `:`: Enter Bayesian MCMC fit mode
- `m`: Enter continuum fitting mode (press space to define regions, enter to fit)
- `M`: Remove continuum fit

### Analysis
- `v`: Calculate equivalent width (click on fitted profile)
- `,`: Assign line ID to selected fit
- `;`: Toggle total fitted profile display
- `w`: Delete selected Gaussian or Voigt fit

### File Operations
- `a`: Save Gaussian fits to CSV
- `A`: Save Voigt fits to CSV
- `S`: Save continuum fits to CSV
- `K`: Load fits from CSV file
- `` ` ``: Save plot as PDF
- `Q`: Quit application

## Data Files

This package includes essential data files:
- `emlines.txt`: Rest-frame emission line catalog with wavelengths and IDs
- `emlines_osc.txt`: Emission lines with oscillator strengths
- `instrument_bands.txt`: Instrument filter band definitions
- `qasapv0.5_v1.key`: Jupyter kernel definition (for notebook support)

## Limitations & Notes

- Error estimates are assumed or derived from the spectrum when not provided
- Column density calculations use fixed atomic masses (configurable in code)
- MCMC fitting requires adequate spectrum resolution for reliable results
- Some features require PyQt5 for full functionality

## Dependencies

See `requirements.txt` for detailed version information.

## Author

Erik Solhaug

## License

Please see LICENSE file or contact author for licensing information.

## Citation

If you use QASAP in your research, please cite this work appropriately.

## Support & Troubleshooting

- For matplotlib display issues: Ensure you have a backend configured (Qt5Agg is recommended)
- For FITS reading errors: Verify file format matches selected `--file_flag`
- For fitting convergence issues: Provide better initial guesses or check data quality

## Version

v0.5 - Current stable release with full Voigt/Gaussian multi-component fitting, Bayesian MCMC, and velocity mode support
