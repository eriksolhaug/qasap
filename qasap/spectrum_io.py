"""
Spectrum I/O module - handles reading various spectrum file formats
"""

import numpy as np
from astropy.io import fits


class SpectrumIO:
    """Handle reading spectrum data from various file formats"""
    
    @staticmethod
    def read_spectrum(fits_file, file_flag=0):
        """
        Read spectrum from file based on file_flag
        
        Parameters
        ----------
        fits_file : str
            Path to spectrum file
        file_flag : int
            File format identifier (0-10)
            
        Returns
        -------
        wav : np.ndarray
            Wavelength array
        spec : np.ndarray
            Flux array
        err : np.ndarray
            Error array
        """
        
        if file_flag == 1:
            # 3-column ASCII: wavelength, flux, error (tab-delimited)
            data = np.genfromtxt(fits_file, comments='#', delimiter='\t')
            wav, spec, err = data[:, 0], data[:, 1], data[:, 2]
            
        elif file_flag == 2:
            # FITS: flux in hdul[0].data, wavelength from header (CRPIX1, CRVAL1, CDELT1)
            with fits.open(fits_file) as hdul:
                spec = hdul[0].data.flatten()
                header = hdul[0].header
                crpix1 = header.get('CRPIX1', 1)
                crval1 = header.get('CRVAL1', 0)
                cdelt1 = header.get('CDELT1', 1)
                wav = crval1 + (np.arange(len(spec)) - (crpix1 - 1)) * cdelt1
                err = np.ones_like(spec) * np.nanstd(spec) * 0.1
                
        elif file_flag == 3:
            # 2-column ASCII: wavelength, flux (error = 10% of flux)
            data = np.genfromtxt(fits_file, comments='#')
            wav, spec = data[:, 0], data[:, 1]
            err = np.ones_like(spec) * np.nanstd(spec) * 0.1
            
        elif file_flag == 4:
            # 4-column ASCII: wavelength, ignored, flux, error
            data = np.genfromtxt(fits_file, comments='#')
            wav, spec, err = data[:, 0], data[:, 2], data[:, 3]
            
        elif file_flag == 5:
            # FITS: hdul[1] with fields 'wave' and 'flux'
            with fits.open(fits_file) as hdul:
                data = hdul[1].data
                wav = data['wave']
                spec = data['flux']
                err = np.ones_like(spec) * np.nanstd(spec) * 0.1
                
        elif file_flag == 6:
            # FITS: hdul[1].data[0] = [wavelength array, flux array]
            with fits.open(fits_file) as hdul:
                wav = hdul[1].data[0]
                spec = hdul[1].data[1]
                err = np.ones_like(spec) * np.nanstd(spec) * 0.1
                
        elif file_flag == 7:
            # FITS: SPECTRUM extension with 'wave', 'flux', 'ivar', 'mask'
            with fits.open(fits_file) as hdul:
                data = hdul['SPECTRUM'].data
                wav = data['wave']
                spec = data['flux']
                ivar = data['ivar']
                err = np.sqrt(1.0 / np.where(ivar > 0, ivar, 1e-10))
                
        elif file_flag == 8:
            # FITS: Three HDUs - wavelength, flux, error
            with fits.open(fits_file) as hdul:
                wav = hdul[0].data.flatten()
                spec = hdul[1].data.flatten()
                err = hdul[2].data.flatten()
                
        elif file_flag == 9:
            # FITS: Custom format 2
            with fits.open(fits_file) as hdul:
                spec = hdul[0].data.flatten()
                header = hdul[0].header
                crpix1 = header.get('CRPIX1', 1)
                crval1 = header.get('CRVAL1', 0)
                cdelt1 = header.get('CDELT1', 1)
                wav = crval1 + (np.arange(len(spec)) - (crpix1 - 1)) * cdelt1
                err = np.ones_like(spec) * np.nanstd(spec) * 0.1
                
        elif file_flag == 10:
            # FITS: Custom format 3
            with fits.open(fits_file) as hdul:
                spec = hdul[0].data.flatten()
                header = hdul[0].header
                crpix1 = header.get('CRPIX1', 1)
                crval1 = header.get('CRVAL1', 0)
                cdelt1 = header.get('CDELT1', 1)
                wav = crval1 + (np.arange(len(spec)) - (crpix1 - 1)) * cdelt1
                err = np.ones_like(spec) * np.nanstd(spec) * 0.1
                
        else:
            # Default: Three HDUs
            with fits.open(fits_file) as hdul:
                wav = hdul[0].data.flatten()
                spec = hdul[1].data.flatten()
                err = hdul[2].data.flatten()
        
        # Clean up NaNs
        mask = ~np.isnan(spec) & ~np.isnan(wav)
        wav = wav[mask]
        spec = spec[mask]
        err = err[mask]
        
        return wav, spec, err
    
    @staticmethod
    def read_lines(filename='emlines.txt'):
        """Read emission line catalog"""
        try:
            data = np.genfromtxt(filename, dtype=str, delimiter=',')
            return data[:, 1].astype(float), data[:, 0]
        except Exception as e:
            print(f"Error reading line catalog: {e}")
            return np.array([]), np.array([])
    
    @staticmethod
    def read_oscillator_strengths(filename='emlines_osc.txt'):
        """Read emission lines with oscillator strengths"""
        try:
            data = np.genfromtxt(filename, dtype=str, delimiter=',')
            return data[:, 1].astype(float), data[:, 0], data[:, 2].astype(float)
        except Exception as e:
            print(f"Error reading oscillator strengths: {e}")
            return np.array([]), np.array([]), np.array([])
    
    @staticmethod
    def read_instrument_bands(filename='instrument_bands.txt'):
        """Read instrument filter definitions"""
        try:
            data = np.genfromtxt(filename, dtype=str, delimiter=',')
            band_ranges = list(zip(data[:, 1].astype(float), data[:, 2].astype(float)))
            return band_ranges, data[:, 0]
        except Exception as e:
            print(f"Error reading instrument bands: {e}")
            return [], []
