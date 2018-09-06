"""
Functions for handling the parameter derivation within ATHOS.

This module provides several functions that constitute the ATHOS workflow, 
including the steps of reading input spectra, calculating flux ratios 
(henceforth FRs), and computing the stellar parameters.

Constants
---------
- 'v_Teff_sys' -- Systematic temperature error found for the adopted method.
- 'v_met_sys' -- Systematic iron abundance error found for the adopted method.
- 'v_logg_sys' -- Systematic surface gravity error found for the adopted method.
- 'Teff_coeff' -- Coefficients for the linear temperature relations in [1].
- 'met_coeff' -- Coefficients for the [Fe/H] relations (Eq. 10) in [1].
- 'logg_coeff' -- Coefficients for the logg relations (Eq. 13) in [1].
- 'stacked_ranges' -- Stacked array with the wavelength information for all FRs.
- 'Teff_cov' -- Covariances for 'Teff_coeff'.
- 'met_cov' -- Covariances for 'met_coeff'.
- 'logg_cov' -- Covariances for 'logg_coeff'.
- 'R_correction_coefficients' -- Coefficients for the resolution corrections 
                                 (Eq. 14 in [1]).
- 'wl_tell' -- Expected positions of tellurics in the topocentric rest frame.

References
----------
.. [1] Astronomy & Astrophysics, "ATHOS: On-the-fly stellar parameter 
determination of FGK stars based on flux ratios from optical spectra", 
M. Hanke, C. J. Hansen, A. Koch, and E. K. Grebel (ADD CORRECT REFERENCE 
BY THE TIME OF PUBLICATION)

"""
import numpy as np

from astropy.io import fits
from pandas import read_csv

v_Teff_sys = 97 ** 2
v_met_sys  = 0.16 ** 2
v_logg_sys = 0.26 ** 2
    
# Load coefficient arrays
Teff_coeff = np.load('./coefficients/Teff_coeffs.npy')
met_coeff  = np.load('./coefficients/met_coeffs.npy')
logg_coeff = np.load('./coefficients/logg_coeffs.npy')

stacked_ranges = np.row_stack((Teff_coeff[:,:3], met_coeff[:,:3], logg_coeff[:,:3]))

Teff_cov = np.load('./coefficients/Teff_covs.npy')
met_cov  = np.load('./coefficients/met_covs.npy')
logg_cov = np.load('./coefficients/logg_covs.npy')

R_correction_coefficients = np.load('./coefficients/R_correction_coefficients.npy')

# Load telluric line positions
wl_tell = np.load('./coefficients/expected_tellurics.npy')
# Only those tellurics in the vicinity (+/- 20 AA) of our FR ranges are considered to speed up telluric rejection
wl_tell = wl_tell[np.any(np.abs(wl_tell[None,:] - stacked_ranges[:,:2].flatten()[:,None]) < 20, axis=0)]

def parallelization_wrapper(file_list, lambda_weight_polys, v_topo, dtype, wunit, tell_rejection, yerr, R, wave_keywd = None, flux_keywd = None):
    """ Run ''analyze_spectrum'' in allocated thread.
    
    Loops through input lists by calling the function ''analyze_spectrum'' for 
    each entry in 'file_list'.
    
    Parameters
    -----------
    file_list : array_like, shape (N, )
        1-D array of either strings (ATHOS will try to load the spectra from 
        disk), or (M, 2) array_like entries containing wavelength and flux 
        columns (in this order).
    lambda_weight_polys : array_like, shape (N, M)
        N 1-D arrays of polynomial coefficients for wavelength-dependent FR 
        weights. Order is highest to lowest degree (see "numpy.polyval").
    v_topo : array_like, shape (N, )
        1-D array of relative velocities of the topocenter. The sign convention 
        is such that if the input spectrum is blue-shifted w.r.t. the 
        topocentric rest frame ''sign(v_topo) = +1'', and ''sign(v_topo) = -1''
        otherwise. 'v_topo' is ignored if ''tell_rejection == False''.
    dtype : {'lin', 'log10', 'ln'}
        'lin':
          Dispersion type is linear.
        'log10':
          Dispersion type is log10(lambda [wunit]).
        'ln':
          Dispersion type is ln(lambda [wunit]).
    wunit : {'nm', 'aa'}
        'nm':
          Dispersion unit is nanometers.
        'aa':
          Dispersion unit is Angstroms.
    tell_rejection : {True, False}
        True:
          Perform masking of tellurics based on topocentric velicities provided
          in 'thread_list'.
        False:
          Skip telluric rejection.
    yerr : 
        Currently not implemented in this wrapper.
    R : float
        Resolution of the input spectra.
        
    Returns
    -------
    results : array_like, shape (N, 9)
        Each row contains Teff, Teff_err_stat, Teff_err_sys, [Fe/H], 
        [Fe/H]_err_stat, [Fe/H]_err_sys, logg, logg_err_stat, and 
        logg_err_sys.
        
    See Also
    --------
    analyze_spectrum : The routine called by this wrapper.
        
    """
    results = []
    for i in range(len(file_list)):
        results.append(list(analyze_spectrum(file_list[i], dtype, wunit, p_weights = lambda_weight_polys[i], tell_rejection = tell_rejection, v_topo = v_topo[i], y_err = yerr, R = R, wave_keywd = wave_keywd, flux_keywd = flux_keywd)))
    
    return results
    

def analyze_spectrum(spec, dtype, wunit, p_weights = None, tell_rejection = False, v_topo = None,\
                     y_err = None, lit_met = None, lit_Teff = None, R = 45000, wave_keywd = None, flux_keywd = None):
    """ Perform the spectrum analysis.
    
    Analyses the input spectrum 'spec' by computing FRs and employing the 
    analytical parameter relations (for details, see paper 
    Hanke et al. 2018[1]).
    
    Parameters
    -----------
    spec : str or array_like, shape (N, 2)
        Either string (ATHOS will try to load the spectrum 'spec' from 
        disk), or (N, 2)-shaped array containing wavelength and flux 
        columns (in this order).
    dtype : {'lin', 'log10', 'ln'}
        'lin':
          Dispersion type is linear.
        'log10':
          Dispersion type is log10(lambda [wunit]).
        'ln':
          Dispersion type is ln(lambda [wunit]).
    wunit : {'nm', 'aa'}
        'nm':
          Dispersion unit is nanometers.
        'aa':
          Dispersion unit is Angstroms.
    p_weights : array_like, shape (M, ), optional
        Polynomial coefficients for wavelength-dependent FR weights. Order is 
        highest (M-1) to lowest degree (see "numpy.polyval").
    tell_rejection : {True, False}, optional
        True:
          Perform masking of tellurics based on the topocentric velocity 
          'v_topo'.
        False:
          Default. Skip telluric rejection.
    v_topo : float, optional
        Relative velocity of the topocenter. The sign convention is such that 
        if the input spectrum is blue-shifted w.r.t. the topocentric rest 
        frame ''sign(v_topo) = +1'', and ''sign(v_topo) = -1'' otherwise. 
        'v_topo' is ignored if ''tell_rejection == False'' (Default).
    yerr : array_like, shape (N, ), optional
        Error spectrum.
    lit_met : float, optional
        Input metallicity. The ATHOS-internal metallicity is overridden.
    lit_Teff : float, optional
        Input temperature. The ATHOS-internal effective temperature is 
        overridden.
    R : float, optional
        Resolution of the input spectrum. By default, ''R = 45000'' is set, 
        i.e. the resolution of the original training sample.
        
    Returns
    -------
    Teff : int 
        Median effective temperature from the available FRs in the Balmer 
        line regions.
    Teff_err_stat : int 
        Median absolute deviation from 'Teff'.
    Teff_err_sys : int 
        Systematic error component for 'Teff'.
    met : float
        Median [Fe/H] from the available metallicitiy-sensitive FRs.
    met_err_stat : float
        Median absolute deviation from 'met'.
    met_err_sys : float
        Systematic error component for 'met'.
    logg : float
        Median surface gravity from the available logg-sensitive FRs.
    logg_err_stat : float
        Median absolute deviation from 'logg'.
    logg_err_sys : float
        Systematic error component for 'logg'.
        
    See Also
    --------
    read_spectrum : Load spectrum file from disk.
        
    References
    ----------
    .. [1] Astronomy & Astrophysics, "ATHOS: On-the-fly stellar parameter 
    determination of FGK stars based on flux ratios from optical spectra", 
    M. Hanke, C. J. Hansen, A. Koch, and E. K. Grebel (ADD CORRECT REFERENCE 
    BY THE TIME OF PUBLICATION)
        
    """
    if isinstance(spec, str):
        x, y = read_spectrum(spec, dtype, wunit, wave_keywd, flux_keywd)
    else:
        x = spec[:,0]
        y = spec[:,1]
        
    if tell_rejection:
        y = reject_telluric_regions(x, y, wl_tell/np.mean(R), v_topo)
    else:
        pass
    
    frs, v_frs = compute_fr(x, y, stacked_ranges[:,0], stacked_ranges[:,1], stacked_ranges[:,2], y_err = y_err)
    if hasattr(R, '__len__'):
        frs -= higher_order_correction(R_correction_coefficients.transpose(), R, frs)
    elif R != 45000:
        frs -= higher_order_correction(R_correction_coefficients.transpose(), R, frs)
    else:
        pass
    
    Teffs, v_Teffs = compute_Teff(Teff_coeff[:,4:], Teff_cov, frs[:Teff_coeff.shape[0]], v_frs[:Teff_coeff.shape[0]])
    Teffs[Teffs < 0] = np.nan
    if hasattr(p_weights, '__len__'):
        mean_Teff, v_mean_Teff = weighted_median(Teffs, np.polyval(p_weights, stacked_ranges[:,0][:Teff_coeff.shape[0]]))
    else:
        mean_Teff = np.nanmedian(Teffs)
        v_mean_Teff = (np.nanmedian(np.abs(Teffs - mean_Teff))) ** 2
    
    if lit_Teff == None:
        Teff_inp = mean_Teff 
    else:
        Teff_inp = lit_Teff
        
    mets, v_mets, v_mets_sys = compute_met(met_coeff[:,3:], met_cov, frs[Teff_coeff.shape[0]:Teff_coeff.shape[0] + met_coeff.shape[0]], \
                               v_frs[Teff_coeff.shape[0]:Teff_coeff.shape[0] + met_coeff.shape[0]], Teff_inp, v_mean_Teff, \
                               v_Teff_sys, v_met_sys)
    if hasattr(p_weights, '__len__'):
        mean_met, v_mean_met = weighted_median(mets, np.polyval(p_weights, stacked_ranges[:,0][Teff_coeff.shape[0]:Teff_coeff.shape[0] + met_coeff.shape[0]]))
    else:
        mean_met        = np.nanmedian(mets)
        v_mean_met      = np.nanmedian((mets - mean_met) ** 2)
    
    
    mean_v_mets_sys = np.nanmedian(v_mets_sys)
    
    if lit_met == None:
        met_inp = mean_met
    else:
        met_inp = lit_met
    
    loggs, v_loggs, v_loggs_sys = compute_logg(logg_coeff[:,3:], logg_cov, frs[Teff_coeff.shape[0] + met_coeff.shape[0]:], v_frs[Teff_coeff.shape[0] + met_coeff.shape[0]:],\
                                  Teff_inp, v_mean_Teff, met_inp, v_mean_met, v_Teff_sys, mean_v_mets_sys, v_logg_sys)
    if hasattr(p_weights, '__len__'):
        mean_logg, v_mean_logg = weighted_median(loggs, np.polyval(p_weights, stacked_ranges[:,0][Teff_coeff.shape[0] + met_coeff.shape[0]:], v_frs[Teff_coeff.shape[0] + met_coeff.shape[0]:]))
    else:
        mean_logg        = np.nanmedian(loggs)
        v_mean_logg      = np.nanmedian((loggs - mean_logg) ** 2)
        
    mean_v_loggs_sys = np.nanmedian(v_loggs_sys)
    
    return round(mean_Teff), round(np.sqrt(v_mean_Teff)), round(np.sqrt(v_Teff_sys)), mean_met, np.sqrt(v_mean_met), np.sqrt(mean_v_mets_sys), mean_logg, np.sqrt(v_mean_logg), np.sqrt(mean_v_loggs_sys)

def read_spectrum(name, dtype, wunit, wave_keywd = None, flux_keywd = None):
    """ Read spectrum from disk.
    
    Based on the file extension, this function will decide how to read the 
    input spectrum. ".fits" extension files will be checked for a binary table 
    in the secondary HDU; else it will be treated as a standard 1D FITS 
    spectrum. ".npy" extensions will be treated as numpy arrays with wavelength 
    and flux information in the first and second column, respectively. All 
    other extensions will be treated as ASCII files and the function will 
    attempt to find the column delimiter automatically based on the first line 
    in the file.
    
    Parameters
    -----------
    name : str
        File path to the input spectrum.
    dtype : {'lin', 'log10', 'ln'}
        'lin':
          Dispersion type is linear.
        'log10':
          Dispersion type is log10(lambda [wunit]).
        'ln':
          Dispersion type is ln(lambda [wunit]).
    wunit : {'nm', 'aa'}
        'nm':
          Dispersion unit is nanometers.
        'aa':
          Dispersion unit is Angstroms.
    wave_keywd : str, optional
        In case the input file extension is ".fits" and the binary table in the 
        secondary HDU does not contain the keywords 'WAVE' or 'FLUX' 
        (alternatively 'FLUX_REDUCED'), this keyword defines the wavelength 
        column.
    flux_keywd : str, optional
        In case the input file extension is ".fits" and the binary table in the 
        secondary HDU does not contain the keywords 'WAVE' or 'FLUX' 
        (alternatively 'FLUX_REDUCED'), this keyword defines the flux column.
        
    Returns
    -------
    x : array_like, shape (N, )
        The dispersion coordinate.
    y : array_like, shape (N, )
        The flux coordinate.        
        
    """
    if name.split('.')[-1] == 'fits':
        with fits.open(name) as hdulist:
            if wave_keywd == None:
                try:
                    x = hdulist[1].data['WAVE'][0]
                    try:
                        y = hdulist[1].data['FLUX'][0]
                    except:
                        y = hdulist[1].data['FLUX_REDUCED'][0]
                except:
                    x = (np.arange(hdulist[0].header['NAXIS1'])) * hdulist[0].header['CDELT1'] + hdulist[0].header['CRVAL1']
                    y = hdulist[0].data
            else:
                x = hdulist[1].data[wave_keywd][0]
                y = hdulist[1].data[flux_keywd][0]
    else:
        if name.split('.')[-1] == 'npy':
            spectrum = np.load(name)
        else:
            with open(name, 'r') as f_in:
                lines = [line.rstrip('\n') for line in f_in]
            while len(lines[-1]) == 0:
                lines = lines[:-1]
            if ',' in lines[0]:
                spectrum = read_csv(name, header=None).values
            elif '\t' in lines[0]:
                spectrum = read_csv(name, sep='\t', header=None).values
            else:
                spectrum = read_csv(name, delim_whitespace=True, header=None).values
       
        x = spectrum[:,0]
        y = spectrum[:,1]

    # convert wavelengths to Angstrom
    if dtype == 'lin':
        pass
    elif dtype == 'log10':
        x = np.power(10, x)
    elif dtype == 'ln':
        x = np.exp(x)

    if wunit == 'aa':
        pass
    elif wunit == 'nm':
        x *= 10
        
    return x, y

def reject_telluric_regions(x, y, rej_width, v_topo):
    """ Set all fluxes that are potentially affected by tellurics to NaN.
    
    Parameters
    -----------
    x : array_like, shape (N, )
        1-D array containing the wavelength information.
    y : array_like, shape (N, )
        1-D array containing the flux information.
    rej_width : float
        Width in Angstroms red- and blueward of each internally stored telluric 
        wavelength to exclude from consideration for the parameter 
        determination.
    v_topo : float
        Relative velocity of the topocenter. The sign convention is such that 
        if the input spectrum is blue-shifted w.r.t. the topocentric rest 
        frame ''sign(v_topo) = +1'', and ''sign(v_topo) = -1'' otherwise.
        
    Returns
    -------
    y_rejected : array_like, shape (N, )
        Copy of 'y' with regions of telluric contamination set to NaN.

    """
    
    half_rej_width  = rej_width / 2
    wl_tell_shifted = wl_tell * (1 - v_topo / 299792.458)
    lower_bounds    = np.searchsorted(x, wl_tell_shifted - half_rej_width, side = 'left')
    upper_bounds    = np.searchsorted(x, wl_tell_shifted + half_rej_width, side = 'right')
    m               = np.zeros(len(x)).astype(bool)
    for l, u in np.column_stack((lower_bounds, upper_bounds)):
        m[l:u] = True
        
    y_rejected    = y.copy()
    y_rejected[m] = np.nan
        
    return y_rejected

def compute_fr(x, y, x_i, x_j, w, y_err = None):
    """ Compute all flux ratios (FRs).
    
    Based on the mean fluxes, <f_i> and <f_j>, in the wavelength ranges of 
    width 'w' around 'x_i' and 'x_j', compute the flux ratios <f_i>/<f_j>.
    
    Parameters
    -----------
    x : array_like, shape (N, )
        1-D array containing the wavelength information.
    y : array_like, shape (N, )
        1-D array containing the flux information.
    x_i : array_like, shape (M, )
        1-D array with the wavelength positions for the numerators of the FRs.
    x_j : array_like, shape (M, )
        1-D array with the wavelength positions for the denominators of the FRs.
    w : array_like, shape (M, )
        1-D array containing the full widths of the wavelength ranges with 
        central positions 'x_i' and 'x_j' to compute the mean fluxes from.
    y_err : array_like, shape (N, ), optional
        Error spectrum.
        
    Returns
    -------
    frs : array_like, shape (M, )
        The computed FRs with unavailable ratios (e.g. due to telluric 
        contamination) set to NaN.
    v_frs : array_like, shape (M, )
        FR variances. These are non-zero only if an error spectrum 'y_err' is 
        provided.
    
    """
    mean_fluxes = mean_flux(x, y, np.concatenate((x_i, x_j)), np.concatenate((w, w)))
    frs         = mean_fluxes[:len(x_i)] / mean_fluxes[len(x_i):]
    
    
    if not hasattr(y_err, '__len__'):
        v_frs = np.zeros(frs.shape)
    else:
        mean_vars = mean_flux(x, y_err**2, np.concatenate((x_i, x_j)), np.concatenate((w, w)))    
        v_frs     = frs ** 2 * ((mean_vars[:len(x_i)] / mean_fluxes[:len(x_i)]) ** 2 + (mean_vars[len(x_i):] / mean_fluxes[len(x_i):]) ** 2)
        
    return frs, v_frs

def mean_flux(x, y, x0, w):
    """ Compute mean fluxes.
    
    Determine the mean flux between the wavelength coordinates ''x0 - w/2'' 
    and ''x0 + w/2'' by linear interpolation of 'y' in 'x'.
    
    Parameters
    -----------
    x : array_like, shape (N, )
        1-D array containing the wavelength information.
    y : array_like, shape (N, )
        1-D array containing the flux information.
    x0 : array_like, shape (M, )
        1-D array of central wavelength positions.
    w : array_like, shape (M, )
        1-D array containing the full widths of the wavelength ranges around 
        'x0'.
        
    Returns
    -------
    mean_flux : ndarray, shape (M, )
        1-D array of mean fluxes.
    
    """
    if not hasattr(x0, '__len__'):
        x0 = np.array([x0])
        w = np.array([w])
        
    if len(y.shape) > 1:
        if y.shape[1] == 1:
            y = y[:,0]
    
    dx = x[1:] - x[:-1]
    dx = np.concatenate((dx,[dx[-1]]))
    
    x_midp        = np.concatenate(([x[0] - dx[0]/2],(x + dx/2)))
    
    nan_mask             = ~np.isfinite(y)
    y_nan_zero           = y.copy()
    y_nan_zero[nan_mask] = 0
    
    cumsum_y           = np.cumsum(y_nan_zero * dx)
    cumsum_y[nan_mask] = np.nan
    cumsum_y           = np.concatenate(([0],cumsum_y))
    interp_points      = np.interp(np.concatenate((x0 + w/2, x0 - w/2)), x_midp, cumsum_y, left=np.nan, right=np.nan)
    
    return (interp_points[:len(x0)] - interp_points[len(x0):]) / w

def higher_order_correction(p, R, fr, R0 = 45000):
    """ Apply resolution correction to FRs.
    
    Given the actual resolution 'R', apply the resolution correction according 
    to Eq. 14 in the paper to the measured 'fr' in order to be conform with 
    the training resolution 'R0'.
    
    Parameters
    -----------
    p : array_like, shape (M, N)
        Correction coefficients.
    R : array_like, shape (N, ), or float
        1-D array containing the resolutions for the individual 'fr', or a 
        single value to be applied to all entries in 'fr'.
    fr : array_like, shape (N, )
        1-D array of measured FRs.
    R0 : float, optional
        The training resolution. This value should not be altered unless a 
        training run with new grid resolution is performed.
        
    Returns
    -------
    corr : ndarray, shape (N, )
        1-D array of resolution-corrected FRs.
    
    """
    lnR = np.log(R) - np.log(R0)
    corr = np.zeros(fr.shape)
    for i in range(0,len(p),2):
        corr += (p[i] * lnR ** 2 + p[i+1] * lnR) * fr ** (len(p)/2 - i/2)
        
    return corr

def compute_Teff(coeffs, covs, frs, v_frs):
    """ Compute effective temperatures from FRs.
    
    Effective temperatures are computed using linear relations with FR. 
    
    Parameters
    -----------
    coeffs : array_like, shape (N, 2)
        Slopes and intercepts.
    covs : array_like, shape (N, 2, 2)
        Covariance matrices of the coefficients.
    frs : array_like, shape (N, )
        1-D array of measured FRs.
    v_frs : array_like, shape (N, )
        1-D array of FR variances.
        
    Returns
    -------
    Teffs : ndarray, shape (N, )
        Individual temperatures for each FR.
    v_Teffs : ndarray, shape (N, )
        Variances for 'Teffs'.
    
    """
    coeffs = coeffs.transpose()       
    Teffs  = np.polyval(coeffs, frs)
    deriv  = np.column_stack((frs,np.ones(len(frs))))
    
    v_Teffs = np.zeros(len(covs))
    for i in range(len(covs)):
        v_Teffs[i] = deriv[i].dot(covs[i].dot(deriv[i]))

    v_Teffs += coeffs[0] ** 2 * v_frs
    
    return Teffs, v_Teffs

def compute_met(coeffs, covs, frs, v_frs, Teff, v_Teff, v_Teff_sys, v_met_sys):
    """ Compute iron abundances, [Fe/H], from FRs.
    
    [Fe/H] results are computed by evaluating Eq. 10 in the paper for each FR.
    
    Parameters
    -----------
    coeffs : array_like, shape (N, 6)
        Coefficients for the [Fe/H] relations.
    covs : array_like, shape (N, 6, 6)
        Covariance matrices of the coefficients.
    frs : array_like, shape (N, )
        1-D array of measured FRs (1st independent variable).
    v_frs : array_like, shape (N, )
        1-D array of FR variances.
    Teff : float
        Input temperature (2nd independent variable).
    v_Teff : float
        Variance of 'Teff'.
    v_Teff_sys : float
        Squared systematic error of 'Teff'.
    v_met_sys : float
        Global squared systematic error of the [Fe/H] relations.
        
    Returns
    -------
    mets : ndarray, shape (N, )
        Individual [Fe/H] findings for each input FR.
    v_mets : ndarray, shape (N, )
        Variances for 'mets'.
    v_mets_sys : ndarray, shape (N, )
        Propagated squared systematic errors for 'mets'.
    
    """
    coeffs = coeffs.transpose()
    
    first_factor  = (coeffs[0] * frs + (coeffs[1] + coeffs[2] * frs) * Teff + coeffs[3])
    exponential   = np.exp(coeffs[4] * (frs - coeffs[5]))
    second_factor = (1 + exponential)
    mets          = first_factor * second_factor
    deriv         = np.column_stack((frs, Teff * np.ones(len(frs)), frs * Teff, np.ones(len(frs)))) * second_factor[:,None]
    deriv         = np.column_stack((deriv, (frs - coeffs[5]) * exponential, -coeffs[4] * exponential)) * first_factor[:,None]
    
    v_mets = np.zeros(len(covs))
    for i in range(len(covs)):
        v_mets[i] = deriv[i].dot(covs[i].dot(deriv[i]))
        
    v_mets    += ((coeffs[0] + coeffs[2] * Teff) * second_factor + first_factor * coeffs[4] * exponential) ** 2 * v_frs
    v_mets    += ((coeffs[1] + coeffs[2] * frs) * second_factor) ** 2 * v_Teff
    v_mets_sys = ((coeffs[1] + coeffs[2] * frs) * second_factor) ** 2 * v_Teff_sys + v_met_sys
    
    return mets, v_mets, v_mets_sys

def compute_logg(coeffs, covs, frs, v_frs, Teff, v_Teff, met, v_met, v_Teff_sys, v_met_sys, v_logg_sys):
    """ Compute surface gravities, logg, from FRs.
    
    logg results are computed by evaluating Eq. 13 in the paper for each FR.
    
    Parameters
    -----------
    coeffs : array_like, shape (N, 4)
        Coefficients for the logg relations.
    covs : array_like, shape (N, 4, 4)
        Covariance matrices of the coefficients.
    frs : array_like, shape (N, )
        1-D array of measured FRs (1st independent variable).
    v_frs : array_like, shape (N, )
        1-D array of FR variances.
    Teff : float
        Input temperature (2nd independent variable).
    v_Teff : float
        Variance of 'Teff'.
    met : float
        Input iron abundance (3rd independent variable)
    v_met : float
        Variance of 'met'.
    v_Teff_sys : float
        Squared systematic error of 'Teff'.
    v_met_sys : float
        Squared systematic error of 'met'.
    v_logg_sys : float
        Global squared systematic error of the logg relations.
        
    Returns
    -------
    loggs : ndarray, shape (N, )
        Individual logg findings for each input FR.
    v_loggs : ndarray, shape (N, )
        Variances for 'loggs'.
    v_loggs_sys : ndarray, shape (N, )
        Propagated squared systematic errors for 'loggs'.
    
    """
    coeffs = coeffs.transpose()
    
    loggs = frs * coeffs[0] + Teff * coeffs[1] + met * coeffs[2] + coeffs[3]
    deriv = np.column_stack((frs, Teff * np.ones(len(frs)), met * np.ones(len(frs)), np.ones(len(frs))))
    
    v_loggs = np.zeros(len(covs))
    for i in range(len(covs)):
        v_loggs[i] = deriv[i].dot(covs[i].dot(deriv[i]))
        
    v_loggs    += coeffs[0] ** 2 * v_frs + coeffs[1] ** 2 * v_Teff + coeffs[2] ** 2 * v_met
    v_loggs_sys = coeffs[1] ** 2 * v_Teff_sys + coeffs[2] ** 2 * v_met_sys + v_logg_sys
    
    return loggs, v_loggs, v_loggs_sys

def weighted_median(y_in, weights = None):
    """ Derive the weighted median.
    
    Parameters
    -----------
    y_in : array_like, shape (N, )
        1-D array of input values.
    weights : array_like, shape (N, ), optional
        Weights for 'y_in'. If no weights are provided, the unweighted median 
        is returned.
        
    Returns
    -------
    med : float
        (Un-)weighted median of 'y_in'.
    mad : float
        Squared, scaled median absolute deviation.
    
    """
    if weights is None:
        med = np.nanmedian(y_in)
        return med, (1.4826 * np.nanmedian(np.abs(y_in - med))) ** 2
    
    else:
        if np.all(np.isnan(y_in)):
            return np.nan, np.nan
        else:
            finite = np.isfinite(y_in)
            argsort = np.argsort(y_in[finite])
            y = y_in[finite][argsort]
            w = weights[finite][argsort]
            p = w.cumsum() / np.nansum(w)
            med = np.interp(0.5, p, y)
            mad = 1.4826 * np.nanmedian(np.abs(y_in - med))
            return med, mad ** 2


