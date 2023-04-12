# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling functions related to cross-correlation operations.

The primary function of this module is to support the .update() method of
Detector() objects. Triggers from continuous waveforms are cross-correlated
for similarity to be sorted into families. This module also has functions
for handling cross-correlation matrices.
"""
import numpy as np
from scipy.fftpack import ifft
# from scipy.fftpack import fft, ifft

# import redpy.cluster
# import redpy.table


def get_correlation_function(window_fft1, window_fft2, sta, detector):
    """
    Calculate the correlation function for a single channel.

    Parameters
    ----------
    window_fft1 : complex ndarray
        Fourier transform of first window on all stations, concatenated.
    window_fft2 : complex ndarray
        Fourier transform of second window on all stations, concatenated.
    sta : int
        Index of station/channel of interest.
    detector : Config object
        Describes the run parameters.

    Returns
    -------
    float ndarray
        Unscaled correlation function.

    """
    win1 = window_fft1[
        sta*detector.get('winlen'):(sta+1)*detector.get('winlen')]
    win2 = window_fft2[
        sta*detector.get('winlen'):(sta+1)*detector.get('winlen')]
    correlation_function = np.real(ifft(win1 * np.conj(win2)))
    return correlation_function


def make_full(detector, rtable_sub, ccc_sub):
    """
    Fill an incomplete correlation matrix.

    In theory, this could probably be done in parallel for very large subsets.
    However, I've run into problems with large datasets (where the parallel
    version would be most useful!) where the array for windowFFT exceeds the
    data size limit imposed by pickling.

    Parameters
    ----------
    rtable_sub : structured array
        Handle to a subset of the Repeaters table.
    ccc_sub : float ndarray
        Existing correlation matrix corresponding to that subset.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    ccc_full : float ndarray
        Filled correlation matrix.

    """
    k = 1
    ccc_full = ccc_sub.copy()
    total_missing = len(np.where(ccc_sub == 0)[0])/2
    for i in np.arange(0, len(rtable_sub)-1):
        for j in np.arange(i+1, len(rtable_sub)):
            if ccc_full[i, j] == 0:
                if k % 100000 == 0:
                    print(f'{(100*k/total_missing):3.2f}% done...')
                cor, _, _ = xcorr_1x1(
                    rtable_sub['windowCoeff'][i],
                    rtable_sub['windowCoeff'][j],
                    rtable_sub['windowFFT'][i],
                    rtable_sub['windowFFT'][j], detector)
                ccc_full[i, j] = cor
                ccc_full[j, i] = cor
                k += 1
    return ccc_full


def subset_matrix(ids_sub, ccc_sparse, return_type='maxrow', ind=-1):
    """
    Transform a subset of the sparse correlation matrix to dense row/matrix.

    Parameters
    ----------
    ids_sub : int ndarray
        'id' numbers of repeaters to use (e.g., sorted family).
    ccc_sparse : float csr_matrix
        Sparse correlation matrix with id as rows/columns.
    return_type : str, optional
        Controls behavior with three options:
            'maxrow' : Returns row of matrix with highest sum.
            'indrow' : Returns row corresponding to 'ind' (e.g., the core).
            'matrix' : Returns the full dense matrix.
    ind : int, optional
        Index of the row to return within the subset (default last row).

    Returns
    -------
    float ndarray
       Either the full correlation matrix or a specified row from it.

    """
    ccc_sub = ccc_sparse[ids_sub, :]
    ccc_sub = ccc_sub[:, ids_sub]
    ccc_sub += ccc_sub.transpose()
    if return_type == 'matrix':
        ccc_sub = ccc_sub.todense()
        ccc_sub = ccc_sub + np.eye(len(ids_sub))
        ccc_array = np.squeeze(np.asarray(ccc_sub))
    else:
        if return_type == 'maxrow':
            ind = np.argmax(ccc_sub.sum(axis=0))
        ccc_array = np.squeeze(np.asarray(ccc_sub[:, ind].todense()))
        ccc_array[ind] = 1  # For autocorrelation
    return ccc_array


def xcorr_1x1(
        window_coeff1, window_coeff2, window_fft1, window_fft2, detector):
    """
    Calculate the cross-correlation coefficient and lag for two windows.

    Order matters for sign of lag, but not cross-correlation coefficient.
    The coefficient returned is the maximum across all stations. Lag is a
    bit more complicated. If the maximum correlation coefficient is above
    the minimum required in detector.get('cmin') on detector.get('ncor') or
    more stations, the lag will be the median lag across the highest
    correlated detector.get('ncor') stations. Otherwise, the lag will be the
    for the single station with the highest coefficient.

    Parameters
    ----------
    window_coeff1 : float ndarray
        Amplitude coefficient of first window on all stations.
    window_coeff2 : float ndarray
        Amplitude coefficient of second window on all stations.
    window_fft1 : complex ndarray
        Fourier transform of first window on all stations, concatenated.
    window_fft2 : complex ndarray
        Fourier transform of second window on all stations, concatenated.

    Returns
    -------
    float
        Maximum cross-correlation coefficient across all stations.
    integer
        Lag corresponding to maximum cross-correlation.
    float
        Cross-correlation coefficient on the detector.get('ncor')-th station.

    """
    station_cors = np.zeros(detector.get('nsta'))
    station_lags = np.zeros(detector.get('nsta'), dtype=int)
    coeffs = window_coeff1 * window_coeff2
    for sta in range(detector.get('nsta')):
        # This is a very expensive calculation!
        correlation_function = get_correlation_function(
            window_fft1, window_fft2, sta, detector)
        indx = np.argmax(correlation_function)
        station_cors[sta] = correlation_function[indx] * coeffs[sta]
        station_lags[sta] = indx
    station_lags[
        station_lags > int(detector.get('winlen')/2)] -= detector.get('winlen')
    maxcor = np.amax(station_cors)
    nthcor = np.sort(np.array(station_cors))[::-1][detector.get('ncor')-1]
    if nthcor >= detector.get('cmin'):
        maxlag = np.median(np.array(station_lags)[np.argsort(
            station_cors)[::-1][0:detector.get('ncor')]])
    else:
        maxlag = station_lags[np.argmax(station_cors)]
    return maxcor, maxlag, nthcor
