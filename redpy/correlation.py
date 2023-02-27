 # REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import numpy as np
from scipy.fftpack import fft, ifft
from scipy.sparse import coo_matrix

import redpy.cluster
import redpy.table


def calculate_window(waveform, windowStart, opt):
    """
    Calculates derived quantities for a window of data for cross-correlation.
        
    Quantities are a scaling coefficient, the Fourier transform, and frequency
    index (FI). The scaling coefficient is the inverse square root of the sum
    of squared amplitudes, which normalizes the cross-correlation coefficient
    to 1.0 for true auto-correlation. FI is the logarithm (base 10) of the
    ratio of the mean spectral amplitudes in two frequency windows (high/low)
    such that FI of 0 is equal amplitudes, positive FI has more amplitude in
    the higher frequencies, and negative FI has more amplitude in the lower
    frequencies. Quantities are derived for each station individually.
    
    Parameters
    ----------
    waveform : float ndarray
        Waveform data for all stations, concatenated.
    windowStart : integer
        Sample of trigger time relative to waveform start.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    windowCoeff : float ndarray
        Scaling coefficient to normalize cross-correlation for each station.
    windowFFT : complex ndarray
        Fourier transforms for each station.
    windowFI : float ndarray
        Frequency index for each station, NaNs where missing data.
    
    """
    
    # Shift window left of the trigger time by 10% of total window length
    windowStart = windowStart - opt.winlen/10
    
    # Initialize output variables
    windowCoeff = np.zeros(opt.nsta)
    windowFFT = np.zeros(opt.winlen*opt.nsta,).astype(np.complex64)
    windowFI = np.zeros(opt.nsta)
    
    for n in range(opt.nsta):
        winstart = int(n*opt.wshape + windowStart)
        winend = int(n*opt.wshape + windowStart + opt.winlen)
        
        fftwin = np.reshape(fft(waveform[winstart:winend]),(opt.winlen,))
        windowFFT[n*opt.winlen:(n+1)*opt.winlen] = fftwin
        
        # !!! Is there a better way to check for partially empty waveforms?
        # !!! This feels like it might throw away too much, as this could
        # !!! potentially only require two samples to be 0
        
        # !!! Proposed change (requires testing) would require fully 1/5th of
        # !!! the waveform to be exactly zero:
        # !!! if np.sort(np.abs(waveform[winstart:winend]))[int(
        # !!!                                             opt.winlen/5)] == 0
        if np.median(np.abs(waveform[winstart:winend]))==0:
            windowFI[n] = np.nan
        else:
            windowCoeff[n] = 1/np.sqrt(sum(
                waveform[winstart:winend] * waveform[winstart:winend]))
            windowFI[n] = np.log10(
                np.mean(np.abs(np.real(fftwin[
                    int(opt.fiupmin*opt.winlen/opt.samprate):
                    int(opt.fiupmax*opt.winlen/opt.samprate)]))
                ) / np.mean(np.abs(np.real(fftwin[
                    int(opt.filomin*opt.winlen/opt.samprate):
                    int(opt.filomax*opt.winlen/opt.samprate)]))
                ))
    
    return windowCoeff, windowFFT, windowFI


def get_window(row, opt):
    """
    Convenience function to get window parameters from a single row in table.
    
    Parameters
    ----------
    row : Row object
        Single row from either Repeaters or Orphans tables
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    windowCoeff : float ndarray
        Scaling coefficient to normalize cross-correlation for each station.
    windowFFT : complex ndarray
        Fourier transforms for each station.
    windowFI : float ndarray
        Frequency index for each station, NaNs where missing data.
    
    """
    
    windowCoeff = row['windowCoeff']
    windowFFT = row['windowFFT']
    windowFI = row['FI']
    
    return windowCoeff, windowFFT, windowFI


def update_window(xtable, rownum, lag, opt, coeff=[], fft=[], fi=[]):
    """
    Convenience function to set window parameters for a single row in a table.
    
    If coeff, fft, and fi are not passed, they are recalculated based on the
    updated trigger time. All three must be passed, otherwise they will be
    recalculated.
    
    Parameters
    ----------
    xtable : Table object
        Handle to either the Repeaters or Orphans tables.
    rownum : integer
        Row number of the table to modify.
    lag : integer
        Number of samples to add to the current trigger time.
    opt : Options object
        Describes the run parameters.
    coeff : float ndarray, optional
        Scaling coefficient to normalize cross-correlation for each station.
    fft : complex ndarray, optional
        Fourier transforms for each station.
    fi : float ndarray, optional
        Frequency index for each station, NaNs where missing data.
    
    """
    
    trigger = int(xtable.cols.windowStart[rownum] + lag)
    
    if not all([len(coeff), len(fft), len(fi)]):
        coeff, fft, fi = calculate_window(xtable[rownum]['waveform'],
            trigger, opt)
    
    # !!! Calculate amplitudes here too
    
    xtable.cols.windowStart[rownum] = trigger
    xtable.cols.windowCoeff[rownum] = coeff
    xtable.cols.windowFFT[rownum] = fft
    xtable.cols.FI[rownum] = fi
    
    xtable.flush()


def get_correlation_function(windowFFT1, windowFFT2, n, opt):
    """
    Calculates the correlation function for a single channel.
    
    Parameters
    ----------
    windowFFT1 : complex ndarray
        Fourier transform of first window on all stations, concatenated.
    windowFFT2 : complex ndarray
        Fourier transform of second window on all stations, concatenated.
    n : int
        Index of station/channel of interest.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    correlation_function : float ndarray
        Unscaled correlation function.
    
    """
    
    win1 = windowFFT1[n*opt.winlen:(n+1)*opt.winlen]
    win2 = windowFFT2[n*opt.winlen:(n+1)*opt.winlen]
    
    correlation_function = np.real(ifft(win1 * np.conj(win2)))
    
    return correlation_function


def xcorr_1x1(windowCoeff1, windowCoeff2, windowFFT1, windowFFT2, opt):
    """
    Calculates the cross-correlation coefficient and lag for two windows.
    
    Order matters for sign of lag, but not cross-correlation coefficient. The
    coefficient returned is the maximum across all stations. Lag is a bit more
    complicated. If the maximum correlation coefficient is above the minimum
    required in opt.cmin on opt.ncor or more stations, the lag will be the
    median lag across the highest correlated opt.ncor stations. Otherwise, the
    lag will be the for the single station with the highest coefficient.
    
    Parameters
    ----------
    windowCoeff1 : float ndarray
        Amplitude coefficient of first window on all stations.
    windowCoeff2 : float ndarray
        Amplitude coefficient of second window on all stations.
    windowFFT1 : complex ndarray
        Fourier transform of first window on all stations, concatenated.
    windowFFT2 : complex ndarray
        Fourier transform of second window on all stations, concatenated.
    
    Returns
    -------
    maxcor : float
        Maximum cross-correlation coefficient across all stations.
    maxlag : integer
        Lag corresponding to maximum cross-correlation.
    nthcor : float
        Cross-correlation coefficient on the opt.ncor-th station.
    
    """
    
    station_cors = np.zeros(opt.nsta)
    station_lags = np.zeros(opt.nsta, dtype=int)
    
    coeffs = windowCoeff1 * windowCoeff2
    
    # Loop over stations
    for n in range(opt.nsta):
        
        # This is a very expensive calculation!
        correlation_function = get_correlation_function(windowFFT1,
                                                        windowFFT2, n, opt)
    
        # Find index of maximum of the correlation function
        indx = np.argmax(correlation_function)
        station_cors[n] = correlation_function[indx] * coeffs[n]
        station_lags[n] = indx
    
    # Deal with wrap-around for determining lag
    station_lags[station_lags > int(opt.winlen/2)] -= opt.winlen
    
    # Find maximum across all stations
    maxcor = np.amax(station_cors)
    
    # Find correlation on opt.ncor-th station
    nthcor = np.sort(np.array(station_cors))[::-1][opt.ncor-1]
    
    # Find best lag to use depending on nthcor
    # !!! Test whether it's better to always have maxlag be median?
    if nthcor >= opt.cmin:
        # Median of best opt.ncor lags if nthcor is good
        maxlag = np.median(np.array(station_lags)[np.argsort(
                                             station_cors)[::-1][0:opt.ncor]])
    else:
        # Only use lag on best station if nthcor isn't good
        maxlag = station_lags[np.argmax(station_cors)]
    
    return maxcor, maxlag, nthcor


def xcorr_1xtable(windowCoeff, windowFFT, xtable, opt):
    """
    Correlates a single event with all events in a table or subtable.
    
    The 'xtable' can be a full table (e.g., the full Orphan table) or a
    selection of rows from a table (e.g., all cores or a single family
    from the Repeaters table). Could potentially be parallelized.
    
    Parameters
    ----------
    windowCoeff : float ndarray
        Amplitude coefficient of single window on all stations, concatenated.
    windowFFT : complex ndarray
        Fourier transform of single window on all stations, concatenated.
    xtable : Table object
        Either a table or subset of a table that contains 'window' columns.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    maxcors : float ndarray
        Maximum cross-correlation coefficient across all stations for each row
        in table.
    maxlags : integer ndarray
        Lag corresponding to maximum cross-correlation for each row in table.
    nthcors : float ndarray
        Cross-correlation coefficient on the opt.ncor-th station for each row
        in table.
    
    """
    
    maxcors = np.zeros(len(xtable))
    maxlags = np.zeros(len(xtable))
    nthcors = np.zeros(len(xtable))
    
    # Loop over rows
    for i in np.arange(0, len(xtable)):
        maxcors[i], maxlags[i], nthcors[i] = xcorr_1x1(windowCoeff,
                                        xtable[i]['windowCoeff'], windowFFT,
                                        xtable[i]['windowFFT'], opt)
    
    return maxcors, maxlags, nthcors


def compare_deleted(trigs, dtable, opt):
    """
    Compares triggers against deleted events, removes from list if correlated.
    
    Parameters
    ----------
    trigs : Trace list
        Triggers to be checked against deleted events.
    dtable : Table object
        Deleted table (i.e., manually removed from rtable)
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    trigs : Trace list
       Triggers that do not match deleted events.
    
    """
    
    for trig in trigs:
        
        # Calculate window
        windowCoeff, windowFFT, windowFI = calculate_window(trig.data,
                                             int(opt.ptrig*opt.samprate), opt)
        
        # Correlate against Deleted table
        maxcor, maxlag, nthcor = xcorr_1xtable(windowCoeff, windowFFT,
                                               dtable, opt)
        
        # If near match found, remove from trigger list. Assumes if it
        # correlates at this level it likely would have correlated with
        # another member of the full family.
        if np.max(maxcor) >= (opt.cmin-0.05):
            trigs.remove(trig)
    
    return trigs


def correlate_remaining_family(rtable, ctable, ftable, rnum, fnum, opt):
    """
    Correlates a known repeater with all events in a family except the core.
    
    This is intended to be run after a new event has correlated above the
    threshold with at least one other event (thus it being in the Repeaters
    table) AND with the core of the family. Convenience function for
    populating the Correlation table.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ctable : Table object
        Handle to the Correlation table.
    ftable : Table object
        Handle to the Families table.
    rnum : integer
        Row of repeater to be compared in Repeaters table.
    fnum : integer
        Family number to compare to.
    opt : Options object
        Describes the run parameters.
    
    """
    
    # Get family subtable and the ids for those events
    family_table = get_family_subtable(rtable, ftable, fnum, opt)
    family_ids = family_table['id']
    
    # Get repeater idnum
    repeater_id = rtable[rnum]['id']
    
    # Correlate repeater with family
    maxcors, maxlags, nthcors = xcorr_1xtable(rtable[rnum]['windowCoeff'],
                                rtable[rnum]['windowFFT'], family_table, opt)
    
    # Write matches to Correlation table
    if np.max(nthcors) >= opt.cmin:
        for i in np.where(nthcors>=opt.cmin)[0]:
            if repeater_id != family_ids[i]:
                redpy.table.populate_correlation(ctable, repeater_id,
                                             family_ids[i], maxcors[i], opt)


def get_family_subtable(rtable, ftable, fnum, opt):
    """
    Gets the 100 most recent and 100 largest remaining events in the family.
    
    Also automatically excludes the current core event.
    # !!! N should be a setting in opt rather than hard-coded
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    fnum : integer
        Family number to query.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    family_table : Table object
        Subset of rtable.
    
    """
    
    n = 100 # !!!
    
    # Get corresponding row numbers for members and cores
    members = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')
    cores = ftable[fnum]['core']
    
    # Exclude the core
    members = np.setdiff1d(members, cores)
    
    if len(members) <= 2*n:
        
        # For small families we can return all of them
        family_table = rtable[members]
        
    else:
        
        # Here we determine which events to bother correlating
        # First figure out the N most recent
        rtimes = rtable[members]['startTimeMPL']
        n_recent = members[np.argsort(rtimes)[-n:]].copy()
        
        # Update members to exclude the most recent
        members = np.setdiff1d(members, n_recent)
        # Take the mean of all channels to punish missing data
        ramps = np.mean(rtable[members]['windowAmp'], axis=1)
        n_largest = members[np.argsort(ramps)[-n:]].copy()
        
        family_table = rtable[np.concatenate((n_recent,n_largest),axis=None)]
    
    return family_table


def append_family_member(ftable, fnum, rnum, opt):
    """
    Adds new member to the end of the list of family members.
    
    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    fnum : integer
        Family number to update.
    rnum : integer
        Member (row) from Repeaters table to append.
    opt : Options object
        Describes the run parameters.
    
    """
    
    # Decode string then append a space and rnum
    ftable.cols.members[fnum] = ftable.cols.members[fnum].decode('utf-8') \
                                                                   + f' {rnum}'
    # Ensure plots are updated
    ftable.cols.printme[fnum] = 1
    ftable.flush()


def compare_trigger_to_orphans(rtable, otable, ctable, ftable, trig, idnum,
        windowCoeff, windowFFT, maxcors, maxlags, nthcors, opt):
    """
    Compares a new trigger to orphans, then chooses how to handle it.
    
    First, the function finds the matches of the new trigger with the orphans.
    If it finds any, it will adopt them and deal with adjusting lags. Then it
    passes the trigger and any adopted orphans along to be checked against
    repeater cores.
    
    Parameters
    ----------    
    rtable : Table object
        Handle to the Repeaters table.
    otable : Table object
        Handle to the Orphans table.
    ctable : Table object
        Handle to the Correlation table.
    trig : Trace object
        New trigger to compare, with data from all stations concatenated.
    idnum : integer
        Unique ID of new trigger.
    windowCoeff : float ndarray
        Amplitude coefficient of trigger on all stations, concatenated.
    windowFFT : complex ndarray
        Fourier transform of trigger on all stations, concatenated.
    maxcors : float ndarray
        Correlation coefficients of trigger to all orphans.
    maxlags : integer ndarray
        Lag of trigger relative to all orphans.
    nthcors : float ndarray
        Correlation coefficients of trigger to all orphans on the opt.ncor-th
        station.
    opt : Options object
        Describes the run parameters.
    
    """
    
    # Loop through potential matches, adjusting windows as necessary
    written = 0
    while len(maxcors[maxcors >= opt.cmin-0.05]) > 0:
        
        # Find best correlated event
        bestcor = np.argmax(maxcors)
        
        # If trigger not written to rtable yet, realign new event
        if written == 0:
            # best_lag is updated until written
            best_lag = maxlags[bestcor]
            windowCoeff1, windowFFT1, windowFI1 = calculate_window(trig.data,
                int(opt.ptrig*opt.samprate + best_lag), opt)
            windowCoeff2, windowFFT2, windowFI2 = get_window(otable[bestcor],
                                                                         opt)
        
        # If written already, realign older orphan to new event
        else:
            windowCoeff2, windowFFT2, windowFI2 = calculate_window(
                otable[bestcor]['waveform'],
                int(opt.ptrig*opt.samprate + best_lag - maxlags[bestcor]),
                opt)
        
        # Correlate with new window alignment
        maxcor_aligned, maxlag_aligned, nthcor_aligned = xcorr_1x1(
                                                 windowCoeff1, windowCoeff2,
                                                 windowFFT1, windowFFT2, opt)
        
        # If actually matches...
        if nthcor_aligned >= opt.cmin:
            
            # Either move the trigger and adopted orphan to the repeater table
            if written == 0:
                redpy.table.populate_repeater(rtable, idnum, trig, opt,
                    int(opt.ptrig*opt.samprate + best_lag))
                redpy.table.move_orphan(rtable, otable, bestcor, opt)
                redpy.table.populate_correlation(ctable, idnum,
                                        rtable[-1]['id'], maxcor_aligned, opt)
                written = 2
            
            # Or update the new window in otable, then move adopted orphan
            else:
                update_window(otable, bestcor, best_lag - maxlags[bestcor],
                    opt, coeff=windowCoeff2, fft=windowFFT2, fi=windowFI2)
                redpy.table.move_orphan(rtable, otable, bestcor, opt)
                redpy.table.populate_correlation(ctable, idnum, 
                                        rtable[-1]['id'], maxcor_aligned, opt)
                written += 1
        
        # Remove best correlated from lists
        maxlags = np.delete(maxlags, bestcor)
        nthcors = np.delete(nthcors, bestcor)
        maxcors = np.delete(maxcors, bestcor)
    
    # If there are no actual matches in the orphans, check trigger with cores
    if written == 0:
        if len(rtable) > 0:
            compare_trigger_to_cores(rtable, otable, ctable, ftable, trig,
                idnum, windowCoeff, windowFFT, opt)
        else:
            redpy.table.populate_orphan(otable, idnum, trig, opt)
    # If there is a match, check new event and its adopted matches with cores
    else:
        compare_adopted_to_cores(rtable, ctable, ftable, written, opt)


def update_with_trigger(rtable, ftable, trig, idnum, windowStart, lag, fnum,
    written, laglist, famlist, opt):
    """
    Updates tables, lags, and families matched with new trigger.
    
    Only updates the Repeaters and Families tables if this trigger hasn't been
    written to the Repeaters table yet, otherwise updates matched family and
    lag arrays.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    trig : Trace object
        Trigger to be added, with data from all stations concatenated.
    idnum : integer
        Unique ID of new trigger.
    windowStart : integer
        Trigger time in samples relative to start of waveform.
    lag : integer
        Lag between trigger time and family.
    fnum : integer
        Family number lag corresponds to.
    written : integer
        Number of triggers written to Repeaters table this iteration.
    laglist : integer list
        Lag between trigger and matched families so far.
    famlist : integer list
        Family numbers matched so far.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    written : integer
        Number of triggers written to Repeaters table this iteration.
    laglist : integer ndarray
        Lag between trigger and matched families so far.
    famlist : integer ndarray
        Family numbers matched so far.
    
    """
    
    if written == 0:
        # Move the trigger to the repeater table
        redpy.table.populate_repeater(rtable, idnum, trig, opt, windowStart)
        # Append to family
        append_family_member(ftable, fnum, len(rtable)-1, opt)
        laglist.append(0)
        written = 1
    else:
        # Keep track of the lag relative to what's been written
        laglist.append(lag)
    
    # Append to family list that needs to be merged
    famlist.append(fnum)
    
    return written, laglist, famlist


def compare_trigger_to_cores(rtable, otable, ctable, ftable, trig, idnum,
        windowCoeff, windowFFT, opt):
    """
    Compares a single unwritten trigger to the cluster cores.
    
    If it matches, it adds it to the best cluster and merges clusters if it
    finds more than one. Otherwise, appends it to the orphan table.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    otable : Table object
        Handle to the Orphans table.
    ctable : Table object
        Handle to the Correlation table.
    trig : Trace object
        New trigger to compare, with data from all stations concatenated.
    idnum : integer
        Unique ID of new trigger.
    windowCoeff : float ndarray
        Amplitude coefficient of trigger on all stations, concatenated.
    windowFFT : complex ndarray
        Fourier transform of trigger on all stations, concatenated.
    opt : Options object
        Describes the run parameters.
    
    """
    
    # Get cores
    fnums = np.arange(ftable.attrs.nClust)
    cores_table = rtable[ftable.cols.core[:]]
    core_windowFFTs = cores_table['windowFFT']
    core_windowCoeffs = cores_table['windowCoeff']
    core_ids = cores_table['id']
    
    # Correlate against the cores
    core_maxcors, core_maxlags, core_nthcors = xcorr_1xtable(windowCoeff,
                                                 windowFFT, cores_table, opt)
    
    written = 0
    laglist = []
    famlist = []
    
    # Loop through potential matching families
    while len(core_maxcors[core_maxcors >= opt.cmin-0.05]) > 0:
        
        bestcor = np.argmax(core_maxcors)
        
        if written == 0:
            # best_lag is updated until written
            best_lag = core_maxlags[bestcor]
            
            # Recalculate windows with this lag
            windowCoeff_new, windowFFT_new, windowFI_new = \
                calculate_window(trig.data,
                                 int(opt.ptrig*opt.samprate + best_lag), opt)
        
        # Cross-correlate with core at appropriate lag
        maxcor_new, maxlag_new, nthcor_new = xcorr_1x1(windowCoeff_new,
                                   core_windowCoeffs[bestcor], windowFFT_new,
                                   core_windowFFTs[bestcor], opt)
        
        # If correlates above threshold with core, it's definitely a match
        if nthcor_new >= opt.cmin:
            
            # Update lists/tables
            written, laglist, famlist = update_with_trigger(rtable, ftable,
                trig, idnum, int(opt.ptrig*opt.samprate + best_lag),
                maxlag_new, fnums[bestcor], written, laglist, famlist, opt)
            
            # Correlate with other members of the family
            redpy.table.populate_correlation(ctable, idnum, core_ids[bestcor],
                                                            maxcor_new, opt)
            correlate_remaining_family(rtable, ctable, ftable, -1,
                                                     fnums[bestcor], opt)
        
        # Since it's close to the core, check the rest of the family
        else:
            
            # Get family
            family_table = get_family_subtable(rtable, ftable, fnums[bestcor],
                                                                          opt)
            family_ids = family_table['id']
            
            # Correlate to family
            maxcors_fam, maxlags_fam, nthcors_fam = xcorr_1xtable(
                            windowCoeff_new, windowFFT_new, family_table, opt)
            
            # If there is even a single match in that family...
            if np.max(nthcors_fam) >= opt.cmin:
                
                # Update lists/tables
                written, laglist, famlist = update_with_trigger(rtable,
                    ftable, trig, idnum, int(opt.ptrig*opt.samprate + \
                    best_lag + maxlags_fam[np.argmax(maxcors_fam)]),
                    maxlags_fam[np.argmax(maxcors_fam)], fnums[bestcor],
                    written, laglist, famlist, opt)
                
                # Populate Correlation table
                for x in np.where(nthcors_fam >= opt.cmin)[0]:
                    redpy.table.populate_correlation(ctable, idnum,
                                           family_ids[x], maxcors_fam[x], opt)
        
        # Remove from arrays when done
        fnums = np.delete(fnums, bestcor)
        core_windowFFTs = np.delete(core_windowFFTs, bestcor, axis=0)
        core_windowCoeffs = np.delete(core_windowCoeffs, bestcor, axis=0)
        core_ids = np.delete(core_ids, bestcor)    
        core_maxlags = np.delete(core_maxlags, bestcor)
        core_maxcors = np.delete(core_maxcors, bestcor)
        
    # If doesn't match anything, append as orphan
    if written == 0:
        redpy.table.populate_orphan(otable, idnum, trig, opt)
        
    # Otherwise, either update core or merge families.
    else:
        if len(famlist) == 1:
            redpy.cluster.update_family(rtable, ctable, ftable, famlist[0],
                                                                         opt)
        else:
            redpy.table.merge_families(rtable, ctable, ftable, famlist,
                                                                laglist, opt)


def compare_adopted_to_cores(rtable, ctable, ftable, written, opt):
    """
    Compares newly adopted repeaters to cores, merges matched families.
    
    Similar to compare_trigger_to_cores() but with the added complication that
    the new trigger has adopted at least one other orphan and now exists in
    the Repeater table.
    
    rtable : Table object
        Handle to the Repeater table.
    ctable : Table object
        Handle to the Correlation table.
    ftable : Table object
        Handle to the Families table.
    written : integer
        Number of new repeaters written to Repeaters table. 
    opt : Options object
        Describes the run parameters.
    
    """
    
    famlist = []
    laglist = []
    
    found = 0
    if len(ftable) >= 1:
        
        # Get cores
        cores_table = rtable[ftable.cols.core[:]]
        core_windowFFTs = cores_table['windowFFT']
        core_windowCoeffs = cores_table['windowCoeff']
        core_ids = cores_table['id']
        fnums = range(ftable.attrs.nClust)
        
        # Get repeater that adopted the others to compare
        windowCoeff = rtable[-written]['windowCoeff']
        windowFFT = rtable[-written]['windowFFT']
        
        # Correlate against the cores
        core_maxcors, core_maxlags, core_nthcors = xcorr_1xtable(windowCoeff,
                                                 windowFFT, cores_table, opt)
        
        # Loop through potential matching families
        while len(core_maxcors[core_maxcors >= opt.cmin-0.05]) > 0:
            
            bestcor = np.argmax(core_maxcors)
            
            if found == 0:
                # best_lag is updated until found
                best_lag = core_maxlags[bestcor]
                
                # Recalculate windows with this lag
                windowCoeff_new, windowFFT_new, windowFI_new = \
                    calculate_window(rtable[-written]['waveform'],
                        int(rtable[-written]['windowStart'] + best_lag), opt)
            
            # Cross-correlate with core at appropriate lag
            maxcor_new, maxlag_new, nthcor_new = xcorr_1x1(windowCoeff_new,
                                   core_windowCoeffs[bestcor], windowFFT_new,
                                   core_windowFFTs[bestcor], opt)
            
            # If correlates above threshold with core, it's definitely a match
            if nthcor_new >= opt.cmin:
                
                # Compare to full family, write to correlation table
                for i in range(-written,0):
                    maxcor, maxlag, nthcor = xcorr_1x1(
                        rtable[i]['windowCoeff'], core_windowCoeffs[bestcor],
                        rtable[i]['windowFFT'], core_windowFFTs[bestcor], opt)
                    if nthcor >= opt.cmin:
                        redpy.table.populate_correlation(ctable,
                            rtable[i]['id'], core_ids[bestcor], maxcor, opt)
                    correlate_remaining_family(rtable, ctable, ftable, i,
                        fnums[bestcor], opt)
                
                if found == 0:
                    
                    # Update found, lag is 0 relative to this family
                    found = 1
                    laglist.append(0)
                    
                    # Assign new trigger time, update windows
                    for i in range(-written,0):
                        update_window(rtable, i, best_lag, opt)
                        append_family_member(ftable, fnums[bestcor],
                                             len(rtable)+i, opt)
                
                # Otherwise, update laglist with lags to prepare merge
                else:
                    laglist.append(maxlag_new)
                
                # Update the family list to be merged
                famlist.append(fnums[bestcor])
            
            # Since it's close to the core, check the rest of the family
            else:
                
                # Get family
                family_table = get_family_subtable(rtable, ftable,
                                                          fnums[bestcor], opt)
                
                # Correlate to family
                maxcors_fam, maxlags_fam, nthcors_fam = xcorr_1xtable(
                            windowCoeff_new, windowFFT_new, family_table, opt)
                
                # If there's at least one match...
                if max(nthcors_fam) >= opt.cmin:
                
                    if found == 0:
                        
                        # Update found, lag is 0 relative to this family
                        found = 1
                        laglist.append(0)
                        
                        # Assign new trigger time, update windows
                        for i in range(-written,0):
                            update_window(rtable, i, best_lag + \
                                maxlags_fam[np.argmax(maxcors_fam)], opt)
                            append_family_member(ftable, fnums[bestcor],
                                                 len(rtable)+i, opt)
                    
                    # Otherwise, update laglist with lags to prepare merge
                    else:
                        laglist.append(maxlags_fam[np.argmax(maxcors_fam)])
                        
                    # Update the family list to be merged
                    famlist.append(fnums[bestcor])
                    
                    # Populate Correlation table
                    for i in np.arange(0, len(maxcors_fam)):
                        if nthcors_fam[i] >= opt.cmin:
                            redpy.table.populate_correlation(ctable,
                                rtable[-written]['id'], family_table[i]['id'],
                                maxcors_fam[i], opt)
            
            # Remove from arrays when done
            fnums = np.delete(fnums, bestcor)
            core_windowFFTs = np.delete(core_windowFFTs, bestcor, axis=0)
            core_windowCoeffs = np.delete(core_windowCoeffs, bestcor, axis=0)
            core_ids = np.delete(core_ids, bestcor)
            core_maxlags = np.delete(core_maxlags, bestcor)
            core_maxcors = np.delete(core_maxcors, bestcor)
    
    # If no matches found, make new family
    if found == 0:
        members = np.arange(len(rtable)-written,len(rtable)).astype(int)
        core = len(rtable)-written
        redpy.table.populate_new_family(rtable, ftable, members, core, opt)
        
    # Otherwise, either update core or merge multiple families
    else:
        if len(famlist) == 1:
            redpy.cluster.update_family(rtable, ctable, ftable, famlist[0],
                                                                          opt)
        else:
            redpy.table.merge_families(rtable, ctable, ftable, famlist,
                                                                 laglist, opt)


def do_comparison(rtable, otable, ctable, ftable, trig, idnum, windowCoeff,
    windowFFT, maxcors, maxlags, nthcors, opt):
    """
    Executes decision for which comparison path to begin with.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    otable : Table object
        Handle to the Orphans table.
    ctable : Table object
        Handle to the Correlation table.
    trig : Trace object
        New trigger to compare, with data from all stations concatenated.
    idnum : integer
        Unique ID of new trigger.
    windowCoeff : float ndarray
        Amplitude coefficient of trigger on all stations, concatenated.
    windowFFT : complex ndarray
        Fourier transform of trigger on all stations, concatenated.
    maxcors : float ndarray
        Correlation coefficients of trigger to all orphans.
    maxlags : integer ndarray
        Lag of trigger relative to all orphans.
    nthcors : float ndarray
        Correlation coefficients of trigger to all orphans on the opt.ncor-th
        station.
    opt : Options object
        Describes the run parameters.
    
    """
    
    # If there's a possible match with orphan(s), run most complex function
    if max(maxcors) > opt.cmin-0.05:
        compare_trigger_to_orphans(rtable, otable, ctable, ftable,
                    trig, idnum, windowCoeff, windowFFT, maxcors, maxlags,
                    nthcors, opt)
    else:
        # Compare that orphan to the cores in the repeater table
        if len(rtable) > 0:
            compare_trigger_to_cores(rtable, otable, ctable, ftable,
                        trig, idnum, windowCoeff, windowFFT, opt)
        # Always populate as an orphan if there are no repeaters yet
        else:
            redpy.table.populate_orphan(otable, idnum, trig, opt)


def correlate_new_triggers(rtable, otable, ctable, ftable, ttimes, trig,
    idnum, troubleshoot, opt):
    """
    Adds a new trigger to the correct table.
    
    Specifically, it checks to ensure the trigger doesn't already exist, then
    correlates it with all orphans, and sorts into the correct table based
    on that result. If troubleshoot=False and a step within the sorting fails
    for some reason, the trigger will be appended to the orphan table.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    otable : Table object
        Handle to the Orphans table.
    ctable : Table object
        Handle to the Correlation table.
    ftable : Table object
        Handle to the Families table.
    ttimes : float ndarray
        Array of times of existing triggers to prevent duplication.
    trig : Trace object
        New trigger to compare, with data from all stations concatenated.
    idnum : integer
        Unique ID of new trigger.
    troubleshoot : bool
        Flag to bypass try/catch and allow code to fail.
    opt : Options object
        Describes the run parameters.
    
    """
    
    windowCoeff, windowFFT, windowFI = calculate_window(trig.data,
                                             int(opt.ptrig*opt.samprate), opt)
    
    # Correlate with the new event with all the orphans
    maxcors, maxlags, nthcors = xcorr_1xtable(windowCoeff, windowFFT,
                                                                  otable, opt)
    
    # Allow the correlation step to fail for troubleshooting
    if troubleshoot:
        do_comparison(rtable, otable, ctable, ftable, trig, idnum,
                       windowCoeff, windowFFT, maxcors, maxlags, nthcors, opt)
    
    # Do not allow a problem to interrupt flow
    else:
        try:
            do_comparison(rtable, otable, ctable, ftable, trig, idnum,
                       windowCoeff, windowFFT, maxcors, maxlags, nthcors, opt)
        except:
            print('Could not properly correlate, troubleshoot with -t')
            redpy.table.populate_orphan(otable, idnum, trig, opt)


def get_matrix(rtable, ctable, opt):
    """
    Turns the contents of the Correlation table into a sparse matrix.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ctable : Table object
        Handle to the Correlation table.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    ids : int ndarray
        'id' column from Repeaters table.
    ccc_sparse : float csr_matrix
        Sparse correlation matrix with id as rows/columns.
    
    """
    
    # Get correlation matrix and ids
    ids = rtable.cols.id[:]
    id1 = ctable.cols.id1[:]
    id2 = ctable.cols.id2[:]
    ccc = ctable.cols.ccc[:]
    
    maxid = np.max((np.max(ids),np.max(id2)))+1
    
    # !!! This is slow-ish, but if there are duplicate entries in the ctable
    # !!! (e.g., from old or maybe interrupted runs), when we use a sparse
    # !!! matrix duplicates get summed, resulting in values far above 1
    rc = np.vstack([id1, id2]).T.copy()
    dt = rc.dtype.descr * 2
    i = np.unique(rc.view(dt), return_index=True)[1]
    
    # Set up sparse correlation matrix in Compressed Sparse Row
    # format for slicing later
    ccc_sparse = coo_matrix((ccc[i], (id1[i], id2[i])),
                            shape=(maxid, maxid)).tocsr()
    
    return ids, ccc_sparse


def subset_matrix(ids_sub, ccc_sparse, opt, return_type='maxrow', ind=-1):
    """
    Transforms a subset of the sparse correlation matrix to dense row/matrix.
    
    Parameters
    ----------
    ids_sub : int ndarray
        'id' numbers of repeaters to use (e.g., sorted family).
    ccc_sparse : float csr_matrix
        Sparse correlation matrix with id as rows/columns.
    opt : Options object
        Describes the run parameters.
    return_type : str, optional
        Controls behavior with three options:
            'maxrow' : Returns row of matrix with highest sum.
            'indrow' : Returns row corresponding to 'ind' (e.g., the core).
            'matrix' : Returns the full dense matrix.
    ind : int, optional
        Index of the row to return within the subset (default last row).
    
    Returns
    -------
    ccc_array : float ndarray
       Either the full correlation matrix or a specified row from it.
    
    """
    
    # Get correlation matrix for family only
    ccc_sub = ccc_sparse[ids_sub,:]
    ccc_sub = ccc_sub[:,ids_sub]
    ccc_sub += ccc_sub.transpose()
    
    if return_type == 'matrix':
        ccc_sub = ccc_sub.todense()
        ccc_sub = ccc_sub + np.eye(len(ids_sub))
        ccc_array = np.squeeze(np.asarray(ccc_sub))
    else:
        if return_type == 'maxrow':
            ind = np.argmax(ccc_sub.sum(axis=0))
        
        ccc_array = np.squeeze(np.asarray(ccc_sub[:,ind].todense()))
        ccc_array[ind] = 1 # For autocorrelation
    
    return ccc_array


def make_full(rtable_sub, ccc_sub, opt):
    """
    Fills an incomplete correlation matrix.
    
    In theory, this could probably be done in parallel for very large subsets.
    However, I've run into problems with large datasets (where the parallel
    version would be most useful!) where the array for windowFFT exceeds the
    data size limit imposed by pickling.
    
    Parameters
    ----------
    rtable_sub : Table object
        Handle to a subset of the Repeaters table.
    ccc_sub : float ndarray
        Existing correlation matrix corresponding to that subset.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    ccc_full : float ndarray
        Filled correlation matrix.
    
    """
    
    k = 1
    ccc_full = ccc_sub.copy()
    total_missing = len(np.where(ccc_sub==0)[0])/2
    
    for i in np.arange(0, len(rtable_sub)-1):
        for j in np.arange(i+1, len(rtable_sub)):
            if ccc_full[i,j]==0:
                if k%100000 is 0:
                    print(f'{(100*k/total_missing):3.2f}% done...')
                # Compute correlation
                cor, lag, nthcor = xcorr_1x1(rtable_sub['windowCoeff'][i],
                                             rtable_sub['windowCoeff'][j],
                                             rtable_sub['windowFFT'][i],
                                             rtable_sub['windowFFT'][j], opt)
                ccc_full[i,j] = cor
                ccc_full[j,i] = cor
                k += 1
    
    return ccc_full

