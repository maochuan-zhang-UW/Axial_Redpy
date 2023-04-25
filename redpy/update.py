# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling functions related to updating the catalog of detections.

The primary function of this module is to support the .update() method of
Detector() objects. The .update() method creates triggers from continuous
waveform data that are cross-correlated and sorted into families.
"""
import time

import matplotlib.dates as mdates
import numpy as np
from obspy import UTCDateTime
from scipy.stats import kurtosis

import redpy.optics
from redpy.correlation import calculate_window, xcorr_1x1, xcorr_1xtable


def clean_junk(detector, trig_list):
    """
    Clean triggers of data spikes, calibration pulses, and teleseisms.

    Specifically, it attempts to weed out spikes and analog calibration
    pulses using kurtosis and outlier ratios; checks for teleseisms that
    have very low frequency index. If force triggering with a specific
    event, a warning will be shown if the event would normally be
    categorized as junk, but otherwise is allowed to pass the tests.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    trig_list : list of Trigger objects
        Trigger list.
    force : bool
        If True, allow forced triggers to pass all tests.

    Returns
    -------
    list of Trigger objects
        Trigger list that does not contain junk.

    # !!! Future plan: store more information in jtable?

    """
    wshape = detector.get('wshape')
    winlen = detector.get('winlen')
    for trig in trig_list:
        trig = trig.generate_window(detector)
    for trig in trig_list.copy():
        if trig.time not in detector.waveforms.event_list:
            n_junk = 0
            n_tele = 0
            for i in range(detector.get('nsta')):
                if trig.freq_index[i] < detector.get('telefi'):
                    n_tele += 1
                data = trig.waveforms[i].data[:wshape]
                data_cut = trig.waveforms[i].slice(
                    trig.time - detector.get('kurtwin')/2,
                    trig.time + detector.get('kurtwin')/2).data
                k_data = np.nan
                k_freq = np.nan
                o_ratio = np.nan
                if np.sum(np.abs(data_cut)) > 0:
                    k_data = kurtosis(data_cut)
                    k_freq = kurtosis(np.abs(
                        trig.fft[i*winlen:(i+1)*winlen]))
                    mad = np.nanmedian(np.abs(data - np.nanmedian(data)))
                    z_ratio = (data - np.nanmedian(data))/mad
                    o_ratio = len(z_ratio[z_ratio > 4.45])/np.array(
                        len(z_ratio)).astype(float)
                    if (k_data >= detector.get('kurtmax')) or (
                        o_ratio >= detector.get('oratiomax')) or (
                            k_freq >= detector.get('kurtfmax')):
                        n_junk += 1
            if (n_junk >= detector.get('ncor')) or (
                    n_tele >= detector.get('teleok')):
                trig.populate(detector, 'jtable', n_junk, n_tele)
                trig_list.remove(trig)
    return trig_list


def compare_deleted(detector, trig_list):
    """
    Compare new triggers against deleted cores, remove matches.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    trig_list : list of Trigger objects
        Trigger list.

    Returns
    -------
    list of Trigger objects
        Trigger list that does not contain matches to deleted events.

    """
    if len(detector.get('dtable')) > 0:
        for trig in trig_list.copy():
            maxcor, _, _ = xcorr_1xtable(
                detector, 'dtable', trig.coeff, trig.fft)
            # If near match found, remove from trigger list. This assumes if
            # it correlates at this level it likely would have correlated
            # with another member of the full family.
            if np.max(maxcor) >= (detector.get('cmin')-0.05):
                trig_list.remove(trig)
    return trig_list


def compare_trigger_to_cores(detector, trig, written=0):
    """
    Compare a new Trigger to Repeaters, then choose how to handle it.

    Triggers are first compared with cores (representative members of a
    Family), and if a near-match to a core is found, it is correlated with
    all members of the Family. If it is a true match, it is adopted into
    that Family. Families are merged if the new Trigger matches more than
    one Family. If no matches are found and the Trigger hasn't adopted other
    Orphans, it is appended as an Orphan.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    trig : Trigger object
        Trigger that adopted orphans.
    written : int, optional
        Number of new Repeaters appended.

    """
    famlist = []
    laglist = []
    found = 0
    if len(detector) > 0:
        print('would do calcs here...')
    if found == 0:
        if written == 0:
            trig.populate(detector, 'otable')
        else:
            populate_new_family(detector, written)
    else:
        if len(famlist) == 1:
            update_family(detector, famlist)
        else:
            merge_families(detector, famlist, laglist)


def compare_trigger_to_orphans(detector, trig, maxcors, maxlags):
    """
    Compare a new Trigger to Orphans, then choose how to handle it.

    If matches are found with existing Orphans, they are adopted and lags
    are adjusted. Then, the new Trigger is compared against Repeaters and
    is adopted into an existing Family, creates a new Family, or is appended
    as an Orphan if no matches are found.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    trig : Trigger object
        New Trigger to sort into the correct table.
    maxcors : float ndarray
        Maximum cross-correlation coefficients across all stations between
        Trigger and Orphans with current window.
    maxlags : integer ndarray
        Lags corresponding to maximum cross-correlation between Trigger and
        Orphans with current window.

    """
    written = 0
    while len(maxcors[maxcors >= detector.get('cmin')-0.05]) > 0:
        bestcor = np.argmax(maxcors)
        if written == 0:
            bestlag = maxlags[bestcor]
        (window_coeff1, window_fft1, window_fi1,
         window_coeff2, window_fft2, window_fi2) = _get_lag_adjusted_windows(
            detector, trig, maxlags, bestlag, bestcor, 'otable', bestcor, 0)
        maxcor_aligned, _, nthcor_aligned = xcorr_1x1(
            detector, window_coeff1, window_coeff2, window_fft1, window_fft2)
        maxcors[bestcor] = 0
        if nthcor_aligned >= detector.get('cmin'):
            if written == 0:
                written = 2
                trig.coeff = window_coeff1
                trig.fft = window_fft1
                trig.freq_index = window_fi1
                trig.start_sample = detector.get('start_sample') + bestlag
                trig.populate(detector, 'rtable')
                _move_orphan_populate_correlation(
                    detector, bestcor, maxcor_aligned, written)
            else:
                written += 1
                _update_window(
                    detector, 'otable', bestcor, bestlag-maxlags[bestcor],
                    window_coeff2, window_fft2, window_fi2)
                _move_orphan_populate_correlation(
                    detector, bestcor, maxcor_aligned, written)
            maxcors = np.delete(maxcors, bestcor)
            maxlags = np.delete(maxlags, bestcor)
    if len(detector.get('rtable')) > 0:
        compare_trigger_to_cores(detector, trig, written)
    else:
        trig.populate(detector, 'otable')


def from_window(detector, window_start, window_end, force):
    """
    Update tables from data in a time window.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    window_start : UTCDateTime object
        Starting time of data window.
    window_end : UTCDateTime object
        Ending time of data window.
    force : bool
        If True, force a trigger to occur at the time of an event in
        the event_list contained in detector.waveforms.

    """
    i_time = time.time()
    stream, d_time = detector.waveforms.get_data(
        detector, window_start, window_end)
    trig_list = detector.waveforms.get_triggers(detector, stream, force)
    trig_list = remove_duplicates(detector, trig_list)
    trig_list = clean_junk(detector, trig_list)
    trig_list = compare_deleted(detector, trig_list)
    for trig in trig_list:
        trigger_to_table(detector, trig)
    if detector.get('verbose'):
        detector.stats()
        print('Time spent this iteration: '
              f'{(time.time()-i_time-d_time):.2f} seconds '
              f'(+{d_time:.2f} seconds getting data)')


def merge_families(detector, famlist, laglist):
    """
    Combine Families that have been merged by adding a new Trigger.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    famlist : int list
        List of families to merge.
    laglist : int list
        List of lags between families and new event.

    """
    nmembers = [len(detector.get_members(fnum)) for fnum in famlist]
    laglist = np.array(laglist)
    laglist -= laglist[np.argmax(nmembers)]
    for i, fnum in enumerate(famlist):
        if laglist[i] != 0:
            for member in detector.get_members(fnum):
                _update_window(detector, 'rtable', member, -laglist[i])
    first_fam = np.min(famlist)
    max_mem = len(detector.get_members(first_fam))
    for fnum in np.sort(famlist)[::-1]:
        if fnum != first_fam:
            max_mem = np.max((max_mem, len(detector.get_members(fnum))))
            member_string = (detector.get(
                'ftable', 'members', first_fam).decode('utf-8') + ' '
                + detector.get('ftable', 'members', fnum).decode('utf-8'))
            detector.set('ftable', member_string, 'members', first_fam)
            detector.get('ftable').remove(fnum)
    detector.set('ftable', -1, 'lastprint', first_fam)
    merge = max_mem/len(detector.get_members(first_fam))
    update_family(detector, first_fam, merge)
    reorder_families(detector)


def populate_new_family(detector, written):
    """
    Populate a new Family with newly written Repeaters.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    written : int
        Number of new Repeaters written.

    """
    members = np.arange(len(detector.get('rtable'))-written,
                        len(detector.get('rtable'))).astype(int)
    core = len(detector.get('rtable'))-written
    row = detector.get('ftable').table.row
    row['members'] = np.array2string(members)[1:-1]
    row['core'] = core
    row['startTime'] = np.min(
        detector.get('rtable', 'startTimeMPL', members))
    row['longevity'] = np.max(
        detector.get('rtable', 'startTimeMPL', members)) - row['startTime']
    row['printme'] = 1
    row['lastprint'] = -1
    detector.get('ftable').append(row)
    if len(detector) > 1:
        reorder_families(detector)


def remove_duplicates(detector, trig_list):
    """
    Remove triggers in list that are too close together.

    Considers triggers that exist in the table already, as well as new
    triggers. Priority is given to existing triggers, and triggers earlier
    in the list (e.g., earlier in time, or sorted by some other property).

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    trig_list : list of Trigger objects
        Trigger list.

    Returns
    -------
    list of Trigger objects
        Trigger list with duplicates removed.

    """
    trig_list = np.array(trig_list)
    if len(trig_list) > 0:
        trig_times = np.array([i.time for i in trig_list])
        duplicates = _find_duplicates(
            detector, trig_times, np.arange(len(trig_times)))
        if len(duplicates) > 0:
            trig_list = np.delete(trig_list, duplicates)
            trig_times = np.delete(trig_times, duplicates)
            # Do again, just in case we missed some
            duplicates = _find_duplicates(
                detector, trig_times, np.arange(len(trig_times)))
            if len(duplicates) > 0:
                trig_list = np.delete(trig_list, duplicates)
                trig_times = np.delete(trig_times, duplicates)
        # Now compare against existing triggers in ttable
        existing = detector.get('ttable', 'startTimeMPL')
        existing = existing[
            (existing > (np.min(trig_times)-100).matplotlib_date)
            & (existing < (np.max(trig_times)+100).matplotlib_date)]
        existing = [UTCDateTime(
            mdates.num2date(i))+detector.get('ptrig') for i in existing]
        rank = np.concatenate((-np.ones(len(existing)),
                               np.arange(len(trig_times)))).astype(int)
        duplicates = _find_duplicates(
            detector, np.concatenate((existing, trig_times)), rank)
        if len(duplicates) > 0:
            return np.delete(trig_list, duplicates).tolist()
    return trig_list.tolist()


def reorder_families(detector):
    """
    Ensure families are ordered by start time.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    order = np.argsort(detector.get('ftable', 'startTime'))
    if (np.arange(len(detector)) != order).any():
        for col in detector.get('ftable').column_names:
            detector.set('ftable', detector.get('ftable', col, order), col)


def trigger_to_table(detector, trig):
    """
    Sort a new trigger into the proper table.

    Specifically, this function correlates the new trigger with all Orphans,
    adopts any matches, correlates with all cores, then with members of any
    matching families. If matches are found, appends to the Repeaters table
    and updates the Families table, otherwise appends as a new Orphan.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    trig : Trigger object
        New Trigger to sort into the correct table.

    """
    maxcors, maxlags, _ = xcorr_1xtable(
        detector, 'otable', trig.coeff, trig.fft)
    # If there's a possible match with an orphan, run most complex function
    if (len(maxcors) > 0) and (np.max(maxcors) > detector.get('cmin')-0.05):
        compare_trigger_to_orphans(detector, trig, maxcors, maxlags)
    else:
        if len(detector.get('rtable')) > 0:
            # Compare the trigger with cores
            compare_trigger_to_cores(detector, trig)
        else:
            # Always populate as an orphan if there are no repeaters yet
            trig.populate(detector, 'otable')


def update_family(detector, fnum, merge=1):
    """Update the Families table."""
    famlen = len(detector.get('ftable', 'members', fnum))
    if famlen > detector.get('ftable').table.attrs.current_max_famlen:
        detector.get('ftable').table.attrs.current_max_famlen = famlen
    fam = detector.get_members(fnum)
    if (len(fam) in [3, 4, 5, 6, 10, 15, 25, 50, 100, 250, 500, 1000, 2500,
                     5000, 10000, 25000, 50000, 100000, 250000, 500000]) or (
            merge <= detector.get('merge_percent')):
        _run_optics(detector, fnum, fam)
    _update_ftable(detector, fnum, fam)


def _find_duplicates(detector, trig_times, rank):
    """Find duplicates within a list of trigger times."""
    order = np.argsort(trig_times)
    spacing = np.diff(trig_times[order])
    rank = rank[order]
    i = np.where(spacing < detector.get('mintrig'))[0]
    return np.unique(np.max(np.vstack((rank[i], rank[i+1])), axis=0))


def _get_lag_adjusted_windows(
        detector, trig, maxlags, bestlag, bestcor, table_type, row, written=0):
    """Get window parameters adjusted for lag."""
    if written == 0:
        if bestlag != 0:
            window_coeff1, window_fft1, window_fi1 = calculate_window(
                detector, trig.concat,
                detector.get('start_sample') + bestlag)
        else:
            window_coeff1, window_fft1, window_fi1 = (
                trig.coeff, trig.fft, trig.freq_index)
        window_coeff2, window_fft2, window_fi2 = _get_window(
            detector, table_type, row)
    else:
        if bestlag - maxlags[bestcor] != 0:
            window_coeff2, window_fft2, window_fi2 = calculate_window(
                detector, detector.get(table_type, 'waveform', row),
                detector.get('start_sample') + bestlag - maxlags[bestcor])
        else:
            window_coeff2, window_fft2, window_fi2 = _get_window(
                detector, table_type, bestcor)
    return (window_coeff1, window_fft1, window_fi1,
            window_coeff2, window_fft2, window_fi2)


def _get_window(detector, table_type, row):
    """Get window parameters."""
    return (detector.get(table_type, 'windowCoeff', row),
            detector.get(table_type, 'windowFFT', row),
            detector.get(table_type, 'FI', row))


def _move_orphan_populate_correlation(
        detector, bestcor, maxcor_aligned, written):
    """Move orphan and populate correlation with new Trigger."""
    detector.get('otable').move(detector.get('rtable'), bestcor)
    detector.get('ctable').append({
        'id1': detector.get('rtable', 'id', -1),
        'id2': detector.get('rtable', 'id', -written),
        'ccc': maxcor_aligned})


def _run_optics(detector, fnum, fam):
    """Run OPTICS to find best core event."""
    ccc_sparse = redpy.correlation.subset_matrix(
        detector.get('rtable', 'id', fam), detector.get_matrix()[1],
        'sparse')
    order = np.argsort(np.squeeze(np.asarray(ccc_sparse.sum(axis=0))))
    fam = fam[order]
    ccc_sparse = ccc_sparse[order, :]
    ccc_sparse = ccc_sparse[:, order]
    distance_matrix = np.ones(len(fam)) - ccc_sparse
    distance_matrix = np.squeeze(np.asarray(distance_matrix))
    distance_matrix[range(len(fam)), range(len(fam))] = 0
    _, core = redpy.optics.OPTICS(distance_matrix).run(1)
    detector.set('ftable', 'core', fnum, fam[core])


def _update_ftable(detector, fnum, fam):
    """Ensure Families table is up to date with members."""
    start_times = detector.get('rtable', 'startTimeMPL', fam)
    mintime = np.min(start_times)
    detector.set('ftable', mintime, 'startTime', fnum)
    detector.set('ftable', np.max(start_times) - mintime, 'longevity', fnum)
    detector.set('ftable', 1, 'printme', fnum)
    detector.set('ftable', 1, 'printme', -1)


def _update_window(
        detector, table_type, row, lag, coeff=None, fft=None, fi=None):
    """Set window parameters for a single row in a table."""
    row = int(row)
    trigger = int(detector.get(table_type, 'windowStart', row) + lag)
    if not all([len(coeff), len(fft), len(fi)]):
        coeff, fft, fi = calculate_window(
            detector, detector.get(table_type, 'waveform', row), trigger)
    detector.set(table_type, trigger, 'windowStart', row)
    detector.set(table_type, coeff, 'windowCoeff', row)
    detector.set(table_type, fft, 'windowFFT', row)
    detector.set(table_type, fi, 'FI', row)
