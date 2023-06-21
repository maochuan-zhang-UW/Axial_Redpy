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
from collections import defaultdict

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
    tracker = defaultdict(list)
    tracker['found'] = False
    tracker['written'] = written
    if len(detector) > 0:
        fnums = np.arange(len(detector))
        core_maxcors, core_maxlags, core_nthcors = xcorr_1xtable(
            detector, 'cores', trig.coeff, trig.fft)
        while np.max(core_maxcors) >= detector.get('cmin')-0.05:
            bestcor = np.argmax(core_maxcors)
            if not tracker['found']:
                bestlag = core_maxlags[bestcor]
                _update_trig_lag(detector, trig, bestlag)
            if bestlag == 0:
                new_maxcor, new_maxlag, new_nthcor = (
                    core_maxcors[bestcor], core_maxlags[bestcor],
                    core_nthcors[bestcor])
            else:
                new_maxcor, new_maxlag, new_nthcor = xcorr_1x1(
                    detector, trig.best_coeff,
                    detector.get('cores', 'windowCoeff', bestcor),
                    trig.best_fft,
                    detector.get('cores', 'windowFFT', bestcor))
            if new_nthcor >= detector.get('cmin'):
                _handle_core_match(
                    detector, trig, tracker,
                    fnums[bestcor], bestlag, new_maxlag, new_maxcor,
                    detector.cores['id'][bestcor])
                tracker['found'] = True
            else:
                _handle_near_match(
                    detector, trig, tracker, fnums[bestcor], bestlag)
            core_maxcors[bestcor] = 0
    if not tracker['found']:
        if tracker['written'] == 0:
            trig.populate(detector, 'otable')
        else:
            populate_new_family(detector, tracker['written'])
    else:
        if len(tracker['famlist']) == 1:
            update_family(detector, tracker['famlist'][0])
        else:
            merge_families(detector, tracker['famlist'], tracker['laglist'])


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
        (window_coeff1, window_fft1, _,
         window_coeff2, window_fft2, window_fi2) = _get_lag_adjusted_windows(
            detector, trig, maxlags, bestlag, bestcor, 'otable', bestcor,
            written)
        maxcor_aligned, _, nthcor_aligned = xcorr_1x1(
            detector, window_coeff1, window_coeff2, window_fft1, window_fft2)
        if nthcor_aligned >= detector.get('cmin'):
            if written == 0:
                written = 2
                trig.coeff, trig.fft, trig.freq_index = (
                    redpy.correlation.calculate_window(
                        detector, trig.concat,
                        detector.get('start_sample') + bestlag))
                trig.start_sample = detector.get('start_sample') + bestlag
                trig.populate(detector, 'rtable')
                _move_orphan_populate_correlation(
                    detector, bestcor, maxcor_aligned, written)
            else:
                written += 1
                _update_window(
                    detector, 'otable', bestcor, bestlag - maxlags[bestcor],
                    window_coeff2, window_fft2, window_fi2)
                _move_orphan_populate_correlation(
                    detector, bestcor, maxcor_aligned, written)
        maxcors = np.delete(maxcors, bestcor)
        maxlags = np.delete(maxlags, bestcor)
    if len(detector.get('rtable')) > 0:
        compare_trigger_to_cores(detector, trig, written)
    else:
        trig.populate(detector, 'otable')


def from_window(detector, window_start, window_end, expire, force):
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
    expire : bool
        If True, expire stale orphans.
    force : bool
        If True, force a trigger to occur at the time of an event in
        the event_list contained in detector.waveforms.

    """
    detector._check_famlen()
    i_time = time.time()
    if detector.get('verbose'):
        print(window_start)
    stream, d_time = detector.waveforms.get_data(
        detector, window_start, window_end)
    trig_list = detector.waveforms.get_triggers(detector, stream, force)
    trig_list = remove_duplicates(detector, trig_list)
    trig_list = clean_junk(detector, trig_list)
    trig_list = compare_deleted(detector, trig_list)
    for trig in trig_list:
        trigger_to_table(detector, trig)
    if expire:
        detector.remove('expire')
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
            detector.set(
                'ftable', bytes(member_string, 'utf-8'), 'members', first_fam)
            detector.get('ftable').remove(fnum)
    _remove_core(detector, np.setdiff1d(famlist, first_fam))
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
    _append_core(detector, core)
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
        for col in detector.get('rtable').column_names:
            detector.set('cores', detector.get('cores', col)[order], col)


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
    members = detector.get_members(fnum)
    if (len(members) in [
        3, 4, 5, 6, 10, 15, 25, 50, 100, 250, 500, 1000, 2500,
        5000, 10000, 25000, 50000, 100000, 250000, 500000]) or (
            merge <= detector.get('merge_ratio')):
        _run_optics(detector, fnum, members)
    _update_ftable(detector, fnum, members)
    _check_core_window(detector, fnum)


def _add_match(detector, trig, tracker, fnum, bestlag, new_maxlag):
    """Add a match to tables based on its status."""
    tracker['famlist'].append(fnum)
    if not tracker['found']:
        tracker['laglist'].append(0)
        if tracker['written'] == 0:
            if new_maxlag:
                trig.coeff, trig.fft, trig.freq_index = (
                    redpy.correlation.calculate_window(
                        detector, trig.concat,
                        trig.start_sample + bestlag + new_maxlag))
                trig.start_sample = trig.start_sample + bestlag + new_maxlag
            else:
                trig.coeff = trig.best_coeff
                trig.fft = trig.best_fft
                trig.freq_index = trig.best_fi
                trig.start_sample = trig.start_sample + bestlag
            trig.populate(detector, 'rtable')
            tracker['written'] = 1
            _append_family_member(detector, fnum, -1)
        else:
            for i in np.arange(-tracker['written'], 0):
                if i == -tracker['written']:
                    _update_window(
                        detector, 'rtable', i, bestlag + new_maxlag)
                else:
                    _update_window(detector, 'rtable', i, bestlag + new_maxlag)
                _append_family_member(detector, fnum, i)
    else:
        tracker['laglist'].append(new_maxlag)


def _append_core(detector, core):
    """Append core of new family to cores subtable."""
    for col in detector.get('rtable').column_names:
        if col in ['FI', 'waveform', 'windowAmp', 'windowCoeff', 'windowFFT']:
            detector.set('cores', np.append(
                detector.get('cores', col),
                [detector.get('rtable', col, core)], axis=0), col)
        else:
            detector.set('cores', np.append(
                detector.get('cores', col),
                detector.get('rtable', col, core)), col)


def _append_family_member(detector, fnum, idx):
    """Append new family member from rtable to ftable."""
    if idx < 0:
        idx = len(detector.get('rtable')) + idx
    member_string = detector.get(
        'ftable', 'members', fnum).decode('utf-8') + f' {idx}'
    detector.set('ftable', bytes(member_string, 'utf-8'), 'members', fnum)


def _check_core_window(detector, fnum):
    """Check that core window hasn't changed."""
    core = detector.get('ftable', 'core', fnum)
    window = detector.get('rtable', 'windowStart', core)
    if detector.get('cores', 'windowStart', fnum) != window:
        for col in ['windowStart', 'windowFFT', 'windowCoeff']:
            detector.set('cores', detector.get('rtable', col, core), col, fnum)


def _correlate_remaining_family(detector, fnum, rnum):
    """Correlate a known repeater with all eligible family members."""
    if rnum < 0:
        rnum = len(detector.get('rtable')) + rnum
    subtable_members = _get_family_subtable(detector, fnum, rnum)
    new_id = detector.get('rtable', 'id', rnum)
    new_coeff = detector.get('rtable', 'windowCoeff', rnum)
    new_fft = detector.get('rtable', 'windowFFT', rnum)
    maxcors, _, nthcors = xcorr_1xtable(
        detector, 'rtable', new_coeff, new_fft, subtable_members)
    if np.max(nthcors) >= detector.get('cmin'):
        for i in np.where(nthcors >= detector.get('cmin'))[0]:
            _populate_correlation(
                detector, new_id, detector.get(
                    'rtable', 'id', subtable_members[i]), maxcors[i])


def _find_duplicates(detector, trig_times, rank):
    """Find duplicates within a list of trigger times."""
    order = np.argsort(trig_times)
    spacing = np.diff(trig_times[order])
    rank = rank[order]
    i = np.where(spacing < detector.get('mintrig'))[0]
    return np.unique(np.max(np.vstack((rank[i], rank[i+1])), axis=0))


def _get_family_subtable(detector, fnum, rnum):
    """Get eligible members of family for correlation."""
    members = np.setdiff1d(detector.get_members(fnum),
                           detector.get('ftable', 'core', fnum))
    members = np.setdiff1d(members, rnum)
    ntotal = (detector.get('corr_nrecent') + detector.get('corr_nyoungest')
              + detector.get('corr_nlargest'))
    if (len(members) <= ntotal) or (ntotal == 0):
        return members
    n_recent = np.array([]).astype(int)
    n_youngest = np.array([]).astype(int)
    n_largest = np.array([]).astype(int)
    if detector.get('corr_nrecent'):
        n_recent = members[
            np.argsort(detector.get('rtable', 'id', members))[
                -detector.get('corr_nrecent'):]]
        members = np.setdiff1d(members, n_recent)
    if detector.get('corr_nyoungest'):
        n_youngest = members[
            np.argsort(detector.get('rtable', 'startTimeMPL', members))[
                -detector.get('corr_nyoungest'):]]
        members = np.setdiff1d(members, n_youngest)
    if detector.get('corr_nlargest'):
        n_largest = members[
            np.argsort(np.mean(
                detector.get('rtable', 'windowAmp', members), axis=1))[
                    -detector.get('corr_nlargest'):]]
    return np.concatenate((n_recent, n_youngest, n_largest), axis=None)


def _get_lag_adjusted_windows(
        detector, trig, maxlags, bestlag, bestcor, table_type, row, written=0):
    """Get window parameters adjusted for lag."""
    window_coeff1, window_fft1, window_fi1 = (
        trig.coeff, trig.fft, trig.freq_index)
    if written == 0:
        if bestlag != 0:
            window_coeff1, window_fft1, window_fi1 = calculate_window(
                detector, trig.concat,
                detector.get('start_sample') + bestlag)
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


def _handle_core_match(detector, trig, tracker,
                       fnum, bestlag, new_maxlag, new_maxcor, core_id):
    """Handle case where trigger matches a core."""
    _add_match(detector, trig, tracker, fnum, bestlag, new_maxlag)
    _populate_correlation(
        detector, detector.get('rtable', 'id', -tracker['written']),
        core_id, new_maxcor)
    for i in np.arange(-tracker['written'], 0):
        _correlate_remaining_family(detector, fnum, i)


def _handle_near_match(detector, trig, tracker, fnum, bestlag):
    """Handle case where a trigger nearly matches a core."""
    subtable_members = _get_family_subtable(detector, fnum, -1)
    maxcors, maxlags, nthcors = xcorr_1xtable(
       detector, 'rtable', trig.best_coeff, trig.best_fft, subtable_members)
    if np.max(nthcors) >= detector.get('cmin'):
        _add_match(detector, trig, tracker, fnum, bestlag,
                   maxlags[np.argmax(maxcors)])
        tracker['found'] = True
        for i in np.where(nthcors >= detector.get('cmin'))[0]:
            _populate_correlation(
                detector, detector.get('rtable', 'id', -tracker['written']),
                detector.get('rtable', 'id', subtable_members[i]), maxcors[i])


def _populate_correlation(detector, id1, id2, ccc):
    """Populate correlation table with measurement."""
    detector.get('ctable').append({
        'id1': np.min((id1, id2)),
        'id2': np.max((id1, id2)),
        'ccc': ccc})


def _move_orphan_populate_correlation(
        detector, bestcor, maxcor_aligned, written):
    """Move orphan and populate correlation with new Trigger."""
    detector.get('otable').move(detector.get('rtable'), bestcor)
    _populate_correlation(
        detector, detector.get('rtable', 'id', -1),
        detector.get('rtable', 'id', -written), maxcor_aligned)


def _remove_core(detector, fnum):
    """Remove family from core subtable."""
    for col in detector.get('rtable').column_names:
        detector.set('cores', np.delete(
            detector.get('cores', col), fnum, axis=0), col)


def _run_optics(detector, fnum, members):
    """Run OPTICS to find best core event."""
    optics_object, members = redpy.optics.run_optics(detector, members)
    core = members[np.argmin(optics_object.reachability_)]
    _update_core(detector, fnum, core)


def _update_core(detector, fnum, rnum):
    """Update the core in ftable and detector.cores."""
    if detector.get('ftable', 'core', fnum) != rnum:
        detector.set('ftable', rnum, 'core', fnum)
        for col in detector.get('rtable').column_names:
            detector.cores[col][fnum] = detector.get('rtable', col, rnum)


def _update_ftable(detector, fnum, members):
    """Ensure Families table is up to date with members."""
    start_times = detector.get('rtable', 'startTimeMPL', members)
    mintime = np.min(start_times)
    detector.set('ftable', mintime, 'startTime', fnum)
    detector.set('ftable', np.max(start_times) - mintime, 'longevity', fnum)
    detector.set('ftable', 1, 'printme', fnum)
    detector.set('ftable', 1, 'printme', -1)


def _update_trig_lag(detector, trig, bestlag):
    """Update trig window based on best lag."""
    if bestlag != 0:
        trig.best_coeff, trig.best_fft, trig.best_fi = calculate_window(
            detector, trig.concat, trig.start_sample + bestlag)
    else:
        trig.best_coeff = trig.coeff
        trig.best_fft = trig.fft
        trig.best_fi = trig.freq_index


def _update_window(
        detector, table_type, row, lag, coeff=None, fft=None, fi=None):
    """Set window parameters for a single row in a table."""
    row = int(row)
    trigger = int(detector.get(table_type, 'windowStart', row) + lag)
    if coeff is None:
        coeff, fft, fi = calculate_window(
            detector, detector.get(table_type, 'waveform', row), trigger)
    detector.set(table_type, trigger, 'windowStart', row)
    detector.set(table_type, coeff, 'windowCoeff', row)
    detector.set(table_type, fft, 'windowFFT', row)
    detector.set(table_type, fi, 'FI', row)
