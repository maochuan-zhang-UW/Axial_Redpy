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
        Triggered events with data from each channel concatenated.
    force : bool
        If True, allow forced triggers to pass all tests.

    Returns
    -------
    list of Trigger objects
        Remaining Triggers that are not considered junk.

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
    for trig in trig_list:
        trig.populate(detector, 'otable')
    if detector.get('verbose'):
        print('Time spent this iteration: '
              f'{(time.time()-i_time-d_time):.2f} seconds '
              f'(+{d_time:.2f} seconds getting data)')


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
        Trigger list to remove duplicates in.

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


def _find_duplicates(detector, trig_times, rank):
    """Find duplicates within a list of trigger times."""
    order = np.argsort(trig_times)
    spacing = np.diff(trig_times[order])
    rank = rank[order]
    i = np.where(spacing < detector.get('mintrig'))[0]
    return np.unique(np.max(np.vstack((rank[i], rank[i+1])), axis=0))
