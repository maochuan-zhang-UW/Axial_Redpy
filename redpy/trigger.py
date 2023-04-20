# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling REDPy Trigger() object.

The Trigger() object holds the waveform data and associated metadata for
a single trigger, and has methods for populating itself into a table.
"""
import matplotlib.dates as mdates
import numpy as np
from obspy import UTCDateTime


class Trigger():
    """
    Primary interface for handling individual triggers.

    Attributes
    ----------
    trig_time : UTCDateTime object
        Time of original trigger.
    waveforms : Stream object
        Ordered waveform data for all stations/channels for the trigger.
    ratio : float
        Maximum value of STA/LTA ratio; used to calculate orphan expiration.
    """

    def __init__(self, detector, time, ratio, stream):
        """
        Slice out waveform and set attribute structure.

        Parameters
        ----------
        detector : Detector object
            Primary interface for handling detections.
        time : UTCDateTime
            Time of original trigger
        ratio : float
            Maximum value of STA/LTA ratio from triggering algorithm.
        stream : Stream object
            Ordered waveform data for all stations/channels for entire
            window used for triggering.

        """
        self.time = time
        self.waveforms = stream.slice(
            time - detector.get('ptrig'),
            time + detector.get('atrig') + 2/detector.get('samprate')
            ).copy()
        self.ratio = ratio
        self.concat = np.array([])
        self.fft = np.array([])
        self.coeff = np.array([])
        self.freq_index = np.array([])

    def __repr__(self):
        """Define representation string."""
        return f'redpy.Trigger() at {self}'

    def __str__(self):
        """Define print string."""
        return str(str(self.time))

    def generate_window(self, detector):
        """Fill out the window-related ."""
        self._concatenate(detector)  # fill self.concat
        # calculate window here...

    def populate(self, detector, table_type):
        """Populate this trigger to a table."""
        print(detector)
        print(f'Add Trigger to {table_type} type')

    def _concatenate(self, detector):
        """Concatenate waveform data together."""
        waves = self.waveforms(self.time - detector.get('ptrig'),
                               self.time + detector.get('atrig')
                               + 2/detector.get('samprate'), pad=True,
                               fill_value=0)
        for i in waves:
            i.data = i.data[:detector.get('wshape')]
            i.data -= np.mean(i.data)
            # !!! i.data[i.data != 0] -= np.mean(i.data[i.data != 0])
            self.concat = np.append(self.concat, i.data)


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
    list of Triggers
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
