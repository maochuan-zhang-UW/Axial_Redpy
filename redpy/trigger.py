# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling REDPy Trigger() object.

The Trigger() object holds the waveform data and associated metadata for
a single trigger, and has methods for populating itself into a table.
"""
import numpy as np

from redpy.correlation import calculate_window


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
        self.coeff, self.fft, self.freq_index = calculate_window(
            detector, self.concat,
            detector.get('ptrig')*detector.get('samprate'))

    def populate(self, detector, table_type, n_junk=0, n_tele=0):
        """Populate this trigger to a table."""
        if table_type == 'jtable':
            if n_junk > 0:
                if n_tele > 0:
                    jtype = 2
                else:
                    jtype = 1
            else:
                jtype = 0
            print(detector.get('jtable'))
            print(f'jtype = {jtype} : {self.time}')

    def _concatenate(self, detector):
        """Concatenate waveform data together."""
        waves = self.waveforms.trim(
            self.time - detector.get('ptrig'),
            self.time + detector.get('atrig') + 2/detector.get('samprate'),
            pad=True, fill_value=0)
        for i in waves:
            i.data = i.data[:detector.get('wshape')]
            if np.sum(i.data != 0):
                i.data[i.data != 0] -= np.mean(i.data[i.data != 0])
            self.concat = np.append(self.concat, i.data)
