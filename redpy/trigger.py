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
        self.start_sample = detector.get('start_sample')
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
        """
        Fill out the window-related attributes.

        This should be saved for after duplicate triggers are handled to
        save time.

        Parameters
        ----------
        detector : Detector object
            Primary interface for handling detections.

        """
        self._concatenate(detector)  # fill self.concat
        self.coeff, self.fft, self.freq_index = calculate_window(
            detector, self.concat,
            detector.get('start_sample'))

    def populate(self, detector, table_type, n_junk=0, n_tele=0):
        """
        Populate this trigger to a table.

        Parameters
        ----------
        detector : Detector object
            Primary interface for handling detections.
        table_type : str
            Table to append the trigger to ('jtable', 'otable', or 'rtable').
        n_junk : int, optional
            If appending to 'jtable', provide the number of 'junk' channels.
        n_tele : int, optional
            If appending to 'jtable', provide the number of 'tele' channels.

        """
        if table_type not in ['jtable', 'otable', 'rtable']:
            raise ValueError("Can't append trigger to {table_type}!")
        if table_type == 'jtable':
            self._populate_junk(detector, n_junk, n_tele)
        else:
            self._populate_trigger(detector)
            if table_type == 'otable':
                self._populate_orphan(detector)
            else:
                self._populate_repeater(detector)
            detector.get('rtable').table.attrs.previd += 1
        detector.get('rtable').table.attrs.ptime = self.time

    def _build_row(self, detector, row):
        """Build a row for rtable/otable based on contents of trigger."""
        row['id'] = detector.get('rtable').table.attrs.previd + 1
        row['startTime'] = self.waveforms[0].stats.starttime.isoformat()
        row['startTimeMPL'] = self.waveforms[0].stats.starttime.matplotlib_date
        row['waveform'] = self.concat
        row['windowStart'] = self.start_sample
        row['windowCoeff'] = self.coeff
        row['windowFFT'] = self.fft
        row['FI'] = self.freq_index
        row['windowAmp'] = self._calculate_window_amplitude(detector)
        return row

    def _calculate_window_amplitude(self, detector):
        """Calculate the maximum amplitudes within the window."""
        amps = np.zeros(detector.get('nsta'))
        winstart = (
            self.time - 0.1*detector.get('winlen')/detector.get('samprate'))
        winend = (
            self.time + 0.75*detector.get('winlen')/detector.get('samprate'))
        for i in range(detector.get('nsta')):
            amps[i] = np.max(np.abs(
                self.waveforms[i].slice(winstart, winend).data))
        return amps

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

    def _get_expiration(self, detector):
        """Get the orphan expiration time based on ratio."""
        maxorph = detector.get('maxorph')
        minorph = detector.get('minorph')
        add_days = np.min([maxorph, self.ratio*(
            (maxorph-minorph)/maxorph) + minorph])
        return (self.waveforms[0].stats.starttime + add_days*86400).isoformat()

    def _populate_junk(self, detector, n_junk, n_tele):
        """Populate the trigger (if unique) to the Junk table."""
        jtimes = detector.get('jtable', 'startTime')
        if n_junk > 0:
            if n_tele > 0:
                jtype = 2
            else:
                jtype = 1
        else:
            jtype = 0
        start_time_min = (
            self.waveforms[0].stats.starttime - 1).isoformat().encode(
                'utf-8')
        start_time_max = (
            self.waveforms[0].stats.starttime + 1).isoformat().encode(
                'utf-8')
        if len(np.where((jtimes > start_time_min) & (
                jtimes < start_time_max))[0]) == 0:
            row = detector.get('jtable').table.row
            row['startTime'] = self.waveforms[0].stats.starttime.isoformat()
            row['waveform'] = self.concat
            row['windowStart'] = int(
                detector.get('start_sample'))
            row['isjunk'] = jtype
            detector.get('jtable').append(row)

    def _populate_orphan(self, detector):
        """Populate the trigger to the Orphans table."""
        row = detector.get('otable').table.row
        row = self._build_row(detector, row)
        row['expires'] = self._get_expiration(detector)
        detector.get('otable').append(row)

    def _populate_repeater(self, detector):
        """Populate the trigger to the Repeaters table."""
        row = detector.get('rtable').table.row
        row = self._build_row(detector, row)
        detector.get('rtable').append(row)

    def _populate_trigger(self, detector):
        """Populate the trigger to the Triggers table."""
        row = detector.get('ttable').table.row
        row['startTimeMPL'] = self.waveforms[0].stats.starttime.matplotlib_date
        detector.get('ttable').append(row)
