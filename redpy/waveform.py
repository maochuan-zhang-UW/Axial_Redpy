# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling REDPy Waveform() object.

The Waveform() object handles loading, downloading, filtering, and then
triggering of continuous waveform data.
"""
import glob
import itertools
import os

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.core.stream import Stream


class Waveform():
    """
    Primary interface for handling waveform data.

    Attributes
    ----------
    event_list : list of datetimes
        List of events to trigger on.
    filekey : DataFrame object or list
        Keys file names of local waveform data to their metadata.
    preload : Stream object or None
        Waveform data held in memory, up to configuration 'preload' days
        in length.
    times : dict
        Dictionary of UTCDateTime objects for keeping track of times.
    """

#     window : Stream object
#         Waveform data for the current processing chunk, up to configuration
#         'nsec' seconds in length.
    def __init__(self, detector, tstart, tend, event_list=None):
        """
        Load first round of waveform data and set attribute structure.

        Parameters
        ----------
        detector : Detector object
            Primary interface for handling detections.
        tstart : str
            Beginning time of span of interest.
        tend : str
            End time of span of interest.
        event_list : list, optional
            List of events to trigger on.

        """
        if event_list:
            self.event_list = event_list
        else:
            self.event_list = []
        self.filekey = []
        self.preload = None
        self.times = {'run_start': UTCDateTime(tstart),
                      'run_end': UTCDateTime(tend),
                      'preload_start': UTCDateTime(tstart),
                      'preload_end': UTCDateTime(tstart)}
        self._get_filekey(detector)
        self._preload_check(
            detector, self.times['run_start'], self.times['run_end'])

    def get_triggers(self, detector, window_start, window_end):  # force=False
        """
        Get triggers within a window.

        Parameters
        ----------
        detector : Detector object
            Primary interface for handling detections.
        window_start : UTCDateTime object
            Beginning time of window.
        window_end : UTCDateTime object
            End time of window.

        Returns
        -------
        list of Trigger objects

        """
        self._preload_check(detector, window_start, window_end)
        # !!! Download/load window
        # !!! -- Preload check
        # !!! Trigger
        # !!! Return list of Trigger objects
        return []

    def update(self, detector, tstart, tend, event_list=None):
        """
        Update the run start and end times for an existing instance.

        Parameters
        ----------
        detector : Detector object
            Primary interface for handling detections.
        tstart : str
            Beginning time of span of interest.
        tend : str
            End time of span of interest.
        event_list : list, optional
            List of events to trigger on.

        """
        if event_list:
            self.event_list = event_list
        else:
            self.event_list = []
        self.times['run_start'] = UTCDateTime(tstart)
        self.times['run_end'] = UTCDateTime(tend)
        self._get_filekey(detector)
        self._preload_check(
            detector, self.times['run_start'], self.times['run_end'])

    def _download_from_webservice(self, detector, window_start, window_end,
                                  do_filter_merge=True):
        """Download window of data from a FDSN webservice."""
        print('Waveform._download_from_webservice()')
        print(f'{detector} {window_start} {window_end} {do_filter_merge}')
        print(f'{self.times}')
        return True

    def _load_from_file(self, detector):
        """Load data from file."""
        stream = Stream()
        nets = detector.get('network')
        stas = detector.get('station')
        locs = detector.get('location')
        chas = detector.get('channel')
        tstart = self.times['preload_start']
        tend = self.times['preload_end']
        for i, sta in enumerate(stas):
            scnl = f'{nets[i]}.{sta}.{chas[i]}.{locs[i]}'
            flist_sub = self.filekey.query(
                f"scnl == '{scnl}' and starttime < '{tend}' "
                f"and endtime > '{tstart}'")['filename'].to_list()
            if len(flist_sub) > 0:
                for file in flist_sub:
                    stream = stream.extend(obspy.read(
                        file, starttime=tstart, endtime=tend))
        return stream

    def _get_filekey(self, detector):
        """
        Read/generate table that keys file names to a subset of metadata.

        Specifically, the metadata of importance are SCNL code, start time,
        and end time. This drastically improves local file read import
        times by reading the headers only once.

        !!! Need to develop a method to update this file instead of
        !!! starting over from scratch if the files have changed.

        Parameters
        ----------
        detector : Detector object
            Primary interface for handling detections.

        """
        if detector.get('server') == 'file':
            flname = os.path.join(detector.get('output_folder'),
                                  'filelist.csv')
            if os.path.exists(flname):
                # If filelist.csv exists, open it
                filekey = pd.read_csv(flname)
            else:
                # If it doesn't exist, create it
                print(f'Indexing {detector.get("filepattern")} files in'
                      f'{detector.get("searchdir")}')
                flist = list(itertools.chain.from_iterable(glob.iglob(
                    os.path.join(root, detector.get('filepattern'))
                    ) for root, _, _ in os.walk(detector.get('searchdir'))))
                filekey = pd.DataFrame(columns=[
                    'filename', 'scnl', 'starttime', 'endtime'], index=range(
                        len(flist)))
                for i, file in enumerate(flist):
                    if detector.get('verbose'):
                        print(file)
                    stmp = obspy.read(file, headonly=True)
                    filekey['filename'][i] = file
                    filekey['scnl'][i] = (
                        f'{stmp[0].stats.network}.{stmp[0].stats.station}.'
                        f'{stmp[0].stats.channel}.{stmp[0].stats.location}')
                    filekey['starttime'][i] = stmp[0].stats.starttime
                    filekey['endtime'][i] = stmp[-1].stats.endtime
                filekey.to_csv(path_or_buf=flname, index=False)
                print('Done indexing!')
                print(f'To force this index to update, remove {flname}')
            buf = (detector.get('maxdt') + detector.get('atrig')
                   + detector.get('ptrig') + 60)
            filekey = filekey.query(
                f"starttime < '{self.times['run_end']+buf}' and "
                f"endtime > '{self.times['run_start']-buf}'")
            self.filekey = filekey

    def _preload_check(self, detector, window_start, window_end):
        """Check if new data need to be 'preloaded' into memory."""
        if (detector.get('preload') > 0) and (len(self.filekey) > 0):
            if (window_end + detector.get('maxdt') >
                self.times['preload_end']) or (
                    window_start < self.times['preload_start']):
                if detector.get('verbose'):
                    print('Loading waveforms into memory...')
                self.times['preload_end'] = (np.min((
                    self.times['run_end'],
                    window_start + detector.get('preload')*86400))
                    + detector.get('atrig') + detector.get('maxdt'))
                self.times['preload_start'] = (window_start
                                               - detector.get('atrig'))
                self.preload = self._load_from_file(detector)
        elif (len(self.event_list) > 0) and (
                window_start >= self.times['preload_end']):
            sub_list = self.event_list[
                (self.event_list > window_start) & (
                    self.event_list < window_end + detector.get('nsec'))]
            if len(sub_list) > 1:
                self.times['preload_end'] = (window_end + sub_list[-1]
                                             - sub_list[0]
                                             + 5*detector.get('atrig'))
                self.times['preload_start'] = (window_start
                                               - 4*detector.get('atrig'))
                self.preload = self._download_from_webservice(
                  detector, self.times['preload_start'],
                  self.times['preload_end'], do_filter_merge=False)
            else:
                self.times['preload_start'] = window_start
                self.times['preload_end'] = window_end
                self.preload = None
