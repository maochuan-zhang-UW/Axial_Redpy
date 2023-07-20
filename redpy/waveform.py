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
import time

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.earthworm import Client as EWClient
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.seedlink import Client as SeedLinkClient
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.signal.trigger import coincidence_trigger

from redpy.trigger import Trigger


class Waveform():
    """
    Primary interface for handling waveform data.

    Attributes
    ----------
    event_list : list of UTCDateTimes
        List of events to trigger on.
    filekey : DataFrame object or list
        Keys file names of local waveform data to their metadata.
    preload : Stream object or None
        Waveform data held in memory, up to configuration 'preload' days
        in length.
    times : dict
        Dictionary of UTCDateTime objects for keeping track of times.
    """

    def __init__(self, detector, tstart, tend, event_list=None):
        """
        Set attribute structure and do initial preload if necessary.

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
        if event_list is not None:
            self.event_list = np.array(
                [UTCDateTime(event) for event in event_list])
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

    def get_data(self, detector, window_start, window_end):
        """
        Download or load a window of data.

        Parameters
        ----------
        detector : Detector object
            Primary interface for handling detections.
        window_start : UTCDateTime object
            Start of window to get data for.
        window_end : UTCDateTime object
            End of window to get data for.

        Returns
        -------
        Stream object
            Waveforms for a window of time.
        float
            Seconds spent getting data.

        """
        t_get = time.time()
        window_start = window_start - detector.get('atrig')
        window_end = window_end + detector.get('atrig')
        self._preload_check(detector, window_start, window_end)
        if self.preload:
            stream = self._extract_from_preload(
                detector, window_start, window_end)
        elif detector.get('server') == 'file':
            stream = self._load_from_file(detector, window_start, window_end)
        else:
            stream = _download_from_client(detector, window_start, window_end)
        stream = _filter_merge(detector, stream)
        if detector.get('maxdt'):
            offsets = np.fromstring(detector.get('offset'), sep=',')
            for i, trace in enumerate(stream):
                trace.stats.starttime = trace.stats.starttime - offsets[i]
        stream = stream.trim(starttime=window_start, endtime=window_end,
                             pad=True, fill_value=0)
        return stream, time.time()-t_get

    def get_triggers(self, detector, stream, force=False):
        """
        Get triggers within a window.

        Parameters
        ----------
        detector : Detector object
            Primary interface for handling detections.
        stream : Stream object
            Waveforms for a window of time to trigger on.

        Returns
        -------
        list of Trigger objects

        """
        start_time = stream[0].stats.starttime
        end_time = stream[0].stats.endtime
        triggers = coincidence_trigger(
            detector.get('trigalg'), detector.get('trigon'),
            detector.get('trigoff'), stream.copy(),
            detector.get('nstac'), sta=detector.get('swin'),
            lta=detector.get('lwin'), details=True)
        # Remove triggers from coincident gaps
        triggers = _gap_check(detector, triggers, stream)
        if force:
            trig_times = self.event_list[(self.event_list > start_time) & (
                self.event_list < end_time)]
            ratios = np.zeros(len(trig_times))
            for i, event in enumerate(trig_times):
                bestmatch = detector.get('mintrig')
                ratios[i] = 0
                for trig in triggers:
                    if np.abs(trig['time']-event) < bestmatch:
                        bestmatch = np.abs(trig['time']-event)
                        ratios[i] = np.max(
                            trig['cft_peaks'])-detector.get('trigoff')
        else:
            trig_times = np.array([trig['time'] for trig in triggers])
            ratios = np.array([np.max(trig['cft_peaks'])-detector.get(
                'trigoff') for trig in triggers])
        trig_list = []
        if len(trig_times) > 0:
            for i, trig in enumerate(trig_times):
                # Enforce time at edges of stream
                if (start_time + detector.get('atrig')
                        <= trig < end_time - detector.get('atrig')):
                    trig_list.append(
                        Trigger(detector, trig, ratios[i], stream))
        return trig_list

    def update_span(self, detector, tstart, tend, event_list=None):
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
        if event_list is not None:  # pragma: no cover
            self.event_list = np.array(
                [UTCDateTime(event) for event in event_list])
        else:
            self.event_list = []
        self.times['run_start'] = UTCDateTime(tstart)
        self.times['run_end'] = UTCDateTime(tend)
        self._get_filekey(detector)
        self._preload_check(
            detector, self.times['run_start'], self.times['run_end'])

    def _extract_from_preload(self, detector, window_start, window_end):
        """Extract waveforms from preload."""
        stream = Stream()
        preload = self.preload.slice(
            starttime=window_start, endtime=window_end+detector.get('maxdt'))
        for scnl in _scnl_list_from_config(detector):
            for trace in preload:
                if scnl == _scnl_from_trace(trace):
                    stream.append(trace.copy())
        return stream

    def _get_filekey(self, detector):
        """
        Read/generate table that keys file names to a subset of metadata.

        Specifically, the metadata of importance are SCNL code, start time,
        and end time. This drastically improves local file read import
        times by reading the headers only once.

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
                    filekey['scnl'][i] = _scnl_from_trace(stmp[0])
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

    def _load_from_file(self, detector, window_start, window_end):
        """Load data from file."""
        stream = Stream()
        for scnl in _scnl_list_from_config(detector):
            flist_sub = self.filekey.query(
                f"scnl == '{scnl}' and starttime < '{window_end}' "
                f"and endtime > '{window_start}'")['filename'].to_list()
            if len(flist_sub) > 0:
                for file in flist_sub:
                    stream = stream.extend(obspy.read(
                        file, starttime=window_start, endtime=window_end))
        return stream

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
                self.preload = self._load_from_file(
                    detector, self.times['preload_start'],
                    self.times['preload_end'])
        elif (len(self.event_list) > 0) and (
                window_start >= self.times['preload_end']):
            sub_list = self.event_list[
                (self.event_list > window_start) & (
                    self.event_list < window_start + detector.get('nsec'))]
            if len(sub_list) > 1:
                self.times['preload_end'] = (
                    window_start + (sub_list[-1] - sub_list[0])
                    + 5*detector.get('atrig'))
                self.times['preload_start'] = (window_start
                                               - 4*detector.get('atrig'))
                self.preload = _download_from_client(
                    detector, self.times['preload_start'],
                    self.times['preload_end'])
            else:  # pragma: no cover
                self.times['preload_start'] = window_start
                self.times['preload_end'] = window_start
                self.preload = None


def _append_empty(detector, stream, scnl):
    """
    Append a Trace to the end of a Stream with SCNL information but no data.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    stream : Stream object
        Stream that will contain Traces for each channel.
    scnl : str
        SCNL information to add.

    Returns
    -------
    Stream object
        Input Stream with empty Trace appended.

    """
    print(f'No data found for {scnl}')
    trace = Trace()
    trace.stats.sampling_rate = detector.get('samprate')
    (trace.stats.network, trace.stats.station,
     trace.stats.channel, trace.stats.location) = scnl.split('.')
    return stream.append(trace)


def _download_from_client(detector, window_start, window_end):
    """Download window of data from a Client (e.g., FDSN webservice)."""
    stream = Stream()
    client = _get_client(detector)
    for nslc in _nslc_list_from_config(detector):
        try:
            stmp = client.get_waveforms(
                *nslc.split('.'), window_start,
                window_end+detector.get('maxdt'))
        except obspy.clients.fdsn.header.FDSNException as exc:
            if 'client does not have a dataselect service' in str(exc):
                raise exc  # Case where service is down rather than no data
            # Try querying one more time
            try:
                stmp = client.get_waveforms(
                    *nslc.split('.'), window_start,
                    window_end+detector.get('maxdt'))
            except Exception:
                stmp = []
        if stmp:
            for trace in stmp:
                trace.stats.location = nslc.split('.')[2]
            stream.extend(stmp.copy())
    return stream


def _filter_merge(detector, stream):
    """Filter and merge so data from each channel is in a single Trace."""
    for trace in stream:
        trace.data = np.where(trace.data == -2**31, 0, trace.data)
    stream = stream.detrend()
    stream = stream.merge(method=1, fill_value=0)
    zeros = [np.where(stream[i].data == 0)[0] for i in range(len(stream))]
    stream = stream.filter('bandpass', freqmin=detector.get('fmin'),
                           freqmax=detector.get('fmax'), corners=2,
                           zerophase=True)
    for i, trace in enumerate(stream):
        trace.data[zeros[i]] = 0
        if trace.stats.sampling_rate != detector.get('samprate'):
            trace = trace.resample(detector.get('samprate'))
    stream_scnls = np.array([_scnl_from_trace(trace) for trace in stream])
    ordered_stream = Stream()
    for scnl in _scnl_list_from_config(detector):
        if len(stream_scnls) > 0:
            idx = np.where(stream_scnls == scnl)
            if len(idx[0]) > 0:
                ordered_stream.append(stream[idx[0][0]])
            else:
                ordered_stream = _append_empty(detector, ordered_stream, scnl)
        else:
            ordered_stream = _append_empty(detector, ordered_stream, scnl)
    return ordered_stream


def _gap_check(detector, triggers, stream):
    """Remove triggers that occur right after a gap."""
    winlen = detector.get('winlen')
    winstart = 0.01*winlen/detector.get('samprate')
    winend = 0.99*winlen/detector.get('samprate')
    for trig in triggers.copy():
        n_gaps = 0
        for waves in stream:
            if waves.id in trig['trace_ids']:
                pretrig = waves.slice(trig['time']-winstart, trig['time']).data
                window = waves.slice(trig['time']-winstart,
                                     trig['time']+winend)
                if (len(np.where(pretrig == 0)[0]) >= 1) or (
                        (len(window) > winlen/5) and (
                            np.sort(np.abs(window))[int(winlen/5)] == 0)):
                    n_gaps += 1
        if n_gaps >= detector.get('nstac'):
            triggers.remove(trig)
    return triggers


def _get_client(detector):
    """
    Decide which Client to use to query data.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    Returns
    -------
    Client object
        Handle to the appropriate Client.

    Raises
    ------
    ValueError
        If server string is not recognized.

    """
    if '://' not in detector.get('server'):
        if '.' not in detector.get('server'):
            return FDSNClient(detector.get('server'))
        return EWClient(detector.get('server'), detector.get('port'))
    if 'fdsnws://' in detector.get('server'):
        server = detector.get('server').split('fdsnws://', 1)[1]
        return FDSNClient(server)
    if 'waveserver://' in detector.get('server'):
        server_str = detector.get('server').split('waveserver://', 1)[1]
        if len(server_str.split(':', 1)) == 2:
            server, port = server_str.split(':', 1)
        else:
            server = server_str
            port = '16017'
        return EWClient(server, int(port))
    if 'seedlink://' in detector.get('server'):
        server_str = detector.get('server').split('seedlink://', 1)[1]
        if len(server_str.split(':', 1)) == 2:
            server, port = server_str.split(':', 1)
        else:
            server = server_str
            port = '18000'
        return SeedLinkClient(server, port=int(port), timeout=1)
    raise ValueError(f'Unrecognized server: {detector.get("server")}')


def _nslc_list_from_config(detector):
    """Define list of NSLCs from configuration."""
    nets = detector.get('network')
    stas = detector.get('station')
    locs = detector.get('location')
    chas = detector.get('channel')
    return [
        f'{nets[i]}.{sta}.{locs[i]}.{chas[i]}' for i, sta in enumerate(stas)]


def _scnl_from_trace(trace):
    """Define SCNL from Trace header."""
    return (f'{trace.stats.network}.{trace.stats.station}.'
            f'{trace.stats.channel}.{trace.stats.location}')


def _scnl_list_from_config(detector):
    """Define list of SCNLs from configuration."""
    nets = detector.get('network')
    stas = detector.get('station')
    locs = detector.get('location')
    chas = detector.get('channel')
    return [
        f'{nets[i]}.{sta}.{chas[i]}.{locs[i]}' for i, sta in enumerate(stas)]
