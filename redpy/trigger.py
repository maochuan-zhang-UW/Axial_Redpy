# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import glob, os, itertools

import numpy as np
import obspy
import pandas as pd

from obspy import UTCDateTime
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy.clients.earthworm import Client as EWClient
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.seedlink import Client as SeedLinkClient
from obspy.signal.trigger import coincidence_trigger
from scipy.fftpack import fft
from scipy.stats import kurtosis

from redpy.correlation import calculate_window

# !!! Be a better coder and address these warnings rather than ignore them
import warnings
warnings.filterwarnings("ignore")


def get_client(config):
    """
    Decides which Client to use to query data.

    Parameters
    ----------
    config : Config object
        Describes the run parameters.

    Returns
    -------
    client : Client object
        Handle to the appropriate Client.

    """

    if '://' not in config.get('server'):
        # Backward compatibility with previous setting files
        if '.' not in config.get('server'):
            client = FDSNClient(config.get('server'))
        else:
            client = EWClient(config.get('server'), config.get('port'))
    # New server syntax (more options and server and port on same variable)
    elif 'fdsnws://' in config.get('server'):
        server = config.get('server').split('fdsnws://',1)[1]
        client = FDSNClient(server)
    elif 'waveserver://' in config.get('server'):
        server_str = config.get('server').split('waveserver://',1)[1]
        try:
            server = server_str.split(':',1)[0]
            port = server_str.split(':',1)[1]
        except:
            server = server_str
            port = '16017'
        client = EWClient(server, int(port))
    elif 'seedlink://' in config.get('server'):
        server_str = config.get('server').split('seedlink://',1)[1]
        try:
            server = server_str.split(':',1)[0]
            port = server_str.split(':',1)[1]
        except:
            server = server_str
            port = '18000'
        client = SeedLinkClient(server, port=int(port), timeout=1)

    return client


def filter_merge(stmp, config):
    """
    Bandpass filter then merge data so each channel is in one Trace.

    This function fundamentally also controls how data gaps are handled. The
    ends are tapered to reduce the likelihood that they will be triggered on
    with STA/LTA.

    !!! This function could probably use some work, as there are still some
    !!! issues with how gaps are handled.

    Parameters
    ----------
    stmp : Stream object
        Stream containing Traces to be filtered/merged.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    stmp : Stream object
        Processed Stream, with each Trace corresponding to a unique channel.

    """

    # Replace -2**31 (Winston NaN token)
    for m in range(len(stmp)):
        stmp[m].data = np.where(stmp[m].data == -2**31, 0, stmp[m].data)

    # Bandpass filter, controlled by config
    stmp = stmp.filter('bandpass', freqmin=config.get('fmin'), freqmax=config.get('fmax'),
                                                    corners=2, zerophase=True)
    # !!! Demean? Detrend?

    # Hann window taper, with window length not to exceed the spacing between
    # consecutive triggers
    stmp = stmp.taper(0.05,type='hann',max_length=config.get('mintrig'))

    # Check for correct sampling rate
    for m in range(len(stmp)):
        if stmp[m].stats.sampling_rate != config.get('samprate'):
            stmp[m] = stmp[m].resample(config.get('samprate'))

    # Merge, filling gaps with zeroes
    stmp = stmp.merge(method=1, fill_value=0)

    return stmp


def append_empty(st, n, config):
    """
    Appends a Trace to the end of a Stream with SCNL information but no data.

    Parameters
    ----------
    st : Stream object
        Stream that will contain Traces for each channel.
    n : integer
        Index of channel within list.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    st : Stream object
        Input Stream with empty Trace appended.

    """

    nets = config.get('network').split(',')
    stas = config.get('station').split(',')
    locs = config.get('location').split(',')
    chas = config.get('channel').split(',')

    print(f'No data found for {nets[n]}.{stas[n]}.{chas[n]}.{locs[n]}')

    trtmp = Trace()
    trtmp.stats.sampling_rate = config.get('samprate')
    trtmp.stats.station = stas[n]
    trtmp.stats.channel = chas[n]
    trtmp.stats.network = nets[n]
    trtmp.stats.location = locs[n]

    st = st.append(trtmp.copy())

    return st


def get_filekey(tstart, tend, config):
    """
    Reads or generates a table that keys file names to a subset of metadata.

    Specifically, the metadata of importance are SCNL code, start time, and
    end time. This drastically improves local file read import times by
    reading the headers only once.

    !!! Need to develop a method to update this file instead of starting over
    !!! from scratch if the files have changed.

    Parameters
    ----------
    tstart : UTCDateTime object
        Beginning time of query.
    tend : UTCDateTime object
        End time of query.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    filekey : DataFrame object
        Keys file names of local waveform data to their metadata.

    """

    if config.get('server') == 'file':
        flname = os.path.join(config.get('output_folder'), 'filelist.csv')
        if os.path.exists(flname):
            # If filelist.csv exists, open it
            filekey = pd.read_csv(flname)
        else:
            # If it doesn't exist, create it
            print(f'Indexing {config.get("filepattern")} files in'
                  f'{config.get("searchdir")}')
            flist = list(itertools.chain.from_iterable(glob.iglob(
                os.path.join(root,config.get('filepattern'))) for root, dirs, files in \
                                                      os.walk(config.get('searchdir'))))
            filekey = pd.DataFrame(columns=['filename', 'scnl', 'starttime',
                      'endtime'], index=range(len(flist)))
            for n, f in enumerate(flist):
                if config.get('verbose'): print(f)
                stmp = obspy.read(f, headonly=True)
                filekey['filename'][n] = f
                filekey['scnl'][n] = '{}.{}.{}.{}'.format(
                                stmp[0].stats.network,stmp[0].stats.station,
                                stmp[0].stats.channel,stmp[0].stats.location)
                filekey['starttime'][n] = stmp[0].stats.starttime
                filekey['endtime'][n] = stmp[-1].stats.endtime
            filekey.to_csv(path_or_buf=flname, index=False)
            print('Done indexing!')
            print(f'To force this index to update, remove {flname}')
        buf = config.get('maxdt') + config.get('atrig') + config.get('ptrig') + 60
        filekey = filekey.query(
                    f"starttime < '{tend+buf}' and endtime > '{tstart-buf}'")
    else:
        filekey = []

    return filekey


def preload_data(tstart, tend, filekey, config):
    """
    Loads waveform data from disk into memory.

    Parameters
    ----------
    tstart : UTCDateTime object
        Beginning time of query.
    tend : UTCDateTime object
        End time of query.
    filekey : DataFrame object
        Keys file names of local waveform data to their metadata.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    preload_waveforms : Stream object
        Preloaded waveform data from files on disk.

    """
    preload_waveforms = Stream()
    nets = config.get('network').split(',')
    stas = config.get('station').split(',')
    locs = config.get('location').split(',')
    chas = config.get('channel').split(',')
    # Load up waveform data from tstart to tend into memory
    for n in range(len(stas)):
        # Format SCNL string
        scnl = f'{nets[n]}.{stas[n]}.{chas[n]}.{locs[n]}'
        # Find list of files to load
        flist_sub = filekey.query(
            f"scnl == '{scnl}' and starttime < '{tend}' "
            f"and endtime > '{tstart}'")['filename'].to_list()
        if len(flist_sub) > 0:
            # Fully load data from file(s)
            for f in flist_sub:
                stmp = obspy.read(f, starttime=tstart, endtime=tend)
                preload_waveforms = preload_waveforms.extend(stmp)
    return preload_waveforms


def preload_check(window_start_time, window_end_time, preload_end_time,
                  run_end_time, filekey, config, preload_waveforms=[],
                  event_list=[]):
    """
    Checks if new data need to be 'preloaded' into memory.

    Nominally this is to load data from local files into memory, however, by
    passing an event_list, if more than one event occurs within 'nsec'
    seconds of the window_start_time, data containing those events will be
    downloaded from a server. This prevents excessive calls to the server for
    many events in a short amount of time.

    Parameters
    ----------
    window_start_time : UTCDateTime object
        Start time of window to process.
    window_end_time : UTCDateTime object
        End time of window to process.
    preload_end_time : UTCDateTime object
        End time of preloaded waveforms.
    run_end_time : UTCDateTime object
        End time of full span of interest.
    filekey : DataFrame object
        Keys file names of local waveform data to their metadata.
    config : Config object
        Describes the run parameters.
    preload_waveforms : Stream object, optional
        Stream containing waveforms 'preloaded' into memory.
    event_list : ndarray of UTCDateTime objects, optional
        List of catalog events to add.

    Returns
    -------
    preload_waveforms : Stream object
        Stream containing waveforms 'preloaded' into memory.
    preload_end_time : UTCDateTime object
        End time of preloaded waveforms.

    """
    if (config.get('preload') > 0) and (len(filekey) > 0):
        if window_end_time+config.get('maxdt') > preload_end_time:
            if config.get('verbose'):
                print('Loading waveforms into memory...')
            preload_end_time = np.min(
                (run_end_time, window_start_time + config.get('preload')*86400)
                ) + config.get('atrig') + config.get('maxdt')
            preload_waveforms = preload_data(window_start_time - config.get('atrig'),
                                             preload_end_time, filekey, config)
    elif (len(event_list) > 0) and (window_start_time >= preload_end_time):
        sub_list = event_list[(event_list > window_start_time) &
                              (event_list < window_end_time
                               + config.get('nsec'))]
        if len(sub_list) > 1:
            preload_end_time = window_end_time + (sub_list[-1] - sub_list[0])
            preload_waveforms = get_data(window_start_time - 4*config.get('atrig'),
                                         preload_end_time + 5*config.get('atrig'),
                                         filekey, [], config,
                                         do_filter_merge=False)
        else:
            preload_end_time = window_end_time
            preload_waveforms = []
    return preload_waveforms, preload_end_time


def initial_data_preload(run_start_time, run_end_time, config):
    """
    Handles the first preload when starting a run.

    Parameters
    ----------
    run_start_time : UTCDateTime object
        Start time of full span of interest.
    run_end_time : UTCDateTime object
        End time of full span of interest.
    config : Config object
        Describes the run parameters

    Returns
    -------
    filekey : DataFrame object
        Keys file names of local waveform data to their metadata.
    preload_waveforms : Stream object
        Stream containing waveforms 'preloaded' into memory.
    preload_end_time : UTCDateTime object
        End time of preloaded waveforms.

    """
    filekey = get_filekey(run_start_time, run_end_time, config)
    preload_waveforms, preload_end_time = preload_check(
        run_start_time, run_end_time, run_start_time, run_end_time,
        filekey, config)
    return filekey, preload_waveforms, preload_end_time


def get_data(tstart, tend, filekey, preload_waveforms, config,
             do_filter_merge=True):
    """
    Download data from web or read from file.

    A note on SAC/miniSEED files: as this makes no assumptions about the
    naming scheme of your data files, please ensure that your headers contain
    the correct SCNL information!

    Parameters
    ----------
    tstart : UTCDateTime object
        Beginning time of query.
    tend : UTCDateTime object
        End time of query.
    filekey : DataFrame object
        Contains filenames, SCNL codes, start times, and end times of waveform
        files on disk. Empty if querying from a server.
    preload_waveforms : Stream object
        Preloaded waveform data from files on disk. Empty if querying from a
        server.
    config : Config object
        Describes the run parameters.
    do_filter_merge : bool, optional
        If True, runs filter_merge() on Streams.

    Returns
    -------
    st : Stream object
        Stream containing continuous, filtered Traces for each channel.

    """

    nets = config.get('network').split(',')
    stas = config.get('station').split(',')
    locs = config.get('location').split(',')
    chas = config.get('channel').split(',')

    st = Stream()

    # Only true if config.get('server') == file and config.get('preload') > 0
    if len(preload_waveforms) > 0:

        # Slice and put in correct order
        preload_waveforms = preload_waveforms.slice(starttime=tstart,
                                                    endtime=tend+config.get('maxdt'))

        for n in range(len(stas)):

            # Format SCNL string
            scnl = f'{nets[n]}.{stas[n]}.{chas[n]}.{locs[n]}'

            stmp = Stream()
            for m in range(len(preload_waveforms)):
                if scnl == '{}.{}.{}.{}'.format(
                        preload_waveforms[m].stats.network,
                        preload_waveforms[m].stats.station,
                        preload_waveforms[m].stats.channel,
                        preload_waveforms[m].stats.location):
                    stmp = stmp.extend([preload_waveforms[m]])

            if len(stmp) > 0:
                if do_filter_merge:
                    stmp = filter_merge(stmp, config)
                st = st.extend(stmp.copy())
            else:
                st = append_empty(st, n, config)

    # Only true if config.get('server') == file and config.get('preload') == 0
    elif len(filekey) > 0:

        # Load directly from file in correct order

        filekey_sub = filekey.query(f"starttime < '{tend+config.get('maxdt')}' \
                                      and endtime > '{tstart}'")

        for n in range(len(stas)):

            # Format SCNL string
            scnl = f'{nets[n]}.{stas[n]}.{chas[n]}.{locs[n]}'

            # Find list of files to load
            flist_sub = filekey_sub.query(
                                    f"scnl == '{scnl}'")['filename'].to_list()

            if len(flist_sub) > 0:

                # Fully load data from file(s)
                stmp = Stream()
                for f in flist_sub:
                    stmp = stmp.extend(obspy.read(f, starttime=tstart,
                                                      endtime=tend+config.get('maxdt')))
                if do_filter_merge:
                    stmp = filter_merge(stmp, config)
                st = st.extend(stmp.copy())

            # If no data found, append an empty trace
            else:
                st = append_empty(st, n, config)

    # config.get('server') != file
    else:

        client = get_client(config)

        for n in range(len(stas)):
            try:
                stmp = client.get_waveforms(nets[n], stas[n], locs[n],
                                      chas[n], tstart, tend+config.get('maxdt'))
                if do_filter_merge:
                    stmp = filter_merge(stmp, config)
            except (obspy.clients.fdsn.header.FDSNException):
                # Try querying again in case timed out on accident
                try:
                    stmp = client.get_waveforms(nets[n], stas[n], locs[n],
                                          chas[n], tstart, tend+config.get('maxdt'))
                    if do_filter_merge:
                        stmp = filter_merge(stmp, config)
                except:
                    stmp = append_empty(Stream(), n, config)

            # Enforce location code
            for i in range(len(stmp)):
                stmp[i].stats.location = locs[n]

            # Last check for length; catches problem with empty waveserver
            if len(stmp) < 1:
                st = append_empty(st, n, config)
            else:
                st.extend(stmp.copy())

    # Edit 'start' time if using offset option
    if config.get('maxdt'):
        dts = np.fromstring(config.get('offset'), sep=',')
        for n, tr in enumerate(st):
            tr.stats.starttime = tr.stats.starttime-dts[n]

    if do_filter_merge:
        st = st.trim(starttime=tstart, endtime=tend, pad=True, fill_value=0)

    return st


def trigger(st, rtable, config, event=None):
    """
    Run triggering algorithm on a stream of data.

    If an event to be added by force is passed, the coincidence triggering
    algorithm is still run and the closest trigger's 'maxratio' is used to
    allow the event to be expired as close as possible to the time it would
    have were it allowed to trigger normally.

    Parameters
    ----------
    st : Stream object
        Stream containing continuous, filtered Traces for each channel.
    rtable : Table object
        Handle to the Repeaters table.
    config : Config object
        Describes the run parameters.
    event : UTCDateTime object, optional
        Catalog event to add by force.

    Returns
    -------
    trigs : Stream object
        Triggered events with data from each channel concatenated.

    """

    t = st[0].stats.starttime

    cft = coincidence_trigger(config.get('trigalg'), config.get('trigon'), config.get('trigoff'), st.copy(),
        config.get('nstac'), sta=config.get('swin'), lta=config.get('lwin'), details=True)
    if event:
        ttimes = [event]
        ratios = [0.0]
        bestmatch = config.get('mintrig')
        for c in cft:
            if np.abs(c['time']-event) < bestmatch:
                bestmatch = np.abs(c['time']-event)
                ratios = [np.max(c['cft_peaks'])-config.get('trigoff')]
    else:
        ttimes = [cft[n]['time'] for n in range(len(cft))]
        ratios = [
            np.max(cft[n]['cft_peaks'])-config.get('trigoff') for n in range(len(cft))]

    if len(ttimes) > 0:

        ind = 0

        # Convert ptime from time of last trigger to seconds before start time
        if rtable.attrs.ptime:
            ptime = (UTCDateTime(rtable.attrs.ptime) - t)
        else:
            ptime = -config.get('mintrig')

        # Loop over triggers
        for n, ttime in enumerate(ttimes):

            # Enforce minimum time between previous known trigger, edges of st
            if (ttime >= t + config.get('atrig')) and (ttime >= t + ptime +
                config.get('mintrig')) and (ttime < t + len(st[0].data)/config.get('samprate') -
                2*config.get('atrig')):

                # Update ptime
                ptime = ttime - t

                # Cut out a copy from st with a few samples of padding
                tr = st.slice(ttime - config.get('ptrig'), ttime + config.get('atrig') + \
                                                        2/config.get('samprate')).copy()

                # Trim, pad with zeros
                tr = tr.trim(ttime - config.get('ptrig'), ttime + config.get('atrig') + \
                                       2/config.get('samprate'), pad=True, fill_value=0)

                for s in range(len(tr)):

                    # Cut out exact number of samples
                    tr[s].data = tr[s].data[0:config.get('wshape')]

                    # Demean
                    tr[s].data -= np.mean(tr[s].data)

                    # !!! Preserve zeroes (replace demean)
                    #tr[s].data[tr[s].data!=0] -= np.mean(
                    #                              tr[s].data[tr[s].data!=0])

                    # Append
                    if s > 0:
                        tr[0].data = np.append(tr[0].data, tr[s].data)

                # Set 'maxratio' for orphan expiration
                tr[0].stats.maxratio = ratios[n]

                # Append to trigs list
                if ind is 0:
                    trigs = Stream(tr[0])
                    ind = ind+1
                else:
                    trigs = trigs.append(tr[0])

        if ind is 0:
            return []
        else:
            rtable.attrs.ptime = (t + ptime).isoformat()
            return trigs
    else:
        return []


def load_and_trigger(rtable, window_start_time, window_end_time, filekey,
                     preload_waveforms, config, event=None):
    """
    Combines getting data for a time window and triggering on that data.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    window_start_time : UTCDateTime object
        Start time of window to process.
    window_end_time : UTCDateTime object
        End time of window to process.
    filekey : DataFrame object
        Keys file names of local waveform data to their metadata.
    preload_waveforms : Stream object
        Preloaded waveform data from files on disk.
    config : Config object
        Describes the run parameters.
    event : UTCDateTime object, optional
        Catalog event to add by force.

    Returns
    -------
    alltrigs : Stream object
        Triggered events with data from each channel concatenated.

    """
    if config.troubleshoot:
        st = get_data(window_start_time-config.get('atrig'), window_end_time,
                                    filekey, preload_waveforms, config)
        alltrigs = trigger(st, rtable, config, event=event)
    else:
        try:
            st = get_data(window_start_time-config.get('atrig'), window_end_time,
                                        filekey, preload_waveforms, config)
            alltrigs = trigger(st, rtable, config, event=event)
        except KeyboardInterrupt:
            print('\nManually interrupting!\n')
            raise KeyboardInterrupt
        except:
            print(('Could not download or trigger data... '
                   'troubleshoot with -t'))
            alltrigs = []
    return alltrigs


def clean_triggers(alltrigs, config, event=None):
    """
    Cleans triggers of data spikes, calibration pulses, and teleseisms.

    Specifically, it attempts to weed out spikes and analog calibration pulses
    using kurtosis and outlier ratios; checks for teleseisms that have very
    low frequency index. If force triggering with a specific event, a warning
    will be shown if the event would normally be categorized as junk, but
    otherwise is allowed to pass the tests.

    Parameters
    ----------
    alltrigs : Stream object
        Triggered events with data from each channel concatenated.
    config : Config object
        Describes the run parameters.
    event : UTCDateTime object, optional
        Catalog event to add by force.

    Returns
    -------
    trigs : Stream object
        Events from alltrigs that passed all tests.
    junk : Stream object
        Events that failed a test.
    jtype : integer list
        List with codes corresponding to which tests failed:
            0: FI too low (possible teleseism)
            1: Kurtosis in time or frequency domains too high (spikes,
                sinewave noise)
            2: Both tests failed

    # !!! Future: Pass more information with jtype, store more in jtable

    """

    trigs = Stream()
    junk = Stream()
    jtype = []

    # Loop over triggers
    for i in range(len(alltrigs)):

        njunk = 0
        ntele = 0

        # Get FI
        windowCoeff, windowFFT, windowFI = calculate_window(
            alltrigs[i].data, int(config.get('ptrig')*config.get('samprate')), config)

        # Loop over channels
        for n in range(config.get('nsta')):

            # Check FI
            fi = windowFI[n]
            if fi<config.get('telefi'):
                ntele+=1

            # Get channel waveform
            dat = alltrigs[i].data[n*config.get('wshape'):(n+1)*config.get('wshape')]

            # Cut out kurtosis window surrounding initial trigger
            datcut = dat[range(int((config.get('ptrig')-config.get('kurtwin')/2)*config.get('samprate')),
                               int((config.get('ptrig')+config.get('kurtwin')/2)*config.get('samprate')))]

            # If not filled with zeros
            if np.sum(np.abs(dat))!=0.0:

                # Calculate kurtosis in window
                k = kurtosis(datcut)

                # Compute kurtosis of frequency amplitude spectrum next
                datf = np.absolute(fft(dat))
                kf = kurtosis(datf)

                # Calculate outlier ratio using z ((data-median)/mad)
                mad = np.nanmedian(np.absolute(dat - np.nanmedian(dat)))
                z = (dat-np.median(dat))/mad

                # Outliers have z > 4.45
                oratio = len(z[z>4.45])/np.array(len(z)).astype(float)

                if (k >= config.get('kurtmax')) or (oratio >= config.get('oratiomax')) or (
                        kf >= config.get('kurtfmax')):
                    njunk+=1

        # Allow if there are enough good stations to correlate
        if njunk <= (config.get('nsta')-config.get('ncor')) and ntele <= config.get('teleok'):
            trigs.append(alltrigs[i])
        else:
            junk.append(alltrigs[i])
            if njunk > 0:
                if ntele > 0:
                    jtype.append(2) # Failed both
                else:
                    jtype.append(1) # Failed kurtosis
            else:
                jtype.append(0) # Failed FI
    if event:
        trigs = alltrigs
        if len(junk) > 0:
            print(f'Forced trigger possible type-{jytpe} junk?')
            junk = Stream()
            jytpe = []
    return trigs, junk, jtype
