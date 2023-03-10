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


def get_client(opt):
    """
    Decides which Client to use to query data.
    
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    client : Client object
        Handle to the appropriate Client.
    
    """
    
    if '://' not in opt.server:
        # Backward compatibility with previous setting files
        if '.' not in opt.server:
            client = FDSNClient(opt.server)
        else:
            client = EWClient(opt.server, opt.port)
    # New server syntax (more options and server and port on same variable)
    elif 'fdsnws://' in opt.server:
        server = opt.server.split('fdsnws://',1)[1]
        client = FDSNClient(server)
    elif 'waveserver://' in opt.server:
        server_str = opt.server.split('waveserver://',1)[1]
        try:
            server = server_str.split(':',1)[0]
            port = server_str.split(':',1)[1]
        except:
            server = server_str
            port = '16017'
        client = EWClient(server, int(port))
    elif 'seedlink://' in opt.server:
        server_str = opt.server.split('seedlink://',1)[1]
        try:
            server = server_str.split(':',1)[0]
            port = server_str.split(':',1)[1]
        except:
            server = server_str
            port = '18000'
        client = SeedLinkClient(server, port=int(port), timeout=1)
        
    return client


def filter_merge(stmp, opt):
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
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    stmp : Stream object
        Processed Stream, with each Trace corresponding to a unique channel.
    
    """
    
    # Replace -2**31 (Winston NaN token)
    for m in range(len(stmp)):
        stmp[m].data = np.where(stmp[m].data == -2**31, 0, stmp[m].data)
    
    # Bandpass filter, controlled by opt
    stmp = stmp.filter('bandpass', freqmin=opt.fmin, freqmax=opt.fmax,
                                                    corners=2, zerophase=True)
    # !!! Demean? Detrend?
    
    # Hann window taper, with window length not to exceed the spacing between
    # consecutive triggers
    stmp = stmp.taper(0.05,type='hann',max_length=opt.mintrig)
    
    # Check for correct sampling rate
    for m in range(len(stmp)):
        if stmp[m].stats.sampling_rate != opt.samprate:
            stmp[m] = stmp[m].resample(opt.samprate)
    
    # Merge, filling gaps with zeroes
    stmp = stmp.merge(method=1, fill_value=0)
    
    return stmp


def append_empty(st, n, opt):
    """
    Appends a Trace to the end of a Stream with SCNL information but no data.
    
    Parameters
    ----------
    st : Stream object
        Stream that will contain Traces for each channel.
    n : integer
        Index of channel within list.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    st : Stream object
        Input Stream with empty Trace appended.
    
    """
    
    nets = opt.network.split(',')
    stas = opt.station.split(',')
    locs = opt.location.split(',')
    chas = opt.channel.split(',')    
    
    print(f'No data found for {nets[n]}.{stas[n]}.{chas[n]}.{locs[n]}')
    
    trtmp = Trace()
    trtmp.stats.sampling_rate = opt.samprate
    trtmp.stats.station = stas[n]
    trtmp.stats.channel = chas[n]
    trtmp.stats.network = nets[n]
    trtmp.stats.location = locs[n]
    
    st = st.append(trtmp.copy())
    
    return st


def get_filekey(tstart, tend, opt):
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
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    filekey : DataFrame object
        Keys file names of local waveform data to their metadata.
    
    """
    
    if opt.server == 'file':
        flname = os.path.join(opt.output_folder, 'filelist.csv')
        if os.path.exists(flname):
            # If filelist.csv exists, open it
            filekey = pd.read_csv(flname)
        else:
            # If it doesn't exist, create it
            print(f'Indexing {opt.filepattern} files in {opt.searchdir}')
            flist = list(itertools.chain.from_iterable(glob.iglob(
                os.path.join(root,opt.filepattern)) for root, dirs, files in \
                                                      os.walk(opt.searchdir)))
            filekey = pd.DataFrame(columns=['filename', 'scnl', 'starttime',
                      'endtime'], index=range(len(flist)))
            for n, f in enumerate(flist):
                if opt.verbose: print(f)
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
        buf = opt.maxdt + opt.atrig + opt.ptrig + 60
        filekey = filekey.query(
                    f"starttime < '{tend+buf}' and endtime > '{tstart-buf}'")
    else:
        filekey = []
    
    return filekey


def preload_data(tstart, tend, filekey, opt):
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
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    preload_waveforms : Stream object
        Preloaded waveform data from files on disk.
    
    """
    preload_waveforms = Stream()
    nets = opt.network.split(',')
    stas = opt.station.split(',')
    locs = opt.location.split(',')
    chas = opt.channel.split(',')
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
                  run_end_time, filekey, opt, preload_waveforms=[],
                  event_list=[]):
    """
    Checks if new data need to be 'preloaded' into memory.
    
    Nominally this is to load data from local files into memory, however, by
    passing an event_list, if more than one event occurs within opt.nsec
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
    opt : Options object
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
    if (opt.preload > 0) and (len(filekey) > 0):
        if window_end_time+opt.maxdt > preload_end_time:
            if opt.verbose: print('Loading waveforms into memory...')
            preload_end_time = np.min(
                (run_end_time, window_start_time + opt.preload*86400)
                ) + opt.atrig + opt.maxdt
            preload_waveforms = preload_data(window_start_time - opt.atrig,
                                             preload_end_time, filekey, opt)
    elif (len(event_list) > 0) and (window_start_time >= preload_end_time):
        sub_list = event_list[(event_list > window_start_time) &
                              (event_list < window_end_time + opt.nsec)]
        if len(sub_list) > 1:
            preload_end_time = window_end_time + (sub_list[-1] - sub_list[0])
            preload_waveforms = get_data(window_start_time - 4*opt.atrig,
                                         preload_end_time + 5*opt.atrig,
                                         filekey, [], opt,
                                         do_filter_merge=False)
        else:
            preload_end_time = window_end_time
            preload_waveforms = []
    return preload_waveforms, preload_end_time


def initial_data_preload(run_start_time, run_end_time, opt):
    """
    Handles the first preload when starting a run.
    
    Parameters
    ----------
    run_start_time : UTCDateTime object
        Start time of full span of interest.
    run_end_time : UTCDateTime object
        End time of full span of interest.
    opt : Options object
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
    filekey = get_filekey(run_start_time, run_end_time, opt)
    preload_waveforms, preload_end_time = preload_check(
        run_start_time, run_end_time, run_start_time, run_end_time,
        filekey, opt)
    return filekey, preload_waveforms, preload_end_time


def get_data(tstart, tend, filekey, preload_waveforms, opt,
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
    opt : Options object
        Describes the run parameters.
    do_filter_merge : bool, optional
        If True, runs filter_merge() on Streams.
    
    Returns
    -------
    st : Stream object
        Stream containing continuous, filtered Traces for each channel.
    
    """
    
    nets = opt.network.split(',')
    stas = opt.station.split(',')
    locs = opt.location.split(',')
    chas = opt.channel.split(',')
    
    st = Stream()
    
    # Only true if opt.server == file and opt.preload > 0
    if len(preload_waveforms) > 0:
        
        # Slice and put in correct order
        preload_waveforms = preload_waveforms.slice(starttime=tstart,
                                                    endtime=tend+opt.maxdt)
        
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
                    stmp = filter_merge(stmp, opt)
                st = st.extend(stmp.copy())
            else:
                st = append_empty(st, n, opt)
    
    # Only true if opt.server == file and opt.preload == 0
    elif len(filekey) > 0:
        
        # Load directly from file in correct order
        
        filekey_sub = filekey.query(f"starttime < '{tend+opt.maxdt}' \
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
                                                      endtime=tend+opt.maxdt))
                if do_filter_merge:
                    stmp = filter_merge(stmp, opt)
                st = st.extend(stmp.copy())
             
            # If no data found, append an empty trace
            else:
                st = append_empty(st, n, opt)
    
    # opt.server != file
    else:
        
        client = get_client(opt)
        
        for n in range(len(stas)):
            try:
                stmp = client.get_waveforms(nets[n], stas[n], locs[n],
                                      chas[n], tstart, tend+opt.maxdt)
                if do_filter_merge:
                    stmp = filter_merge(stmp, opt)
            except (obspy.clients.fdsn.header.FDSNException):
                # Try querying again in case timed out on accident
                try:
                    stmp = client.get_waveforms(nets[n], stas[n], locs[n],
                                          chas[n], tstart, tend+opt.maxdt)
                    if do_filter_merge:
                        stmp = filter_merge(stmp, opt)
                except:
                    stmp = append_empty(Stream(), n, opt)
            
            # Enforce location code
            for i in range(len(stmp)):
                stmp[i].stats.location = locs[n]
            
            # Last check for length; catches problem with empty waveserver
            if len(stmp) < 1:
                st = append_empty(st, n, opt)
            else:
                st.extend(stmp.copy())
    
    # Edit 'start' time if using offset option
    if opt.maxdt:
        dts = np.fromstring(opt.offset, sep=',')
        for n, tr in enumerate(st):
            tr.stats.starttime = tr.stats.starttime-dts[n]
    
    if do_filter_merge:
        st = st.trim(starttime=tstart, endtime=tend, pad=True, fill_value=0)
    
    return st


def trigger(st, rtable, opt, event=None):
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
    opt : Options object
        Describes the run parameters.
    event : UTCDateTime object, optional
        Catalog event to add by force.
    
    Returns
    -------
    trigs : Stream object
        Triggered events with data from each channel concatenated.
    
    """
    
    t = st[0].stats.starttime
    
    cft = coincidence_trigger(opt.trigalg, opt.trigon, opt.trigoff, st.copy(),
        opt.nstaC, sta=opt.swin, lta=opt.lwin, details=True)
    if event:
        ttimes = [event]
        ratios = [0.0]
        bestmatch = opt.mintrig
        for c in cft:
            if np.abs(c['time']-event) < bestmatch:
                bestmatch = np.abs(c['time']-event)
                ratios = [np.max(c['cft_peaks'])-opt.trigon]
    else:
        ttimes = [cft[n]['time'] for n in range(len(cft))]
        ratios = [
            np.max(cft[n]['cft_peaks'])-opt.trigon for n in range(len(cft))]
    
    if len(ttimes) > 0:
        
        ind = 0
        
        # Convert ptime from time of last trigger to seconds before start time
        if rtable.attrs.ptime:
            ptime = (UTCDateTime(rtable.attrs.ptime) - t)
        else:
            ptime = -opt.mintrig
        
        # Loop over triggers
        for n, ttime in enumerate(ttimes):
            
            # Enforce minimum time between previous known trigger, edges of st
            if (ttime >= t + opt.atrig) and (ttime >= t + ptime +
                opt.mintrig) and (ttime < t + len(st[0].data)/opt.samprate -
                2*opt.atrig):
                
                # Update ptime
                ptime = ttime - t
                
                # Cut out a copy from st with a few samples of padding
                tr = st.slice(ttime - opt.ptrig, ttime + opt.atrig + \
                                                        2/opt.samprate).copy()
                
                # Trim, pad with zeros
                tr = tr.trim(ttime - opt.ptrig, ttime + opt.atrig + \
                                       2/opt.samprate, pad=True, fill_value=0)
                
                for s in range(len(tr)):
                    
                    # Cut out exact number of samples
                    tr[s].data = tr[s].data[0:opt.wshape]
                    
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
                     preload_waveforms, opt, event=None):
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
    opt : Options object
        Describes the run parameters.
    event : UTCDateTime object, optional
        Catalog event to add by force.
    
    Returns
    -------
    alltrigs : Stream object
        Triggered events with data from each channel concatenated.
    
    """
    if opt.troubleshoot:
        st = get_data(window_start_time-opt.atrig, window_end_time,
                                    filekey, preload_waveforms, opt)
        alltrigs = trigger(st, rtable, opt, event=event)
    else:
        try:
            st = get_data(window_start_time-opt.atrig, window_end_time,
                                        filekey, preload_waveforms, opt)
            alltrigs = trigger(st, rtable, opt, event=event)
        except KeyboardInterrupt:
            print('\nManually interrupting!\n')
            raise KeyboardInterrupt
        except:
            print(('Could not download or trigger data... '
                   'troubleshoot with -t'))
            alltrigs = []
    return alltrigs


def clean_triggers(alltrigs, opt, event=None):
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
    opt : Options object
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
            alltrigs[i].data, int(opt.ptrig*opt.samprate), opt)
        
        # Loop over channels
        for n in range(opt.nsta):
            
            # Check FI
            fi = windowFI[n]
            if fi<opt.telefi:
                ntele+=1
            
            # Get channel waveform
            dat = alltrigs[i].data[n*opt.wshape:(n+1)*opt.wshape]
            
            # Cut out kurtosis window surrounding initial trigger
            datcut = dat[range(int((opt.ptrig-opt.kurtwin/2)*opt.samprate),
                               int((opt.ptrig+opt.kurtwin/2)*opt.samprate))]
            
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
                
                if (k >= opt.kurtmax) or (oratio >= opt.oratiomax) or (
                        kf >= opt.kurtfmax):
                    njunk+=1
        
        # Allow if there are enough good stations to correlate
        if njunk <= (opt.nsta-opt.ncor) and ntele <= opt.teleok:
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
