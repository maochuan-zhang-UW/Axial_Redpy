# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import glob, os, itertools

import numpy as np
import obspy

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
        Describes run parameters.
    
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
        Describes run parameters.
    
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
    Appends an empty Trace to the end of a Stream with proper SCNL information.
    
    Parameters
    ----------
    st : Stream object
        Stream that will contain Traces for each channel.
    n : integer
        Index of channel within list.
    opt : Options object
        Describes run parameters.
    
    Returns
    -------
    st : Stream object
        Input Stream with empty Trace appended.
    """
    
    nets = opt.network.split(',')
    stas = opt.station.split(',')
    locs = opt.location.split(',')
    chas = opt.channel.split(',')    
    
    print('No data found for {}.{}.{}.{}'.format(nets[n], stas[n], chas[n],
                                                                   locs[n]))
    
    trtmp = Trace()
    trtmp.stats.sampling_rate = opt.samprate
    trtmp.stats.station = stas[n]
    trtmp.stats.channel = chas[n]
    trtmp.stats.network = nets[n]
    trtmp.stats.location = locs[n]
    
    st = st.append(trtmp.copy())
    
    return st


def get_data(tstart, tend, opt):
    """
    Download data from web or read from file.
    
    A note on SAC/miniSEED files: as this makes no assumptions about the naming
    scheme of your data files, please ensure that your headers contain the
    correct SCNL information!
    
    Parameters
    ----------
    tstart : UTCDateTime object
        Beginning time of query.
    tend : UTCDateTime object
        End time of query.
    opt : Options object
        Describes run parameters.
    
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
    
    if opt.server == 'file':
        
        # !!! This method of loading from file is extremely slow!
        # !!! Recommend only doing the header load ONCE rather than within
        # !!! each time loop, create a dataframe/csvfile that can be passed
        # !!! to this function that has starttime, endtime, scnl, filepath
        # !!! that can be more rapidly searched than this monstrosity.        
        
        # Generate list of files
        flist = list(itertools.chain.from_iterable(glob.iglob(os.path.join(
                root,opt.filepattern)) for root, dirs, files in os.walk(
                                                              opt.searchdir)))
        
        # Determine which subset of files to load based on start and end times
        # and station name; we'll fully deal with stations below
        flist_sub = []
        for f in flist:
            # Load header only
            stmp = obspy.read(f, headonly=True)
            # Check if station is contained in the stas list
            if stmp[0].stats.station in stas:
                # Check if contains either start or end time
                ststart = stmp[0].stats.starttime
                stend = stmp[-1].stats.endtime
                if (ststart<=tstart and tstart<=stend) or (ststart<=tend and
                    tend<=stend) or (tstart<=stend and ststart<=tend):
                    flist_sub.append(f)
        
        # Fully load data from file
        stmp = Stream()
        for f in flist_sub:
            tmp = obspy.read(f, starttime=tstart, endtime=tend+opt.maxdt)
            if len(tmp) > 0:
                stmp = stmp.extend(tmp)
        
        # Filter and merge
        stmp = filter_merge(stmp, opt)
        
        # !!! Being able to search by SCNL above will also let me not have
        # !!! to worry about ordering issues here.
        
        # Only grab stations/channels that we want and in order
        netlist = []
        stalist = []
        chalist = []
        loclist = []
        for s in stmp:
            stalist.append(s.stats.station)
            chalist.append(s.stats.channel)
            netlist.append(s.stats.network)
            loclist.append(s.stats.location)
        
        # Find match of SCNL in header or fill empty
        for n in range(len(stas)):
            for m in range(len(stalist)):
                if (stas[n] in stalist[m] and chas[n] in chalist[m] and 
                             nets[n] in netlist[m] and locs[n] in loclist[m]):
                    st = st.append(stmp[m])
            if len(st) == n:
                st = append_empty(st, n, opt)
    
    else:
        
        client = get_client(opt)
        
        for n in range(len(stas)):
            try:
                stmp = client.get_waveforms(nets[n], stas[n], locs[n], chas[n],
                        tstart, tend+opt.maxdt)
                stmp = filter_merge(stmp, opt)
            except (obspy.clients.fdsn.header.FDSNException):
                # Try querying again in case timed out on accident
                try:
                    stmp = client.get_waveforms(nets[n], stas[n], locs[n],
                            chas[n], tstart, tend+opt.maxdt)
                    stmp = filter_merge(stmp, opt)
                except:
                    stmp = append_empty(Stream(), n, opt)
            
            # Last check for length; catches problem with empty waveserver
            if len(stmp) != 1:
                st = append_empty(st, n, opt)
            else:
                st.extend(stmp.copy())
    
    # Edit 'start' time if using offset option
    if opt.maxdt:
        dts = np.fromstring(opt.offset, sep=',')
        for n, tr in enumerate(st):
            tr.stats.starttime = tr.stats.starttime-dts[n]
    
    st = st.trim(starttime=tstart, endtime=tend, pad=True, fill_value=0)
    
    return st


def trigger(st, rtable, opt):
    """
    Run triggering algorithm on a stream of data.
    
    Parameters
    ----------
    st : Stream object
        Stream containing continuous, filtered Traces for each channel.
    rtable : Table object
        Handle to the Repeaters table.
    opt : Options object
        Describes run parameters.
    
    Returns
    -------
    trigs : Stream object
        Triggered events with data from each channel concatenated.
    """
    
    t = st[0].stats.starttime
    
    # !!! Force trigger to load ttimes, ratios from file instead
    cft = coincidence_trigger(opt.trigalg, opt.trigon, opt.trigoff, st.copy(),
        opt.nstaC, sta=opt.swin, lta=opt.lwin, details=True)
    ttimes = [cft[n]['time'] for n in range(len(cft))]
    ratios = [np.max(cft[n]['cft_peaks'])-opt.trigon for n in range(len(cft))]
    
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
                    #                                tr[s].data[tr[s].data!=0])
                    
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


def clean_triggers(alltrigs, opt):
    """
    Cleans triggers of data spikes, calibration pulses, and teleseisms.
    
    Specifically, it attempts to weed out spikes and analog calibration pulses
    using kurtosis and outlier ratios; checks for teleseisms that have very low
    frequency index.
    
    Parameters
    ----------
    alltrigs : Stream object
        Triggered events with data from each channel concatenated.
    opt : Options object
        Describes run parameters.
    
    Returns
    -------
    trigs : Stream object
        Events from alltrigs that passed all tests.
    junk : Stream object
        Events that failed a test.
    jtype : integer list
        List with codes corresponding to which tests failed:
            0: FI too low (possible teleseism)
            1: Kurtosis in time or frequency domains too high (spikes, sinewave)
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
        windowCoeff, windowFFT, windowFI = calculate_window(alltrigs[i].data,
            int(opt.ptrig*opt.samprate), opt)
        
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
                
                if k >= opt.kurtmax or oratio >= opt.oratiomax or \
                                                          kf >= opt.kurtfmax:
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
    
    return trigs, junk, jtype
