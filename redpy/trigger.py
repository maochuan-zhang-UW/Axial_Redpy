# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import glob, os, itertools

import numpy as np

from obspy import UTCDateTime
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy.clients.fdsn import Client
from obspy.clients.earthworm import Client as EWClient
from obspy.clients.seedlink import Client as SeedLinkClient
from obspy.signal.trigger import coincidence_trigger
from scipy.fftpack import fft
from scipy.stats import kurtosis

# !!! Be a better coder and address these warnings rather than ignore them
import warnings
warnings.filterwarnings("ignore")

def getData(tstart, tend, opt):
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
    stC : Stream object
        Copy of st.
    """
    
    nets = opt.network.split(',')
    stas = opt.station.split(',')
    locs = opt.location.split(',')
    chas = opt.channel.split(',')
    
    st = Stream()
    
    if opt.server == 'file':
        
        # Generate list of files
        if opt.server == 'file':
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
        stmp = stmp.filter('bandpass', freqmin=opt.fmin, freqmax=opt.fmax,
            corners=2, zerophase=True)
        stmp = stmp.taper(0.05,type='hann',max_length=opt.mintrig)
        for m in range(len(stmp)):
            if stmp[m].stats.sampling_rate != opt.samprate:
                stmp[m] = stmp[m].resample(opt.samprate)
        stmp = stmp.merge(method=1, fill_value=0)
        
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
                print("Couldn't find {}.{}.{}.{}".format(stas[n], chas[n], 
                                                            nets[n], locs[n]))
                trtmp = Trace()
                trtmp.stats.sampling_rate = opt.samprate
                trtmp.stats.station = stas[n]
                st = st.append(trtmp.copy())
    
    else:
        
        if '://' not in opt.server:
            # Backward compatibility with previous setting files
            if '.' not in opt.server:
                client = Client(opt.server)
            else:
                client = EWClient(opt.server, opt.port)
        # New server syntax (more options and server and port on same variable)
        elif 'fdsnws://' in opt.server:
            server = opt.server.split('fdsnws://',1)[1]
            client = Client(server)
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
        
        for n in range(len(stas)):
            try:
                stmp = client.get_waveforms(nets[n], stas[n], locs[n], chas[n],
                        tstart, tend+opt.maxdt)
                for m in range(len(stmp)):
                    stmp[m].data = np.where(stmp[m].data == -2**31, 0,
                        stmp[m].data) # replace -2**31 (Winston NaN token)
                stmp = stmp.filter('bandpass', freqmin=opt.fmin,
                    freqmax=opt.fmax, corners=2, zerophase=True)
                stmp = stmp.taper(0.05,type='hann',max_length=opt.mintrig)
                for m in range(len(stmp)):
                    if stmp[m].stats.sampling_rate != opt.samprate:
                        stmp[m] = stmp[m].resample(opt.samprate)
                stmp = stmp.merge(method=1, fill_value=0)
            except (obspy.clients.fdsn.header.FDSNException):
                try: # try again
                    stmp = client.get_waveforms(nets[n], stas[n], locs[n],
                            chas[n], tstart, tend+opt.maxdt)
                    for m in range(len(stmp)):
                        stmp[m].data = np.where(stmp[m].data == -2**31, 0,
                            stmp[m].data) # replace -2**31 (Winston NaN token)
                    stmp = stmp.filter('bandpass', freqmin=opt.fmin,
                        freqmax=opt.fmax, corners=2, zerophase=True)
                    stmp = stmp.taper(0.05,type='hann',max_length=opt.mintrig)
                    for m in range(len(stmp)):
                        if stmp[m].stats.sampling_rate != opt.samprate:
                            stmp[m] = stmp[m].resample(opt.samprate)
                    stmp = stmp.merge(method=1, fill_value=0)
                except (obspy.clients.fdsn.header.FDSNException):
                    print('No data found for {0}.{1}'.format(stas[n],nets[n]))
                    trtmp = Trace()
                    trtmp.stats.sampling_rate = opt.samprate
                    trtmp.stats.station = stas[n]
                    stmp = Stream().extend([trtmp.copy()])
            
            # Last check for length; catches problem with empty waveserver
            if len(stmp) != 1:
                print('No data found for {0}.{1}'.format(stas[n],nets[n]))
                trtmp = Trace()
                trtmp.stats.sampling_rate = opt.samprate
                trtmp.stats.station = stas[n]
                stmp = Stream().extend([trtmp.copy()])
            
            st.extend(stmp.copy())
    
    # Edit 'start' time if using offset option
    if opt.maxdt:
        dts = np.fromstring(opt.offset, sep=',')
        for n, tr in enumerate(st):
            tr.stats.starttime = tr.stats.starttime-dts[n]
    
    st = st.trim(starttime=tstart, endtime=tend, pad=True, fill_value=0)
    stC = st.copy()
    
    return st, stC


def trigger(st, stC, rtable, opt):
    """
    Run triggering algorithm on a stream of data.
    
    Parameters
    ----------
    st : Stream object
        Stream containing continuous, filtered Traces for each channel.
    stC : Stream object
        Copy of st.
    rtable : Table object
        Handle to the Repeaters table.
    opt : Options object
        Describes run parameters.
    
    Returns
    -------
    trigs : Stream object
        Triggered events with data from each channel concatenated.
    """
    
    tr = st[0]
    t = tr.stats.starttime
    
    cft = coincidence_trigger(opt.trigalg, opt.trigon, opt.trigoff, stC,
        opt.nstaC, sta=opt.swin, lta=opt.lwin, details=True)
    
    if len(cft) > 0:
        
        ind = 0
        
        # Slice out the data from st and save the maximum STA/LTA ratio value
        # for use in orphan expiration
        
        # Convert ptime from time of last trigger to seconds before start time
        if rtable.attrs.ptime:
            ptime = (UTCDateTime(rtable.attrs.ptime) - t)
        else:
            ptime = -opt.mintrig
            
        for n in range(len(cft)):
            
            ttime = cft[n]['time'] # This is a UTCDateTime, not samples
            
            if (ttime >= t + opt.atrig) and (ttime >= t + ptime +
                opt.mintrig) and (ttime < t + len(tr.data)/opt.samprate -
                2*opt.atrig):
                
                ptime = ttime - t
                
                # Cut out and append all data to first trace
                tmp = st.slice(ttime - opt.ptrig, ttime + opt.atrig)
                ttmp = tmp.copy()
                ttmp = ttmp.trim(ttime - opt.ptrig, ttime + opt.atrig + 0.05,
                    pad=True, fill_value=0)
                ttmp[0].data = ttmp[0].data[0:opt.wshape] - np.mean(
                    ttmp[0].data[0:opt.wshape])
                for s in range(1,len(ttmp)):
                    ttmp[0].data = np.append(ttmp[0].data, ttmp[s].data[
                        0:opt.wshape] - np.mean(ttmp[s].data[0:opt.wshape]))
                ttmp[0].stats.maxratio = np.max(cft[n]['cft_peaks'])
                if ind is 0:
                    trigs = Stream(ttmp[0])
                    ind = ind+1
                else:
                    trigs = trigs.append(ttmp[0])
        
        if ind is 0:
            return []
        else:
            rtable.attrs.ptime = (t + ptime).isoformat()
            return trigs
    else:
        return []


def dataClean(alltrigs, opt, flag=1):
    """
    Cleans data of data spikes, calibration pulses, and teleseisms.
    
    Specifically, it attempts to weed out spikes and analog calibration pulses
    using kurtosis and outlier ratios; checks for teleseisms that have very low
    frequency index.
    
    Parameters
    ----------
    alltrigs : Stream object
        Triggered events with data from each channel concatenated.
    opt : Options object
        Describes run parameters.
    flag : integer, optional
        Flag for whether to use window or full waveform. 1 if defining window
        to check, 0 to check whole waveform for spikes (note that different
        threshold values should be used for different window lengths).
    
    Returns
    -------
    trigs : Stream object
        Events from alltrigs that passed all tests.
    junk : Stream object
        Events that failed both FI and kurtosis tests.
    junkFI : Stream object
        Events that failed FI test (teleseisms).
    junkKurt : Stream object
        Events that failed kurtosis test (data spikes, calibration pulses).
    """
    
    trigs=Stream()
    junkFI=Stream()
    junkKurt=Stream()
    junk=Stream()
    for i in range(len(alltrigs)):
        
        njunk = 0
        ntele = 0
        
        for n in range(opt.nsta):
            
            dat = alltrigs[i].data[n*opt.wshape:(n+1)*opt.wshape]
            if flag == 1:
                datcut=dat[range(int((opt.ptrig-opt.kurtwin/2)*opt.samprate),
                    int((opt.ptrig+opt.kurtwin/2)*opt.samprate))]
            else:
                datcut=dat
            
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
                orm = len(z[z>4.45])/np.array(len(z)).astype(float)
                
                if k >= opt.kurtmax or orm >= opt.oratiomax or \
                                                          kf >= opt.kurtfmax:
                    njunk+=1
                
                winstart = int(opt.ptrig*opt.samprate - opt.winlen/10)
                winend = int(opt.ptrig*opt.samprate - opt.winlen/10 + \
                             opt.winlen)
                fftwin = np.reshape(fft(dat[winstart:winend]),(opt.winlen,))
                if np.median(np.abs(dat[winstart:winend]))!=0:
                    fi = np.log10(np.mean(np.abs(np.real(
                        fftwin[int(opt.fiupmin*opt.winlen/opt.samprate):int(
                        opt.fiupmax*opt.winlen/opt.samprate)])))/np.mean(
                        np.abs(np.real(fftwin[int(
                        opt.filomin*opt.winlen/opt.samprate):int(
                        opt.filomax*opt.winlen/opt.samprate)]))))
                    if fi<opt.telefi:
                        ntele+=1
        
        # Allow if there are enough good stations to correlate
        if njunk <= (opt.nsta-opt.ncor) and ntele <= opt.teleok:
            trigs.append(alltrigs[i])
        else:
            if njunk > 0:
                if ntele > 0:
                    junk.append(alltrigs[i])
                else:
                    junkKurt.append(alltrigs[i])
            else:
                junkFI.append(alltrigs[i])
    
    return trigs, junk, junkFI, junkKurt
