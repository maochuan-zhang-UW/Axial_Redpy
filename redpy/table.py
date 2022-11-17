# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import sys

import datetime
import matplotlib.dates as mdates
import numpy as np
from obspy import UTCDateTime
from obspy.core.trace import Trace
from tables import *

import redpy.correlation


def Repeaters(opt):
    """
    Creates a dictionary defining the columns in the 'Repeaters' table.
    
    The columns are as follows:
        id           : integer, unique ID number for the event.
        startTime    : string, UTC time of start of the waveform.
        startTimeMPL : float, matplotlib datenumber associated with start time.
        waveform     : float ndarray, filtered waveform data for each station,
                           concatenated.
        windowStart  : integer, window start time, in samples from start of
                           waveform.
        windowCoeff  : float ndarray, amplitude scaling for cross-correlation
                           for each station.
        windowFFT    : complex ndarray, Fourier transform of window for each
                           station, concatenated.
        windowAmp    : float ndarray, maximum amplitude in first half of window
                           for each station.
        FI           : float ndarray, frequency index for each station.
    
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    dict
        Dictionary of column definitions.
    """
    
    repeaters_dictionary = {
        "id"           : Int32Col(shape=(), pos=0),
        "startTime"    : StringCol(itemsize=32, pos=1),
        "startTimeMPL" : Float64Col(shape=(), pos=2),
        "waveform"     : Float32Col(shape=(opt.wshape*opt.nsta,), pos=3),
        "windowStart"  : Int32Col(shape=(), pos=4),
        "windowCoeff"  : Float64Col(shape=(opt.nsta,), pos=5),
        "windowFFT"    : ComplexCol(shape=(opt.winlen*opt.nsta,),
                                    itemsize=16, pos=6),
        "windowAmp"    : Float64Col(shape=(opt.nsta,), pos=7),
        "FI"           : Float64Col(shape=(opt.nsta,), pos=8)
        }
    
    return repeaters_dictionary


def Orphans(opt):
    """
    Creates a dictionary defining the columns in the 'Orphans' table.
    
    The columns are as follows:
        id           : integer, unique ID number for the event.
        startTime    : string, UTC time of start of the waveform.
        startTimeMPL : float, matplotlib datenumber associated with start time.
        waveform     : float ndarray, filtered waveform data for each station,
                           concatenated.
        windowStart  : integer, window start time, in samples from start of
                           waveform.
        windowCoeff  : float ndarray, amplitude scaling for cross-correlation
                           for each station.
        windowFFT    : complex ndarray, Fourier transform of window for each
                           station, concatenated.
        windowAmp    : float ndarray, maximum amplitude in first half of window
                           for each station.
        FI           : float ndarray, frequency index for each station.
        expires      : string, UTC time of expiration date.
    
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    dict
        Dictionary of column definitions.
    """
    
    orphans_dictionary = {
        "id"           : Int32Col(shape=(), pos=0),
        "startTime"    : StringCol(itemsize=32, pos=1),
        "startTimeMPL" : Float64Col(shape=(), pos=2),
        "waveform"     : Float32Col(shape=(opt.wshape*opt.nsta,), pos=3),
        "windowStart"  : Int32Col(shape=(), pos=4),
        "windowCoeff"  : Float64Col(shape=(opt.nsta,), pos=5),
        "windowFFT"    : ComplexCol(shape=(opt.winlen*opt.nsta,),
                                    itemsize=16, pos=6),
        "windowAmp"    : Float64Col(shape=(opt.nsta,), pos=7),
        "FI"           : Float64Col(shape=(opt.nsta,), pos=8),
        "expires"      : StringCol(itemsize=32, pos=9)
        }

    return orphans_dictionary


def Triggers(opt):
    """
    Creates a dictionary defining the columns in the 'Triggers' table.
    
    The columns are as follows:
        startTimeMPL : float, matplotlib datenumber associated with start time.
    
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    dict
        Dictionary of column definitions.
    """
    
    triggers_dictionary = {
        "startTimeMPL" : Float64Col(shape=(), pos=0)
        }
        
    return triggers_dictionary


def Deleted(opt):
    """
    Creates a dictionary defining the columns in the 'Deleted' table.
    
    The columns are as follows:
        id           : integer, unique ID number for the event.
        startTime    : string, UTC time of start of the waveform.
        startTimeMPL : float, matplotlib datenumber associated with start time.
        waveform     : float ndarray, filtered waveform data for each station,
                           concatenated.
        windowStart  : integer, window start time, in samples from start of
                           waveform.
        windowCoeff  : float ndarray, amplitude scaling for cross-correlation
                           for each station.
        windowFFT    : complex ndarray, Fourier transform of window for each
                           station, concatenated.
        windowAmp    : float ndarray, maximum amplitude in first half of window
                           for each station.
        FI           : float ndarray, frequency index for each station.
    
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    dict
        Dictionary of column definitions.
    """
    
    deleted_dictionary = {
        "id"           : Int32Col(shape=(), pos=0),
        "startTime"    : StringCol(itemsize=32, pos=1),
        "startTimeMPL" : Float64Col(shape=(), pos=2),
        "waveform"     : Float32Col(shape=(opt.wshape*opt.nsta,), pos=3),
        "windowStart"  : Int32Col(shape=(), pos=4),
        "windowCoeff"  : Float64Col(shape=(opt.nsta,), pos=5),
        "windowFFT"    : ComplexCol(shape=(opt.winlen*opt.nsta,),
                                    itemsize=16, pos=6),
        "windowAmp"    : Float64Col(shape=(opt.nsta,), pos=7),
        "FI"           : Float64Col(shape=(opt.nsta,), pos=8)
        }

    return deleted_dictionary


def Junk(opt):
    """
    Creates a dictionary defining the columns in the 'Junk' table.
    
    The columns are as follows:
        isjunk      : integer, code for which flags were raised.
        startTime   : string, UTC time of start of the waveform.
        waveform    : float ndarray, filtered waveform data for each station,
                           concatenated.
        windowStart : integer, window start time, in samples from start of
                           waveform.
        
        !!! Rename isjunk !!!
    
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    dict
        Dictionary of column definitions.
    """
    
    junk_dictionary = {
        "isjunk"      : Int32Col(shape=(), pos=0),
        "startTime"   : StringCol(itemsize=32, pos=1),
        "waveform"    : Float32Col(shape=(opt.wshape*opt.nsta,), pos=2),
        "windowStart" : Int32Col(shape=(), pos=3)
        }
        
    return junk_dictionary


def Correlation(opt):
    """
    Creates a dictionary defining the columns in the 'Correlation' table.

    The columns are as follows:
    
        id1 : integer, unique ID number for the first event.
        id2 : integer, unique ID number for the second event.
        ccc : float, cross-correlation coefficient between those two events.
    
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    dict
        Dictionary of column definitions.
    """
    
    correlation_dictionary = {
        "id1" : Int32Col(shape=(), pos=0),
        "id2" : Int32Col(shape=(), pos=1),
        "ccc" : Float64Col(shape=(), pos=2)
    }

    return correlation_dictionary


def Families(opt):
    """
    Creates a dictionary defining the columns in the 'Families' table.
    
    The columns are as follows:
        members   : string, rows in the repeater table that contain members of
                        this family as an ordered list converted to a string.
        core      : integer, row in the repeater table that corresponds to the
                        current 'core' event.
        startTime : float, matplotlib datenumber associated with start time of
                        first event.
        longevity : float, number of days between occurrence of first and last
                        event.
        printme   : integer, describes whether the family has been updated
                        since the last plotting/printing call.
        lastprint : integer, family number when previously plotted/printed.
        
        !!! Rename startTime/printme/lastprint !!!
    
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    dict
        Dictionary of column definitions.
    """
    
    # !!! members itemsize should be defined in opt !!!
    families_dictionary = {
        "members"   : StringCol(itemsize=1000000, shape=(), pos=0),
        "core"      : Int32Col(shape=(), pos=1),
        "startTime" : Float64Col(shape=(), pos=2),
        "longevity" : Float64Col(shape=(), pos=3),
        "printme"   : Int32Col(shape=(), pos=4),
        "lastprint" : Int32Col(shape=(), pos=5)
    }

    return families_dictionary

    
def initializeTable(opt):
    """
    Initializes and creates the hdf5 table file on disk.
     
    Saves table to file and closes it. Always overwrites an existing file.
            
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    """

    h5file = open_file(opt.filename, mode="w", title=opt.title)
    group = h5file.create_group("/", opt.groupName, opt.title)

    rtable = h5file.create_table(group, "repeaters", Repeaters(opt),
                                 "Repeater Catalog")
    rtable.attrs.scnl = [opt.station, opt.channel, opt.network, opt.location]
    rtable.attrs.samprate = opt.samprate
    rtable.attrs.windowLength = opt.winlen
    rtable.attrs.ptrig = opt.ptrig
    rtable.attrs.atrig = opt.atrig
    rtable.attrs.fmin = opt.fmin
    rtable.attrs.fmax = opt.fmax
    rtable.attrs.previd = 0
    rtable.attrs.ptime = 0
    rtable.flush()
    
    otable = h5file.create_table(group, "orphans", Orphans(opt),
                                 "Orphan Catalog")
    otable.flush()
    
    ttable = h5file.create_table(group, "triggers", Triggers(opt),
                                 "Trigger Catalog")
    ttable.flush()
    
    jtable = h5file.create_table(group, "junk", Junk(opt),
                                 "Junk Catalog")
    jtable.flush()
    
    dtable = h5file.create_table(group, "deleted", Deleted(opt),
                                 "Manually Deleted Events")
    dtable.flush()

    ctable = h5file.create_table(group, "correlation", Correlation(opt),
                                 "Correlation Matrix")
    ctable.flush()
    
    ftable = h5file.create_table(group, "families", Families(opt),
                                 "Families Table")
    ftable.attrs.nClust = 0
    ftable.flush()

    h5file.close()


def openTable(opt):
    """
    Convenience function to open the hdf5 file and access the tables in it.
    
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    h5file : File object
        Handle to the h5 file.
    rtable : Table object
        Handle to the Repeaters table.
    otable : Table object
        Handle to the Orphans table.
    ttable : Table object
        Handle to the Triggers table.
    ctable : Table object
        Handle to the Correlation table.
    jtable : Table object
        Handle to the Junk table.
    dtable : Table object
        Handle to the Deleted table.
    ftable : Table object
        Handle to the Families table.
    """

    h5file = open_file(opt.filename, "a")
    
    rtable = eval('h5file.root.'+ opt.groupName + '.repeaters')
    otable = eval('h5file.root.'+ opt.groupName + '.orphans')
    ctable = eval('h5file.root.'+ opt.groupName + '.correlation')
    ttable = eval('h5file.root.'+ opt.groupName + '.triggers')
    jtable = eval('h5file.root.'+ opt.groupName + '.junk')
    dtable = eval('h5file.root.'+ opt.groupName + '.deleted')
    ftable = eval('h5file.root.'+ opt.groupName + '.families')
    
    return h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable


def calcAmps(data, windowStart, opt):
    """
    Calculates the maximum waveform amplitudes for 'windowAmp'.
    
    Calculations are done for the first half of the window for each station.
    
    Parameters
    ----------
    data : float ndarray
        Waveform data for all stations, appended together.
    windowStart : integer
        ...
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    amps : float list
        Array of maximum amplitudes.
    """
    # !!! CHANGE THIS TO NDARRAY !!!
    amps = []
    for n in range(opt.nsta):
        amps.append(max(abs(data[(n*opt.wshape)+windowStart:int(
            (n*opt.wshape)+windowStart+opt.winlen/2)])))
    
    return amps
    
    
def populateRepeater(rtable, ftable, idnum, trig, opt, windowStart=-1):
    """
    Populates a new row in the 'Repeater' table from trigger.
    
    !!! Why is ftable called here?
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeater table.
    idnum : integer
        Unique ID number given to this event.
    trig : Trace object
        Output from triggering function, with data from all stations appended.
    opt : Options object
        Describes the run parameters.
    windowStart : integer, optional
        Triggering time in samples from start, defaults to opt.ptrig seconds.
    """
    
    # Create an empty row
    rrow = rtable.row
    
    if windowStart == -1:
        windowStart = int(opt.ptrig*opt.samprate)
    
    rrow['id'] = idnum
    rrow['startTime'] = trig.stats.starttime.isoformat()
    # Deal with difference in format string if exactly at a second
    try:
        rrow['startTimeMPL'] = mdates.date2num(datetime.datetime.strptime(
            trig.stats.starttime.isoformat(), '%Y-%m-%dT%H:%M:%S.%f'))
    except ValueError:
        rrow['startTimeMPL'] = mdates.date2num(datetime.datetime.strptime(
            trig.stats.starttime.isoformat(), '%Y-%m-%dT%H:%M:%S'))
    rrow['waveform'] = trig.data
    rrow['windowStart'] = windowStart
    rrow['windowCoeff'], rrow['windowFFT'], rrow['FI'] = \
        redpy.correlation.calcWindow(trig.data, windowStart, opt)
    rrow['windowAmp'] = calcAmps(trig.data, windowStart, opt)
    
    rrow.append()  
    rtable.flush()  

    
def populateOrphan(otable, idnum, trig, opt):
    """
    Populates a new row in the 'Orphans' table from trigger.
    
    This function also determines the expiration date based on the STA/LTA
    amplitude from triggering
    
    Parameters
    ----------
    otable : Table object
        Handle to the Orphans table.
    idnum : integer
        Unique ID number given to this event.
    trig : Trace object
        Output from triggering function, with data from all stations appended.
    opt : Options object
        Describes the run parameters.
    """
    
    orow = otable.row
    
    windowStart = int(opt.ptrig*opt.samprate)
    
    orow['id'] = idnum
    orow['startTime'] = trig.stats.starttime.isoformat()
    try:
        orow['startTimeMPL'] = mdates.date2num(datetime.datetime.strptime(
            trig.stats.starttime.isoformat(), '%Y-%m-%dT%H:%M:%S.%f'))
    except ValueError:
        orow['startTimeMPL'] = mdates.date2num(datetime.datetime.strptime(
            trig.stats.starttime.isoformat(), '%Y-%m-%dT%H:%M:%S'))
    orow['waveform'] = trig.data
    orow['windowStart'] = windowStart
    orow['windowCoeff'], orow['windowFFT'], orow['FI'] = \
        redpy.correlation.calcWindow(trig.data, windowStart, opt)
    orow['windowAmp'] = calcAmps(trig.data, windowStart, opt)
    
    # Determine expiration date based on STA/LTA amplitude
    adddays = np.min([opt.maxorph,((opt.maxorph-opt.minorph)/opt.maxorph)*(
        trig.stats.maxratio-opt.trigon)+opt.minorph])
    orow['expires'] = (trig.stats.starttime+adddays*86400).isoformat()
    
    orow.append()
    otable.flush()


def populateTriggers(ttable, trigs, ttimes, opt):    
    """
    Populates new rows in the 'Triggers' table from a list of triggers.
    
    !!! Change to have it return only triggers that were not found as
    !!! duplicates? I have to do that check a lot down the line; should
    !!! only have to do it once...
    
    Parameters
    ----------
    ttable : Table object
        Handle to the Triggers table.
    trigs : list of Trace objects
        Output from triggering function, with data from all stations appended.
    ttimes : float ndarray
        Array of times of existing triggers to prevent duplication.
    opt : Options object
        Describes the run parameters.
    """
    
    for t in trigs:        
        try:
            trigtime = mdates.date2num(
                datetime.datetime.strptime(t.stats.starttime.isoformat(),
                '%Y-%m-%dT%H:%M:%S.%f'))
        except ValueError:
            trigtime = mdates.date2num(
                datetime.datetime.strptime(t.stats.starttime.isoformat(),
                '%Y-%m-%dT%H:%M:%S'))
        
        if not len(np.intersect1d(
                   np.where(ttimes > trigtime - opt.mintrig/86400),
                   np.where(ttimes < trigtime + opt.mintrig/86400))):
            trigger = ttable.row
            trigger['startTimeMPL'] = trigtime
            trigger.append()
            ttable.flush()


def populateJunk(jtable, trig, isjunk, opt):
    """
    Populates a new row in the 'Junk' table from trigger.
    
    !!! Rename isjunk !!!
    
    Parameters
    ----------
    jtable : Table object
        Handle to the Junk table.
    trig : Trace object
        Output from triggering function, with data from all stations appended.
    isjunk : integer
        Flag corresponding to the type of junk.
    opt : Options object
        Describes the run parameters.
    """
    
    jrow = jtable.row
    
    jrow['startTime'] = trig.stats.starttime.isoformat()
    jrow['waveform'] = trig.data
    jrow['windowStart'] = int(opt.ptrig*opt.samprate)
    jrow['isjunk'] = isjunk
    jrow.append()
    jtable.flush()


def appendCorrelation(ctable, id1, id2, ccc, opt):
    """
    Appends a new value to the 'Correlation' table.
    
    Automatically puts the smaller of the two id numbers first, and only 
    appends if the correlation value is greater than the minimum required.
    
    Parameters
    ----------
    ctable : Table object
        Handle to the Correlation table.
    id1 : int
        Unique id number of first trigger.
    id2 : int
        Unique id number of second trigger.
    ccc : float
        Cross-correlation coefficients between the two triggers.
    opt : Options object
        Describes the run parameters.
    """
    
    if (ccc >= opt.cmin) and (id1!=id2):
        crow = ctable.row
        crow['id1'] = min(id1, id2)
        crow['id2'] = max(id1, id2)
        crow['ccc'] = ccc
        crow.append()
        ctable.flush()


def moveOrphan(rtable, otable, ftable, oindex, opt):
    """
    Moves a row from the 'Orphans' table to the 'Repeater' table.
    
    !!! Again, why is ftable called here?
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeater table.
    otable : Table object
        Handle to the Orphans table.
    ftable : Table object
        !!! to be removed !!!
    oindex : integer
        Row in otable to move
    opt : 
        Describes the run parameters.    
    """
    
    rrow = rtable.row
    orow = otable[oindex]
    
    rrow['id'] = orow['id']
    rrow['startTime'] = orow['startTime']
    rrow['startTimeMPL'] = orow['startTimeMPL']
    rrow['waveform'] = orow['waveform']
    rrow['windowStart'] = orow['windowStart']
    if len(otable) > 1:
        rrow['windowCoeff'] = orow['windowCoeff']
        rrow['windowFFT'] = orow['windowFFT']
        rrow['FI'] = orow['FI']
    else:
        # Deal with edge case where this is the last remaining orphan in the
        # otable. The table can't be empty, so we set the windowCoeff to be
        # 0 so it will never correlate, and will expire as soon as a new
        # orphan is found.
        coeff, fft, fi = redpy.correlation.calcWindow(orow['waveform'],
                                                      orow['windowStart'], opt)
        rrow['windowCoeff'] = coeff
        rrow['windowFFT'] = fft
        rrow['FI'] = fi
        otable.cols.windowCoeff[oindex] = 0
        otable.cols.expires[oindex] = (UTCDateTime(
                                       orow['startTime'])-86400).isoformat()
    rrow['windowAmp'] = orow['windowAmp']
    
    rrow.append()
        
    if len(otable) > 1:
        otable.remove_row(oindex)
    
    rtable.flush()   
    otable.flush()  
    

def clearExpiredOrphans(otable, opt, tend): # !!! Reverse order of opt, tend
    """
    Deletes orphans that have passed their expiration date.
    
    Parameters
    ----------   
    otable : Table object
        Handle to the Orphans table.
    opt : Options object
        Describes the run parameters.
    tend : UTCDateTime object
        Time to remove orphans older than.
    """
    
    expired = np.empty(0) #!!! astype(int) ? then remove int()s later !!!
    for n in range(len(otable)):
        if otable.cols.expires[n].decode('utf-8') < tend.isoformat():
            expired = np.append(expired,n)
    
    if len(expired) > 0:
        if len(expired) != len(otable):
            for n in range(len(expired)-1,-1,-1):
                otable.remove_row(int(expired[n]))        
        else:
            print('Warning: All orphans expired...')
            for n in range(len(idx)-1,0,-1):
                otable.remove_row(int(expired[n]))
        otable.flush()
    

def createNewFamily(rtable, ftable, members, core, opt):
    """
    Creates new family from two or more events.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeater table.
    ftable : Table object
        Handle to the Families table.
    members : integer ndarray
        Array of row indices in rtable that are family members.
    core : integer
        Row index of core event in rtable.
    opt : Options object
        Describes the run parameters.
    """
    
    frow = ftable.row
    frow['members'] = np.array2string(members)[1:-1]
    frow['core'] = core
    frow['startTime'] = np.min(rtable[members]['startTimeMPL'])
    frow['longevity'] = np.max(rtable[members]['startTimeMPL']) - np.min(
        rtable[members]['startTimeMPL'])
    frow['printme'] = 1
    frow['lastprint'] = -1
    frow.append()
    ftable.attrs.nClust+=1
    ftable.flush()
    
    if len(ftable)>1:
        reorderFamilies(ftable, opt)
        # Reprint second to last family (so html navigation updates)
        # !!! why only second to last and not last? !!!
        ftable.cols.printme[-2] = 1        
    
    
def mergeFamilies(rtable, ctable, ftable, wfam, wlag, opt):
    """
    Combines families that have been merged by adding a new event.
    
    !!! Confirm this is without problems !!!
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeater table.
    ctable : Table object
        Handle to the Correlation table.
    ftable : Table object
        Handle to the Families table.
    wfam : integer list
    
    wlag : integer list
        
    opt : Options object
        Describes the run parameters.
    """
    
    # Determine which family is largest, use that as base family
    nmem = []
    for n in range(len(wfam)):
        nmem.append(len(np.fromstring(ftable[wfam[n]]['members'], dtype=int,
                                      sep=' ')))
    wlag = np.array(wlag)
    wlag = wlag - wlag[np.argmax(nmem)]
    
    # Adjust lags
    for n in range(len(wfam)):
        if wlag[n]!=0:
            members = np.fromstring(ftable[wfam[n]]['members'], dtype=int,
                                    sep=' ')
            for m in members:
                rtable.cols.windowStart[m] = \
                    rtable.cols.windowStart[m] - wlag[n]
                rtable.cols.windowCoeff[m], rtable.cols.windowFFT[m], \
                    rtable.cols.FI[m] = redpy.correlation.calcWindow(
                    rtable.cols.waveform[m], rtable.cols.windowStart[m], opt)
            rtable.flush()
    
    # Perform merge, cluster        
    f1 = min(wfam)
    for n in range(len(wfam))[::-1]:
        f2 = np.sort(wfam)[n]
        if f2!=f1:            
            ftable.cols.members[f1] = ftable.cols.members[f1].decode(
                'utf-8')+' '+ftable[f2]['members'].decode('utf-8')
            ftable.remove_row(f2)
            ftable.attrs.nClust-=1
    ftable.cols.printme[f1] = 1
    ftable.cols.lastprint[f1] = -1            
    reorderFamilies(ftable, opt)
    redpy.cluster.runFamOPTICS(rtable, ctable, ftable, f1, opt)

    
def reorderFamilies(ftable, opt):
    """
    Ensures families are ordered by start time.
    
    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    opt : Options object
        Describes the run parameters.
    """
    
    startTimes = ftable.cols.startTime[:]
    order = np.argsort(startTimes)
    x = np.arange(len(ftable))
        
    if (x!=order).any():
        members = ftable.cols.members[:]
        cores = ftable.cols.core[:]
        longevity = ftable.cols.longevity[:]
        printme = ftable.cols.printme[:]
        lastprint = ftable.cols.lastprint[:]
        for n in np.where(x!=order)[0]:
            ftable.cols.startTime[n] = startTimes[order[n]]
            ftable.cols.members[n] = members[order[n]]
            ftable.cols.longevity[n] = longevity[order[n]]
            ftable.cols.core[n] = cores[order[n]]
            ftable.cols.printme[n] = printme[order[n]]
            ftable.cols.lastprint[n] = lastprint[order[n]]
            
        ftable.flush()
    

def removeFamilies(rtable, ctable, dtable, ftable, cnums, opt):
    """
    Removes families from catalog.
    
    Specifically, it removes the families from the Families table, removes the
    cross-correlation values from the Correlation table for members of those
    families, moves the core of the families into the Deleted table, and
    removes the rest of the members from the Repeaters table.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeater table.
    ctable : Table object 
        Handle to the Correlation table.
    dtable : Table object
        Handle to the Deleted table.
    ftable : Table object
        Handle to the Families table.
    cnums : integer list
        List of clusters/rows to remove from the families table.
    opt : Options object
        Describes the run parameters.
    """

    cnums = np.sort(cnums)
    old = list(range(len(rtable)))
    transform = np.zeros((len(rtable),)).astype(int)    
    ids = rtable.cols.id[:]
    members = np.array([])
    oldcores = ftable.cols.core[:]
    for cnum in cnums[::-1]: # Process in reverse
        members = np.append(members, np.fromstring(ftable[cnum]['members'],
                                                   dtype=int, sep=' '))
        ftable.remove_row(cnum)
        ftable.flush()
    ftable.attrs.nClust-=len(cnums)
    members = np.sort(members).astype('uint32')
    
    # Populate cores in dtable before removing them from rtable
    cores = rtable[np.intersect1d(members, oldcores)]
    for core in cores:
        drow = dtable.row
        drow['id'] = core['id']
        drow['startTime'] = core['startTime'][0]
        drow['startTimeMPL'] = core['startTimeMPL']
        drow['waveform'] = core['waveform']
        drow['windowStart'] = core['windowStart']
        drow['windowCoeff'] = core['windowCoeff']
        drow['windowFFT'] = core['windowFFT']
        drow['windowAmp'] = core['windowAmp']
        drow['FI'] = core['FI']
        drow.append()
    
    for m in members[::-1]:
        rtable.remove_row(m)
        old.remove(m)
    
    # !!! Functionalize finding rows in the correlation table? !!!    
    ids = ids[members]
    id2 = ctable.cols.id2[:]
    idxc = np.where(np.in1d(id2,ids))[0]
    for c in idxc[::-1]:
        ctable.remove_row(c)
    
    new = range(len(rtable))
    transform[old] = new
    
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(linewidth=sys.maxsize)
    for n in range(len(ftable)):
        fmembers = np.fromstring(ftable[n]['members'], dtype=int, sep=' ')
        core = ftable[n]['core']
        ftable.cols.members[n] = np.array2string(transform[fmembers])[1:-1]
        ftable.cols.core[n] = transform[core]
        ftable.flush()
    
    ftable.cols.printme[-1] = 1
    rtable.flush()
    dtable.flush()


def checkMPL(rtable, ftable, ttable, otable, dtable, opt):
    """
    Checks for mismatch in reference epoch with current matplotlib version.
    
    This check is done because the reference epoch was changed in matplotlib
    v3.3, so a table generated using matplotlib <v3.3 will have dates in
    the future if the user has updated past v3.3. The opposite is also true,
    with dates far in the past if a table was generated with >=v3.3 and
    current version is <v3.3.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    ttable : Table object
        Handle to the Triggers table
    otable : Table object
        Handle to the Orphan table
    dtable : Table object
        Handle to the Deleted table
    opt : Options object describing station/run parameters
        
    """
    
    if len(ttable) > 0:

        # Check and fix matplotlib datetime issues
        if ttable.cols.startTimeMPL[0] > mdates.date2num(
            np.datetime64('now')):
            # Explicitly assumes the first trigger will have the issue.
        
            print('Found matplotlib version mismatch! Fixing...')
            reftime = mdates.date2num(np.datetime64('now'))
            epoch = mdates.date2num(np.datetime64('0000-12-31'))
        
            # Fix triggers
            for i in range(len(ttable.cols.startTimeMPL[:])):
                if ttable.cols.startTimeMPL[i] > reftime:
                    ttable.cols.startTimeMPL[i] = \
                        ttable.cols.startTimeMPL[i] + epoch
        
            # Fix orphans
            for i in range(len(otable.cols.startTimeMPL[:])):
                if otable.cols.startTimeMPL[i] > reftime:
                    otable.cols.startTimeMPL[i] = \
                        otable.cols.startTimeMPL[i] + epoch
        
            # Fix repeaters
            for i in range(len(rtable.cols.startTimeMPL[:])):
                if rtable.cols.startTimeMPL[i] > reftime:
                    rtable.cols.startTimeMPL[i] = \
                        rtable.cols.startTimeMPL[i] + epoch
                
            # Fix deleted
            for i in range(len(dtable.cols.startTimeMPL[:])):
                if dtable.cols.startTimeMPL[i] > reftime:
                    dtable.cols.startTimeMPL[i] = \
                        dtable.cols.startTimeMPL[i] + epoch
        
            # Fix families
            for i in range(len(ftable.cols.startTime[:])):
                if ftable.cols.startTime[i] > reftime:
                    ftable.cols.startTime[i] = \
                        ftable.cols.startTime[i] + epoch
        
        elif ttable.cols.startTimeMPL[0] < mdates.date2num(
            np.datetime64('1900-01-01')):
            # Same problem, but with the opposite sense, in case current
            # version is outdated.
        
            print('Found matplotlib version mismatch! Fixing...')
            reftime = mdates.date2num(np.datetime64('1900-01-01'))
            epoch = mdates.date2num(np.datetime64('1970-01-01'))
        
            # Fix triggers
            for i in range(len(ttable.cols.startTimeMPL[:])):
                if ttable.cols.startTimeMPL[i] < reftime:
                    ttable.cols.startTimeMPL[i] = \
                        ttable.cols.startTimeMPL[i] + epoch
        
            # Fix orphans
            for i in range(len(otable.cols.startTimeMPL[:])):
                if otable.cols.startTimeMPL[i] < reftime:
                    otable.cols.startTimeMPL[i] = \
                        otable.cols.startTimeMPL[i] + epoch
        
            # Fix repeaters
            for i in range(len(rtable.cols.startTimeMPL[:])):
                if rtable.cols.startTimeMPL[i] < reftime:
                    rtable.cols.startTimeMPL[i] = \
                        rtable.cols.startTimeMPL[i] + epoch
                
            # Fix deleted
            for i in range(len(dtable.cols.startTimeMPL[:])):
                if dtable.cols.startTimeMPL[i] < reftime:
                    dtable.cols.startTimeMPL[i] = \
                        dtable.cols.startTimeMPL[i] + epoch
        
            # Fix families
            for i in range(len(ftable.cols.startTime[:])):
                if ftable.cols.startTime[i] < reftime:
                    ftable.cols.startTime[i] = \
                        ftable.cols.startTime[i] + epoch

