# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import os
import sys
import time

import matplotlib.dates as mdates
import numpy as np
from obspy import UTCDateTime
from obspy.core.trace import Trace
from tables import *

import redpy.correlation


def Repeaters(opt):
    """
    Creates a dictionary defining the columns in the Repeaters table.
    
    The columns are as follows:
        id           : integer, unique ID number for the event.
        startTime    : string, UTC time of start of the waveform.
        startTimeMPL : float, matplotlib datenumber associated with startTime.
        waveform     : float ndarray, filtered waveform data for each station,
                           concatenated.
        windowStart  : integer, trigger time, in samples from start of 
                           waveform.
        windowCoeff  : float ndarray, amplitude scaling for cross-correlation
                           for each station.
        windowFFT    : complex ndarray, Fourier transform of window for each
                           station, concatenated.
        windowAmp    : float ndarray, maximum amplitude in first half of
                           window for each station.
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
    Creates a dictionary defining the columns in the Orphans table.
    
    The columns are as follows:
        id           : integer, unique ID number for the event.
        startTime    : string, UTC time of start of the waveform.
        startTimeMPL : float, matplotlib datenumber associated with startTime.
        waveform     : float ndarray, filtered waveform data for each station,
                           concatenated.
        windowStart  : integer, trigger time, in samples from start of
                           waveform.
        windowCoeff  : float ndarray, amplitude scaling for cross-correlation
                           for each station.
        windowFFT    : complex ndarray, Fourier transform of window for each
                           station, concatenated.
        windowAmp    : float ndarray, maximum amplitude in first half of
                           window for each station.
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
    Creates a dictionary defining the columns in the Triggers table.
    
    The columns are as follows:
        startTimeMPL : float, matplotlib datenumber associated with startTime.
    
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
    Creates a dictionary defining the columns in the Deleted table.
    
    The columns are as follows:
        id           : integer, unique ID number for the event.
        startTime    : string, UTC time of start of the waveform.
        startTimeMPL : float, matplotlib datenumber associated with startTime.
        waveform     : float ndarray, filtered waveform data for each station,
                           concatenated.
        windowStart  : integer, trigger time, in samples from start of
                           waveform.
        windowCoeff  : float ndarray, amplitude scaling for cross-correlation
                           for each station.
        windowFFT    : complex ndarray, Fourier transform of window for each
                           station, concatenated.
        windowAmp    : float ndarray, maximum amplitude in first half of
                           window for each station.
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
    Creates a dictionary defining the columns in the Junk table.
    
    The columns are as follows:
        isjunk      : integer, code for which flags were raised.
        startTime   : string, UTC time of start of the waveform.
        waveform    : float ndarray, filtered waveform data for each station,
                           concatenated.
        windowStart : integer, trigger time, in samples from start of
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
    Creates a dictionary defining the columns in the Correlation table.

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
    Creates a dictionary defining the columns in the Families table.
    
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
    
    families_dictionary = {
        "members"   : StringCol(itemsize=opt.max_famlen, shape=(), pos=0),
        "core"      : Int32Col(shape=(), pos=1),
        "startTime" : Float64Col(shape=(), pos=2),
        "longevity" : Float64Col(shape=(), pos=3),
        "printme"   : Int32Col(shape=(), pos=4),
        "lastprint" : Int32Col(shape=(), pos=5)
    }

    return families_dictionary

    
def initialize_table(opt):
    """
    Initializes and creates the hdf5 table file on disk.
    
    Saves table to file and closes it. Always overwrites an existing file.
    
    Parameters
    ----------
    opt : Options object
        Describes the run parameters.
    
    """
    
    if opt.verbose: print("Writing hdf5 table: {}".format(opt.filename))
    
    # Open file
    h5file = open_file(opt.filename, mode="w", title=opt.title)
    group = h5file.create_group("/", opt.groupName, opt.title)
    
    # Create repeaters table, populate attributes based on settings in opt
    # that can be checked to ensure compatibility with current configuration
    rtable = h5file.create_table(group, "repeaters", Repeaters(opt),
                                 "Repeater Catalog")
    rtable.attrs.scnl = [opt.station, opt.channel, opt.network, opt.location]
    rtable.attrs.samprate = opt.samprate
    rtable.attrs.windowLength = opt.winlen
    rtable.attrs.ptrig = opt.ptrig
    rtable.attrs.atrig = opt.atrig
    rtable.attrs.fmin = opt.fmin
    rtable.attrs.fmax = opt.fmax
    # previd keeps track of the most recently used unique id number 
    rtable.attrs.previd = 0
    # ptime keeps track of the most recently added trigger
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
    ftable.attrs.nClust = 0 # Number of clusters
    ftable.attrs.allowed_max_famlen = opt.max_famlen
    ftable.attrs.current_max_famlen = 0
    ftable.flush()

    h5file.close()


def open_with_cfg(configfile, verbose=False, troubleshoot=False):
    """
    Convenience function to open the hdf5 file and opt with a config file.
    
    Parameters
    ----------
    configfile : str
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.
    troubleshoot : bool, optional
        Escape try/except statements to diagnose problems.
    
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
    opt : Options object
        Describes the run parameters.
    
    """
    
    opt = redpy.config.Options(configfile, verbose, troubleshoot)
    
    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable = \
        redpy.table.open_table(opt)
    
    return h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt


def open_table(opt):
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
    
    if opt.verbose: print(f'Opening hdf5 table: {opt.filename}')
    
    h5file = open_file(opt.filename, "a")
    
    rtable = eval('h5file.root.' + opt.groupName + '.repeaters')
    otable = eval('h5file.root.' + opt.groupName + '.orphans')
    ctable = eval('h5file.root.' + opt.groupName + '.correlation')
    ttable = eval('h5file.root.' + opt.groupName + '.triggers')
    jtable = eval('h5file.root.' + opt.groupName + '.junk')
    dtable = eval('h5file.root.' + opt.groupName + '.deleted')
    ftable = eval('h5file.root.' + opt.groupName + '.families')
    
    # Check for MPL version mismatch
    check_epoch_date(rtable, ftable, ttable, otable, dtable, opt)
    
    # Check attributes in ftable, fill if missing
    ftable_compatibility_check(ftable, opt)
    
    print_stats(rtable, otable, ftable, opt)
    
    return h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable


def print_stats(rtable, otable, ftable, opt):
    """
    Prints the current number of orphans, repeaters, and families.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    otable : Table object
        Handle to the Orphans table.
    ftable : Table object
        Handle to the Families table.
    opt : Options object
        Describes the run parameters.
    
    """
    if opt.verbose:
        print(f'Number of orphans   : {len(otable)}')
        print(f'Number of repeaters : {len(rtable)}')
        print(f'Number of families  : {ftable.attrs.nClust}')


def calculate_window_amplitude(data, trigger_sample, opt):
    """
    Calculates the maximum waveform amplitudes for 'windowAmp'.
    
    Calculations are done for the first half of the window for each station.
    
    Parameters
    ----------
    data : float ndarray
        Waveform data for all stations, appended together.
    trigger_sample : integer
        Trigger time in samples.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    amps : float list
        Array of maximum amplitudes.
    
    """
    
    # !!! Enable shift by 10% later
    window_start = trigger_sample #- opt.winlen/10
    
    amps = np.zeros(opt.nsta)
    for n in range(opt.nsta):
        amps[n] = np.max(np.abs(data[int((n*opt.wshape) + window_start):int(
            (n*opt.wshape) + window_start + opt.winlen/2)]))
    
    # !!! Also use the whole window instead of first half
    
    return amps
    

def populate_triggers(ttable, trigs, ttimes, opt):
    """
    Populates new rows in the Triggers table from a list of triggers.
    
    Parameters
    ----------
    ttable : Table object
        Handle to the Triggers table.
    trigs : list of Trace objects
        Output from triggering function, with data from all stations appended.
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    trigs : list of Trace objects
        Triggers without duplicates found in Triggers table or too close to
        previous triggers.
    
    """
    
    for t in trigs:
        
        trigtime = t.stats.starttime.matplotlib_date
        if not len(np.intersect1d(
                   np.where(ttimes > trigtime - opt.mintrig/86400),
                   np.where(ttimes < trigtime + opt.mintrig/86400))):
            
            trigger = ttable.row
            trigger['startTimeMPL'] = trigtime
            trigger.append()
            ttable.flush()
            
        else:
            trigs.remove(t)
    
    return trigs


def populate_junk(jtable, trig, isjunk, opt):
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

    
def populate_orphan(otable, idnum, trig, opt):
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
    orow['startTimeMPL'] = trig.stats.starttime.matplotlib_date
    orow['waveform'] = trig.data
    orow['windowStart'] = windowStart
    orow['windowCoeff'], orow['windowFFT'], orow['FI'] = \
        redpy.correlation.calculate_window(trig.data, windowStart, opt)
    orow['windowAmp'] = calculate_window_amplitude(trig.data, windowStart,
        opt)
    
    # Determine expiration date based on STA/LTA amplitude
    add_days = np.min([opt.maxorph,((opt.maxorph-opt.minorph)/opt.maxorph)*(
        trig.stats.maxratio)+opt.minorph])
    orow['expires'] = (trig.stats.starttime+add_days*86400).isoformat()
    
    orow.append()
    otable.flush()


def clear_expired_orphans(otable, end_time, opt):
    """
    Deletes orphans that have passed their expiration date.
    
    Parameters
    ----------
    otable : Table object
        Handle to the Orphans table.
    end_time : UTCDateTime object
        Time to remove orphans older than.
    opt : Options object
        Describes the run parameters.
    
    """
    
    expired = np.empty(0).astype(int)
    for n in range(len(otable)):
        if otable.cols.expires[n].decode('utf-8') < end_time.isoformat():
            expired = np.append(expired,n)
    
    if len(expired) > 0:
        if len(expired) != len(otable):
            for n in range(len(expired)-1,-1,-1):
                otable.remove_row(expired[n])
        else:
            for n in range(len(expired)-1,0,-1):
                otable.remove_row(expired[n])
            # Deal with edge case where the last remaining orphan is slated
            # for removal. The table can't be empty, so we set the windowCoeff
            # to be 0 so it will never correlate, and set it to expire as
            # soon as a new orphan is found.
            otable.cols.windowCoeff[0] = 0
            otable.cols.expires[0] = (UTCDateTime(otable.cols.startTime[0]) \
                                                  - 86400).isoformat()
        otable.flush()
    

def move_orphan(rtable, otable, oindex, opt):
    """
    Moves a row from the 'Orphans' table to the 'Repeater' table.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    otable : Table object
        Handle to the Orphans table.
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
        # 0 so it will never correlate, and set it to expire as soon as a new
        # orphan is found.
        coeff, fft, fi = redpy.correlation.calculate_window(orow['waveform'],
                                                    orow['windowStart'], opt)
        rrow['windowCoeff'] = coeff
        rrow['windowFFT'] = fft
        rrow['FI'] = fi
        otable.cols.windowCoeff[oindex] = 0*otable.cols.windowCoeff[oindex]
        otable.cols.expires[oindex] = (UTCDateTime(
                                       orow['startTime'])-86400).isoformat()
    rrow['windowAmp'] = orow['windowAmp']
    
    rrow.append()
    
    if len(otable) > 1:
        otable.remove_row(oindex)
    
    rtable.flush()
    otable.flush()


def populate_repeater(rtable, idnum, trig, opt, windowStart=-1):
    """
    Populates a new row in the 'Repeater' table from trigger.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    idnum : integer
        Unique ID number given to this event.
    trig : Trace object
        Output from triggering function, with data from all stations appended.
    opt : Options object
        Describes the run parameters.
    windowStart : integer, optional
        Trigger time in samples from start of waveform, defaults to
        opt.ptrig seconds.
    
    """
    
    # Create an empty row
    rrow = rtable.row
    
    if windowStart == -1:
        windowStart = int(opt.ptrig*opt.samprate)
    
    rrow['id'] = idnum
    rrow['startTime'] = trig.stats.starttime.isoformat()
    rrow['startTimeMPL'] = trig.stats.starttime.matplotlib_date
    rrow['waveform'] = trig.data
    rrow['windowStart'] = windowStart
    rrow['windowCoeff'], rrow['windowFFT'], rrow['FI'] = \
        redpy.correlation.calculate_window(trig.data, windowStart, opt)
    rrow['windowAmp'] = calculate_window_amplitude(trig.data, windowStart,
        opt)
    
    rrow.append()
    rtable.flush()
    

def populate_correlation(ctable, id1, id2, ccc, opt):
    """
    Populates a new row in the 'Correlation' table.
    
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


def populate_new_family(rtable, ftable, members, core, opt):
    """
    Populates a new family from two or more events.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
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
        reorder_families(ftable, opt)
    
    
def reorder_families(ftable, opt):
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
        # Get all the rows
        members = ftable.cols.members[:]
        cores = ftable.cols.core[:]
        longevity = ftable.cols.longevity[:]
        printme = ftable.cols.printme[:]
        lastprint = ftable.cols.lastprint[:]
        
        # Rearrange them
        ftable.cols.startTime[:] = startTimes[order]
        ftable.cols.members[:] = members[order]
        ftable.cols.longevity[:] = longevity[order]
        ftable.cols.core[:] = cores[order]
        ftable.cols.printme[:] = printme[order]
        ftable.cols.lastprint[:] = lastprint[order]
        
        ftable.flush()
    
    
def merge_families(rtable, ctable, ftable, famlist, laglist, opt):
    """
    Combines families that have been merged by adding a new event.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ctable : Table object
        Handle to the Correlation table.
    ftable : Table object
        Handle to the Families table.
    famlist : integer list
        List of families to merge.
    laglist : integer list
        List of lags between families and new event.
    opt : Options object
        Describes the run parameters.
    
    """
    
    # Determine which family is largest, use that as base family
    nmembers = []
    for n in range(len(famlist)):
        nmembers.append(len(np.fromstring(ftable[famlist[n]]['members'],
                                          dtype=int, sep=' ')))
    laglist = np.array(laglist)
    laglist = laglist - laglist[np.argmax(nmembers)]
    
    # Adjust laglist relative to largest family, update windows
    for n in range(len(famlist)):
        if laglist[n]!=0:
            members = np.fromstring(ftable[famlist[n]]['members'], dtype=int,
                                    sep=' ')
            for m in members:
                redpy.correlation.update_window(rtable, m, -laglist[n], opt)
            rtable.flush()
    
    # Perform merge, run clustering to find best new core
    f1 = min(famlist)
    maxmem = ftable.cols.members[f1].decode('utf-8').count(' ')+1
    for f2 in np.sort(famlist)[::-1]:
        if f2!=f1:
            maxmem = np.max((maxmem, ftable.cols.members[f2].decode(
                                                       'utf-8').count(' ')+1))
            ftable.cols.members[f1] = ftable.cols.members[f1].decode(
                'utf-8')+' '+ftable[f2]['members'].decode('utf-8')
            ftable.remove_row(f2)
            ftable.attrs.nClust -= 1
    ftable.cols.lastprint[f1] = -1
    merge = maxmem/(ftable.cols.members[f1].decode('utf-8').count(' ')+1)
    redpy.cluster.update_family(rtable, ctable, ftable, f1, opt, merge=merge)
    reorder_families(ftable, opt)


def remove_all_junk(jtable, opt):
    """
    Remove all but the last row of the junk table.
    
    Parameters
    ----------
    jtable : Table object
        Handle to the Junk table.
    opt : Options object
        Describes the run parameters.
    
    """
    if opt.verbose:
        print('Removing junk...')
    if len(jtable) > 1:
        # We have to leave at least one row
        for i in range(len(jtable)-1, 0, -1):
            jtable.remove_row(i)
        jtable.flush()
    else:
        if opt.verbose:
            print('No junk to remove!')


def remove_families(rtable, ctable, dtable, ftable, remove_clusters, opt,
    verbose=False):
    """
    Removes families from catalog.
    
    Specifically, it removes the families from the Families table, removes the
    cross-correlation values from the Correlation table for members of those
    families, moves the core of the families into the Deleted table, and
    removes the rest of the members from the Repeaters table.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ctable : Table object
        Handle to the Correlation table.
    dtable : Table object
        Handle to the Deleted table.
    ftable : Table object
        Handle to the Families table.
    remove_clusters : integer list
        List of clusters/rows to remove from the Families table.
    opt : Options object
        Describes the run parameters.
    verbose : bool, optional
        Enable additional print statements.
    
    """
    
    if verbose: print('Getting family members to remove...')
    remove_clusters = np.sort(remove_clusters)[::-1] # Process in reverse
    old_rrows = list(range(len(rtable)))
    transform = np.zeros((len(rtable),)).astype(int)
    old_cores = ftable.cols.core[:]
    
    # Get members of each cluster and remove row in Families table
    members = np.array([])
    for cluster in remove_clusters:
        members = np.append(members, np.fromstring(ftable[cluster]['members'],
                                                   dtype=int, sep=' '))
        ftable.remove_row(cluster)
        ftable.flush()
    ftable.attrs.nClust-=len(remove_clusters)
    members = np.sort(members).astype('uint32')
    
    # !!! Provide option to not do this step for small families?
    # Populate cores in dtable before removing them from rtable
    if verbose: print('Moving cores to deleted table...')
    cores = rtable[np.intersect1d(members, old_cores)]
    for core in cores:
        drow = dtable.row
        drow['id'] = core['id']
        drow['startTime'] = core['startTime'][0] #!!! why the [0]?
        drow['startTimeMPL'] = core['startTimeMPL']
        drow['waveform'] = core['waveform']
        drow['windowStart'] = core['windowStart']
        drow['windowCoeff'] = core['windowCoeff']
        drow['windowFFT'] = core['windowFFT']
        drow['windowAmp'] = core['windowAmp']
        drow['FI'] = core['FI']
        drow.append()
    
    # !!! Functionalize finding rows in the Correlation table? !!!
    if verbose: print('Updating correlation table...')
    ids = rtable.cols.id[:]
    ids = ids[members]
    id2 = ctable.cols.id2[:]
    idxc = np.where(np.in1d(id2,ids))[0]
    for c in idxc[::-1]:
        ctable.remove_row(c)
    
    if verbose: print('Updating repeater table...')    
    # Remove rows from table and list
    for m in members[::-1]:
        rtable.remove_row(m)
        old_rrows.remove(m)
    
    if verbose: print('Updating family table...')
    # Update members of Families table with new row locations
    transform[old_rrows] = range(len(rtable))
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
    
    if verbose: print('Done removing families!')


def remove_small_families(rtable, ctable, dtable, ftable, ttable, minmembers,
    maxdays, seedtime, opt, verbose=False, list_only=False):
    """
    Searches for old, small families and removes them.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ctable : Table object
        Handle to the Correlation table.
    dtable : Table object
        Handle to the Deleted table.
    ftable : Table object
        Handle to the Families table.
    ttable : Table object
        Handle to the Triggers table.
    minmembers : integer
        Minimum number of members needed to keep a family.
    maxdays : float !!! change to "maxage"
        Maximum age relative to seedtime (days).
    seedtime : string
        Date from which to measure maxdays.
    opt : Options object
        Describes the run parameters.
    verbose : bool, optional
        Enable additional print statements.
    list_only : bool, optional
        Skip deletion step and return only the list.
    
    Returns
    -------
    removed_families : 
        List of family numbers that were (or were slated to be) removed.
    
    """
    
    # !!! Need to consider further how to better incorporate the verbosity
    # !!! and listing to better match the rest of the code
    
    # If using list_only mode, automatically invoke verbose mode too
    if list_only:
        verbose = True
    
    if verbose:
        print("\n::: table.removeSmallFamilies()")
        print("::: - minmembers : {}".format(minmembers))
        print("::: - maxdays    : {}".format(maxdays))
        print("::: - seedtime   : {}\n".format(seedtime))
        print("::: Member count per Family :::")
        print("#{:>12s} | {:>12s} | {:>12s} | {:<12s}".format(
            "Family #", "Members", "Age (d)", "Fate"))
    
    removed_families = []  # list of families to be removed
    nremoved = 0  # total number of repeaters removed
    for i in range(len(ftable)):
        # Initialize fate of family as "keep," for printing purposes only
        fate = "keep"
        # Number of repeaters in the family
        n = len(np.fromstring(ftable[i]['members'], dtype=int, sep=' '))
        
        # a = age in days (measured from family beginning), see above
        a = (seedtime - (UTCDateTime(mdates.num2date(ftable[i]["startTime"])
             ))) / 86400
        
        # If family has too few members and is too old
        if n < minmembers and a > maxdays:
            removed_families.append(i)  # Append it to the list to remove
            fate = "REMOVE"  # Update fate
            nremoved += n  # Keep track of total number of repeaters removed
        if verbose:
            print("#{:>12d} | {:12d} | {:>12.2f} |  {:<12s}".format(i, n, a,
                                                                        fate))
    
    if verbose:
        print("\nRemoved families    : {}".format(removed_families))
        print("# Families removed  : {}/{}".format(len(removed_families),
                                                                len(ftable)))
        percent_removed = nremoved/len(rtable)*100
        print("# Repeaters removed : {}/{} ({:2.1f}%)\n".format(nremoved,
                                               len(rtable), percent_removed))
    
    if list_only:
        # If list_only (-l), do not execute removeFamilies()
        print("Families listed were not removed!")
        print("Remove -l flag to actually modify table.")
    else:
        # removeFamilies() if there are families to remove
        if removed_families:
            remove_families(rtable, ctable, dtable, ftable, removed_families,
                                                                opt, verbose)
        else:
            if verbose:
                print("No families to remove.")
    
    return removed_families


def check_epoch_date(rtable, ftable, ttable, otable, dtable, opt):
    """
    Checks for mismatch in reference epoch with current matplotlib version.
    
    This check is done because the reference epoch was changed in matplotlib
    v3.3, so a table generated using matplotlib <v3.3 will have datenumbers in
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
        Handle to the Triggers table.
    otable : Table object
        Handle to the Orphans table.
    dtable : Table object
        Handle to the Deleted table.
    opt : Options object
        Describes the run parameters.
    
    """
    
    if len(ttable) > 0:
        
        if ttable.cols.startTimeMPL[0] > mdates.date2num(
            np.datetime64('now')):
            # Explicitly assumes the first trigger will have the issue
            
            print('Found matplotlib version mismatch! Fixing...')
            reftime = mdates.date2num(np.datetime64('now'))
            epoch = mdates.date2num(np.datetime64('0000-12-31'))
            
            # Loop over tables and entries in tables to fix
            for table in [ttable, otable, rtable, dtable]:
                for i in range(len(table)):
                    if table.cols.startTimeMPL[i] > reftime:
                        table.cols.startTimeMPLp[i] = \
                            table.cols.startTimeMPL[i] + epoch
                table.flush()
            
            # Fix ftable separately because startTime instead of startTimeMPL
            for i in range(len(ftable.cols.startTime[:])):
                if ftable.cols.startTime[i] > reftime:
                    ftable.cols.startTime[i] = \
                        ftable.cols.startTime[i] + epoch
            ftable.flush()
        
        elif ttable.cols.startTimeMPL[0] < mdates.date2num(
            np.datetime64('1900-01-01')):
            # Same problem, but with the opposite sense, in case current
            # version is outdated
            
            print('Found matplotlib version mismatch! Fixing...')
            reftime = mdates.date2num(np.datetime64('1900-01-01'))
            epoch = mdates.date2num(np.datetime64('1970-01-01'))
            
            # Loop over tables and entries in tables to fix
            for table in [ttable, otable, rtable, dtable]:
                for i in range(len(table.cols.startTimeMPL[:])):
                    if table.cols.startTimeMPL[i] < reftime:
                        table.cols.startTimeMPL[i] = \
                            table.cols.startTimeMPL[i] + epoch
                table.flush()
            
            # Fix ftable separately because startTime instead of startTimeMPL
            for i in range(len(ftable.cols.startTime[:])):
                if ftable.cols.startTime[i] < reftime:
                    ftable.cols.startTime[i] = \
                        ftable.cols.startTime[i] + epoch


def ftable_compatibility_check(ftable, opt):
    """
    Compatibility check to fill ftable.attrs.*_max_famlen on an old table.
    
    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    opt : Options object
        Describes the run parameters.
    
    """
    
    if not 'allowed_max_famlen' in ftable.attrs._f_list('user'):
        ftable.attrs.allowed_max_famlen = 1000000
        ftable.attrs.current_max_famlen = np.max(
            [len(ftable[i]['members']) for i in range(len(ftable))])


def check_famlen(h5file, rtable, otable, ttable, ctable, jtable, dtable,
                                                                 ftable, opt):
    """
    Checks if the string holding family members is too long and expands table.
    
    When the current maximum family string length exceeds 45% of the current
    allotted maximum length, the Families table needs to be expanded.
    
    Unfortunately, in order to do that we need to copy everything over into
    a new file. Additionally, we use 45% instead of something higher like
    95% to allow for the edge case that the largest family and the second
    largest family with similar lengths are merged.
    
    It's still possible this is too generous based on how often the check is
    done.
    
    Parameters
    ----------
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
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    h5file : File object
        Updated handle to the h5 file.
    rtable : Table object
        Updated handle to the Repeaters table.
    otable : Table object
        Updated handle to the Orphans table.
    ttable : Table object
        Updated handle to the Triggers table.
    ctable : Table object
        Updated handle to the Correlation table.
    jtable : Table object
        Updated handle to the Junk table.
    dtable : Table object
        Updated handle to the Deleted table.
    ftable : Table object
        Updated handle to the Families table.
    opt : Options object
        Updated description of the run parameters.
    
    """
    
    if ftable.attrs.current_max_famlen >= 0.45*ftable.attrs.allowed_max_famlen:
        print('Approaching maximum family length Families table can hold!')
        print('Automatically expanding hdf5 file to compensate...')
        
        h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
            expand_table(h5file, ftable, opt,
                         max_famlen=3*ftable.attrs.allowed_max_famlen)
        
    return h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt


def set_ftable_columns(ftable, opt, plotall=False, resetlp=False, startfam=0,
                       endfam=0):
    """
    Reset 'printme' and 'lastprint' columns of Families table.

    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    opt : Options object
        Describes the run parameters.
    plotall : bool, optional
        If True, completely resets 'printme' column so all families are output.
    resetlp : bool, optional
        If True, sets 'lastprint' column to match row index.
    startfam : int, optional
        Starting family to generate plots for. May be negative to count
        backward from last family.
    endfam : int, optional
        Ending family to generate plots for. May be negative to count backward
        from last family.

    """
    if plotall:
        if opt.verbose:
            print('Resetting plotting column...')
        ftable.cols.printme[:] = np.ones(ftable.attrs.nClust)
    if resetlp:
        if opt.verbose:
            print('Resetting last print column...')
        ftable.cols.lastprint[:] = np.arange(ftable.attrs.nClust)
    if startfam or endfam:
        if startfam < 0:
            startfam = ftable.attrs.nClust + startfam
        if endfam < 0:
            endfam = ftable.attrs.nClust + endfam
        if (startfam > endfam) and endfam:
            raise ValueError('startfam is larger than endfam!')
        if startfam == endfam:
            print('startfam is equal to endfam; no plots will be produced.')
        if startfam >= ftable.attrs.nClust-1:
            raise ValueError('startfam is larger than the number of available '
                             f'families ({ftable.attrs.nClust})!')
        if endfam > ftable.attrs.nClust:
            raise ValueError('endfam is larger than the number of available '
                             f'families ({ftable.attrs.nClust})!')
        if startfam < 0:
            raise ValueError('startfam cannot be less than '
                             f'-{ftable.attrs.nClust}')
        ftable.cols.printme[:] = np.zeros(ftable.attrs.nClust)
        if startfam and not endfam:
            ftable.cols.printme[startfam:] = np.ones(
                ftable.attrs.nClust - startfam)
        elif endfam and not startfam:
            ftable.cols.printme[:endfam] = np.ones(endfam)
        else:
            ftable.cols.printme[startfam:endfam] = np.ones(endfam - startfam)


def expand_table(h5file, ftable, optfrom, optto=None, max_famlen=None,
                                                              do_plot=False):
    """
    Expands an existing table to make more room by copying data to a new one.
    
    Has the capacity to expand the Families table to lengthen the maximum
    allowed string length (and therefore members in a family) as well as allow
    new channels to be appended to the end of tables that contain waveforms.
    
    Parameters
    ----------
    h5file : File object
        Handle to the h5 file.
    ftable : Table object
        Handle to the Families table.
    optfrom : Options object
        Describes the current run parameters.
    optto : Options object, optional
        Describes the transformed run parameters.
    max_famlen : int, optional
        New maximum family string length.
    do_plot : bool, optional
        If true, update the 'printme' column in ftable
    
    Returns
    -------
    h5fileto : File object
        Updated handle to the h5 file.
    rtableto : Table object
        Updated handle to the Repeaters table.
    otableto : Table object
        Updated handle to the Orphans table.
    ttableto : Table object
        Updated handle to the Triggers table.
    ctableto : Table object
        Updated handle to the Correlation table.
    jtableto : Table object
        Updated handle to the Junk table.
    dtableto : Table object
        Updated handle to the Deleted table.
    ftableto : Table object
        Updated handle to the Families table.
    optto : Options object
        Updated description of the run parameters.
    
    """
    
    if max_famlen:
        optto = optfrom.copy()
        optto.max_famlen = max_famlen
    else:
        max_famlen = optto.max_famlen
    
    # Check to ensure optto.max_famlen is long enough to hold longest string
    while max_famlen < 3*ftable.attrs.current_max_famlen:
        max_famlen *= 3
    
    # Check if additional channels need to be appended
    dsta = optto.nsta - optfrom.nsta
    # Complain if there are fewer, as this isn't supported
    if dsta < 0:
        raise ValueError(f'New config file must have nsta >= {optfrom.nsta}')
    
    # Close currently open table
    h5file.close()
    
    # Rename and change filename in optfrom to point to the .old version
    os.rename(optfrom.filename,'{}.old'.format(optfrom.filename))
    
    # Change filename in optfrom to point to the .old version
    optfrom.filename = '{}.old'.format(optfrom.filename)
    
    # Initialize new table
    redpy.table.initialize_table(optto)
    
    # Open tables
    h5filefrom, rtablefrom, otablefrom, ttablefrom, ctablefrom, jtablefrom, \
        dtablefrom, ftablefrom = redpy.table.open_table(optfrom)
    h5fileto, rtableto, otableto, ttableto, ctableto, jtableto, \
        dtableto, ftableto = redpy.table.open_table(optto)
    
    # Do all the copying!
    print('Copying data into new table... please wait...')
    
    for rfrom in rtablefrom.iterrows():
        rto = rtableto.row
        # These stay the same
        rto['id'] = rfrom['id']
        rto['startTime'] = rfrom['startTime']
        rto['startTimeMPL'] = rfrom['startTimeMPL']
        rto['windowStart'] = rfrom['windowStart']
        # These can be extended
        rto['windowAmp'] = np.append(rfrom['windowAmp'] ,np.zeros(dsta))
        rto['windowCoeff'] = np.append(rfrom['windowCoeff'], np.zeros(dsta))
        rto['FI'] = np.append(rfrom['FI'], np.empty(dsta)*np.nan)
        rto['waveform'] = np.append(rfrom['waveform'],
                                    np.zeros(dsta*optto.wshape))
        rto['windowFFT'] = np.append(rfrom['windowFFT'],
                                    np.zeros(dsta*optto.winlen))
        rto.append()
    rtableto.attrs.ptime = rtablefrom.attrs.ptime
    rtableto.attrs.previd = rtablefrom.attrs.previd
    rtableto.flush()
    
    for ofrom in otablefrom.iterrows():
        oto = otableto.row
        # These stay the same
        oto['id'] = ofrom['id']
        oto['startTime'] = ofrom['startTime']
        oto['startTimeMPL'] = ofrom['startTimeMPL']
        oto['windowStart'] = ofrom['windowStart']
        oto['expires'] = ofrom['expires']
        # These can be extended
        oto['windowAmp'] = np.append(ofrom['windowAmp'], np.zeros(dsta))
        oto['windowCoeff'] = np.append(ofrom['windowCoeff'], np.zeros(dsta))
        oto['FI'] = np.append(ofrom['FI'], np.empty(dsta)*np.nan)
        oto['waveform'] = np.append(ofrom['waveform'],
                                    np.zeros(dsta*optto.wshape))
        oto['windowFFT'] = np.append(ofrom['windowFFT'],
                                    np.zeros(dsta*optto.winlen))
        oto.append()
    otableto.flush()
    
    for tfrom in ttablefrom.iterrows():
        tto = ttableto.row
        # This stays the same
        tto['startTimeMPL'] = tfrom['startTimeMPL']
        tto.append()
    ttableto.flush()
    
    for dfrom in dtablefrom.iterrows():
        dto = dtableto.row
        # These stay the same
        dto['id'] = dfrom['id']
        dto['startTime'] = dfrom['startTime']
        dto['startTimeMPL'] = dfrom['startTimeMPL']
        dto['windowStart'] = dfrom['windowStart']
        # These can be extended
        dto['windowAmp'] = np.append(dfrom['windowAmp'], np.zeros(dsta))
        dto['windowCoeff'] = np.append(dfrom['windowCoeff'], np.zeros(dsta))
        dto['FI'] = np.append(dfrom['FI'], np.empty(dsta)*np.nan)
        dto['waveform'] = np.append(dfrom['waveform'],
                                    np.zeros(dsta*optto.wshape))
        dto['windowFFT'] = np.append(dfrom['windowFFT'],
                                    np.zeros(dsta*optto.winlen))
        dto.append()
    dtableto.flush()
    
    for jfrom in jtablefrom.iterrows():
        jto = jtableto.row
        # These stay the same
        jto['startTime'] = jfrom['startTime']
        jto['windowStart'] = jfrom['windowStart']
        jto['isjunk'] = jfrom['isjunk']
        # This can be extended
        jto['waveform'] = np.append(jfrom['waveform'],
                                    np.zeros(dsta*optto.wshape))
        jto.append()
    jtableto.flush()
    
    for cfrom in ctablefrom.iterrows():
        cto = ctableto.row
        # All stay the same
        cto['id1'] = cfrom['id1']
        cto['id2'] = cfrom['id2']
        cto['ccc'] = cfrom['ccc']
        cto.append()
    ctableto.flush()
    
    for ffrom in ftablefrom.iterrows():
        fto = ftableto.row
        # All stay the same, except maybe printme
        fto['members'] = ffrom['members']
        fto['core'] = ffrom['core']
        fto['startTime'] = ffrom['startTime']
        fto['longevity'] = ffrom['longevity']
        fto['lastprint'] = ffrom['lastprint']
        if do_plot:
            fto['printme'] = 1
        else:
            fto['printme'] = ffrom['printme']
        fto.append()
    ftableto.attrs.nClust = ftablefrom.attrs.nClust
    ftableto.attrs.current_max_famlen = ftablefrom.attrs.current_max_famlen
    ftableto.attrs.allowed_max_famlen = max_famlen
    ftableto.flush()
    
    # Clean up old table
    h5filefrom.close()
    os.remove(optfrom.filename)
    
    return h5fileto, rtableto, otableto, ttableto, ctableto, jtableto, \
        dtableto, ftableto, optto


def update_tables(h5file, rtable, otable, ttable, ctable, jtable, dtable,
                  ftable, ttimes, filekey, preload_waveforms, preload_end_time,
                  run_end_time, window_start_time, window_end_time, opt,
                  event_list=[], event=None):
    """
    Primary processing loop to update the tables with data in a time window.
    
    This function handles downloading, triggering, and populating the tables
    given a time window (window_start_time, window_end_time).
    
    Parameters
    ----------
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
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    filekey : DataFrame object
        Keys file names of local waveform data to their metadata.
    preload_waveforms : Stream object
        Stream containing waveforms 'preloaded' into memory.
    preload_end_time : UTCDateTime object
        End time of preloaded waveforms.
    run_end_time : UTCDateTime object
        End time of full span of interest.
    window_start_time : UTCDateTime object
        Start time of window to process.
    window_end_time : UTCDateTime object
        End time of window to process.
    opt : Options object
        Describes the run parameters.
    event_list : ndarray of UTCDateTime objects, optional
        List of catalog events to add.
    event : UTCDateTime object, optional
        Catalog event to add by force.
    
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
    preload_waveforms : Stream object
        Stream containing waveforms 'preloaded' into memory.
    preload_end_time : UTCDateTime object
        End time of preloaded waveforms.
    opt : Options object
        Describes the run parameters.
    
    """
    # Check to make sure we have space
    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
        check_famlen(h5file, rtable, otable, ttable, ctable, jtable, dtable,
                     ftable, opt)
    # Check if we need to preload more data
    preload_waveforms, preload_end_time = redpy.trigger.preload_check(
        window_start_time, window_end_time, preload_end_time, run_end_time,
        filekey, opt, preload_waveforms=preload_waveforms,
        event_list=event_list)
    # Download and trigger
    alltrigs = redpy.trigger.load_and_trigger(
        rtable, window_start_time, window_end_time, filekey,
        preload_waveforms, opt, event=event)
    # Populate tables with triggers as appropriate
    populate_tables(rtable, otable, ttable, ctable, jtable, dtable, ftable,
                    ttimes, alltrigs, opt, event=event)
    return (h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable,
            preload_waveforms, preload_end_time, opt)


def populate_tables(rtable, otable, ttable, ctable, jtable, dtable, ftable,
                    ttimes, alltrigs, opt, event=None):
    """
    Populates tables with new triggers as appropriate.
    
    Parameters
    ----------
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
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    alltrigs : Stream object
        Stream of new triggers to process.
    opt : Options object
        Describes the run parameters.
    event : UTCDateTime object, optional
        Catalog event to add by force.
    
    """
    trigs, junk, jtype = redpy.trigger.clean_triggers(alltrigs, opt,
                                                      event=event)
    # !!! This step already goes through and calculates the window, can I
    # !!! pass that down the line to save some duplicate calculations?
    # Save junk triggers in separate table for quality checking purposes
    for i in range(len(junk)):
        populate_junk(jtable, junk[i], jtype[i], opt)
    # Append times of triggers to ttable to compare total seismicity later
    trigs = populate_triggers(ttable, trigs, ttimes, opt)
    # Check triggers against deleted events
    if len(dtable) > 0:
        trigs = redpy.correlation.compare_deleted(trigs, dtable, opt)
    if len(trigs) > 0:
        idnum = rtable.attrs.previd
        if len(trigs) == 1:
            ostart = 0
            if len(otable) == 0:
                # First trigger goes to orphans table
                populate_orphan(otable, 0, trigs[0], opt)
                ostart = 1
            else:
                idnum += 1
                redpy.correlation.correlate_new_triggers(
                    rtable, otable, ctable, ftable, ttimes,
                    trigs[0], idnum, opt)
        else:
            ostart = 0
            if len(otable) == 0:
                # First trigger goes to orphans table
                populate_orphan(otable, 0, trigs[0], opt)
                ostart = 1
            # Loop through remaining triggers
            for i in range(ostart,len(trigs)):
                idnum += 1
                redpy.correlation.correlate_new_triggers(
                    rtable, otable, ctable, ftable, ttimes,
                    trigs[i], idnum, opt)
        rtable.attrs.previd = idnum


def update_with_event_list(h5file, rtable, otable, ttable, ctable, jtable,
                         dtable, ftable, event_list, opt, force=False,
                         expire=False):
    """
    Update tables based on catalog events provided in a list.

    Parameters
    ----------
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
    event_list : ndarray of UTCDateTime objects
        List of catalog events to add.
    opt : Options object
        Describes the run parameters.
    force : bool, optional
        Force trigger at times specified in event_list.
    expire : bool, optional
        If True, expire orphans after adding each event.

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
    opt : Options object
        Describes the run parameters.

    """
    run_start_time = event_list[0] - 4*opt.atrig
    run_end_time = event_list[-1] +  5*opt.atrig + opt.maxdt
    filekey, preload_waveforms, preload_end_time = \
        redpy.trigger.initial_data_preload(run_start_time, run_end_time, opt)
    if rtable.attrs.ptime:
        rtable.attrs.ptime = UTCDateTime(run_start_time)
    for event_time in event_list:
        if opt.verbose:
            print(event_time)
        window_start_time = event_time - 4*opt.atrig
        window_end_time = event_time + 5*opt.atrig + opt.maxdt
        if len(ttable) > 0:
            ttimes = ttable.cols.startTimeMPL[:]
        else:
            ttimes = 0
        event = event_time if force else None
        h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, \
            preload_waveforms, preload_end_time, opt = update_tables(
                h5file, rtable, otable, ttable, ctable, jtable, dtable,
                ftable, ttimes, filekey, preload_waveforms, preload_end_time,
                run_end_time, window_start_time, window_end_time, opt,
                event_list=event_list, event=event)
        if expire:
            clear_expired_orphans(otable, window_end_time, opt)
        print_stats(rtable, otable, ftable, opt)
    return h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt


def update_with_continuous(h5file, rtable, otable, ttable, ctable, jtable,
        dtable, ftable, opt, start_time=None, end_time=None, nsec=None):
    """
    Update tables with continuous data for a fixed time period.

    Parameters
    ----------
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
    opt : Options object
        Describes the run parameters.
    start_time : str, optional
        Starting time. If not provided, will default to either the end of the
        previous run time or opt.nsec seconds prior to end_time.
    end_time : str, optional
        Ending time. If not provided, will default to now.
    nsec : int, optional
        Temporarily override opt.nsec with this value.

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
    opt : Options object
        Describes the run parameters.

    """
    if nsec:
        opt.nsec = nsec
    if end_time:
        run_end_time = UTCDateTime(end_time)
    else:
        run_end_time = UTCDateTime()
    if start_time:
        run_start_time = UTCDateTime(start_time)
        if rtable.attrs.ptime:
            rtable.attrs.ptime = UTCDateTime(run_start_time)
    else:
        if rtable.attrs.ptime:
            run_start_time = UTCDateTime(rtable.attrs.ptime)
        else:
            run_start_time = run_end_time-opt.nsec
    if run_start_time > run_end_time:
        raise ValueError(
            f'Start {run_start_time} is after end {run_end_time}!')
    if len(ttable) > 0:
        ttimes = ttable.cols.startTimeMPL[:]
    else:
        ttimes = 0
    # Load data from file
    filekey, preload_waveforms, preload_end_time = \
        redpy.trigger.initial_data_preload(run_start_time, run_end_time, opt)
    i = 0
    while run_start_time + i*opt.nsec < run_end_time:
        t_iter = time.time()
        window_start_time = run_start_time + i*opt.nsec
        window_end_time = min(run_start_time+(i+1)*opt.nsec,
                              run_end_time) + opt.atrig + opt.maxdt
        print(window_start_time)
        h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, \
            preload_waveforms, preload_end_time, opt = \
            redpy.table.update_tables(
                h5file, rtable, otable, ttable, ctable, jtable, dtable,
                ftable, ttimes, filekey, preload_waveforms,
                preload_end_time, run_end_time, window_start_time,
                window_end_time, opt)
        i += 1
        redpy.table.clear_expired_orphans(otable, window_end_time, opt)
        redpy.table.print_stats(rtable, otable, ftable, opt)
        if opt.verbose:
            print('Time spent this iteration: '
                  f'{(time.time()-t_iter)/60:.3f} minutes')
    print(f'Caught up to: {window_end_time-opt.atrig}')
    return h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt
