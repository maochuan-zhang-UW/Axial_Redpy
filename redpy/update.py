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


def calculate_window_amplitude(data, trigger_sample, config):
    """
    Calculates the maximum waveform amplitudes for 'windowAmp'.

    Calculations are done for the first half of the window for each station.

    Parameters
    ----------
    data : float ndarray
        Waveform data for all stations, appended together.
    trigger_sample : int
        Trigger time in samples.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    amps : float list
        Array of maximum amplitudes.

    """

    # !!! Enable shift by 10% later
    window_start = trigger_sample #- config.get('winlen')/10

    amps = np.zeros(config.get('nsta'))
    for n in range(config.get('nsta')):
        amps[n] = np.max(np.abs(data[int((n*config.get('wshape')) + window_start):int(
            (n*config.get('wshape')) + window_start + config.get('winlen')/2)]))

    # !!! Also use the whole window instead of first half

    return amps


def populate_triggers(ttable, trigs, ttimes, config):
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
    config : Config object
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
                   np.where(ttimes > trigtime - config.get('mintrig')/86400),
                   np.where(ttimes < trigtime + config.get('mintrig')/86400))):

            trigger = ttable.row
            trigger['startTimeMPL'] = trigtime
            trigger.append()
            ttable.flush()

        else:
            trigs.remove(t)

    return trigs


def populate_junk(jtable, trig, isjunk, config):
    """
    Populates a new row in the 'Junk' table from trigger.

    !!! Rename isjunk !!!

    Parameters
    ----------
    jtable : Table object
        Handle to the Junk table.
    trig : Trace object
        Output from triggering function, with data from all stations appended.
    isjunk : int
        Flag corresponding to the type of junk.
    config : Config object
        Describes the run parameters.

    """

    jrow = jtable.row

    jrow['startTime'] = trig.stats.starttime.isoformat()
    jrow['waveform'] = trig.data
    jrow['windowStart'] = int(config.get('ptrig')*config.get('samprate'))
    jrow['isjunk'] = isjunk
    jrow.append()
    jtable.flush()


def populate_orphan(otable, idnum, trig, config):
    """
    Populates a new row in the 'Orphans' table from trigger.

    This function also determines the expiration date based on the STA/LTA
    amplitude from triggering

    Parameters
    ----------
    otable : Table object
        Handle to the Orphans table.
    idnum : int
        Unique ID number given to this event.
    trig : Trace object
        Output from triggering function, with data from all stations appended.
    config : Config object
        Describes the run parameters.

    """

    orow = otable.row

    windowStart = int(config.get('ptrig')*config.get('samprate'))

    orow['id'] = idnum
    orow['startTime'] = trig.stats.starttime.isoformat()
    orow['startTimeMPL'] = trig.stats.starttime.matplotlib_date
    orow['waveform'] = trig.data
    orow['windowStart'] = windowStart
    orow['windowCoeff'], orow['windowFFT'], orow['FI'] = \
        redpy.correlation.calculate_window(trig.data, windowStart, config)
    orow['windowAmp'] = calculate_window_amplitude(trig.data, windowStart,
        config)

    # Determine expiration date based on STA/LTA amplitude
    add_days = np.min([config.get('maxorph'),((config.get('maxorph')-config.get('minorph'))/config.get('maxorph'))*(
        trig.stats.maxratio)+config.get('minorph')])
    orow['expires'] = (trig.stats.starttime+add_days*86400).isoformat()

    orow.append()
    otable.flush()


def clear_expired_orphans(otable, end_time, config):
    """
    Deletes orphans that have passed their expiration date.

    Parameters
    ----------
    otable : Table object
        Handle to the Orphans table.
    end_time : UTCDateTime object
        Time to remove orphans older than.
    config : Config object
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


def move_orphan(rtable, otable, oindex, config):
    """
    Moves a row from the 'Orphans' table to the 'Repeater' table.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    otable : Table object
        Handle to the Orphans table.
    oindex : int
        Row in otable to move
    config :
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
                                                    orow['windowStart'], config)
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


def populate_repeater(rtable, idnum, trig, config, windowStart=-1):
    """
    Populates a new row in the 'Repeater' table from trigger.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    idnum : int
        Unique ID number given to this event.
    trig : Trace object
        Output from triggering function, with data from all stations appended.
    config : Config object
        Describes the run parameters.
    windowStart : int, optional
        Trigger time in samples from start of waveform, defaults to
        config.get('ptrig') seconds.

    """

    # Create an empty row
    rrow = rtable.row

    if windowStart == -1:
        windowStart = int(config.get('ptrig')*config.get('samprate'))

    rrow['id'] = idnum
    rrow['startTime'] = trig.stats.starttime.isoformat()
    rrow['startTimeMPL'] = trig.stats.starttime.matplotlib_date
    rrow['waveform'] = trig.data
    rrow['windowStart'] = windowStart
    rrow['windowCoeff'], rrow['windowFFT'], rrow['FI'] = \
        redpy.correlation.calculate_window(trig.data, windowStart, config)
    rrow['windowAmp'] = calculate_window_amplitude(trig.data, windowStart,
        config)

    rrow.append()
    rtable.flush()


def populate_correlation(ctable, id1, id2, ccc, config):
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
    config : Config object
        Describes the run parameters.

    """

    if (ccc >= config.get('cmin')) and (id1!=id2):
        crow = ctable.row
        crow['id1'] = min(id1, id2)
        crow['id2'] = max(id1, id2)
        crow['ccc'] = ccc
        crow.append()
        ctable.flush()


def populate_new_family(rtable, ftable, members, core, config):
    """
    Populates a new family from two or more events.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    members : int ndarray
        Array of row indices in rtable that are family members.
    core : int
        Row index of core event in rtable.
    config : Config object
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
    ftable.flush()

    if len(ftable)>1:
        reorder_families(ftable, config)


def reorder_families(ftable, config):
    """
    Ensures families are ordered by start time.

    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    config : Config object
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


def merge_families(rtable, ctable, ftable, famlist, laglist, config):
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
    famlist : int list
        List of families to merge.
    laglist : int list
        List of lags between families and new event.
    config : Config object
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
                redpy.correlation.update_window(rtable, m, -laglist[n], config)
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
    ftable.cols.lastprint[f1] = -1
    merge = maxmem/(ftable.cols.members[f1].decode('utf-8').count(' ')+1)
    redpy.cluster.update_family(rtable, ctable, ftable, f1, config, merge=merge)
    reorder_families(ftable, config)


def set_ftable_columns(ftable, config, plotall=False, resetlp=False, startfam=0,
                       endfam=0):
    """
    Reset 'printme' and 'lastprint' columns of Families table.

    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    config : Config object
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
        if config.get('verbose'):
            print('Resetting plotting column...')
        ftable.cols.printme[:] = np.ones(len(ftable))
    if resetlp:
        if config.get('verbose'):
            print('Resetting last print column...')
        ftable.cols.lastprint[:] = np.arange(len(ftable))
    if startfam or endfam:
        if startfam < 0:
            startfam = len(ftable) + startfam
        if endfam < 0:
            endfam = len(ftable) + endfam
        if (startfam > endfam) and endfam:
            raise ValueError('startfam is larger than endfam!')
        if startfam == endfam:
            print('startfam is equal to endfam; no plots will be produced.')
        if startfam >= len(ftable)-1:
            raise ValueError('startfam is larger than the number of available '
                             f'families ({len(ftable)})!')
        if endfam > len(ftable):
            raise ValueError('endfam is larger than the number of available '
                             f'families ({len(ftable)})!')
        if startfam < 0:
            raise ValueError('startfam cannot be less than '
                             f'-{len(ftable)}')
        ftable.cols.printme[:] = np.zeros(len(ftable))
        if startfam and not endfam:
            ftable.cols.printme[startfam:] = np.ones(
                len(ftable) - startfam)
        elif endfam and not startfam:
            ftable.cols.printme[:endfam] = np.ones(endfam)
        else:
            ftable.cols.printme[startfam:endfam] = np.ones(endfam - startfam)


def update_tables(h5file, rtable, otable, ttable, ctable, jtable, dtable,
                  ftable, ttimes, filekey, preload_waveforms, preload_end_time,
                  run_end_time, window_start_time, window_end_time, config,
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
    config : Config object
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
    config : Config object
        Describes the run parameters.

    """
    # Check to make sure we have space
    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, config = \
        check_famlen(h5file, rtable, otable, ttable, ctable, jtable, dtable,
                     ftable, config)
    # Check if we need to preload more data
    preload_waveforms, preload_end_time = redpy.trigger.preload_check(
        window_start_time, window_end_time, preload_end_time, run_end_time,
        filekey, config, preload_waveforms=preload_waveforms,
        event_list=event_list)
    # Download and trigger
    alltrigs = redpy.trigger.load_and_trigger(
        rtable, window_start_time, window_end_time, filekey,
        preload_waveforms, config, event=event)
    # Populate tables with triggers as appropriate
    populate_tables(rtable, otable, ttable, ctable, jtable, dtable, ftable,
                    ttimes, alltrigs, config, event=event)
    return (h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable,
            preload_waveforms, preload_end_time, config)


def populate_tables(rtable, otable, ttable, ctable, jtable, dtable, ftable,
                    ttimes, alltrigs, config, event=None):
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
    config : Config object
        Describes the run parameters.
    event : UTCDateTime object, optional
        Catalog event to add by force.

    """
    trigs, junk, jtype = redpy.trigger.clean_triggers(alltrigs, config,
                                                      event=event)
    # !!! This step already goes through and calculates the window, can I
    # !!! pass that down the line to save some duplicate calculations?
    # Save junk triggers in separate table for quality checking purposes
    for i in range(len(junk)):
        populate_junk(jtable, junk[i], jtype[i], config)
    # Append times of triggers to ttable to compare total seismicity later
    trigs = populate_triggers(ttable, trigs, ttimes, config)
    # Check triggers against deleted events
    if len(dtable) > 0:
        trigs = redpy.correlation.compare_deleted(trigs, dtable, config)
    if len(trigs) > 0:
        idnum = rtable.attrs.previd
        if len(trigs) == 1:
            ostart = 0
            if len(otable) == 0:
                # First trigger goes to orphans table
                populate_orphan(otable, 0, trigs[0], config)
                ostart = 1
            else:
                idnum += 1
                redpy.correlation.correlate_new_triggers(
                    rtable, otable, ctable, ftable, ttimes,
                    trigs[0], idnum, config)
        else:
            ostart = 0
            if len(otable) == 0:
                # First trigger goes to orphans table
                populate_orphan(otable, 0, trigs[0], config)
                ostart = 1
            # Loop through remaining triggers
            for i in range(ostart,len(trigs)):
                idnum += 1
                redpy.correlation.correlate_new_triggers(
                    rtable, otable, ctable, ftable, ttimes,
                    trigs[i], idnum, config)
        rtable.attrs.previd = idnum


def update_with_event_list(h5file, rtable, otable, ttable, ctable, jtable,
                         dtable, ftable, event_list, config, force=False,
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
    config : Config object
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
    config : Config object
        Describes the run parameters.

    """
    run_start_time = event_list[0] - 4*config.get('atrig')
    run_end_time = event_list[-1] +  5*config.get('atrig') + config.get('maxdt')
    filekey, preload_waveforms, preload_end_time = \
        redpy.trigger.initial_data_preload(run_start_time, run_end_time, config)
    if rtable.attrs.ptime:
        rtable.attrs.ptime = UTCDateTime(run_start_time)
    for event_time in event_list:
        if config.get('verbose'):
            print(event_time)
        window_start_time = event_time - 4*config.get('atrig')
        window_end_time = event_time + 5*config.get('atrig') + config.get('maxdt')
        if len(ttable) > 0:
            ttimes = ttable.cols.startTimeMPL[:]
        else:
            ttimes = 0
        event = event_time if force else None
        h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, \
            preload_waveforms, preload_end_time, config = update_tables(
                h5file, rtable, otable, ttable, ctable, jtable, dtable,
                ftable, ttimes, filekey, preload_waveforms, preload_end_time,
                run_end_time, window_start_time, window_end_time, config,
                event_list=event_list, event=event)
        if expire:
            clear_expired_orphans(otable, window_end_time, config)
        print_stats(rtable, otable, ftable, config)
    return h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, config


def update_with_continuous(h5file, rtable, otable, ttable, ctable, jtable,
        dtable, ftable, config, start_time=None, end_time=None, nsec=None):
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
    config : Config object
        Describes the run parameters.
    start_time : str, optional
        Starting time. If not provided, will default to either the end of the
        previous run time or "nsec" seconds prior to end_time.
    end_time : str, optional
        Ending time. If not provided, will default to now.
    nsec : int, optional
        Temporarily override "nsec" from config with this value.

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
    config : Config object
        Describes the run parameters.

    """
    if nsec:
        config.set_config('nsec', nsec)
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
            run_start_time = run_end_time-config.get('nsec')
    if run_start_time > run_end_time:
        raise ValueError(
            f'Start {run_start_time} is after end {run_end_time}!')
    if len(ttable) > 0:
        ttimes = ttable.cols.startTimeMPL[:]
    else:
        ttimes = 0
    # Load data from file
    filekey, preload_waveforms, preload_end_time = \
        redpy.trigger.initial_data_preload(run_start_time, run_end_time, config)
    i = 0
    while run_start_time + i*config.get('nsec') < run_end_time:
        t_iter = time.time()
        window_start_time = run_start_time + i*config.get('nsec')
        window_end_time = min(run_start_time+(i+1)*config.get('nsec'),
                              run_end_time) + config.get('atrig') + config.get('maxdt')
        print(window_start_time)
        h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, \
            preload_waveforms, preload_end_time, config = \
            redpy.table.update_tables(
                h5file, rtable, otable, ttable, ctable, jtable, dtable,
                ftable, ttimes, filekey, preload_waveforms,
                preload_end_time, run_end_time, window_start_time,
                window_end_time, config)
        i += 1
        redpy.table.clear_expired_orphans(otable, window_end_time, config)
        redpy.table.print_stats(rtable, otable, ftable, config)
        if config.get('verbose'):
            print('Time spent this iteration: '
                  f'{(time.time()-t_iter)/60:.3f} minutes')
    print(f'Caught up to: {window_end_time-config.get("atrig")}')
    return h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, config
