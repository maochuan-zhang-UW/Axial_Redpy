# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import argparse
import time

import numpy as np
from obspy import UTCDateTime

import redpy


def main():
    """
    Backfills table with data from the past.
    
    Run this script to fill the table with data from the past. If a start time
    is not specified, it will check the attributes of the repeater table to
    pick up where it left off. Additionally, if this is the first run and a
    start time is not specified, it will assume one time chunk prior to the
    end time. If an end time is not specified, "now" is assumed. The end time
    updates at the end of each time chunk processed (default: by hour, set in
    configuration). This script can be run as a cron job that will pick up
    where it left off if a chunk is missed. Use -n if you are backfilling with
    a large amount of time; it will consume less time downloading the data in
    small chunks if NSEC is an hour or a day instead of a few minutes, but at
    the cost of keeping orphans for longer.
    
    usage: backfill.py [-h] [-v] [-t] [-s STARTTIME] [-e ENDTIME]
                       [-c CONFIGFILE] [-n NSEC]
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         increase written print statements
      -t, --troubleshoot    run in troubleshoot mode (without try/except)
      -s STARTTIME, --starttime STARTTIME
                            optional start time to begin filling
                            (yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)
      -e ENDTIME, --endtime ENDTIME
                            optional end time to end filling
                            (yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)
      -c CONFIGFILE, --configfile CONFIGFILE
                            use configuration file named CONFIGFILE instead of
                            default settings.cfg
      -n NSEC, --nsec NSEC  overwrite opt.nsec from configuration file with
                            NSEC this run only
    
    """
    t_func = time.time()
    args = backfill_parse()
    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
        redpy.table.open_with_cfg(args.configfile, args.verbose)
    
    # Deal with input arguments
    if args.nsec: opt.nsec = args.nsec
    if args.endtime:
        tend = UTCDateTime(args.endtime)
    else:
        tend = UTCDateTime()
    if args.starttime:
        tstart = UTCDateTime(args.starttime)
        if rtable.attrs.ptime:
            rtable.attrs.ptime = UTCDateTime(tstart)
    else:
        if rtable.attrs.ptime:
            tstart = UTCDateTime(rtable.attrs.ptime)
        else:
            tstart = tend-opt.nsec
    if tstart > tend: raise ValueError(f'Start {tstart} is after end {tend}!')
    
    if len(ttable) > 0:
        ttimes = ttable.cols.startTimeMPL[:]
    else:
        ttimes = 0
    
    # Create or read in file key to improve local file load times
    filekey, tend_preload = redpy.trigger.get_filekey(tstart, tend, opt)
    
    n = 0
    rlen = len(rtable)
    while tstart + n*opt.nsec < tend:
        t_iter = time.time()
        
        starttime = tstart+n*opt.nsec
        endtime = np.min((tstart+(n+1)*opt.nsec, tend))+opt.atrig+opt.maxdt
        
        print(starttime)
        
        ####
        
        # Check to make sure we have space
        h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
            redpy.table.check_famlen(h5file, rtable, otable, ttable, ctable,
                                     jtable, dtable, ftable, opt)
        
        # Preload check
        if (opt.preload > 0) and (len(filekey) > 0):
            print(f'{endtime+opt.maxdt} {tend_preload}')
            if endtime+opt.maxdt > tend_preload:
                if opt.verbose: print('Loading waveforms into memory...')
                tend_preload = np.min((tend,
                    starttime+opt.preload*86400))+opt.atrig+opt.maxdt
                st_preload = redpy.trigger.preload_data(
                    starttime-opt.atrig, tend_preload, filekey, opt)
        else:
            st_preload = []
        
        # Download and trigger
        if args.troubleshoot:
            st = redpy.trigger.get_data(starttime-opt.atrig, endtime,
                                        filekey, st_preload, opt)
            alltrigs = redpy.trigger.trigger(st, rtable, opt)
        else:
            try:
                st = redpy.trigger.get_data(starttime-opt.atrig, endtime,
                                            filekey, st_preload, opt)
                alltrigs = redpy.trigger.trigger(st, rtable, opt)
            except KeyboardInterrupt:
                print('\nManually interrupting!\n')
                raise KeyboardInterrupt
            except:
                print(('Could not download or trigger data... '
                       'troubleshoot with -t'))
                alltrigs = []
        
        # Clean out data spikes etc.
        trigs, junk, jtype = redpy.trigger.clean_triggers(alltrigs, opt)
        
        # !!! This step already goes through and calculates the window, can I 
        # !!! pass that down the line to save some duplicate calculations?
        
        # Save junk triggers in separate table for quality checking purposes
        for i in range(len(junk)):
            redpy.table.populate_junk(jtable, junk[i], jtype[i], opt)
        
        # Append times of triggers to ttable to compare total seismicity later
        trigs = redpy.table.populate_triggers(ttable, trigs, ttimes, opt)
        
        # Check triggers against deleted events
        if len(dtable) > 0:
            trigs = redpy.correlation.compare_deleted(trigs, dtable, opt)
        
        if len(trigs) > 0:
            id = rtable.attrs.previd
            if len(trigs) == 1:
                ostart = 0
                if len(otable) == 0:
                    # First trigger goes to orphans table
                    redpy.table.populate_orphan(otable, 0, trigs[0], opt)
                    ostart = 1
                else:
                    id += 1
                    redpy.correlation.correlate_new_triggers(
                        rtable, otable, ctable, ftable, ttimes,
                        trigs[0], id, args.troubleshoot, opt)
            else:
                ostart = 0
                if len(otable) == 0:
                    # First trigger goes to orphans table
                    redpy.table.populate_orphan(otable, 0, trigs[0], opt)
                    ostart = 1
                # Loop through remaining triggers
                for i in range(ostart,len(trigs)):
                    id += 1
                    redpy.correlation.correlate_new_triggers(
                        rtable, otable, ctable, ftable, ttimes,
                        trigs[i], id, args.troubleshoot, opt)
            rtable.attrs.previd = id
        ####
        
        redpy.table.clear_expired_orphans(otable, tstart+(n+1)*opt.nsec, opt)
        redpy.table.print_stats(rtable, otable, ftable, opt)
        n += 1
        if opt.verbose: print('Time spent this iteration: '
                              f'{(time.time()-t_iter)/60:.3f} minutes')
    print(f'Caught up to: {endtime-opt.atrig}')
    redpy.plotting.generate_all_outputs(rtable, ftable, ttable, ctable,
                                        otable, opt)
    if opt.verbose: print('Closing table...')
    h5file.close()
    if opt.verbose: print('Total time spent: '
                          f'{(time.time()-t_func)/60:.3f} minutes')
    print('Done')


def backfill_parse():
    """
    Defines and parses acceptable command line inputs for backfill.py.
    
    Returns
    -------
    args : ArgumentParser Object
    
    """
    parser = argparse.ArgumentParser(
        description='Backfills table with data from the past.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-t', "--troubleshoot", action='store_true',
                        default=False,
                        help='run in troubleshoot mode (without try/except)')
    parser.add_argument('-s', '--starttime',
                        help=('optional start time to begin filling '
                              '(yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)'))
    parser.add_argument('-e', '--endtime',
                        help=('optional end time to end filling '
                              '(yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)'))
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-n', '--nsec', type=int,
                        help=('overwrite opt.nsec from configuration file '
                              'with NSEC this run only'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

