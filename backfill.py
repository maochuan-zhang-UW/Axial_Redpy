# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import argparse
import redpy
import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime
from obspy.core.stream import Stream
import time

"""
Run this script to fill the table with data from the past. If a start time is not
specified, it will check the attributes of the repeater table to pick up where it left
off. Additionally, if this is the first run and a start time is not specified, it will
assume one time chunk prior to the end time. If an end time is not specified, "now" is
assumed. The end time updates at the end of each time chunk processed (default: by hour,
set in configuration). This script can be run as a cron job that will pick up where it
left off if a chunk is missed. Use -n if you are backfilling with a large amount of time;
it will consume less time downloading the data in small chunks if NSEC is an hour or a day
instead of a few minutes, but at the cost of keeping orphans for longer.

usage: backfill.py [-h] [-v] [-t] [-s STARTTIME] [-e ENDTIME] [-c CONFIGFILE] [-n NSEC]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -t, --troubleshoot    run in troubleshoot mode (without try/except)
  -s STARTTIME, --starttime STARTTIME
                        optional start time to begin filling (YYYY-MM-DDTHH:MM:SS)
  -e ENDTIME, --endtime ENDTIME
                        optional end time to end filling (YYYY-MM-DDTHH:MM:SS)
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -n NSEC, --nsec NSEC  overwrite opt.nsec from configuration file with NSEC this run only
"""

t = time.time()

parser = argparse.ArgumentParser(description=
    "Backfills table with data from the past")
parser.add_argument("-v", "--verbose", action="count", default=0,
    help="increase written print statements")
parser.add_argument("-t", "--troubleshoot", action="store_true", default=False,
    help="run in troubleshoot mode (without try/except)")
parser.add_argument("-s", "--starttime",
    help="optional start time to begin filling (YYYY-MM-DDTHH:MM:SS)")
parser.add_argument("-e", "--endtime",
    help="optional end time to end filling (YYYY-MM-DDTHH:MM:SS)")
parser.add_argument("-c", "--configfile",
    help="use configuration file named CONFIGFILE instead of default settings.cfg")
parser.add_argument("-n", "--nsec", type=int,
    help="overwrite opt.nsec from configuration file with NSEC this run only")
args = parser.parse_args()

if args.configfile:
    opt = redpy.config.Options(args.configfile)
    if args.verbose: print("Using config file: {0}".format(args.configfile))
else:
    opt = redpy.config.Options("settings.cfg")
    if args.verbose: print("Using config file: settings.cfg")

if args.nsec:
    opt.nsec = args.nsec

if args.verbose: print("Opening hdf5 table: {0}".format(opt.filename))
h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable = \
    redpy.table.open_table(opt)

# Check for MPL version mismatch
redpy.table.check_epoch_date(rtable, ftable, ttable, otable, dtable, opt)

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

if len(ttable) > 0:
    ttimes = ttable.cols.startTimeMPL[:]
else:
    ttimes = 0

# Create or read in file key to improve file load times
if opt.server == 'file':
    
    filekey = redpy.trigger.get_filekey(opt, args)
    
    # Subset filekey to only time of interest
    filekey = filekey.query("starttime < '{}' \
                        and endtime > '{}'".format(
                        tend+opt.maxdt+opt.atrig+opt.ptrig+60,
                        tstart-opt.atrig-60))
    
    # Set start time of preload (if to be used) to tstart
    tend_preload = tstart
    
else:
    filekey = []

n = 0
rlen = len(rtable)
while tstart+n*opt.nsec < tend:

    ti = time.time()
    print(tstart+n*opt.nsec)
    
    # Check if we need to preload waveform data from file into memory
    if (opt.preload > 0) and (len(filekey) > 0):
        if np.min([tend+opt.maxdt,tstart+(n+1)*opt.nsec+opt.maxdt]) > tend_preload:
            if args.verbose:
                print('Loading waveforms into memory...')
                
            # Determine end time to load
            tend_preload = np.min([tend+opt.maxdt,
                tstart+n*opt.nsec+opt.preload*86400+opt.maxdt])
            
            # Load into memory
            st_preload = redpy.trigger.preload_data(
                tstart+n*opt.nsec-opt.atrig, tend_preload, filekey, opt)
    else:
        st_preload = []
    
    # Download and trigger
    if args.troubleshoot:
        endtime = tstart+(n+1)*opt.nsec+opt.atrig
        if endtime > tend:
            endtime = tend
        st = redpy.trigger.get_data(tstart+n*opt.nsec-opt.atrig, endtime,
                                                     filekey, st_preload, opt)
        alltrigs = redpy.trigger.trigger(st, rtable, opt)
    else:
        try:
            endtime = tstart+(n+1)*opt.nsec+opt.atrig
            if endtime > tend:
                endtime = tend
            st = redpy.trigger.get_data(tstart+n*opt.nsec-opt.atrig, endtime,
                                                     filekey, st_preload, opt)
            alltrigs = redpy.trigger.trigger(st, rtable, opt)
        except KeyboardInterrupt:
            print('\nManually interrupting!\n')
            raise KeyboardInterrupt
        except:
            print('Could not download or trigger data... troubleshoot with -t')
            alltrigs = []
    
    # Clean out data spikes etc.
    trigs, junk, jtype = redpy.trigger.clean_triggers(alltrigs, opt)
    
    # !!! This step already goes through and calculates the window, can I 
    # !!! pass that down the line to save some duplicate calculations?
    
    # Save junk triggers in separate table for quality checking purposes
    for i in range(len(junk)):
        redpy.table.populate_junk(jtable, junk[i], jtype[i], opt)
    
    # Append times of triggers to ttable to compare total seismicity later
    redpy.table.populate_triggers(ttable, trigs, ttimes, opt)
    
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
                id = id + 1
                redpy.correlation.correlate_new_triggers(rtable, otable, ctable, ftable, ttimes,
                    trigs[0], id, args.troubleshoot, opt)
        else:
            ostart = 0
            if len(otable) == 0:
                # First trigger goes to orphans table
                redpy.table.populate_orphan(otable, 0, trigs[0], opt)
                ostart = 1
            # Loop through remaining triggers
            for i in range(ostart,len(trigs)):
                id = id + 1
                redpy.correlation.correlate_new_triggers(rtable, otable, ctable, ftable, ttimes,
                    trigs[i], id, args.troubleshoot, opt)
        rtable.attrs.previd = id

    redpy.table.clear_expired_orphans(otable, tstart+(n+1)*opt.nsec, opt)

    # Print some stats
    if args.verbose:
        print("Length of Orphan table: {}".format(len(otable)))
        if len(rtable) > 1:
            print("Number of repeaters: {}".format(len(rtable)))
            print("Number of clusters: {}".format(ftable.attrs.nClust))

    # Update tend if an end date is not specified so this will run until it is fully
    # caught up, instead of running to when the script was originally run.
    if not args.endtime:
        tend = UTCDateTime()

    n = n+1

    if args.verbose: print("Time spent this iteration: {} minutes".format(
        (time.time()-ti)/60))

print("Caught up to: {}".format(endtime-opt.atrig))

if args.verbose: print("Updating plots...")
redpy.plotting.generate_all_outputs(rtable, ftable, ttable, ctable, otable, opt)

if args.verbose: print("Closing table...")
h5file.close()

print("Total time spent: {} minutes".format((time.time()-t)/60))
if args.verbose: print("Done")
