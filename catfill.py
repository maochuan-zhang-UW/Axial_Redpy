# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import argparse
import redpy
import numpy as np
import obspy
from obspy import UTCDateTime
import time
import pandas as pd

"""
Run this script to fill the table with data from the past using a catalog of events.
 
usage: catfill.py [-h] [-v] [-c CONFIGFILE] csvfile

positional arguments:
  csvfile               catalog csv file with a 'Time UTC' column of event times

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -t, --troubleshoot    run in troubleshoot mode (without try/except)
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
"""

t = time.time()

parser = argparse.ArgumentParser(description=
    "Backfills table with data from the past")
parser.add_argument("csvfile",
    help="catalog csv file with a 'Time UTC' column of event times")
parser.add_argument("-v", "--verbose", action="count", default=0,
    help="increase written print statements")
parser.add_argument("-t", "--troubleshoot", action="store_true", default=False,
    help="run in troubleshoot mode (without try/except)")
parser.add_argument("-c", "--configfile",
    help="use configuration file named CONFIGFILE instead of default settings.cfg")
args = parser.parse_args()

if args.configfile:
    opt = redpy.config.Options(args.configfile)
    if args.verbose: print("Using config file: {0}".format(args.configfile))
else:
    opt = redpy.config.Options("settings.cfg")
    if args.verbose: print("Using config file: settings.cfg")

if args.verbose: print("Opening hdf5 table: {0}".format(opt.filename))
h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable = \
    redpy.table.open_table(opt)

# Check for MPL version mismatch
redpy.table.check_epoch_date(rtable, ftable, ttable, otable, dtable, opt)

# Read in csv file using pandas
df = pd.read_csv(args.csvfile)
# Grab event times from 'Time UTC' column, convert to datetimes also
eventlist = pd.to_datetime(df['Time UTC']).tolist()
# Sort so events are processed in order of occurrence
eventlist.sort()

# Create or read in file key to improve file load times
if opt.server == 'file':
    
    filekey = redpy.trigger.get_filekey(opt, args)
    
    tstart = UTCDateTime(eventlist[0])-5*opt.atrig
    tend = UTCDateTime(eventlist[-1])+5*opt.atrig
    
    # Subset filekey to only time of interest
    filekey = filekey.query("starttime < '{}' \
                        and endtime > '{}'".format(
                        tend+opt.maxdt+opt.atrig+opt.ptrig+60,
                        tstart-opt.atrig-60))
    
    # Set start time of preload (if to be used) to just before first event
    tend_preload = tstart
    
else:
    filekey = []


for event in eventlist:
    
    etime = UTCDateTime(event)
    if len(ttable) > 0:
        ttimes = ttable.cols.startTimeMPL[:]
    else:
        ttimes = 0
    
    # Check if we need to preload waveform data from file into memory
    if (opt.preload > 0) and (len(filekey) > 0):
        if etime+5*opt.atrig > tend_preload:
            if args.verbose:
                print('Loading waveforms into memory...')
                
            # Determine end time to load
            tend_preload = np.min([tend+opt.maxdt,
                etime+opt.preload*86400+opt.maxdt])
            
            # Load into memory
            st_preload = redpy.trigger.preload_data(
                etime-5*opt.atrig, tend_preload, filekey, opt)
    else:
        st_preload = []
    
    
    if args.verbose: print(etime)
    
    # Download and trigger
    if args.troubleshoot:
        st = redpy.trigger.get_data(etime-5*opt.atrig,
                                        etime+5*opt.atrig, filekey, st_preload, opt)
        alltrigs = redpy.trigger.trigger(st, rtable, opt)
        # Reset ptime for refilling later
        rtable.attrs.ptime = []
    else:
        try:
            st = redpy.trigger.get_data(etime-5*opt.atrig,
                                            etime+5*opt.atrig, filekey, st_preload, opt)
            alltrigs = redpy.trigger.trigger(st, rtable, opt)
            # Reset ptime for refilling later
            rtable.attrs.ptime = []
        except KeyboardInterrupt:
            print('\nManually interrupting!\n')
            raise KeyboardInterrupt
        except:
            print('Could not download or trigger data... troubleshoot with -t')
            alltrigs = []
    
    # Clean out data spikes etc.
    trigs, junk, jtype = redpy.trigger.clean_triggers(alltrigs, opt)
    
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
                redpy.correlation.correlate_new_triggers(rtable, otable, ctable, ftable,
                    ttimes, trigs[0], id, args.troubleshoot, opt)
        else:
            ostart = 0
            if len(otable) == 0:
                # First trigger goes to orphans table
                redpy.table.populate_orphan(otable, 0, trigs[0], opt)
                ostart = 1        
            # Loop through remaining triggers
            for i in range(ostart,len(trigs)):  
                id = id + 1
                redpy.correlation.correlate_new_triggers(rtable, otable, ctable, ftable,
                    ttimes, trigs[i], id, args.troubleshoot, opt)
        rtable.attrs.previd = id        
    
    # Don't expire orphans in the catalog?
    # redpy.table.clearExpiredOrphans(otable, opt, tstart+(n+1)*opt.nsec)
    
    # Print some stats
    if args.verbose:
        print("Length of Orphan table: {}".format(len(otable)))
        if len(rtable) > 1:
            print("Number of repeaters: {}".format(len(rtable)))
            print("Number of clusters: {}".format(ftable.attrs.nClust))

if len(rtable) > 1:
    if args.verbose: print("Creating plots...")
    redpy.plotting.create_plots(rtable, ftable, ttable, ctable, otable, opt)
else:
    print("No repeaters to plot.")

if args.verbose: print("Closing table...")
h5file.close()

if args.verbose: print("Total time spent: {} minutes".format((time.time()-t)/60))
if args.verbose: print("Done")