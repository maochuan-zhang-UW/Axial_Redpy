# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import argparse
import time

import numpy as np
import pandas as pd
from obspy import UTCDateTime

import redpy


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

parser = argparse.ArgumentParser(description=
    "Backfills table with data from the past")
parser.add_argument("csvfile",
    help="catalog csv file with a 'Time UTC' column of event times")
parser.add_argument("-v", "--verbose", action="store_true", default=False,
    help="increase written print statements")
parser.add_argument("-t", "--troubleshoot", action="store_true", default=False,
    help="run in troubleshoot mode (without try/except)")
parser.add_argument("-c", "--configfile", default="settings.cfg",
    help="use configuration file named CONFIGFILE instead of default settings.cfg")
args = parser.parse_args()

t = time.time()

print(args.verbose)

h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
    redpy.table.open_with_cfg(args.configfile, args.verbose)

# Read in csv file using pandas
df = pd.read_csv(args.csvfile)
# Grab event times from 'Time UTC' column, convert to datevent_times also
eventlist = pd.to_datetime(df['Time UTC']).tolist()
# Sort so events are processed in order of occurrence
eventlist.sort()
tstart = UTCDateTime(eventlist[0])-5*opt.atrig
tend = UTCDateTime(eventlist[-1])+5*opt.atrig

if len(ttable) > 0:
    ttimes = ttable.cols.startTimeMPL[:]
else:
    ttimes = 0

# Create or read in file key to improve local file load times
filekey, tend_preload = redpy.trigger.get_filekey(tstart, tend, opt)

for event in eventlist:
    event_time = UTCDateTime(event)
    if opt.verbose: print(event_time)
    
    starttime = event_time - 4*opt.atrig
    endtime = event_time + 5*opt.atrig + opt.maxdt
    
    if len(ttable) > 0:
        ttimes = ttable.cols.startTimeMPL[:]
    else:
        ttimes = 0
    
    ####
    
    # Check to make sure we have space
    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
        redpy.table.check_famlen(h5file, rtable, otable, ttable, ctable,
                                                  jtable, dtable, ftable, opt)
    
    # Check if we need to preload waveform data from file into memory
    if (opt.preload > 0) and (len(filekey) > 0):
        if endtime+opt.maxdt > tend_preload:
            if opt.verbose: print('Loading waveforms into memory...')
            tend_preload = np.min((tend,
                starttime+opt.preload*86400)) + opt.maxdt
            st_preload = redpy.trigger.preload_data(
                starttime, tend_preload, filekey, opt)
    else:
        st_preload = []
    
    # Download and trigger
    if args.troubleshoot:
        st = redpy.trigger.get_data(starttime-opt.atrig, endtime, filekey,
                                    st_preload, opt)
        alltrigs = redpy.trigger.trigger(st, rtable, opt)
    else:
        try:
            st = redpy.trigger.get_data(starttime-opt.atrig, endtime, filekey,
                                        st_preload, opt)
            alltrigs = redpy.trigger.trigger(st, rtable, opt)
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
                id += 1
                redpy.correlation.correlate_new_triggers(rtable, otable, ctable, ftable,
                    ttimes, trigs[i], id, args.troubleshoot, opt)
        rtable.attrs.previd = id        
    ####
    
    # Reset ptime for refilling later
    rtable.attrs.ptime = []
    
    # Don't expire orphans in the catalog?
    # redpy.table.clearExpiredOrphans(otable, opt, tstart+(n+1)*opt.nsec)
    
    redpy.table.print_stats(rtable, otable, ftable, opt)

redpy.plotting.generate_all_outputs(rtable, ftable, ttable, ctable, otable, opt)

if opt.verbose: print("Closing table...")
h5file.close()

if opt.verbose: print("Total time spent: {} minutes".format((time.time()-t)/60))
if opt.verbose: print("Done")