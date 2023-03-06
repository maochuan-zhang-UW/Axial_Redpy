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
    redpy.table.open_with_cfg(args.configfile, args.verbose, args.troubleshoot)

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
filekey = redpy.trigger.get_filekey(tstart, tend, opt)
tend_preload = tstart
st_preload = []

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
    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, \
        st_preload, tend_preload, opt = redpy.table.update_tables(
            h5file, rtable, otable, ttable, ctable, jtable, dtable,
            ftable, ttimes, filekey, st_preload, tend_preload, tend,
            starttime, endtime, opt)
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