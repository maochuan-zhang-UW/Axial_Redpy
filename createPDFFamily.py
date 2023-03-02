# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import redpy.config
import redpy.table
import argparse
import os
import matplotlib
import matplotlib.dates as mdates
import numpy as np

"""
Run this script to manually produce editable PDF versions of family pages in the clusters
directory (same location as fam*.png) with custom time span

usage: createPDFFamily.py [-h] [-v] [-c CONFIGFILE] N [N ...]

positional arguments:
  N                     family number(s) to be plotted

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -s STARTTIME, --starttime STARTTIME
                        earliest time to plot, defaults to first event
  -e ENDTIME, --endtime ENDTIME
                        latest time to plot, defaults to last event
"""

parser = argparse.ArgumentParser(description=
    "Run this script to manually produce editable PDF versions of family pages in the "+
    "clusters directory (same location as fam*.png)")
parser.add_argument('famnum', metavar='N', type=int, nargs='+',
    help="family number(s) to be plotted")
parser.add_argument("-v", "--verbose", action="count", default=0,
    help="increase written print statements")
parser.add_argument("-c", "--configfile",
    help="use configuration file named CONFIGFILE instead of default settings.cfg")
parser.add_argument("-s", "--starttime",
    help="earliest time to plot, defaults to first trigger")
parser.add_argument("-e", "--endtime",
    help="latest time to plot, defaults to last trigger")
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

# Load into memory
windowStart = rtable.cols.windowStart[:]
rtimes_mpl = rtable.cols.startTimeMPL[:]+windowStart/opt.samprate/86400
rtimes = np.array([mdates.num2date(rtime) for rtime in rtimes_mpl])
windowAmp = rtable.cols.windowAmp[:][:,opt.printsta]
ids, ccc_sparse = redpy.correlation.get_matrix(rtable, ctable, opt)

# Process arguments
if args.starttime:
    tmin = matplotlib.dates.date2num(np.datetime64(args.starttime))
else:
    tmin = 0

if args.endtime:
    tmax = matplotlib.dates.date2num(np.datetime64(args.endtime))
else:
    tmax = 0

# Initialize figure layout
fig, axes, bboxes = redpy.plotting.initialize_family_image(opt)

for fnum in args.famnum:
    if args.verbose: print("Creating PDF for family {}...".format(fnum))
    redpy.plotting.assemble_family_image(bboxes, rtable, ftable, rtimes,
        rtimes_mpl, windowAmp, ids, ccc_sparse, 'pdf', 100, fnum, tmin, tmax,
        opt)

if args.verbose: print("Closing table...")
h5file.close()
if args.verbose: print("Done")
