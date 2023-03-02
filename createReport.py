# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import redpy.config
import redpy.table
import redpy.plotting
import argparse
import os
import numpy as np
import redpy.correlation
import matplotlib.dates as mdates

"""
Run this script to manually produce a more detailed 'report' page for a given family
(or families)

usage: createReport.py [-h] [-v] [-o] [-c CONFIGFILE] N [N ...]

positional arguments:
  N                     family number(s) to be reported on

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -o, --ordered         order plots by OPTICS
  -m, --matrixtofile    save correlation matrix to file
  -s, --skip            skip recalculating the full correlation matrix
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
"""

parser = argparse.ArgumentParser(description=
    "Run this script to manually produce a more detailed 'report' page for a given " +
    "family (or families)")
parser.add_argument('famnum', metavar='N', type=int, nargs='+',
    help="family number(s) to be reported on")
parser.add_argument("-v", "--verbose", action="count", default=0,
    help="increase written print statements")
parser.add_argument("-o", "--ordered", action="count", default=0,
    help="order plots by OPTICS")
parser.add_argument("-m", "--matrixtofile", action="count", default=0,
    help="save correlation matrix to file")
parser.add_argument("-s", "--skip", action="count", default=0,
    help="skip recalculating the full correlation matrix")
parser.add_argument("-c", "--configfile", default="settings.cfg",
    help="use configuration file named CONFIGFILE instead of default settings.cfg")
args = parser.parse_args()

h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
    redpy.table.open_with_cfg(args.configfile, args.verbose)

if args.verbose: print("Creating folder to store files '{}{}/reports'".format(
    opt.outputPath, opt.groupName))
    
try:
    os.mkdir('{}{}/reports'.format(opt.outputPath,opt.groupName))
except OSError:
    print("Folder exists.")

# Load from table to pass in case multiple families are to be plotted
windowStart = rtable.cols.windowStart[:]
rtimes_mpl = rtable.cols.startTimeMPL[:]+windowStart/opt.samprate/86400
rtimes = np.array([mdates.num2date(rtime) for rtime in rtimes_mpl])
windowAmps = rtable.cols.windowAmp[:]
fi = rtable.cols.FI[:]
ids, ccc_sparse = redpy.correlation.get_matrix(rtable, ctable, opt)

for fnum in args.famnum:
    if args.verbose: print("Creating report for family {}...".format(fnum))
    redpy.plotting.create_report(rtable, ftable, rtimes, rtimes_mpl,
        windowAmps, fi, ids, ccc_sparse, fnum, args.ordered, args.skip,
        args.matrixtofile, opt)

if args.verbose: print("Closing table...")
h5file.close()
if args.verbose: print("Done")
