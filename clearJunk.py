# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import redpy.config
import redpy.table
import redpy.plotting
import argparse
import numpy as np
import os

"""
Run this script to clear the contents of the junk table.

usage: clearJunk.py [-h] [-v] [-c CONFIGFILE]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
"""

parser = argparse.ArgumentParser(description=
    "Run this script to clear the contents of the junk table.")
parser.add_argument("-v", "--verbose", action="count", default=0,
    help="increase written print statements")
parser.add_argument("-c", "--configfile", default="settings.cfg",
    help="use configuration file named CONFIGFILE instead of default settings.cfg")
args = parser.parse_args()

h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
    redpy.table.open_with_cfg(args.configfile, args.verbose)

if args.verbose: print("Removing junk...")

if len(jtable) > 1:
# This will remove all but the last row (have to leave one)
    for n in range(len(jtable)-1,0,-1):
        jtable.remove_row(n)
    jtable.flush()
else:
    if args.verbose: print("No junk to remove!")

if args.verbose: print("Closing table...")
h5file.close()
if args.verbose: print("Done")