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
Run this script to output the contents of the junk table for troubleshooting.

usage: plotJunk.py [-h] [-v] [-c CONFIGFILE]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
"""

parser = argparse.ArgumentParser(description=
    "Run this script to output the contents of the junk table for troubleshooting.")
parser.add_argument("-v", "--verbose", action="count", default=0,
    help="increase written print statements")
parser.add_argument("-c", "--configfile", default="settings.cfg",
    help="use configuration file named CONFIGFILE instead of default settings.cfg")
args = parser.parse_args()

h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
    redpy.table.open_with_cfg(args.configfile, args.verbose)

if args.verbose: print("Creating folder to store junk images named '{}{}'/junk".format(
    opt.outputPath,opt.groupName))
try:
    os.mkdir('{}{}/junk'.format(opt.outputPath,opt.groupName))
except OSError:
    print("Folder exists.")

if args.verbose: print("Creating junk plots...")
redpy.plotting.create_junk_images(jtable, opt)

if args.verbose: print("Closing table...")
h5file.close()
if args.verbose: print("Done")