# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import redpy.config
import redpy.table
import argparse
import os
import numpy as np
import time

"""
Run this script to create space for additional stations while preserving data in an
existing table or to change the directory name for a run. Additional stations should
always be included at the end of the station list; reordering that list is currently not
supported. Running this script will overwrite any existing table with the same name
defined by filename in the new .cfg file. If the table names in both .cfg files are the
same, the original table will be renamed and then deleted. All output files are also
remade to reflect the additional station, unless flagged otherwise.

usage: extendTable.py [-h] [-v] [-n] CONFIGFILE_FROM CONFIGFILE_TO

positional arguments:
  CONFIGFILE_FROM       old .cfg file corresponding to table to be copied from
  CONFIGFILE_TO         new .cfg file corresponding to table to be copied to

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -n, --noplot          do not re-render plots after extending
"""

parser = argparse.ArgumentParser(description=
    "Create space for additional stations based on an existing table")
parser.add_argument("-v", "--verbose", action="count", default=0,
    help="increase written print statements")
parser.add_argument("-n", "--noplot", action="count", default=0,
    help="do not re-render plots after extending")
parser.add_argument('cfgfrom', metavar='CONFIGFILE_FROM', type=str, nargs=1,
    help="old .cfg file corresponding to table to be copied from")
parser.add_argument('cfgto', metavar='CONFIGFILE_TO', type=str, nargs=1,
    help="new .cfg file corresponding to table to be copied to")

args = parser.parse_args()

do_plot = not args.noplot

t = time.time()

if args.verbose: print("Using old config file: {0}".format(args.cfgfrom[0]))
optfrom = redpy.config.Options(args.cfgfrom)

if args.verbose: print("Using new config file: {0}".format(args.cfgto[0]))
optto = redpy.config.Options(args.cfgto)

if args.verbose: print("Opening hdf5 table: {0}".format(optfrom.filename))
h5filefrom, rtablefrom, otablefrom, ttablefrom, ctablefrom, jtablefrom, \
    dtablefrom, ftablefrom = redpy.table.open_table(optfrom)

h5fileto, rtableto, otableto, ttableto, ctableto, jtableto, dtableto, \
    ftableto, optto = redpy.table.expand_table(h5filefrom, ftablefrom, optfrom,
                                   optto=optto, do_plot=do_plot)

if do_plot:
    if args.verbose: print("Creating plots...")
    redpy.plotting.generate_all_outputs(rtableto, ftableto, ttableto, 
                                                    ctableto, otableto, optto)

if args.verbose: print("Closing table...")
h5fileto.close()

if args.verbose: print(f"Done in {time.time()-t:.3f} seconds")
