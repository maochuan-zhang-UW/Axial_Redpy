# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import redpy.config
import redpy.table
import argparse

"""
Run this script to manually remove families/clusters (e.g., correlated noise that made it
past the 'junk' detector). Reclusters and remakes images when done.

usage: removeFamily.py [-h] [-v] [-c CONFIGFILE] N [N ...]

positional arguments:
  N                     family number(s) to be moved and deleted

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
"""

parser = argparse.ArgumentParser(description=
    "Run this script to manually remove families/clusters")
parser.add_argument('famnum', metavar='N', type=int, nargs='+',
    help="family number(s) to be moved and deleted")
parser.add_argument("-v", "--verbose", action="store_true", default=False,
    help="increase written print statements")
parser.add_argument("-c", "--configfile", default="settings.cfg",
    help="use configuration file named CONFIGFILE instead of default settings.cfg")
args = parser.parse_args()

h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
    redpy.table.open_with_cfg(args.configfile, args.verbose)

oldnClust = ftable.attrs.nClust

redpy.table.remove_families(rtable, ctable, dtable, ftable, args.famnum, opt, args.verbose)

if opt.verbose: print("Creating plots...")
redpy.plotting.generate_all_outputs(rtable, ftable, ttable, ctable, otable, opt)

if opt.verbose: print("Cleaning up .html files...")
redpy.plotting.remove_old_html(oldnClust, ftable.attrs.nClust, opt)

if opt.verbose: print("Closing table...")
h5file.close()
if opt.verbose: print("Done")