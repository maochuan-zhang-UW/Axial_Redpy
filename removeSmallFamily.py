# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import redpy.config
import redpy.table
import argparse

import matplotlib.dates as mdates
from obspy import UTCDateTime


"""
Run this script to remove small families/clusters (i.e., families that have less than M members and are more than D days
old). Reclusters and remakes images when done. This module works by determining the families that need to be removed,
and then it passes those families to removeFamily.py. Note: Removing families from datasets with manny families and
repeaters may take a significant amount of time.
usage: removeSmallFamily.py [-h] [-v] [-c CONFIGFILE] [-m MINMEMBERS] [-d MAXDAYS] [-t SEEDTIME]
optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -m, --MINMEMBERS      minimum family size to keep (default: 5)
  -d, --MAXDAYS         maximum age of a family (days) to keep; measured from first member in family
                        in other words: keep families less than or equal to d days old
                        (default: 0; i.e., removes all small families)
  -t  --SEEDTIME        Time from which to compute families' ages (default: last trigger time UTC)
                        If a family started after the seedtime, it will be kept
  -l --LIST             Lists families to keep and remove, but does not actually modify anything. Automatically uses
                        verbose mode.
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
"""


parser = argparse.ArgumentParser(description=
    "Run this script to manually remove small families/clusters")
parser.add_argument("-v", "--verbose", action="count", default=0,
    help="increase written print statements")
parser.add_argument("-c", "--configfile", default="settings.cfg",
    help="use configuration file named CONFIGFILE instead of default settings.cfg")
parser.add_argument("-m", "--minmembers", type=int, default=5,
    help="minimum family size to keep")
parser.add_argument("-d", "--maxdays", type=int, default=0,
    help="maximum age of a family to be saved (default: 0, i.e., removes every small family regardless of age")
parser.add_argument("-t", "--seedtime", default=False,
    help="time from which to compute families' repose times (YYYY-MM-DDTHH:MM:SS) (deafult: last trigger time UTC)")
parser.add_argument("-l", "--list", action="store_true", default=False,
    help="list families to keep and remove, but do not execute")
args = parser.parse_args()


def main(args):

    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
        redpy.table.open_with_cfg(args.configfile, args.verbose)

    oldnClust = ftable.attrs.nClust
    
    # Convert args.seedtime to UTCDateTime
    if args.seedtime:
        seedtime = UTCDateTime(args.seedtime)
    else:
        # Last trigger
        seedtime = UTCDateTime(mdates.num2date(ttable[-1]['startTimeMPL']))

    # Determines which families to remove, sends to table.removeFamilies(), outputs number of families removed
    cnums = redpy.table.remove_small_families(rtable, ctable, dtable, ftable, ttable, args.minmembers, args.maxdays,
                                    seedtime, opt, list_only=args.list, verbose=args.verbose)

    if len(cnums) > 0:
        # Only update plots if there are families removed
        if args.verbose: print("Creating plots...")
        redpy.plotting.generate_all_outputs(rtable, ftable, ttable, ctable, otable, opt)

        if args.verbose: print("Cleaning up old .html & .png files...")
        redpy.plotting.remove_old_html(oldnClust, ftable.attrs.nClust, opt)
    else:
        if args.verbose: print("No families removed. No plots to update...")

    if args.verbose: print("Closing table...")
    h5file.close()
    if args.verbose: print("Done")


if __name__ == "__main__":
    main(args)