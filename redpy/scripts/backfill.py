# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Fill tables using continuous data.

Run this script to fill the table with data from the past. If a start time
is not specified, it will check the attributes of the repeater table to
pick up where it left off. Additionally, if this is the first run and a
start time is not specified, it will assume one time chunk prior to the
end time. If an end time is not specified, "now" is assumed. The end time
updates at the end of each time chunk processed (default: by hour, set in
configuration). This script can be run as a cron job that will pick up
where it left off if a chunk is missed. Use -n if you are backfilling with
a large amount of time; it will consume less time downloading the data in
small chunks if NSEC is an hour or a day instead of a few minutes, but at
the cost of keeping orphans for longer.

usage: redpy-backfill [-h] [-v] [-s STARTTIME] [-e ENDTIME]
                      [-c CONFIGFILE] [-n NSEC]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -s STARTTIME, --starttime STARTTIME
                        optional start time to begin filling (yyyy-mm-dd or
                        yyyy-mm-ddTHH:MM:SS)
  -e ENDTIME, --endtime ENDTIME
                        optional end time to end filling (yyyy-mm-dd or
                        yyyy-mm-ddTHH:MM:SS)
  -n NSEC, --nsec NSEC  overwrite "nsec" from configuration file with NSEC
                        this run only
"""
import argparse
import time

import redpy


def backfill(configfile='settings.cfg', verbose=False, starttime=None,
             endtime=None, nsec=None):
    """
    Update tables defined in configfile with continuous data.

    If this is the first run and a start time is not specified, it will
    assume one time chunk prior to the end time. If an end time is not
    specified, "now" is assumed. The end time updates at the end of each
    time chunk processed (default: by hour, set in configuration), so
    consecutive calls can pick up where the last one left off.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.
    starttime : str, optional
        Starting time. If not provided, will default to either the end of
        the previous run time or "nsec" seconds prior to end_time.
    endtime : str, optional
        Ending time. If not provided, will default to now.
    nsec : int, optional
        Temporarily overwrite "nsec" from config with this value.

    """
    t_start = time.time()
    detector = redpy.Detector(configfile, verbose, opened=True)
    detector.update('backfill', starttime, endtime, nsec=nsec)
    detector.close()
    print(f'Total time spent: {time.time()-t_start:.2f} seconds')


def main():
    """Handle run from the command line."""
    args = parse()
    backfill(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        prog='redpy-backfill',
        description='Fill tables using continuous data.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-s', '--starttime',
                        help=('optional start time to begin filling '
                              '(yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)'))
    parser.add_argument('-e', '--endtime',
                        help=('optional end time to end filling '
                              '(yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)'))
    parser.add_argument('-n', '--nsec', type=int,
                        help=('overwrite "nsec" from configuration file '
                              'with NSEC this run only'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
