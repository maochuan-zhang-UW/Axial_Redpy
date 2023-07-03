# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
r"""
Fill tables with data using a catalog.

Run this script to fill the table with data from the past using a catalog
of known events to limit the amount of waveforms to process. By default,
catfill does not expire orphans, allowing backfill to be run over the same
time period and for orphaned catalog events to still exist. This behavior
can be overridden with -x to expire orphans after each event time. When
using -f, a trigger will be forced at the given time and 'junk' filtering
will be skipped, however, the minimum allowed time between events is still
enforced.

usage: redpy-catfill [-h] [-v] [-a] [-f] [-q] [-x] [-c CONFIGFILE]
                     [-d DELIMITER] [-n NAME] [-s STARTTIME] [-e ENDTIME]
                     csvfile

positional arguments:
  csvfile               catalog csv file with a column of event times or
                        file to write a queried catalog to disk

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -a, --arrival         estimate and use the P-wave arrival time to the
                        center of the network; requires a location to be
                        included in the catalog
  -f, --force           force a trigger at time specified in catalog instead
                        of using default STA/LTA triggering
  -q, --query           queries external catalog for local seismicity as
                        defined in the config file and saves output to
                        csvfile
  -x, --expire          expire orphans
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -d DELIMITER, --delimiter DELIMITER
                        define custom DELIMITER between columns instead of
                        default "," (e.g., "\t" for tabs, " " for spaces, or
                        "|" for pipes)
  -n NAME, --name NAME  define custom time column NAME instead of default
                        "Time"
  -s STARTTIME, --starttime STARTTIME
                        subsets catalog to begin at STARTTIME (yyyy-mm-dd or
                        yyyy-mm-ddTHH:MM:SS)
  -e ENDTIME, --endtime ENDTIME
                        subsets catalog to end at ENDTIME (yyyy-mm-dd or
                        yyyy-mm-ddTHH:MM:SS)
"""
import argparse
import time

import redpy


def catfill(configfile='settings.cfg', csvfile='catalog.csv', verbose=False,
            arrival=False, force=False, query=False, expire=False,
            delimiter=',', name='Time', starttime=None, endtime=None):
    """
    Update tables defined in configfile with a catalog of known events.

    This catalog can be a local .csv-like file, or you may query from an
    FDSN webservice, which will save locally to 'catalog.csv' by default.
    If you wish to calculate arrivals, a location must be provided in the
    catalog (i.e., columns for Latitude, Longitude, and Depth). You may
    specify custom names for the 'Time' column and delimiter.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    csvfile : str, optional
        Name of catalog csv file to read or save to.
    verbose : bool, optional
        Enable additional print statements.
    arrival : bool, optional
        Calculate and use P-wave arrival to center of network.
    force : bool, optional
        Force trigger at times specified in catalog instead of using default
        STA/LTA triggering.
    query : bool, optional
        Query an external webservice with parameters in configfile.
    expire : bool, optional
        Expire orphans.
    delimiter : str, optional
        Custom delimiter between columns in csvfile.
    name : str, optional
        Custom name of event time column.
    starttime : str, optional
        Subsets catalog to begin at this time.
    endtime : str, optional
        Subsets catalog to end at this time.

    """
    t_start = time.time()
    detector = redpy.Detector(configfile, verbose, opened=True)
    if query:
        detector.locate('arrivals', starttime, endtime, outfile=csvfile)
    try:
        event_list = redpy.locate.event_times_from_catalog(
            detector, csvfile, name, starttime, endtime,
            arrival, delimiter)
    except KeyError as exc:
        raise KeyError(
            f'Could not find "{name}" column in {csvfile}. Check file, '
            'column name, and delimiter! Use -h for help.') from exc
    detector.update(
        'catfill', starttime, endtime, event_list, force, expire)
    detector.close()
    print(f'Total time spent: {time.time()-t_start:.2f} seconds')


def main():
    """Handle run from the command line."""
    args = parse()
    catfill(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        description='Fill tables with data using a catalog.')
    parser.add_argument('csvfile',
                        help=('catalog csv file with a column of event times '
                              'or file to write a queried catalog to disk'))
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-a', '--arrival', action='store_true', default=False,
                        help=('estimate and use the P-wave arrival time to '
                              'the center of the network; requires a location '
                              'to be included in the catalog'))
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help=('force a trigger at time specified in catalog '
                              'instead of using default STA/LTA triggering'))
    parser.add_argument('-q', '--query', action='store_true', default=False,
                        help=('queries external catalog for local seismicity '
                              'as defined in the config file and saves output '
                              'to csvfile'))
    parser.add_argument('-x', '--expire', action='store_true', default=False,
                        help='expire orphans')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-d', '--delimiter', default=',',
                        help=('define custom DELIMITER between columns '
                              'instead of default "," (e.g., "\\t" for tabs, '
                              '" " for spaces, or "|" for pipes)'))
    parser.add_argument('-n', '--name', default='Time',
                        help=('define custom time column NAME instead of '
                              'default "Time"'))
    parser.add_argument('-s', '--starttime', default=None,
                        help=('subsets catalog to begin at STARTTIME '
                              '(yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)'))
    parser.add_argument('-e', '--endtime', default=None,
                        help=('subsets catalog to end at ENDTIME '
                              '(yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
