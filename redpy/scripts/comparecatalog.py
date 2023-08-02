# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
r"""
Compare REDPy catalog with external catalog.

Run this script to compare an independent text catalog with a REDPy catalog.
The input catalog is appended with the best matching event, with columns for
times, time difference (negative being that REDPy triggered early), which
family or trigger type the best match corresponds to, and for repeaters,
the frequency index and amplitudes. The combined catalog is saved
separately, and by default in the current directory as 'matches.csv'.

usage: redpy-compare-catalog [-h] [-v] [-a] [-i] [-j] [-c CONFIGFILE]
                             [-d DELIMITER] [-m MAXDTOFFSET] [-n NAME]
                             [-o OUTFILE] catfile

positional arguments:
  catfile               catalog file

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -a, --arrival         estimate and use the P-wave arrival time to the
                        center of the network; requires a location to be
                        included in the catalog
  -i, --include_missing include REDPy detections without a match
  -j, --junk            include matches with triggers considered junk
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -d DELIMITER, --delimiter DELIMITER
                        define custom DELIMITER between columns instead of
                        default "," (e.g., "\t" for tabs, " " for spaces, or
                        "|" for pipes)
  -m MAXDTOFFSET, --maxdtoffset MAXDTOFFSET
                        define custom time offset (in seconds) instead of
                        default 75% of window length
  -n NAME, --name NAME  define custom time column NAME instead of default
                        "Time"
  -o OUTFILE, --outfile OUTFILE
                        define custom output file name and location
"""
import argparse

from redpy.detector import Detector


def compare_catalog(catfile, arrival=False, configfile='settings.cfg',
                    delimiter=',', include_missing=False, junk=False,
                    maxdtoffset=-1., name='Time', outfile='matches.csv',
                    verbose=False):
    """
    Compare an independent text catalog with a REDPy catalog.

    The input catalog is appended with the best matching event, with columns
    for times, time difference (negative being that REDPy triggered early),
    which family or trigger type the best event corresponds to, and for
    repeaters, the frequency index and amplitudes. The combined catalog is
    saved separately, and by default in the current directory.

    Parameters
    ----------
    catfile : str
        File name of catalog to compare with.
    arrival : bool, optional
        If True,
    configfile : str, optional
        Name of configuration file to read.
    delimiter : str, optional
        Delimiter between columns of the catalog.
    include_missing : bool, optional
        Include REDPy detections without a match.
    junk : bool, optional
        Include matches with triggers considered junk.
    maxdtoffset : float, optional
        Maximum time offset in seconds to be considered a match. If
        negative, reverts to the default of 75% of the correlation window
        length, which is also the minimum time between consecutive triggers.
    name : str, optional
        Name of the 'Time' column in catfile.
    outfile : str, optional
        Path and filename to write output catalog to.
    verbose : bool, optional
        Enable additional print statements.

    """
    detector = Detector(configfile, verbose)
    _ = detector.locate('compare', catfile, arrival, delimiter,
                        include_missing, junk, maxdtoffset, name, outfile)
    # Returns the matched catalog as a pandas DataFrame.
    detector.close()


def main():
    """Handle run from the command line."""
    args = parse()
    compare_catalog(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser Object

    """
    parser = argparse.ArgumentParser(
        prog='redpy-compare-catalog',
        description='Compare REDPy catalog with external catalog.')
    parser.add_argument('catfile', help='catalog file')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-a', '--arrival', action='store_true', default=False,
                        help=('estimate and use the P-wave arrival time to '
                              'the center of the network; requires a location '
                              'to be included in the catalog'))
    parser.add_argument('-i', '--include_missing', action='store_true',
                        default=False, help=(
                            'include REDPy detections without a match'))
    parser.add_argument('-j', '--junk', action='store_true', default=False,
                        help='include matches with triggers considered junk')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-d', '--delimiter', default=',',
                        help=('define custom DELIMITER between columns '
                              'instead of default "," (e.g., "\\t" for tabs, '
                              '" " for spaces, or "|" for pipes)'))
    parser.add_argument('-m', '--maxdtoffset', default=-1., type=float,
                        help=('define custom time offset (in seconds) '
                              'instead of default 75%% of window length'))
    parser.add_argument('-n', '--name', default='Time',
                        help=('define custom time column NAME instead of '
                              'default "Time"'))
    parser.add_argument('-o', '--outfile', default='matches.csv',
                        help='define custom output file name and location')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
