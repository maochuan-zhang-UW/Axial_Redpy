# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Find families with distant catalog matches.

Run this script to print out the families with a minimum percentage of
regional and/or teleseismic matches contained in their .html output files
that can then be copy/pasted into remove_family.py. An optional table is
printed that summarizes matches of each type. A custom 'FINDPHRASE' may
be given to find matches in the location string.

usage: distant_families.py [-h] [-v] [-c CONFIGFILE] [-f FINDPHRASE]
                           [-p PERCENT]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements, including a table
                        of matches
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -f FINDPHRASE, --findphrase FINDPHRASE
                        phrase to explicitly find in location string
  -p PERCENT, --percent PERCENT
                        minimum percentage of regional/teleseismic matches,
                        default 90
"""
import argparse

import redpy


def distant_families(configfile='settings.cfg', verbose=False, findphrase='',
                     percent=90):
    """
    Find families with distant catalog matches by parsing their .html files.

    Prints family numbers that match the criteria that can then be copied as
    arguments to other removal functions, while also allowing the user to
    vet those matches prior to removal.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements, including a summary table.
    findphrase : str, optional
        Specific phrase to find in location string.
    percent : float, optional
        Minimum percentage of matches required to add a family to the list;
        90% by default.

    """
    detector = redpy.Detector(configfile, verbose)
    _ = detector.locate('distant', findphrase, percent)
    # Returns a dictionary that contains the families that match the criteria
    # as numpy arrays that could be fed to detector.remove() directly.


def main():
    """Handle run from the command line."""
    args = parse()
    distant_families(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        description='Find families with distant catalog matches.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help=('increase written print statements, including '
                              'a table of matches'))
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-f', '--findphrase', default='',
                        help='phrase to explicitly find in location string')
    parser.add_argument('-p', '--percent', type=float, default=90,
                        help=('minimum percentage of regional/teleseismic '
                              'matches, default 90'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
