# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Remove "small" families with few members.

Run this script to remove small families (i.e., families that have less than
MINMEMBERS members and are more than MAXAGE days old). Remakes images when
done. Optionally, the list of families that meet those criteria may be
printed to screen without removing them.

Note: Removing families from large datasets may take a significant amount of
time.

usage: redpy-remove-small-family [-h] [-v] [-l] [-c CONFIGFILE]
                                 [-m MINMEMBERS] [-a MAXAGE] [-t SEEDTIME]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -l, --listonly        list families to keep and to remove, but do not
                        execute removal
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -m MINMEMBERS, --minmembers MINMEMBERS
                        minimum family size to keep; 5 by default
  -a MAXAGE, --maxage MAXAGE
                        maximum age of a family to be saved in days; 0 by
                        default removes every small family that began before
                        SEEDTIME
  -t SEEDTIME, --seedtime SEEDTIME
                        time from which to compute family age times (yyyy-
                        mm-dd or yyyy-mm-ddTHH:MM:SS); last trigger time by
                        default
"""
import argparse

import redpy


def remove_small_family(configfile='settings.cfg', minmembers=5, maxage=0,
                        seedtime='', listonly=False, verbose=False):
    """
    Remove "small" families with few members.

    Note that the way this function is written, cores for these families
    are not added to the 'Deleted' table, which allows repeats to attempt
    to create a new family instead of being discarded as matching a deleted
    family. That family will be missing the earlier repeats, but with the
    benefit of not keeping a small family around for a long time.

    Parameters
    ----------
    configfile : str, optional
        Configuration file to read.
    minmembers : int, optional
        Minimum number of family members in order to be kept; 5 by default.
    maxage : int, optional
        Maximum age of a family to be saved in days, calculated as the
        difference between 'seedtime' and the most recent member. This
        allows recent small families time to grow and removes truly "stale"
        families. 0 by default, which removes all small families that
        began before 'seedtime.'
    seedtime : str, optional
        Reference date for age calculation; defaults to last trigger time.
    listonly : bool, optional
        Only print a list of families that would be removed.
    verbose : bool, optional
        Enable additional print statements.

    """
    detector = redpy.Detector(configfile, verbose, opened=True)
    if listonly:
        detector.set('verbose', True)
    small_families = detector.get_small_families(minmembers, maxage, seedtime)
    if (len(small_families) > 0) and not listonly:
        detector.remove('family', small_families, skip_dtable=True)
        detector.output()
    else:
        if detector.get('verbose'):
            print('No families removed.')
        if listonly:
            print('Rerun without -l to remove families listed.')
    detector.close()


def main():
    """Handle run from the command line."""
    args = parse()
    remove_small_family(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        prog='redpy-remove-small-family',
        description='Remove "small" families with few members.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-l', '--listonly', action='store_true', default=False,
                        help=('list families to keep and to remove, but do '
                              'not execute removal'))
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-m', '--minmembers', type=int, default=5,
                        help='minimum family size to keep; 5 by default')
    parser.add_argument('-a', '--maxage', type=float, default=0,
                        help=('maximum age of a family to be saved in days; '
                              '0 by default removes every small family '
                              'that began before SEEDTIME'))
    parser.add_argument('-t', '--seedtime', default='',
                        help=('time from which to compute family age '
                              'times (yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS); '
                              'last trigger time by default'))
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
