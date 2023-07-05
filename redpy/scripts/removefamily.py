# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Manually remove one or more families.

Run this script to remove families that are not of interest such as
correlated noise that made it past the 'junk' detector or regional
seismicity. The cores from these families are moved to the "deleted" table
and, if new triggers correlate with them, they are not considered further.
Ensures outputs are up to date.

usage: redpy-remove-family [-h] [-v] [-c CONFIGFILE] N [N ...]

positional arguments:
  N                     family number(s) to be moved and deleted

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
"""
import argparse

import redpy


def remove_family(fam_list, configfile='settings.cfg', verbose=False):
    """
    Manually remove one or more families.

    The cores from these families are moved to the "deleted" table, and, if
    new triggers correlate with them, they are not considered further.

    Parameters
    ----------
    fam_list : int list
        List of family numbers to remove.
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.

    """
    detector = redpy.Detector(configfile, verbose, opened=True)
    detector.remove('family', fam_list)
    detector.output()
    detector.close()


def main():
    """Handle run from the command line."""
    args = parse()
    remove_family(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        prog='redpy-remove-family',
        description='Manually remove one or more families.')
    parser.add_argument('fam_list', metavar='N', type=int, nargs='+',
                        help='family number(s) to be moved and deleted')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
