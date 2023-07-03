# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Produce a detailed "report" page for one or more families.

These report pages resemble the family images, but with more details and are
more interactive. All waveforms are plotted instead of just the core and
stack, and the timelines can be zoomed and panned. At the bottom are two
correlation matrices, one depicting what values are stored in the file, and
what that matrix would look like if it were filled with all possible pairs.
Note that for very large families (1000+ members) this matrix can take a
long time to calculate, and may be bypassed with -s.

usage: redpy-create-report [-h] [-v] [-o] [-c CONFIGFILE] N [N ...]

positional arguments:
  N                     family number(s) to be reported on

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -o, --ordered         order plots by OPTICS instead of time
  -m, --matrixtofile    save correlation matrix to file
  -s, --skip            skip recalculating the full correlation matrix
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
"""
import argparse
import os

import redpy


def create_report(fam_list, configfile='settings.cfg', verbose=False,
                  ordered=False, matrixtofile=False, skip=False):
    """
    Produce a detailed "report" page for one or more families.

    Parameters
    ----------
    fam_list : int list
        List of family numbers to produce reports for.
    configfile : str
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.
    ordered : bool, optional
        If True, order plots by OPTICS instead of by time.
    matrixtofile : bool, optional
        If True, save correlation matrix to local .npy file.
    skip : bool, optional
        If True, skip recalculating the full correlation matrix.

    """
    detector = redpy.Detector(configfile, verbose, opened=True)
    detector.output('report', fnum=fam_list, ordered=ordered,
                    skip_recalculate_ccc=skip, matrixtofile=matrixtofile)
    detector.close()


def main():
    """Handle run from the command line."""
    args = parse()
    create_report(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser Object

    """
    parser = argparse.ArgumentParser(
        description=('Produce a detailed "report" page for one or more '
                     'families.'))
    parser.add_argument('fam_list', metavar='N', type=int, nargs='+',
                        help='family number(s) to be reported on')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-o', '--ordered', action='store_true', default=False,
                        help='order plots by OPTICS instead of time')
    parser.add_argument('-m', '--matrixtofile', action='store_true',
                        default=False, help='save correlation matrix to file')
    parser.add_argument('-s', '--skip', action='store_true', default=False,
                        help='skip recalculating the full correlation matrix')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
