# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Clear contents of the "junk" table.

usage: clearJunk.py [-h] [-v] [-c CONFIGFILE]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
"""
import argparse

import redpy


def clear_junk(configfile='settings.cfg', verbose=False):
    """
    Remove all "junk" from hdf5 file defined in configuration file.

    Closes file when done.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.

    """
    h5file, _, _, _, _, jtable, _, _, opt = redpy.table.open_with_cfg(
        configfile, verbose)
    redpy.table.remove_all_junk(jtable, opt)
    h5file.close()


def main():
    """Handle run from the command line."""
    args = parse()
    clear_junk(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    args : ArgumentParser Object

    """
    parser = argparse.ArgumentParser(
        description='Clear contents of the "junk" table.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
