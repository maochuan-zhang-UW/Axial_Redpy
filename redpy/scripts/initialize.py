# Copyright (C) 2016-2018  Alicia Hotovec-Ellis (ahotovec@gmail.com)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Initialize and overwrite hdf5 table using configuration.

Running this script will overwrite an existing table with the same name
defined by filename in the .cfg file!

usage: redpy-initialize [-h] [-v] [-c CONFIGFILE]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg

"""
import argparse

from redpy.detector import Detector


def initialize(configfile='settings.cfg', verbose=False):
    """
    Initialize tables defined in configfile, overwriting existing tables.

    Additionally, create folder structure for outputs.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.

    """
    detector = Detector(configfile, verbose)
    detector.initialize()
    detector.close()


def main():
    """Handle run from the command line."""
    args = parse()
    initialize(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        prog='redpy-initialize',
        description='Initialize and overwrite hdf5 table using configuration.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
