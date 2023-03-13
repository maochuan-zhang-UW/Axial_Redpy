# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2018  Alicia Hotovec-Ellis (ahotovec@gmail.com)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Initialize and overwrite hdf5 table using configuration.

Running this script will overwrite an existing table with the same name
defined by filename in the .cfg file!

usage: initialize.py [-h] [-v] [-c CONFIGFILE]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg

"""
import argparse
import os

import redpy


def initialize(configfile='settings.cfg', verbose=False):
    """
    Initialize tables defined in configfile, overwriting any existing
    tables.

    Additionally, create folder structure for outputs.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.

    """
    opt = redpy.config.Options(configfile, verbose)
    create_output_folders(opt)
    redpy.table.initialize_table(opt)


def create_output_folders(opt):
    """
    Create folder structure for outputs.

    Parameters
    ----------
    opt : Options object
        Describes the run parameters.

    """
    if opt.verbose:
        print(f'Creating folders to store outputs...\n{opt.output_folder}')
    try:
        os.mkdir(opt.output_folder)
    except OSError as exc:
        if opt.verbose:
            print(exc)
    subfolder = os.path.join(opt.output_folder, 'clusters')
    if opt.verbose:
        print(subfolder)
    try:
        os.mkdir(subfolder)
    except OSError as exc:
        if opt.verbose:
            print(exc)


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
    args : ArgumentParser object

    """
    parser = argparse.ArgumentParser(
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
