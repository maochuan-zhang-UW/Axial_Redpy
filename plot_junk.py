# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Output the contents of the junk table for troubleshooting.

usage: plot_junk.py [-h] [-v] [-c CONFIGFILE]

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


def plot_junk(configfile='settings.cfg', verbose=False):
    """
    Output the contents of the junk table for troubleshooting.

    Creates images in a folder "junk" as well as a flat catalog "junk.txt"
    both in the main outputs directory for the run.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.

    """
    detector = redpy.Detector(configfile, verbose, opened=True)
    detector.output('junk')
    detector.close()


def create_junk_folder(config):
    """
    Create folder structure for outputs.

    Parameters
    ----------
    config : Config object
        Describes the run parameters.

    """
    if config.get('verbose'):
        print('Creating folder to store junk images...')
    subfolder = os.path.join(config.get('output_folder'), 'junk')
    if config.get('verbose'):
        print(subfolder)
    try:
        os.mkdir(subfolder)
    except OSError as exc:
        if config.get('verbose'):
            print(exc)


def main():
    """Handle run from the command line."""
    args = parse()
    plot_junk(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    args : ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        description=('Output the contents of the junk table for '
                     'troubleshooting.'))
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
