# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Write median family locations to file.

Run this script to write the median location of each family to a .csv file.
This currently only parses the existing .html files rather than querying
a catalog directly. Default behavior only uses locations for local matched
earthquakes, but the -d and -r flags allow distant matches to be considered.

usage: write_family_locations.py [-h] [-v] [-d] [-r] [-c CONFIGFILE]
                                  [-o OUTFILE]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -d, --distant         include distant (regional and teleseismic) matches
                        in addition to local seismicity
  -r, --regional        include regional matches in addition to local
                        seismicity
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -o OUTFILE, --outfile OUTFILE
                        define custom output file OUTFILE instead of default
                        famlocs.csv

"""
import argparse
import os

import redpy


def write_family_locations(configfile='settings.cfg', outfile='famlocs.csv',
                           distant=False, regional=False, verbose=False):
    """
    Write median family locations to a .csv file.

    Parameters
    ----------
    things

    """
    config = redpy.Config(configfile, verbose)
    outfile = os.path.join(config.get('output_folder'), outfile)
    df = redpy.catalog.get_median_locations(config, regional, distant)
    df.to_csv(outfile, index_label='Family')
    if config.get('verbose'):
        print(f'Done writing to {outfile}')


def main():
    """Handle run from the command line."""
    args = parse()
    write_family_locations(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    args : ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        description='Write median family locations to file.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-d', '--distant', action='store_true', default=False,
                        help=('include distant (regional and teleseismic) '
                              'matches in addition to local seismicity'))
    parser.add_argument('-r', '--regional', action='store_true', default=False,
                        help=('include regional matches in addition to local '
                              'seismicity'))
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-o', '--outfile', default='famlocs.csv',
                        help=('define custom output file OUTFILE instead of '
                              'default famlocs.csv'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
