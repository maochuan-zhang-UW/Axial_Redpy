# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Create publication-quality, editable .pdf versions of family images.

Run this script to manually produce files in the families directory
(same location as fam*.png), optionally with a custom time span.

usage: create_pdf_family.py [-h] [-v] [-c CONFIGFILE] [-s STARTTIME]
                            [-e ENDTIME] N [N ...]

positional arguments:
  N                     family number(s) to be plotted

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead
                        of default settings.cfg
  -s STARTTIME, --starttime STARTTIME
                        earliest time to plot (yyyy-mm-dd or
                        yyyy-mm-ddTHH:MM:SS); defaults to first event
  -e ENDTIME, --endtime ENDTIME
                        latest time to plot (yyyy-mm-dd or
                        yyyy-mm-ddTHH:MM:SS); defaults to last event
"""
import argparse

import redpy


def create_pdf_family(fam_list, configfile='settings.cfg', verbose=False,
                      starttime='', endtime=''):
    """
    Create publication-quality, editable .pdf versions of family images.

    These images are saved in the same location as fam*.png (i.e., in
    the families directory) as fam*.pdf. A custom start and/or end time
    can be defined to zoom the time span to a span of interest.

    Parameters
    ----------
    fam_list : int list
        List of families to produce files for.
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.
    starttime : str, optional
        Earliest time to plot; defaults to first event.
    endtime : str, optional
        Latest time to plot; defaults to last event.

    """
    detector = redpy.Detector(configfile, verbose, opened=True)
    detector.output(
        'pdf_family', fnum=fam_list, starttime=starttime, endtime=endtime)
    detector.close()


def main():
    """Handle run from the command line."""
    args = parse()
    create_pdf_family(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser Object

    """
    parser = argparse.ArgumentParser(
        description=('Create publication-quality, editable .pdf versions of '
                     'family images.'))
    parser.add_argument('fam_list', metavar='N', type=int, nargs='+',
                        help='family number(s) to be plotted')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-s', '--starttime',
                        help=('earliest time to plot (yyyy-mm-dd or '
                              'yyyy-mm-ddTHH:MM:SS); defaults to first event'))
    parser.add_argument('-e', '--endtime',
                        help=('latest time to plot (yyyy-mm-dd or '
                              'yyyy-mm-ddTHH:MM:SS); defaults to last event'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
