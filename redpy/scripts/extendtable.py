# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Copy data from an existing table into a new, larger table.

Run this script to create space for additional stations while preserving
data in an existing table or to change the directory name for a run.
Additional stations should always be included at the end of the station
list; reordering that list is currently not supported. Running this
script will overwrite any existing table with the same name defined by
filename in the new .cfg file. If the table names in both .cfg files are the
same, the original table will be renamed and then deleted. All output files
are also remade to reflect the additional station, unless flagged otherwise.

usage: redpy-extend-table [-h] [-v] [-n] CONFIGFILE_FROM CONFIGFILE_TO

positional arguments:
  CONFIGFILE_FROM       old .cfg file corresponding to table to be copied
                        from
  CONFIGFILE_TO         new .cfg file corresponding to table to be copied to

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -n, --noplot          do not re-render plots after extending
"""
import argparse

from redpy.config import Config
from redpy.detector import Detector


def extend_table(cfgfrom, cfgto, verbose=False, noplot=False):
    """
    Copy data from existing table into a new, larger table.

    Parameters
    ----------
    cfgfrom : str
        Configuration file corresponding to table to copy from.
    cfgto : str
        Configuration file corresponding to table to copy to.
    verbose : bool, optional
        Increase written print statements.
    noplot : bool, optional
        If True, skip plotting once done copying.

    """
    if isinstance(cfgto, list):  # pragma: no cover
        cfgto = cfgto[0]
    if isinstance(cfgfrom, list):  # pragma: no cover
        cfgfrom = cfgfrom[0]
    config_to = Config(cfgto, verbose)
    detector = Detector(cfgfrom, verbose)
    detector.expand(config_to, update_outputs=not noplot)
    detector.close()


def main():
    """Handle run from the command line."""
    args = parse()
    extend_table(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser Object

    """
    parser = argparse.ArgumentParser(
        prog='redpy-extend-table',
        description=(
            'Copy data from existing table into a new, larger table.'))
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="increase written print statements")
    parser.add_argument('-n', '--noplot', action='store_true', default=False,
                        help='do not re-render plots after extending')
    parser.add_argument('cfgfrom', metavar='CONFIGFILE_FROM', type=str,
                        nargs=1, help=(
                            'old .cfg file corresponding to table to be '
                            'copied from'))
    parser.add_argument('cfgto', metavar='CONFIGFILE_TO', type=str, nargs=1,
                        help=('new .cfg file corresponding to table to be '
                              'copied to'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
