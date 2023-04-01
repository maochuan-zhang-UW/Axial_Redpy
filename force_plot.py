# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Update plots by force.

Run this script to force plotting. Can be used immediately after a run has
been interrupted or after updating settings. Note that the -s and -e
settings follow Python convention for arrays, i.e., start at 0 and do not
include the ending number. For example, -s 1 -e 5 would replot 1, 2, 3,
and 4. Negative numbers may be used to count backward from the last family.

usage: force_plot.py [-h] [-v] [-a] [-f] [-l] [-r] [-c CONFIGFILE]
                     [-s STARTFAM] [-e ENDFAM]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -a, --plotall         render everything, not just updated families
  -f, --famplot         only render the family plots, not .html pages
  -l, --html            only render the .html pages, not family plots
  -r, --resetlp         reset the "last print" column (use for "missing
                        file" errors)
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -s STARTFAM, --startfam STARTFAM
                        manual starting family to replot (assumes first
                        family if not set)
  -e ENDFAM, --endfam ENDFAM
                        manual (noninclusive) ending family to replot
                        (assumes last family if not set)

"""
import argparse

import redpy


def force_plot(configfile='settings.cfg', verbose=False, plotall=False,
               famplot=False, html=False, resetlp=False, startfam=0,
               endfam=0):
    """
    Generate plots (or a subset of plots) for a table defined in configfile.

    Additional control on which outputs to produce is allowed via optional
    parameters, including forcing everything to be produced regardless of
    updated status.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.
    plotall : bool, optional
        If True, completely resets 'printme' column so all families are
        output.
    famplot : bool, optional
        If True, only generates family image plots.
    html : bool, optional
        If True, only generates .html pages.
    resetlp : bool, optional
        If True, resets the 'lastprint' column.
    startfam : int, optional
        Starting family to generate plots for. May be negative to count
        backward from last family.
    endfam : int, optional
        Ending family to generate plots for. May be negative to count
        backward from last family.

    """
    h5file, rtable, otable, ttable, ctable, _, _, ftable, config = \
        redpy.table.open_with_cfg(configfile, verbose)
    redpy.table.set_ftable_columns(ftable, config, plotall, resetlp, startfam,
                                   endfam)
    if config.get('verbose'):
        print('Creating requested plots...')
    if famplot or html:
        redpy.plotting.generate_subset_outputs(rtable, ftable, ttable, ctable,
                                               config, famplot, html)
    else:
        redpy.plotting.generate_all_outputs(rtable, ftable, ttable, ctable,
                                            otable, config)
    h5file.close()


def main():
    """Handle run from the command line."""
    args = parse()
    force_plot(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    args : ArgumentParser object

    """
    parser = argparse.ArgumentParser(description='Update plots by force.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-a', '--plotall', action='store_true', default=False,
                        help='render everything, not just updated families')
    parser.add_argument('-f', '--famplot', action='store_true', default=False,
                        help='only render the family plots, not .html pages')
    parser.add_argument('-l', '--html', action='store_true', default=False,
                        help='only render the .html pages, not family plots')
    parser.add_argument('-r', '--resetlp', action='store_true', default=False,
                        help=('reset the "last print" column (use for '
                              '"missing file" errors)'))
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-s', '--startfam', type=int, default=0,
                        help=('manual starting family to replot (assumes '
                              'first family if not set)'))
    parser.add_argument('-e', '--endfam', type=int, default=0,
                        help=('manual (noninclusive) ending family to replot '
                              '(assumes last family if not set)'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
