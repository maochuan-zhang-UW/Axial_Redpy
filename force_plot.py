# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Update plots by force.

Run this script to force plotting. Can be used immediately after a run has
been interrupted or after updating settings. Note that the -s and -e settings
follow Python convention for arrays, i.e., start at 0 and do not include the
ending number. For example, -s 1 -e 5 would replot 1, 2, 3, and 4. Negative
numbers may be used to count backward from the last family.

usage: force_plot.py [-h] [-v] [-a] [-f] [-l] [-r] [-c CONFIGFILE]
                     [-s STARTFAM] [-e ENDFAM]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -a, --plotall         render everything, not just updated families
  -f, --famplot         only render the family plots, not .html pages
  -l, --html            only render the .html pages, not family plots
  -r, --resetlp         reset the "last print" column (use for "missing file"
                        errors)
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -s STARTFAM, --startfam STARTFAM
                        manual starting family to replot (assumes first family
                        if not set)
  -e ENDFAM, --endfam ENDFAM
                        manual (noninclusive) ending family to replot (assumes
                        last family if not set)

"""
import argparse

import numpy as np
import matplotlib.dates as mdates

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
        If True, completely resets 'printme' column so all families are output.
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
        Ending family to generate plots for. May be negative to count backward
        from last family.

    """
    h5file, rtable, otable, ttable, ctable, _, _, ftable, opt = \
        redpy.table.open_with_cfg(configfile, verbose)
    set_ftable_columns(ftable, opt, plotall, resetlp, startfam, endfam)
    if opt.verbose:
        print("Creating requested plots...")
    if famplot or html:
        generate_subset_outputs(rtable, ftable, ttable, ctable, opt, famplot,
                                html)
    else:
        redpy.plotting.generate_all_outputs(rtable, ftable, ttable, ctable,
                                            otable, opt)
    h5file.close()


def set_ftable_columns(ftable, opt, plotall=False, resetlp=False, startfam=0,
                       endfam=0):
    """
    Reset 'printme' and 'lastprint' columns of Families table.

    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    opt : Options object
        Describes the run parameters.
    plotall : bool, optional
        If True, completely resets 'printme' column so all families are output.
    resetlp : bool, optional
        If True, sets 'lastprint' column to match row index.
    startfam : int, optional
        Starting family to generate plots for. May be negative to count
        backward from last family.
    endfam : int, optional
        Ending family to generate plots for. May be negative to count backward
        from last family.

    """
    if plotall:
        if opt.verbose:
            print('Resetting plotting column...')
        ftable.cols.printme[:] = np.ones(ftable.attrs.nClust)
    if resetlp:
        if opt.verbose:
            print('Resetting last print column...')
        ftable.cols.lastprint[:] = np.arange(ftable.attrs.nClust)
    if startfam or endfam:
        if startfam < 0:
            startfam = ftable.attrs.nClust + startfam
        if endfam < 0:
            endfam = ftable.attrs.nClust + endfam
        if (startfam > endfam) and endfam:
            raise ValueError('startfam is larger than endfam!')
        if startfam == endfam:
            print('startfam is equal to endfam; no plots will be produced.')
        if startfam >= ftable.attrs.nClust-1:
            raise ValueError('startfam is larger than the number of available '
                             f'families ({ftable.attrs.nClust})!')
        if endfam > ftable.attrs.nClust:
            raise ValueError('endfam is larger than the number of available '
                             f'families ({ftable.attrs.nClust})!')
        if startfam < 0:
            raise ValueError('startfam cannot be less than '
                             f'-{ftable.attrs.nClust}')
        ftable.cols.printme[:] = np.zeros(ftable.attrs.nClust)
        if startfam and not endfam:
            ftable.cols.printme[startfam:] = np.ones(
                ftable.attrs.nClust - startfam)
        elif endfam and not startfam:
            ftable.cols.printme[:endfam] = np.ones(endfam)
        else:
            ftable.cols.printme[startfam:endfam] = np.ones(endfam - startfam)


def generate_subset_outputs(rtable, ftable, ttable, ctable, opt, famplot=True,
                            html=True):
    """
    Generate family plot images and/or .html pages.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    ttable : Table object
        Handle to the Triggers table.
    ctable : Table object
        Handle to the Correlation table.
    opt : Options object
        Describes the run parameters.
    famplot : bool, optional
        If True, generates family plot images.
    html : bool, optional
        If True, generates .html pages.

    """
    windowStart = rtable.cols.windowStart[:]
    windowAmps = rtable.cols.windowAmp[:]
    rtimes_mpl = (rtable.cols.startTimeMPL[:]
                  + windowStart[:]/opt.samprate/86400)
    rtimes = np.array([mdates.num2date(rtime) for rtime in rtimes_mpl])
    if famplot:
        ids, ccc_sparse = redpy.correlation.get_matrix(rtable, ctable, opt)
        redpy.plotting.create_family_images(rtable, ftable, rtimes, rtimes_mpl,
                                            windowAmps, ids, ccc_sparse, opt)
    if html:
        if opt.checkComCat:
            ttimes = ttable.cols.startTimeMPL[:] + opt.ptrig/opt.samprate/86400
            external_catalogs = redpy.catalog.prepare_catalog(ttimes, opt)
        else:
            external_catalogs = []
        fi = np.nanmean(rtable.cols.FI[:], axis=1)
        redpy.plotting.create_family_html(
            rtable, ftable, rtimes, rtimes_mpl, windowAmps, fi,
            external_catalogs, opt)
    ftable.cols.printme[:] = np.zeros(ftable.attrs.nClust)
    ftable.cols.lastprint[:] = np.arange(ftable.attrs.nClust)


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
