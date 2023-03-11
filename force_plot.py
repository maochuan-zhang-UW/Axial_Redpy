# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Update plots by force.

Run this script to force plotting. Can be used after killing mid-run or
updating settings. Note that the -s and -e settings follow Python convention
for arrays, i.e., start at 0 and do not include the ending number. For
example, -s 1 -e 5 would replot 1, 2, 3, and 4.

usage: force_plot.py [-h] [-v] [-a] [-f] [-l] [-r] [-c CONFIGFILE]
                     [-s STARTFAM] [-e ENDFAM]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -a, --all             replot everything, not just updated families
  -f, --famplot         only replot the family plots, not .html files
  -l, --html            only render the .html files, not any images
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
import os

import numpy as np
import matplotlib.dates as mdates

import redpy


def force_plot(configfile='settings.cfg', verbose=False, plotall=False,
               famplot=False, html=False, resetlp=False, startfam=0,
               endfam=0):
    """
    
    """
    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
    redpy.table.open_with_cfg(configfile, verbose)

    if plotall:
        if opt.verbose:
            print('Resetting plotting column...')
        ftable.cols.printme[:] = np.ones(ftable.attrs.nClust)

    if resetlp:
        if opt.verbose:
            print('Resetting last print column...')
        ftable.cols.lastprint[:] = np.arange(ftable.attrs.nClust)
    
    if startfam or endfam:
        ftable.cols.printme[:] = np.zeros(ftable.attrs.nClust)
        if startfam and not endfam:
            ftable.cols.printme[startfam:ftable.attrs.nClust] = np.ones(
                ftable.attrs.nClust-startfam)
        elif endfam and not startfam:
            ftable.cols.printme[0:endfam] = np.ones(endfam)
        else:
            ftable.cols.printme[startfam:endfam] = np.ones(endfam-startfam)

    if opt.verbose:
        print("Creating requested plots...")

    if famplot or html:
        windowStart = rtable.cols.windowStart[:]
        windowAmps = rtable.cols.windowAmp[:]
        rtimes_mpl = rtable.cols.startTimeMPL[:]+windowStart[:]/opt.samprate/86400
        rtimes = np.array([mdates.num2date(rtime) for rtime in rtimes_mpl])

    if famplot:
        # Get correlation matrix and ids
        ids, ccc_sparse = redpy.correlation.get_matrix(rtable, ctable, opt)
        redpy.plotting.create_family_images(rtable, ftable, rtimes, rtimes_mpl,
                                  windowAmps, ids, ccc_sparse, opt)
    if html:
        if opt.checkComCat==True:
            ttimes = ttable.cols.startTimeMPL[:] + opt.ptrig/opt.samprate/86400
            external_catalogs = redpy.catalog.prepare_catalog(ttimes, opt)
        else:
            external_catalogs = []
        fi = np.nanmean(rtable.cols.FI[:], axis=1)
        redpy.plotting.create_family_html(rtable, ftable, rtimes, rtimes_mpl,
                                          windowAmps, fi, external_catalogs, opt)

    if html or famplot:
        ftable.cols.printme[:] = np.zeros(ftable.attrs.nClust)
        ftable.cols.lastprint[:] = np.arange(ftable.attrs.nClust)
    else:
        redpy.plotting.generate_all_outputs(rtable, ftable, ttable, ctable,
                                                                       otable, opt)
    h5file.close()


def main():
    """Handle run from the command line."""
    args = parse()
    force_plot(**vars(args))
    print('Done')


def parse():
    """
    """
    parser = argparse.ArgumentParser(description='Update plots by force.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-a', '--plotall', action='store_true', default=False,
                        help='replot everything, not just updated families')
    parser.add_argument('-f', '--famplot', action='store_true', default=False,
                        help='only replot the family plots, not .html files')
    parser.add_argument('-l', '--html', action='store_true', default=False,
                        help='only render the .html files, not any images')
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
