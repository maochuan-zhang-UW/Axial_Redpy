# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Create a publication-quality, editable .pdf version of overview page.

Run this script to manually produce overview.pdf in the output directory
(same location as overview.html), optionally with a custom time span and/or
different options from the configuration file. Note that BINSIZE applies to
both the 'rate' and 'occurrence'/'occurrencefi' histograms.

usage: create_pdf_overview.py [-h] [-v] [-u] [-b BINSIZE] [-c CONFIGFILE]
                              [-s STARTTIME] [-e ENDTIME] [-m MINMEMBERS]
                              [-o OCCURHEIGHT] [-f FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -u, --usehrs          use hours instead of days for definition of BINSIZE
  -b BINSIZE, --binsize BINSIZE
                        custom time bin size
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -s STARTTIME, --starttime STARTTIME
                        earliest time to plot (yyyy-mm-dd or
                        yyyy-mm-ddTHH:MM:SS), defaults to first trigger
  -e ENDTIME, --endtime ENDTIME
                        latest time to plot(yyyy-mm-dd or
                        yyyy-mm-ddTHH:MM:SS), defaults to last trigger
  -m MINMEMBERS, --minmembers MINMEMBERS
                        minimum number of members required to include family
                        in occurrence plot
  -o OCCURHEIGHT, --occurheight OCCURHEIGHT
                        integer multiplier for how much taller the
                        occurrence plot should be compared to other plots;
                        defaults to 3
  -f PLOTFORMAT, --plotformat PLOTFORMAT
                        comma separated list of subplots
"""
import argparse

import numpy as np
from obspy import UTCDateTime

import redpy


def create_pdf_overview(
        configfile='settings.cfg', verbose=False, usehrs=False, binsize=0.,
        starttime='', endtime='', minmembers=0, occurheight=3, plotformat=''):
    """
    Create a publication-quality, editable .pdf version of overview page.

    An 'overview.pdf' file will be created in the primary output directory.
    A custom time span may be used to zoom in on a time period of interest.
    Note that the setting 'binsize' applies to both the 'rate' and
    'occurrence'/'occurrencefi' histograms.

    Parameters
    ----------
    configfile : str, optional
        Configuration file to read.
    verbose : bool, optional
        Enable additional print statements.
    usehrs : bool, optional
        Use hours (instead of days) to define binsize.
    binsize : float, optional
        Histogram bin size; defaults to opt.dybin days.
    starttime : str, optional
        Earliest time to plot; defaults to first trigger.
    endtime : str, optional
        Latest time to plot; defaults to last trigger.
    minmembers : int, optional
        Minimum number of members required to include family in occurrence
        plot; defaults to opt.minplot.
    occurheight : int, optional
        Integer multiplier for how much taller the occurrence plot(s) should
        be compared to other plots; defaults to 3.
    plotformat : str, optional
        Comma separated list of subplots to use with same format as
        opt.plotformat; defaults to 'eqrate,fi,occurrence,longevity'.

    """
    h5file, rtable, _, ttable, ctable, _, _, ftable, opt = \
        redpy.table.open_with_cfg(configfile, verbose)
    if starttime:
        tmin = UTCDateTime(starttime).matplotlib_date
    else:
        tmin = 0
    if endtime:
        tmax = UTCDateTime(endtime).matplotlib_date
    else:
        tmax = 0
    if binsize:
        if usehrs:
            binsize = binsize/24
    else:
        binsize = opt.dybin
    if not minmembers:
        minmembers = opt.minplot
    if not plotformat:
        plotformat = 'eqrate,fi,occurrence,longevity'
    if opt.verbose:
        print('Creating overview.pdf in main output directory...')
    rtimes, rtimes_mpl, _, ttimes, fi, _, _ = \
        redpy.plotting.get_plotting_columns(rtable, ttable, ctable, opt,
                                            load_cmatrix=False)
    fi = np.nanmean(fi, axis=1)
    redpy.plotting.assemble_pdf_overview(
        rtable, ftable, ttimes, rtimes, rtimes_mpl, fi, tmin, tmax, binsize,
        minmembers, occurheight, plotformat, opt)
    h5file.close()


def main():
    """Handle run from the command line."""
    args = parse()
    create_pdf_overview(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    args : ArgumentParser Object

    """
    parser = argparse.ArgumentParser(
        description=('Create a publication-quality, editable .pdf version of '
                     'overview page.'))
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-u', '--usehrs', action='store_true', default=False,
                        help=('use hours instead of days for definition of '
                              'BINSIZE'))
    parser.add_argument('-b', '--binsize', type=float, default=0.,
                        help='custom time bin size')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-s', '--starttime',
                        help=('earliest time to plot (yyyy-mm-dd or '
                              'yyyy-mm-ddTHH:MM:SS), defaults to first '
                              'trigger'))
    parser.add_argument('-e', '--endtime',
                        help='latest time to plot (yyyy-mm-dd or '
                              'yyyy-mm-ddTHH:MM:SS), defaults to last trigger')
    parser.add_argument('-m', '--minmembers', type=int, default=0,
                        help=('minimum number of members required to include '
                              'family in occurrence plot'))
    parser.add_argument('-o', '--occurheight', type=int, default=3,
                        help=('integer multiplier for how much taller the '
                              'occurrence plot should be compared to other '
                              'plots; defaults to 3'))
    parser.add_argument('-f', '--plotformat',
                        help='comma separated list of subplots')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
