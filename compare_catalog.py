# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
r"""
Compare REDPy catalog with external catalog.

Run this script to compare an independent text catalog with a REDPy catalog.
The input catalog is appended with the best matching event, with columns for
times, time difference (negative being that REDPy triggered early), which
family or trigger type the best match corresponds to, and for repeaters,
the frequency index and amplitudes. The combined catalog is saved
separately, and by default in the current directory.

usage: compare_catalog.py [-h] [-v] [-c CONFIGFILE] [-d DELIMITER]
                          [-m MAXDTOFFSET] [-n NAME] [-o OUTFILE]
                          catfile

positional arguments:
  catfile               catalog file

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -d DELIMITER, --delimiter DELIMITER
                        define custom DELIMITER between columns instead of
                        default "," (e.g., "\t" for tabs, " " for spaces, or
                        "|" for pipes)
  -m MAXDTOFFSET, --maxdtoffset MAXDTOFFSET
                        define custom time offset (in seconds) instead of
                        default 75% of window length
  -n NAME, --name NAME  define custom time column NAME instead of default
                        "Time"
  -o OUTFILE, --outfile OUTFILE
                        define custom output file name and location
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from obspy import UTCDateTime

import redpy


def compare_catalog(catfile, configfile='settings.cfg', delimiter=',',
                    maxdtoffset=-1., name='Time', outfile='', verbose=False):
    """
    Compare an independent text catalog with a REDPy catalog.

    The input catalog is appended with the best matching event, with columns
    for times, time difference (negative being that REDPy triggered early),
    which family or trigger type the best event corresponds to, and for
    repeaters, the frequency index and amplitudes. The combined catalog is
    saved separately, and by default in the current directory.

    Parameters
    ----------
    catfile : str
        File name of catalog to compare with.
    configfile : str, optional
        Name of configuration file to read.
    delimiter : str, optional
        Delimiter between columns of the catalog.
    maxdtoffset : float, optional
        Maximum time offset in seconds to be considered a match. If
        negative, reverts to the default of 75% of the correlation window
        length, which is also the minimum time between consecutive triggers.
    name : str, optional
        Name of the 'Time' column in catfile.
    outfile : str, optional
        Path and filename to write output catalog to.
    verbose : bool, optional
        Enable additional print statements.

    """
    catalog = pd.read_csv(catfile, sep=delimiter)
    catalog['Match Time'] = pd.to_datetime(catalog[name], utc=True)
    catalog['Trigger Time'] = ''
    catalog['Trigger Time'] = pd.to_datetime(catalog['Trigger Time'], utc=True)
    catalog['dt (s)'] = ''
    catalog['Family'] = ''
    catalog['FI'] = ''
    catalog['Amplitudes'] = ''
    df, opt = build_event_dataframe(configfile, verbose)
    if maxdtoffset < 0:
        maxdtoffset = opt.mintrig
    if not outfile:
        outfile = f'matches_{getattr(opt, "groupName")}.csv'
    if getattr(opt, 'verbose'):
        print('Matching...')
    for i in range(len(catalog)):
        if getattr(opt, 'verbose'):
            if i % 1000 == 0 and i > 0:
                print(f'{100.0*i/len(catalog):3.2f}% complete')
        time_delta = df['Trigger Time'] - catalog['Match Time'][i]
        idx = np.argmin(np.abs(time_delta))
        if np.abs(time_delta[idx].total_seconds()) <= maxdtoffset:
            catalog['Trigger Time'][i] = df['Trigger Time'][idx]
            catalog['dt (s)'][i] = time_delta[idx].total_seconds()
            catalog['Family'][i] = df['Family'][idx]
            catalog['FI'][i] = df['FI'][idx]
            catalog['Amplitudes'][i] = df['Amplitudes'][idx]
    if getattr(opt, 'verbose'):
        print(f'Saving to {outfile}')
    catalog.to_csv(outfile, index=False, date_format='%Y-%m-%dT%H:%M:%S.%fZ')


def build_event_dataframe(configfile='settings.cfg', verbose=False):
    """
    Create a DataFrame from triggers and repeaters on disk.

    Specifically, this DataFrame has columns for the trigger time, family
    number (or whether it is a trigger (former orphans or deleted events),
    current orphan, or categorized as junk), frequency index, and amplitudes
    on all stations.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    verbose : bool, optional
        Enable additional print statements.

    Returns
    -------
    df : DataFrame object
        Tabular summary of all triggers and subset of metadata.
    opt : Options object
        Describes the run parameters.

    """
    h5file, rtable, otable, ttable, _, jtable, _, ftable, opt = \
        redpy.table.open_with_cfg(configfile, verbose)
    if getattr(opt, 'verbose'):
        print('Building event table...')
    jdates = [UTCDateTime(j).matplotlib_date for j in jtable.cols.startTime[:]]
    rtimes_mpl = rtable.cols.startTimeMPL[:]
    fi_mean = np.nanmean(rtable.cols.FI[:], axis=1)
    amps = rtable.cols.windowAmp[:]
    df = pd.DataFrame(columns=['Trigger Time', 'Family', 'FI', 'Amplitudes'],
                      index=np.concatenate((ttable.cols.startTimeMPL[:],
                                            np.array(jdates))))
    df['Family'] = 'trigger'
    df['Family'][jdates] = 'junk'
    df['Family'][otable.cols.startTimeMPL[:]] = 'orphan'
    df['Trigger Time'] = mdates.num2date(
        df.index + opt.ptrig/86400)
    df['Trigger Time'][rtimes_mpl] = mdates.num2date(
        rtimes_mpl + rtable.cols.windowStart[:]/getattr(opt, 'samprate')/86400)
    for fam in range(ftable.attrs.nClust):
        members = np.fromstring(ftable[fam]['members'], dtype=int, sep=' ')
        df['Family'][rtimes_mpl[members]] = fam
        df['FI'][rtimes_mpl[members]] = fi_mean[members]
        df['Amplitudes'][rtimes_mpl[members]] = amps[members, :].tolist()
    df.reset_index(drop=True, inplace=True)
    h5file.close()
    return df, opt


def main():
    """Handle run from the command line."""
    args = parse()
    compare_catalog(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    args : ArgumentParser Object

    """
    parser = argparse.ArgumentParser(
        description='Compare REDPy catalog with external catalog.')
    parser.add_argument('catfile', help='catalog file')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-d', '--delimiter', default=',',
                        help=('define custom DELIMITER between columns '
                              'instead of default "," (e.g., "\\t" for tabs, '
                              '" " for spaces, or "|" for pipes)'))
    parser.add_argument('-m', '--maxdtoffset', default=-1.,
                        help=('define custom time offset (in seconds) '
                              'instead of default 75%% of window length'))
    parser.add_argument('-n', '--name', default='Time',
                        help=('define custom time column NAME instead of '
                              'default "Time"'))
    parser.add_argument('-o', '--outfile', default='',
                        help='define custom output file name and location')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
