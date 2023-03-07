# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import argparse
import time

import numpy as np
import pandas as pd
from obspy import UTCDateTime

import redpy


def main():
    """
    Backfills table with data from the past given a catalog.
    
    Run this script to fill the table with data from the past using a catalog
    of known events to limit the amount of waveforms to process.
    
    usage: catfill.py [-h] [-v] [-t] [-c CONFIGFILE] csvfile
    
    positional arguments:
      csvfile               catalog csv file with a "Time UTC" column of event
                            times
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         increase written print statements
      -t, --troubleshoot    run in troubleshoot mode (without try/except)
      -c CONFIGFILE, --configfile CONFIGFILE
                            use configuration file named CONFIGFILE instead of
                            default settings.cfg
    
    """
    t_func = time.time()
    args = catfill_parse()
    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
        redpy.table.open_with_cfg(args.configfile, args.verbose,
                                  args.troubleshoot)
    
    # Read in csv file using pandas
    df = pd.read_csv(args.csvfile)
    event_list = np.array([UTCDateTime(ev) for ev in df['Time UTC']])
    event_list.sort()
    run_start_time = event_list[0] - 4*opt.atrig
    run_end_time = event_list[-1] +  5*opt.atrig + opt.maxdt
    
    # Create or read in file key to improve local file load times
    filekey, preload_waveforms, preload_end_time = \
        redpy.trigger.initial_data_preload(run_start_time, run_end_time, opt)
    
    for event_time in event_list:
        if opt.verbose: print(event_time)
        window_start_time = event_time - 4*opt.atrig
        window_end_time = event_time + 5*opt.atrig + opt.maxdt
        if len(ttable) > 0:
            ttimes = ttable.cols.startTimeMPL[:]
        else:
            ttimes = 0
        
        h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, \
            preload_waveforms, preload_end_time, opt = \
                redpy.table.update_tables(h5file, rtable, otable, ttable,
                    ctable, jtable, dtable, ftable, ttimes, filekey,
                    preload_waveforms, preload_end_time, run_end_time,
                    window_start_time, window_end_time, opt,
                    event_list=event_list)
        
        redpy.table.print_stats(rtable, otable, ftable, opt)
    redpy.plotting.generate_all_outputs(rtable, ftable, ttable, ctable,
                                        otable, opt)
    if opt.verbose: print('Closing table...')
    h5file.close()
    if opt.verbose: print('Total time spent: '
                          f'{(time.time()-t_func)/60:.3f} minutes')
    print('Done')


def catfill_parse():
    """
    Defines and parses acceptable command line inputs for catfill.py.
    
    Returns
    -------
    args : ArgumentParser Object
    
    """
    parser = argparse.ArgumentParser(
        description='Backfills table with data from the past given a catalog.')
    parser.add_argument('csvfile',
                        help=('catalog csv file with a "Time UTC" column of '
                              'event times'))
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-t', "--troubleshoot", action='store_true',
                        default=False,
                        help='run in troubleshoot mode (without try/except)')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

