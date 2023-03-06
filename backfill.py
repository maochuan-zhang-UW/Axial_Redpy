# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import argparse
import time

from obspy import UTCDateTime

import redpy


def main():
    """
    Backfills table with data from the past.
    
    Run this script to fill the table with data from the past. If a start time
    is not specified, it will check the attributes of the repeater table to
    pick up where it left off. Additionally, if this is the first run and a
    start time is not specified, it will assume one time chunk prior to the
    end time. If an end time is not specified, "now" is assumed. The end time
    updates at the end of each time chunk processed (default: by hour, set in
    configuration). This script can be run as a cron job that will pick up
    where it left off if a chunk is missed. Use -n if you are backfilling with
    a large amount of time; it will consume less time downloading the data in
    small chunks if NSEC is an hour or a day instead of a few minutes, but at
    the cost of keeping orphans for longer.
    
    usage: backfill.py [-h] [-v] [-t] [-s STARTTIME] [-e ENDTIME]
                       [-c CONFIGFILE] [-n NSEC]
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         increase written print statements
      -t, --troubleshoot    run in troubleshoot mode (without try/except)
      -s STARTTIME, --starttime STARTTIME
                            optional start time to begin filling
                            (yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)
      -e ENDTIME, --endtime ENDTIME
                            optional end time to end filling
                            (yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)
      -c CONFIGFILE, --configfile CONFIGFILE
                            use configuration file named CONFIGFILE instead of
                            default settings.cfg
      -n NSEC, --nsec NSEC  overwrite opt.nsec from configuration file with
                            NSEC this run only
    
    """
    t_func = time.time()
    args = backfill_parse()
    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
        redpy.table.open_with_cfg(args.configfile, args.verbose,
                                  args.troubleshoot)
    
    # Deal with input arguments
    if args.nsec: opt.nsec = args.nsec
    if args.endtime:
        run_end_time = UTCDateTime(args.endtime)
    else:
        run_end_time = UTCDateTime()
    if args.starttime:
        run_start_time = UTCDateTime(args.starttime)
        if rtable.attrs.ptime:
            rtable.attrs.ptime = UTCDateTime(run_start_time)
    else:
        if rtable.attrs.ptime:
            run_start_time = UTCDateTime(rtable.attrs.ptime)
        else:
            run_start_time = run_end_time-opt.nsec
    if run_start_time > run_end_time:
        raise ValueError(
            f'Start {run_start_time} is after end {run_end_time}!')
    
    if len(ttable) > 0:
        ttimes = ttable.cols.startTimeMPL[:]
    else:
        ttimes = 0
    
    # Load data from file
    filekey, preload_waveforms, preload_end_time = \
        redpy.trigger.initial_data_preload(run_start_time, run_end_time, opt)
    
    n = 0
    while run_start_time + n*opt.nsec < run_end_time:
        t_iter = time.time()
        window_start_time = run_start_time + n*opt.nsec
        window_end_time = min(run_start_time+(n+1)*opt.nsec,
                              run_end_time) + opt.atrig + opt.maxdt
        print(window_start_time)
        
        h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, \
            preload_waveforms, preload_end_time, opt = \
                redpy.table.update_tables(
                    h5file, rtable, otable, ttable, ctable, jtable, dtable,
                    ftable, ttimes, filekey, preload_waveforms,
                    preload_end_time, run_end_time, window_start_time,
                    window_end_time, opt)
        
        n += 1
        redpy.table.clear_expired_orphans(otable, window_end_time, opt)
        redpy.table.print_stats(rtable, otable, ftable, opt)
        if opt.verbose: print('Time spent this iteration: '
                              f'{(time.time()-t_iter)/60:.3f} minutes')
    print(f'Caught up to: {window_end_time-opt.atrig}')
    redpy.plotting.generate_all_outputs(rtable, ftable, ttable, ctable,
                                        otable, opt)
    if opt.verbose: print('Closing table...')
    h5file.close()
    if opt.verbose: print('Total time spent: '
                          f'{(time.time()-t_func)/60:.3f} minutes')
    print('Done')


def backfill_parse():
    """
    Defines and parses acceptable command line inputs for backfill.py.
    
    Returns
    -------
    args : ArgumentParser Object
    
    """
    parser = argparse.ArgumentParser(
        description='Backfills table with data from the past.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-t', "--troubleshoot", action='store_true',
                        default=False,
                        help='run in troubleshoot mode (without try/except)')
    parser.add_argument('-s', '--starttime',
                        help=('optional start time to begin filling '
                              '(yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)'))
    parser.add_argument('-e', '--endtime',
                        help=('optional end time to end filling '
                              '(yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)'))
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-n', '--nsec', type=int,
                        help=('overwrite opt.nsec from configuration file '
                              'with NSEC this run only'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

