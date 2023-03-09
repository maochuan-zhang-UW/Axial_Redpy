# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import argparse
import time

import numpy as np
from obspy import UTCDateTime

import redpy


def main():
    """
    Backfills table with data from the past given a catalog.
    
    Run this script to fill the table with data from the past using a catalog
    of known events to limit the amount of waveforms to process. By default,
    catfill does not expire orphans, allowing backfill to be run over the
    same time period and for orphaned catalog events to still exist. This
    behavior can be overridden with -x to expire orphans after each event
    time. When using -f, a trigger will be forced at the given time and
    'junk' filtering will be skipped, however, the minimum allowed time
    between events is still enforced.
    
    usage: catfill.py [-h] [-v] [-t] [-a] [-f] [-q] [-x] [-c CONFIGFILE]
                      [-d DELIMITER] [-n NAME] [-s STARTTIME] [-e ENDTIME]
                      csvfile
    
    positional arguments:
      csvfile               catalog csv file with a column of event times or
                            file to write a queried catalog to disk
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         increase written print statements
      -t, --troubleshoot    run in troubleshoot mode (without try/except)
      -a, --arrival         estimate and use the P-wave arrival time to the
                            center of the network; requires a location to be
                            included in the catalog
      -f, --force           force a trigger at time specified in catalog
                            instead of using default STA/LTA triggering
      -q, --query           queries external catalog for local seismicity
                            as defined in the config file and saves output
                            to csvfile
      -x, --expire          expire orphans
      -c CONFIGFILE, --configfile CONFIGFILE
                            use configuration file named CONFIGFILE instead of
                            default settings.cfg
      -d DELIMITER, --delimiter DELIMITER
                            define custom DELIMITER between columns instead
                            of default "," (e.g., "\t" for tabs, " " for
                            spaces, or "|" for pipes)
      -n NAME, --name NAME  define custom time column NAME instead of default
                            "Time"
      -s STARTTIME, --starttime STARTTIME
                            subsets catalog to begin at STARTTIME
                            (yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)
      -e ENDTIME, --endtime ENDTIME
                            subsets catalog to end at ENDTIME
                            (yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)
    
    """
    t_func = time.time()
    args = catfill_parse()
    h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, opt = \
        redpy.table.open_with_cfg(args.configfile, args.verbose,
                                  args.troubleshoot)
    if args.query:
        if not args.endtime:
            args.endtime = UTCDateTime()
            if opt.verbose:
                print(f'Defaulting to end time of "now" ({args.endtime})')
        if not args.starttime:
            args.starttime = args.endtime - opt.nsec
            if opt.verbose:
                print(f'Defaulting to start time of {opt.nsec} seconds '
                      f'before end time ({args.starttime})')
        catalog = redpy.catalog.query_external(
            'local', UTCDateTime(args.starttime),
            UTCDateTime(args.endtime), opt, arrivals=args.arrival)
        if len(catalog) == 0:
            print('No events found!')
            quit()
        catalog.to_csv(args.csvfile, index=False, sep=args.delimiter)
    try:
        event_list = redpy.catalog.get_event_times_from_csv(
            args.csvfile, args.name, args.delimiter, opt,
            start_time=args.starttime, end_time=args.endtime,
            arrivals=args.arrival)
    except KeyError:
        print(f'Could not find "{args.name}" column in {args.csvfile}. '
              'Check file, column name, and delimiter! Use -h for help.')
        exit()
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
        
        event = event_time if args.force else None
        h5file, rtable, otable, ttable, ctable, jtable, dtable, ftable, \
            preload_waveforms, preload_end_time, opt = \
                redpy.table.update_tables(h5file, rtable, otable, ttable,
                    ctable, jtable, dtable, ftable, ttimes, filekey,
                    preload_waveforms, preload_end_time, run_end_time,
                    window_start_time, window_end_time, opt,
                    event_list=event_list, event=event)
        
        if args.expire:
            lot = len(otable)
            redpy.table.clear_expired_orphans(otable, window_end_time, opt)
            if lot > len(otable):
                if opt.verbose: print(f'Expired {lot-len(otable)} orphan(s)')
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
                        help=('catalog csv file with a column of event times '
                              'or file to write a queried catalog to disk'))
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-t', '--troubleshoot', action='store_true',
                        default=False,
                        help='run in troubleshoot mode (without try/except)')
    parser.add_argument('-a', '--arrival', action='store_true', default=False,
                        help=('estimate and use the P-wave arrival time to '
                              'the center of the network; requires a location '
                              'to be included in the catalog'))
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help=('force a trigger at time specified in catalog '
                              'instead of using default STA/LTA triggering'))
    parser.add_argument('-q', '--query', action='store_true', default=False,
                        help=('queries external catalog for local seismicity '
                             'as defined in the config file and saves output '
                             'to csvfile'))
    parser.add_argument('-x', '--expire', action='store_true', default=False,
                        help='expire orphans')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-d', '--delimiter', default=',',
                        help=('define custom DELIMITER between columns '
                              'instead of default "," (e.g., "\\t" for tabs, '
                              '" " for spaces, or "|" for pipes)'))
    parser.add_argument('-n', '--name', default='Time',
                        help=('define custom time column NAME instead of '
                              'default "Time"'))
    parser.add_argument('-s', '--starttime', default=None,
                        help=('subsets catalog to begin at STARTTIME '
                              '(yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)'))
    parser.add_argument('-e', '--endtime', default=None,
                        help=('subsets catalog to end at ENDTIME '
                              '(yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS)'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

