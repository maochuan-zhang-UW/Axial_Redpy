# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import os

import matplotlib.dates as mdates
import numpy as np

from obspy import UTCDateTime

import redpy.correlation


def catalog_family(ftable, rtimes, opt):
    """
    Prints simple catalog of family members to text file.
    
    Columns of this catalog correspond to family number and event time, sorted
    chronologically within each family. Event time corresponds to the current
    best alignment, rather than when the event originally triggered.
    
    ftable : Table object
        Handle to the Families table.
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    opt : Options object
        Describes the run parameters.
    
    """
    
    outfile = os.path.join(opt.output_folder, 'catalog.txt')
    
    with open(outfile, 'w') as f:
        
        f.write('Family\tEvent Time (UTC)\n')
        for fnum in range(ftable.attrs.nClust):
            
            fam = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')
            
            for i in np.argsort(rtimes[fam]):
                format_time = UTCDateTime(rtimes[fam[i]]).isoformat()
                f.write(f'{fnum}\t{format_time}\n')


def catalog_triggers(ttimes, opt):
    """
    Prints simple catalog of all triggers to text file.
    
    Event times in this file correspond to the original trigger times.
    
    Parameters
    ----------
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    opt : Options object
        Describes the run parameters.
    
    """
    
    outfile = os.path.join(opt.output_folder, 'triggers.txt')
    
    with open(outfile, 'w') as f:
        
        f.write('Trigger Time (UTC)\n')
        for ttime in np.sort(ttimes):
            format_time = UTCDateTime(mdates.num2date(ttime)).isoformat()
            f.write(f'{format_time}\n')


def catalog_orphans(otable, opt):
    """
    Prints simple catalog of current orphans to text file.
    
    Event times in this file correspond to the original trigger times.
    
    Parameters
    ----------
    otable : Table object
        Handle to the Orphans table.
    opt : Options object
        Describes the run parameters.
    
    """
    
    outfile = os.path.join(opt.output_folder, 'orphancatalog.txt')
    
    startTimes = otable.cols.startTime[:]
    
    with open(outfile, 'w') as f:
        
        f.write('Trigger Time (UTC)\n')
        for i in np.argsort(startTimes):
            format_time = (UTCDateTime(startTimes[i]) + \
                                           opt.ptrig/opt.samprate).isoformat()
            f.write(f'{format_time}\n')


def catalog_junk(jtable, opt):
    """
    Print simple catalog of junk table to text file for debugging.

    Columns of this catalog correspond to the original trigger time and a
    code corresponding to which 'type' of junk that clean_triggers() thought
    it was.

    Parameters
    ----------
    jtable : Table object
        Handle to the Junk table.
    opt : Options object
        Describes the run parameters.

    """
    outfile = os.path.join(opt.output_folder, 'junk.txt')
    if opt.verbose:
        print(f'Writing junk catalog to {outfile}...')
    startTimes = jtable.cols.startTime[:]
    jtype = jtable.cols.isjunk[:]
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write('Trigger Time (UTC)\tJunk Code\n')
        for i in np.argsort(startTimes):
            format_time = (UTCDateTime(startTimes[i])
                           + opt.ptrig/opt.samprate).isoformat()
            f.write(f'{format_time}\t{jtype[i]}\n')


def catalog_cores(ftable, rtimes, opt):
    """
    Prints simple catalog of current core events to text file.
    
    Columns of this catalog correspond to family number and event time. Event
    time corresponds to the current best alignment, rather than when the
    event originally triggered.
    
    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    opt : Options object
        Describes the run parameters.
    
    """
    
    outfile = os.path.join(opt.output_folder, 'cores.txt')
    
    with open(outfile, 'w') as f:
        
        f.write('Family\tEvent Time (UTC)\n')
        for fnum in range(ftable.attrs.nClust):
            
            core = ftable[fnum]['core']
            format_time = UTCDateTime(rtimes[core]).isoformat()
            f.write(f'{fnum}\t{format_time}\n')


def catalog_verbose(ftable, rtimes, rtimes_mpl, windowAmps, fi, ids,
                                                             ccc_sparse, opt):
    """
    Prints detailed catalog of family members to text file.
    
    Like the simple catalog, events are sorted by event time within each
    family. Additional columns correspond to frequency index,
    cross-correlation coefficient with respect to the event with the highest
    sum (matching the family plots) and with the current core event, time
    since previous event (dt), and the amplitudes on all channels (grouped
    with [ square brackets ]).
    
    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    windowAmps : float ndarray
        'windowAmp' column from rtable for all stations.
    fi : float ndarray
        Frequency index values for repeaters.
    ids : int ndarray
        'id' column from Repeaters table.
    ccc_sparse : float csr_matrix
        Sparse correlation matrix with id as rows/columns.
    opt: Options object describing station/run parameters
    
    """
    
    outfile = os.path.join(opt.output_folder, 'catalog.txt')
    
    with open(outfile, 'w') as f:
        
        f.write('Family\tEvent Time (UTC)\tFI\t')
        f.write('ccc_max\tccc_core\tdt (hr)\t[ Amplitudes ]\n')
        for fnum in range(ftable.attrs.nClust):
            
            fam = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')
            catalogind = np.argsort(rtimes_mpl[fam])
            fam = fam[catalogind]
            corenum = ftable[fnum]['core']
            catalog = rtimes_mpl[fam]
            
            # Get dt
            spacing = np.diff(catalog)*24
            
            # Get correlation values for maximum sum along row (to match
            # family plots) and with the current core event
            ccc_max = redpy.correlation.subset_matrix(ids[fam], ccc_sparse,
                opt, return_type='maxrow')
            ccc_core = redpy.correlation.subset_matrix(ids[fam], ccc_sparse,
                opt, return_type='indrow', ind=np.where(fam==corenum)[0][0])
            
            for i, member in enumerate(fam):
                evTime = UTCDateTime(rtimes[member])
                amp = windowAmps[member,:]
                if i == 0:
                    dt = np.nan
                else:
                    dt = spacing[i-1]
                f.write(f'{fnum}\t{evTime.isoformat()}\t')
                f.write(f'{fi[member]:4.3f}\t')
                f.write(f'{ccc_max[i]:3.2f}\t{ccc_core[i]:3.2f}\t')
                f.write(f'{dt:12.6f}\t[')
                for a in amp:
                    f.write(f' {a:10.2f}')
                f.write(' ]\n')


def catalog_swarm(ftable, ttimes, rtimes, opt):
    
    """
    Writes a .csv file for use in annotating events in Swarm v2.8.5+.
    
    Format for Swarm is 'Date Time, STA CHA NET LOC, label'
    The SCNL defaults to whichever station was chosen for the preview,
    which can be changed by a global search/replace in a text editor.
    The label name is the same as the folder name (groupName) followed by
    the family number. Highlighting families of interest in a different
    color can be done by editing the EventClassifications.config file in
    the Swarm folder, and adding a line for each cluster of interest
    followed by a hex code for color, such as:
        default1, #ffff00
    to highlight family 1 from the 'default' run in yellow compared to other
    repeaters in red/orange.
    
    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    opt : Options object
        Describes the run parameters.
    
    """
    
    nets = opt.network.split(',')
    stas = opt.station.split(',')
    locs = opt.location.split(',')
    chas = opt.channel.split(',')
    
    catalogfile = os.path.join(opt.output_folder, 'swarm.csv')
    triggerfile = os.path.join(opt.output_folder, 'triggerswarm.csv')
    
    with open(catalogfile, 'w') as f:
        
        for fnum in range(ftable.attrs.nClust):
            fam = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')
            for i in np.argsort(rtimes[fam]):
                
                f.write("{}, {} {} {} {}, {}{}\n".format(UTCDateTime(
                    rtimes[fam][i]).isoformat(sep=' '), stas[opt.printsta],
                    chas[opt.printsta], nets[opt.printsta],
                    locs[opt.printsta], opt.groupName,fnum))
                
    with open(triggerfile, 'w') as f:
    
        for ttime in np.sort(ttimes):
            f.write("{}, {} {} {} {}, trigger\n".format((UTCDateTime(
                mdates.num2date(ttime))).isoformat(sep=' '),
                stas[opt.printsta],chas[opt.printsta],nets[opt.printsta],
                    locs[opt.printsta]))
