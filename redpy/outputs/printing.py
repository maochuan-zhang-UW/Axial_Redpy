# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling functions related to creating text catalogs.

The primary function of this module is to support the .output() method of
Detector() objects. The .output() method generates various images and .html
files so the user may easily browse and export the contents of REDPy's
detections.
"""
import os

import matplotlib.dates as mdates
import numpy as np
from obspy import UTCDateTime

from redpy.correlation import subset_matrix


def catalog_cores(detector):
    """
    Print simple catalog of current core events to text file.

    Columns of this catalog correspond to family number and event time. Event
    time corresponds to the current best alignment, rather than when the
    event originally triggered.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    outfile = os.path.join(detector.get('output_folder'), 'cores.txt')
    with open(outfile, 'w', encoding='utf-8') as file:
        file.write('Family\tEvent Time (UTC)\n')
        for fnum in range(len(detector)):
            core = detector.get('ftable', 'core', fnum)
            format_time = UTCDateTime(
                detector.get('plotvars')['rtimes'][core]).isoformat()
            file.write(f'{fnum}\t{format_time}\n')


def catalog_family(detector):
    """
    Print simple catalog of family members to text file.

    Columns of this catalog correspond to family number and event time,
    sorted chronologically within each family. Event time corresponds to
    the current best alignment, rather than when the event originally
    triggered.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    outfile = os.path.join(detector.get('output_folder'), 'catalog.txt')
    with open(outfile, 'w', encoding='utf-8') as file:
        file.write('Family\tEvent Time (UTC)\n')
        for fnum in range(len(detector)):
            fam = detector.get_members(fnum)
            format_times = [UTCDateTime(i).isoformat() for i in np.sort(
                detector.get('plotvars')['rtimes'][fam])]
            for format_time in format_times:
                file.write(f'{fnum}\t{format_time}\n')


def catalog_junk(detector):
    """
    Print simple catalog of junk table to text file for debugging.

    Columns of this catalog correspond to the original trigger time and a
    code corresponding to which 'type' of junk that clean_triggers() thought
    it was.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    outfile = os.path.join(detector.get('output_folder'), 'junk.txt')
    type_strings = ['freq', 'kurt', 'both']
    if detector.get('verbose'):
        print(f'Writing junk catalog to {outfile}...')
    start_times = detector.get('jtable', 'startTime')
    jtype = detector.get('jtable', 'isjunk')
    with open(outfile, 'w', encoding='utf-8') as file:
        file.write('Trigger Time (UTC)\tJunk Type\n')
        for i in np.argsort(start_times):
            format_time = (UTCDateTime(start_times[i])
                           + detector.get('ptrig')).isoformat()
            file.write(f'{format_time}\t{type_strings[jtype[i]]}\n')


def catalog_orphans(detector):
    """
    Print simple catalog of current orphans to text file.

    Event times in this file correspond to the original trigger times.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    outfile = os.path.join(detector.get('output_folder'), 'orphancatalog.txt')
    with open(outfile, 'w', encoding='utf-8') as file:
        file.write('Trigger Time (UTC)\n')
        for time in np.sort(detector.get('otable', 'startTime')):
            format_time = (
                UTCDateTime(time) + detector.get('ptrig')
                / detector.get('samprate')).isoformat()
            file.write(f'{format_time}\n')


def catalog_swarm(detector):
    """
    Print .csv files for use in annotating events in Swarm v2.8.5+.

    Format for Swarm is 'Date Time, STA CHA NET LOC, label'
    The SCNL defaults to whichever station was chosen for the preview,
    which can be changed by a global search/replace in a text editor.
    The label name is the same as the folder name (groupname) followed by
    the family number. Highlighting families of interest in a different
    color can be done by editing the EventClassifications.config file in
    the Swarm folder, and adding a line for each family of interest
    followed by a hex code for color, such as:
        default1, #ffff00
    to highlight family 1 from the 'default' run in yellow compared to
    other repeaters in red/orange.

    Separate files for repeaters and triggers are rendered.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    scnl = (f"{detector.get('station')[detector.get('printsta')]} "
            f"{detector.get('channel')[detector.get('printsta')]} "
            f"{detector.get('network')[detector.get('printsta')]} "
            f"{detector.get('location')[detector.get('printsta')]}")
    catalogfile = os.path.join(detector.get('output_folder'), 'swarm.csv')
    triggerfile = os.path.join(detector.get('output_folder'),
                               'triggerswarm.csv')
    with open(catalogfile, 'w', encoding='utf-8') as file:
        for fnum in range(len(detector)):
            fam = detector.get_members(fnum)
            for time in np.sort(detector.get('plotvars')['rtimes'][fam]):
                file.write(
                    f"{UTCDateTime(time).isoformat(sep=' ')}, "
                    f"{scnl}, {detector.get('groupname')}{fnum}\n")
    with open(triggerfile, 'w', encoding='utf-8') as file:
        for time in np.sort(detector.get('plotvars')['ttimes']):
            file.write(
                f"{UTCDateTime(mdates.num2date(time)).isoformat(sep=' ')}, "
                f"{scnl}, trigger\n")


def catalog_triggers(detector):
    """
    Print simple catalog of all triggers to text file.

    Event times in this file correspond to the original trigger times.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    outfile = os.path.join(detector.get('output_folder'), 'triggers.txt')
    with open(outfile, 'w', encoding='utf-8') as file:
        file.write('Trigger Time (UTC)\n')
        for ttime in np.sort(detector.get('plotvars')['ttimes']):
            format_time = UTCDateTime(mdates.num2date(ttime)).isoformat()
            file.write(f'{format_time}\n')


def catalog_verbose(detector):
    """
    Print detailed catalog of family members to a .csv file.

    Like the simple catalog, events are sorted by event time within each
    family. Additional columns correspond to frequency index,
    cross-correlation coefficient with respect to the event with the highest
    sum (matching the family plots) and with the current core event, time
    since previous event (dt), and the amplitudes on all channels (grouped
    with [ square brackets ]).

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    outfile = os.path.join(detector.get('output_folder'), 'catalog.csv')
    ids = detector.get('plotvars')['ids']
    ccc_sparse = detector.get('plotvars')['ccc_sparse']
    with open(outfile, 'w', encoding='utf-8') as file:
        file.write('Family,Event Time (UTC),FI,'
                   'ccc_max,ccc_core,dt (hr),[ Amplitudes ]\n')
        for fnum in range(len(detector)):
            fam = detector.get_members(fnum)
            catalog = detector.get('plotvars')['rtimes_mpl'][fam]
            fam = fam[np.argsort(catalog)]
            catalog = np.sort(catalog)
            spacing = np.concatenate(([np.nan], np.diff(catalog)*24))
            ccc_max = subset_matrix(
                ids[fam], ccc_sparse, return_type='maxrow')
            ccc_core = subset_matrix(
                ids[fam], ccc_sparse, return_type='indrow', ind=np.where(
                    fam == detector.get('ftable', 'core', fnum))[0][0])
            for i, member in enumerate(fam):
                time = UTCDateTime(
                    detector.get("plotvars")["rtimes"][member]).isoformat()
                file.write(
                    f'{fnum},{time},'
                    f'{detector.get("plotvars")["mean_fi"][member]:4.3f},'
                    f'{ccc_max[i]:3.2f},{ccc_core[i]:3.2f},'
                    f'{spacing[i]:12.6f},[ ')
                for amp in detector.get('rtable', 'windowAmp', member):
                    file.write(f'{amp:10.2f}')
                file.write(' ]\n')


def generate_catalogs(detector):
    """
    Generate catalog files based on contents of Detector.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    catalog_triggers(detector)
    catalog_orphans(detector)
    if len(detector) and np.sum(detector.get('ftable', 'printme')):
        if detector.get('verbosecatalog'):
            catalog_verbose(detector)
        else:
            catalog_family(detector)
        catalog_swarm(detector)
        catalog_cores(detector)
