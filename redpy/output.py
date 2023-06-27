# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling functions related to creating outputs.

The primary function of this module is to support the .output() method of
Detector() objects. The .output() method generates various images and .html
files so the user may easily browse and export the contents of REDPy's
detections. Functions in this module specifically are called directly by
.output(), and then call other functions in redpy.outputs.
"""
import glob
import os

import matplotlib
import matplotlib.dates as mdates
import numpy as np
from obspy import UTCDateTime

import redpy.outputs

# Adjust mpl rcParams
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 8.0
matplotlib.rcParams['pdf.fonttype'] = 42


def force(detector, **kwargs):
    """
    Force plots to be re-rendered, regardless of current status.

    The keyword arguments for this function must be among the keyword
    arguments for generate() and set_print_cols(). If none are given,
    the keyword argument 'plotall' is set to True.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    **kwargs
        Arbitrary keyword arguments.

    """
    kwargs_print_cols = {'resetlp': True, 'plotall': False,
                         'startfam': 0, 'endfam': 0}
    kwargs_generate = dict.fromkeys(
        ['catalogs', 'timelines', 'images', 'html'], True)
    plotall = True
    for key, value in kwargs.items():
        if key in kwargs_print_cols:
            kwargs_print_cols[key] = value
            plotall = False
        elif key in kwargs_generate:
            kwargs_generate[key] = value
        else:
            raise KeyError(f'{key} is not an accepted keyword argument, '
                           f'must be in {list(kwargs_generate.keys())} or '
                           f'{list(kwargs_print_cols.keys())}')
    if plotall:  # Enforce default behavior to plot everything.
        kwargs_print_cols['plotall'] = True
    set_print_cols(detector, **kwargs_print_cols)
    generate(detector, **kwargs_generate)


def generate(detector, catalogs=True, timelines=True, images=True, html=True):
    """
    Generate updated catalogs, images, and .html pages.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    catalogs : bool, optional
        If True, write catalogs.
    timelines : bool, optional
        If True, create Bokeh timelines.
    images : bool, optional
        If True, create family images.
    html : bool, optional
        If True, write family .html files.

    """
    if len(detector.get('ttable')):
        if detector.get('verbose'):
            print('Updating outputs...')
        set_plotvars(detector)
        if catalogs:
            redpy.outputs.printing.generate_catalogs(detector)
        if timelines:
            redpy.outputs.timeline.generate_timelines(detector)
        if images:
            redpy.outputs.image.generate_images(detector)
        if html:
            redpy.outputs.html.generate_html(detector)
        set_print_cols(detector)
        remove_old_files(detector)
    else:
        print('No triggers, not creating outputs!')


def junk(detector):
    """
    Make junk outputs (catalog and images).

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    redpy.outputs.printing.catalog_junk(detector)
    redpy.outputs.image.create_junk_images(detector)


def pdf_family(detector, fnum=None, starttime='', endtime=''):
    """
    Make a .pdf version of the family plot.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    fnum : int, int list, or int ndarray
        Family number or numbers to make plots for.
    starttime : str, optional
        Adjust starting time for timeline plots.
    endtime : str, optional
        Adjust ending time for timeline plots.

    """
    if fnum is None:
        raise ValueError("Specify at least one family number with fnum=")
    if isinstance(fnum, int):
        fnum = [fnum]
    tmin = 0
    tmax = 0
    if starttime:
        tmin = UTCDateTime(starttime).matplotlib_date
    if endtime:
        tmax = UTCDateTime(endtime).matplotlib_date
    redpy.output.set_plotvars(detector)
    _, bboxes = redpy.outputs.image.initialize_family_image(detector)
    for fam in fnum:
        if detector.get('verbose'):
            print(f'Creating fam{fam}.pdf...')
        redpy.outputs.image.assemble_family_image(
            detector, fam, tmin, tmax, bboxes, 'pdf', 100)


def pdf_timeline(detector, starttime='', endtime='', binsize=0, usehrs=False,
                 minmembers=0, occurheight=3,
                 plotformat='eqrate,fi,occurrence,longevity'):
    """
    Make a .pdf version of the overview timeline.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    starttime : str, optional
        Adjust starting time for timeline plots; defaults to time of first
        trigger.
    endtime : str, optional
        Adjust ending time for timeline plots; defaults to time of last
        trigger.
    binsize : float, optional
        Temporal bin size (in days) for both the rate and occurrence plots.
        Defaults to 'dybin' configuration.
    usehrs : bool, optional
        If True, define binsize in hours instead of days.
    minmembers : int, optional
        Minimum number of members in a family to be included in the
        occurrence plot; defaults to 'minplot' configuration.
    occurheight : int, optional
        How much taller the occurrence plot should be than the other
        subplots.
    plotformat : str
        Comma-separated list of plots to include.

    """
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
        binsize = detector.get('dybin')
    if not plotformat:
        plotformat = 'eqrate,fi,occurrence,longevity'
    if not minmembers:
        minmembers = detector.get('minplot')
    redpy.output.set_plotvars(detector)
    redpy.outputs.timeline.generate_pdf_timeline(
        detector, tmin, tmax, binsize, minmembers, occurheight, plotformat)


def remove_old_files(detector):
    """
    Remove .html and .png files from deleted or moved family pages.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    tmplist = glob.glob(
        os.path.join(detector.get('output_folder'), 'families', '*.tmp'))
    for tmp in tmplist:
        os.rename(tmp, tmp[0:-4])
    flist = glob.glob(os.path.join(detector.get(
        'output_folder'), 'families', '*.html'))
    for file in flist:
        fnum = int(os.path.split(file)[1].split('.')[0])
        if fnum >= len(detector):
            os.remove(file)
            for itype in [f'{fnum}.png', f'fam{fnum}.png', f'map{fnum}.png']:
                img = os.path.join(os.path.split(file)[0], itype)
                if os.path.exists(img):
                    os.remove(img)


def report(detector, fnum=None, ordered=False, skip_recalculate_ccc=False,
           matrixtofile=False):
    """
    Create more detailed 'report' family pages.

    Reports are generated in the reports/ directory, named by family number.
    They include images of all waveforms on each station/channel stored,
    interactive versions of the family timelines, and correlation matrices.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    fnum : int, int list, int ndarray
        Family number(s) to render report(s) for.
    ordered : bool, optional
        If True, order waveforms and correlation matrix by OPTICS instead of
        by time.
    skip_recalculate_ccc : bool, optional
        If True, do not calculate the full cross-correlation matrix. This is
        recommended for very large families as the number of calculations
        required to fill the matrix can be significant.
    matrixtofile : bool, optional
        If True, save the full cross-correlation matrix to a .npy file in
        the reports/ directory.

    """
    if fnum is None:
        raise ValueError("Specify at least one family number with fnum=")
    if isinstance(fnum, int):
        fnum = [fnum]
    redpy.output.set_plotvars(detector)
    for i in fnum:
        redpy.outputs.report.create_report(
            detector, i, ordered, skip_recalculate_ccc, matrixtofile)


def set_print_cols(detector, resetlp=True, plotall=False, startfam=0,
                   endfam=0):
    """
    Set 'printme' and 'lastprint' columns of Families table.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    resetlp : bool, optional
        If True, sets 'lastprint' column to match row index.
    plotall : bool, optional
        If True, completely resets 'printme' column so all families are
        output.
    startfam : int, optional
        Starting family to generate plots for. May be negative to count
        backward from last family.
    endfam : int, optional
        Ending family to generate plots for. May be negative to count
        backward from last family.

    """
    if plotall:
        if detector.get('verbose'):
            print('Resetting plotting column...')
        detector.set('ftable', np.ones(len(detector)), 'printme')
    else:
        detector.set('ftable', np.zeros(len(detector)), 'printme')
    if resetlp:
        detector.set('ftable', np.arange(len(detector)), 'lastprint')
    if startfam or endfam:
        if startfam < 0:
            startfam = len(detector) + startfam
        if endfam < 0:
            endfam = len(detector) + endfam
        if (startfam > endfam) and endfam:
            raise ValueError('startfam is larger than endfam!')
        if startfam >= len(detector)-1:
            raise ValueError('startfam is larger than the number of available '
                             f'families ({len(detector)})!')
        if endfam > len(detector):
            raise ValueError('endfam is larger than the number of available '
                             f'families ({len(detector)})!')
        if startfam < 0:
            raise ValueError('startfam cannot be less than '
                             f'-{len(detector)}')
        if startfam and not endfam:
            endfam = len(detector)
        detector.set('ftable', np.ones(endfam - startfam), 'printme',
                     np.arange(startfam, endfam))


def set_plotvars(detector):
    """
    Load commonly called variables into Detector's 'plotvars' dictionary.

    These variables are:

        'rtimes' : datetime ndarray
            Trigger times of all repeaters as datetimes.
        'rtimes_mpl' : float ndarray
            Trigger times of all repeaters as matplotlib dates.
        'ttimes' : float ndarray
            Trigger times of all triggers as matplotlib dates.
        'mean_fi' : float ndarray
            Mean frequency index across all available channels.
        'ids' : int ndarray
            'id' column from Repeaters table.
        'ccc_sparse' : float csr_matrix
            Sparse correlation matrix with id as rows/columns.

    Several variables are also 'remembered' because they will be called
    into memory a lot.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    detector.get('rtable').remember('windowStart')
    detector.get('rtable').remember('windowAmp')
    detector.get('ftable').remember('startTime')
    detector.get('ftable').remember('longevity')
    seconds_to_days = 1 / 86400
    samples_to_days = seconds_to_days / detector.get('samprate')
    window_start = detector.get('rtable', 'windowStart') * samples_to_days
    rtimes_mpl = detector.get('rtable', 'startTimeMPL') + window_start
    detector.get('plotvars')['rtimes_mpl'] = rtimes_mpl
    detector.get('plotvars')['rtimes'] = np.array(
        [mdates.num2date(rtime) for rtime in rtimes_mpl])
    detector.get('plotvars')['ttimes'] = detector.get(
        'ttable', 'startTimeMPL') + detector.get('ptrig') * seconds_to_days
    detector.get('plotvars')['mean_fi'] = np.nanmean(
        detector.get('rtable', 'FI'), axis=1)
    detector.get('plotvars')['amps'] = detector.get(
        'rtable', 'windowAmp')[:, detector.get('printsta')]
    if len(detector):
        (detector.get('plotvars')['ids'],
         detector.get('plotvars')['ccc_sparse']) = detector.get_matrix()
    else:
        detector.get('plotvars')['ids'] = np.array([])
        detector.get('plotvars')['ccc_sparse'] = np.array([])
