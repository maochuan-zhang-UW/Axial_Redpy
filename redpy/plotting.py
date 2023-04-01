# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import glob, os, shutil
import datetime

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from bokeh.models import Arrow, ColorBar, ColumnDataSource, Div
from bokeh.models import HoverTool, Label, LinearColorMapper, LogColorMapper
from bokeh.models import LogTicker, OpenURL, Panel, Range1d, Span, Tabs
from bokeh.models import TapTool, VeeHead
from bokeh.models.glyphs import Line
from bokeh.models.formatters import LogTickFormatter
from bokeh.palettes import all_palettes, inferno
from bokeh.plotting import figure, gridplot, output_file, save
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.transforms import offset_copy
from obspy import UTCDateTime
from obspy.geodetics import kilometers2degrees
#from tables import *

import redpy
from redpy.optics import *

matplotlib.use('Agg')

# Adjust rcParams
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 8.0
matplotlib.rcParams['pdf.fonttype'] = 42


def get_plotting_columns(rtable, ttable, ctable, config, load_ttimes=True,
        load_fi=True, load_cmatrix=True):
    """
    Load several commonly used columns for plotting from tables into memory.

    If any of the load_* arguments are False, the returned variables will be
    empty. This can help reduce file read overhead when only a subset of these
    columns are needed.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ttable : Table object
        Handle to the Triggers table.
    ctable : Table object
        Handle to the Correlation table.
    config : Config object
        Describes the run parameters.
    load_ttimes : bool, optional
        If True, return ttimes filled.
    load_fi : bool, optional
        If True, return fi filled.
    load_cmatrix : bool, optional
        If True, return ids and ccc_sparse filled.

    Returns
    -------
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    windowAmps : float ndarray
        "windowAmps" column from Repeaters table.
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    fi : float ndarray
        "FI" column from Repeaters table.
    ids : int ndarray
        'id' column from Repeaters table.
    ccc_sparse : float csr_matrix
        Sparse correlation matrix with id as rows/columns.

    """
    windowStart = rtable.cols.windowStart[:]
    windowAmps = rtable.cols.windowAmp[:]
    rtimes_mpl = rtable.cols.startTimeMPL[:] + windowStart/config.get('samprate')/86400
    rtimes = np.array([mdates.num2date(rtime) for rtime in rtimes_mpl])
    if load_ttimes:
        ttimes = ttable.cols.startTimeMPL[:] + config.get('ptrig')/config.get('samprate')/86400
    else:
        ttimes = []
    if load_fi:
        fi = rtable.cols.FI[:]
    else:
        fi = []
    if load_cmatrix:
        ids, ccc_sparse = redpy.correlation.get_matrix(rtable, ctable, config)
    else:
        ids = []
        ccc_sparse = []
    return rtimes, rtimes_mpl, windowAmps, ttimes, fi, ids, ccc_sparse


def generate_all_outputs(rtable, ftable, ttable, ctable, otable, config):
    """
    Creates all output files.

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
    otable : Table object
        Handle to the Orphans table.
    config : Config object
        Describes the run parameters.

    """
    if config.get('verbose'): print('Updating plots...')

    # Call columns that are used multiple times into memory
    rtimes, rtimes_mpl, windowAmps, ttimes, _, _, _ = \
        get_plotting_columns(rtable, ttable, ctable, config, load_fi=False,
                             load_cmatrix=False)

    if config.get('checkcomcat')==True:
        external_catalogs = redpy.catalog.prepare_catalog(ttimes, config)
    else:
        external_catalogs = []

    # Write trigger and orphan catalogs
    redpy.printing.catalog_triggers(ttimes, config)
    redpy.printing.catalog_orphans(otable, config)

    # If there is at least one family
    if len(rtable) > 1:

        # Render Bokeh timelines
        fi = np.nanmean(rtable.cols.FI[:], axis=1)
        create_timelines(rtable, ftable, ttimes, rtimes, rtimes_mpl, fi, config)

        # If there are changes to the catalog
        if np.sum(ftable.cols.printme[:]):

            # Get correlation matrix and ids
            ids, ccc_sparse = redpy.correlation.get_matrix(rtable, ctable, config)

            # Write repeater-related catalogs
            if config.get('verbosecatalog') == True:
                # !!! pass columns here
                redpy.printing.catalog_verbose(ftable, rtimes, rtimes_mpl,
                                         windowAmps, fi, ids, ccc_sparse, config)
            else:
                redpy.printing.catalog_family(ftable, rtimes, config)
            redpy.printing.catalog_swarm(ftable, ttimes, rtimes, config)
            redpy.printing.catalog_cores(ftable, rtimes, config)

            # Make images
            create_core_images(rtable, ftable, config)
            create_family_images(rtable, ftable, rtimes, rtimes_mpl,
                                             windowAmps, ids, ccc_sparse, config)

            # Make HTML files
            create_family_html(rtable, ftable, rtimes, rtimes_mpl, windowAmps,
                                                   fi, external_catalogs, config)

            # Reset printing columns
            ftable.cols.printme[:] = np.zeros((len(ftable),))
            ftable.cols.lastprint[:] = np.arange(len(ftable))

    else:
        print('Nothing to plot!')

    # Rename any .tmp files created in create_core_images()
    tmplist = glob.glob(os.path.join(config.get('output_folder'), 'clusters','*.tmp'))
    for tmp in tmplist:
        os.rename(tmp,tmp[0:-4])


def generate_subset_outputs(rtable, ftable, ttable, ctable, config, famplot=True,
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
    config : Config object
        Describes the run parameters.
    famplot : bool, optional
        If True, generates family plot images.
    html : bool, optional
        If True, generates .html pages.

    """
    rtimes, rtimes_mpl, windowAmps, _, _, _, _ = get_plotting_columns(
        rtable, ttable, ctable, config, load_ttimes=False, load_fi=False,
        load_cmatrix=False)
    if famplot:
        ids, ccc_sparse = redpy.correlation.get_matrix(rtable, ctable, config)
        redpy.plotting.create_family_images(rtable, ftable, rtimes, rtimes_mpl,
                                            windowAmps, ids, ccc_sparse, config)
    if html:
        if config.get('checkcomcat'):
            ttimes = ttable.cols.startTimeMPL[:] + config.get('ptrig')/config.get('samprate')/86400
            external_catalogs = redpy.catalog.prepare_catalog(ttimes, config)
        else:
            external_catalogs = []
        fi = np.nanmean(rtable.cols.FI[:], axis=1)
        redpy.plotting.create_family_html(
            rtable, ftable, rtimes, rtimes_mpl, windowAmps, fi,
            external_catalogs, config)
    ftable.cols.printme[:] = np.zeros(ftable.attrs.nClust)
    ftable.cols.lastprint[:] = np.arange(ftable.attrs.nClust)


def create_timelines(rtable, ftable, ttimes, rtimes, rtimes_mpl, fi, config):
    """
    Creates HTML bokeh timelines: overview, overview_recent, and meta_recent.

    The primary purpose of this function is to sort out the differences in
    the unique behavior of each Bokeh timeline type.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    config : Config object
        Describes the run parameters.

    """

    # Load table columns into memory to reduce I/O overhead
    # !!! This variable technically needs the windowStart added to it
    # !!! Same for ttimes needing config.get('ptrig')...

    longevity = ftable.cols.longevity[:]
    famstarts = ftable.cols.startTime[:]

    if config.get('bokehendtime') == 'now':
        maxtime = UTCDateTime().matplotlib_date
    else:
        maxtime = np.max(ttimes)

    for i, file in enumerate(['overview', 'overview_recent', 'meta_recent']):

        # Set parameters unique to each timeline type
        if i == 0: # overview.html
            # Give a little extra padding to mintime
            mintime = np.min(ttimes)-config.get('winlen')/config.get('samprate')/86400
            barpad = (maxtime-mintime)*0.01
            plotformat = config.get('plotformat')
            binsize_hist = config.get('dybin')
            binsize_occur = config.get('occurbin')
            minplot = config.get('minplot')
            fixedheight = config.get('fixedheight')
            htmltitle = f'{config.get("title")} Overview'
            divtitle = f'<h1>{config.get("title")}</h1>'
        elif i == 1: # overview_recent.html
            mintime = maxtime-config.get('recplot')
            barpad = config.get('recplot')*0.01
            plotformat = config.get('plotformat')
            binsize_hist = config.get('hrbin')/24
            binsize_occur = config.get('recbin')
            minplot = 0
            fixedheight = config.get('fixedheight')
            htmltitle = f'{config.get("title")} Overview - Last {config.get("recplot"):.1f} Days'
            divtitle = f'<h1>{config.get("title")} - Last {config.get("recplot"):.1f} Days</h1>'
        else: # meta_recent.html
            mintime = maxtime-config.get('mrecplot')
            barpad = config.get('mrecplot')*0.01
            plotformat = config.get('plotformat').replace(',','+')
            binsize_hist = config.get('mhrbin')/24
            binsize_occur = config.get('mrecbin')
            minplot = config.get('mminplot')
            fixedheight = True
            htmltitle = f'{config.get("title")} Overview - Last {config.get("mrecplot"):.1f} Days'
            divtitle = f"""<h1>{config.get('title')} - Last {config.get('mrecplot'):.1f} Days |
                          <a href='overview.html' style='color:red'
                          target='_blank'>Full Overview</a> |
                          <a href='overview_recent.html'
                          style='color:red' target='_blank'>Recent</a>
                          </h1>"""

        filepath = os.path.join(config.get('output_folder'), f'{file}.html')

        assemble_bokeh_timeline(ftable, rtimes, rtimes_mpl, fi, longevity,
            famstarts, ttimes, barpad, plotformat, binsize_hist,
            binsize_occur, mintime, maxtime, minplot, fixedheight, filepath,
            htmltitle, divtitle, config)


def create_core_images(rtable, ftable, config):
    """
    Plots core waveforms as *.png files in the clusters folder.

    Used for hovering in timeline and header for HTML family pages.

    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    config : Config object
        Describes the run parameters.

    """

    opath = os.path.join(config.get('output_folder'), 'clusters')

    # Deal with renaming files to reduce plotting time overhead
    for n in range(len(ftable))[::-1]:
        if ftable[n]['lastprint'] != n and ftable[n]['printme'] == 0:
            lastprint = ftable[n]['lastprint']
            os.rename(os.path.join(opath,f'{lastprint}.png'),
                      os.path.join(opath,f'{n}.png.tmp'))
            os.rename(os.path.join(opath,f'fam{lastprint}.png'),
                      os.path.join(opath,f'fam{n}.png.tmp'))

    # Iterate and plot
    cores = rtable[ftable.cols.core[:]]
    n = -1
    for r in cores:
        n = n+1
        if ftable[n]['printme'] == 1:
            data = prep_wiggle(r['waveform'], config.get('printsta'), r['windowStart'],
                               r['windowAmp'][config.get('printsta')], config)
            wiggle_plot(data, (5, 1), os.path.join(opath, f'{n}.png'), config)


def create_junk_images(jtable, config):
    """
    Create images of waveforms contained in the junk table.

    File names correspond to the trigger time and the flag for the type of
    junk it was flagged as. Also writes a catalog to "junk.txt".

    Parameters
    ----------
    jtable : Table object
        Handle to the Junk table.
    config : Config object
        Describes the run parameters.

    """
    redpy.printing.catalog_junk(jtable, config)
    if config.get('verbose'):
        print('Creating junk plots...')
    for r in jtable:
        data = np.array([])
        for s in range(config.get('nsta')):
            # Concatenate all channels together
            data = np.append(data, prep_wiggle(r['waveform'], s,
                      r['windowStart'] + int(config.get('ptrig')*config.get('samprate')), 0, config))
        jtime = (UTCDateTime(r['startTime']) + config.get('ptrig')).strftime(
                '%Y%m%d%H%M%S')
        jtype = r['isjunk']
        wiggle_plot(data, (15, 0.5), os.path.join(config.get('output_folder'), 'junk',
                                                  f'{jtime}-{jtype}.png'), config)


def create_family_images(rtable, ftable, rtimes, rtimes_mpl, windowAmps, ids,
                                                            ccc_sparse, config):
    """
    Creates multi-paneled family plots for all families that need plotting.

    This function wraps assemble_family_image() and outputs all files as *.png
    files in the clusters folder.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    windowAmps : float ndarray
        'windowAmp' column from rtable for all stations.
    ids : int ndarray
        'id' column from Repeaters table.
    ccc_sparse : float csr_matrix
        Sparse correlation matrix with id as rows/columns.
    config : Config object
        Describes the run parameters.

    """

    fig, axes, bboxes = initialize_family_image(config)

    for fnum in range(ftable.attrs.nClust):

        if ftable[fnum]['printme'] != 0:

            assemble_family_image(bboxes, rtable, ftable, rtimes, rtimes_mpl,
                windowAmps[:,config.get('printsta')], ids, ccc_sparse, 'png', 100, fnum,
                0, 0, config)


def create_family_html(rtable, ftable, rtimes, rtimes_mpl, windowAmps, fi,
                                                      external_catalogs, config):
    """
    Creates the HTML for the individual family pages.

    HTML will hold navigation, images, and basic statistics. May also include
    location information if an external catalog is queried.

    rtable : Table object
        Handle to the Repeaters table.
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
    external_catalogs : list of DataFrame objects
        External catalogs to check against, with 'Arrivals_' columns.
    config : Config object
        Describes the run parameters.

    """

    # Load into memory
    if config.get('checkcomcat'):
        windowAmp = windowAmps[:,config.get('printsta')]

    printme = ftable.cols.printme[:]
    lastprint = ftable.cols.lastprint[:]

    for fnum in range(ftable.attrs.nClust):

        if printme[fnum] != 0 or lastprint[fnum] != fnum:

            with open(os.path.join(config.get('output_folder'), 'clusters',
                                   f'{fnum}.html'), 'w') as f:

                write_html_header(f, ftable, fnum, rtimes, rtimes_mpl, fi, config)

                if config.get('checkcomcat'):
                    redpy.catalog.match_external(
                        windowAmp, ftable, fnum, f, rtimes,
                        external_catalogs, config)

                f.write('</center></body></html>')


def create_report(rtable, ftable, rtimes, rtimes_mpl, windowAmps, fi, ids,
    ccc_sparse, fnum, ordered, skip_recalculate_ccc, matrixtofile, config):
    """
    Creates more detailed output plots for a single family in '/reports'.

    rtable : Table object
        Handle to the Repeaters table.
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
    fnum : int
        Family to be inspected.
    ordered : int
        1 if members should be ordered by OPTICS, 0 if by time.
    skip_recalculate_ccc : int
        1 if user wishes to skip recalculating the full correlation matrix.
    matrixtofile : int
        1 if correlation should be written to file.
    config : Config object
        Describes the run parameters.

    """
    if config.get('verbose'): print(f'Creating report for family {fnum}...')
    rpath = os.path.join(config.get('output_folder'), 'reports')

    # Derive family-specific variables
    corenum = ftable[fnum]['core']
    fam = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')
    fam = fam[np.argsort(rtimes_mpl[fam])]
    catalog = rtimes_mpl[fam]
    spacing = np.diff(catalog)*24
    ccc_fam = redpy.correlation.subset_matrix(ids[fam], ccc_sparse, config,
                                                         return_type='matrix')

    # Copy static preview image in case cluster changes
    shutil.copy(os.path.join(config.get('output_folder'), 'clusters', f'{fnum}.png'),
                os.path.join(rpath, f'{fnum}-report.png'))

    if not skip_recalculate_ccc:
        # Warn the user about the very large matrix
        if len(fam) > 1000:
            print('There are a lot of members in this family!' + \
                  ' Consider using the -s option to skip recalculating' + \
                  ' the cross-correlation matrix...' )
        # Fill in full correlation matrix
        print('Computing full correlation matrix; this will take time' + \
              ' if the family is large')
        ccc_full = redpy.correlation.make_full(rtable[fam], ccc_fam, config)
    else:
        ccc_full = ccc_fam.copy()

    # Make Bokeh plots
    plotformat = 'amplitude,spacing,correlation' # Pass this as an option?
    core_idx = np.where(fam==corenum)[0]
    assemble_bokeh_timeline_report(rtimes, rtimes_mpl, fam, core_idx,
        plotformat, fnum, config, windowAmps=windowAmps, ccc_full=ccc_full)

    # optional OPTICS ordering
    if ordered:
        # Order by OPTICS rather than by time
        D = 1-ccc_full
        s = np.argsort(sum(D))[::-1]
        D = D[s,:]
        D = D[:,s]
        fam = fam[s]
        ccc_fam = ccc_fam[s,:]
        ccc_fam = ccc_fam[:,s]
        ccc_full = ccc_full[s,:]
        ccc_full = ccc_full[:,s]
        ttree = setOfObjects(D)
        prep_optics(ttree,1)
        build_optics(ttree,1)
        order = np.array(ttree._ordered_list)
        fam = fam[order]
        ccc_fam = ccc_fam[order,:]
        ccc_fam = ccc_fam[:,order]
        ccc_full = ccc_full[order,:]
        ccc_full = ccc_full[:,order]

    # optional save to file
    if matrixtofile:
        np.save(os.path.join(rpath,f'{fnum}-ccc_full.npy'), ccc_full)
        np.save(os.path.join(rpath,f'{fnum}-evTimes.npy'), rtimes[fam])

    # Plot correlation matrix
    correlation_matrix_plot(ccc_fam, ccc_full, rtimes_mpl, fam, ordered,
                            skip_recalculate_ccc, os.path.join(rpath,
                            f'{fnum}-reportcmat.png'), config)

    # Waveform images
    wiggle_plot_all(rtable, rtimes_mpl, fam, ordered, os.path.join(rpath,
                    f'{fnum}-reportwaves.png'), config)

    # HTML page
    with open(os.path.join(rpath, f'{fnum}-report.html'), 'w') as f:

        write_html_header(f, ftable, fnum, rtimes, rtimes_mpl, fi, config,
                          report=True)
        f.write('</center></body></html>')


def assemble_bokeh_timeline(ftable, rtimes, rtimes_mpl, fi, longevity,
        famstarts, ttimes, barpad, plotformat, binsize_hist, binsize_occur,
        mintime, maxtime, minplot, fixedheight, filepath, htmltitle, divtitle,
        config):
    """
    Assembles an interactive HTML timeline with given parameters using Bokeh.

    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    rtimes : datetime ndarray
        Times of repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of repeaters as matplotlib dates.
    fi : float ndarray
        Frequency index values for repeaters.
    longevity : float ndarray
        Longevity values for all families.
    famstarts : float ndarray
        Start times of all families as matplotlib dates.
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    barpad : float
        Horizontal padding for hover and arrows (usually ~1% of window range).
    plotformat : str
        Formatted list of plots to be rendered, separated by ',' or '+' where
        ',' denotes a new row and '+' groups the plots in tabs.
    binsize_hist : float
        Width (in days) of time bins for rate plot histogram.
    binsize_occur : float
        Width (in days) of time bins for occurrence plot histogram.
    mintime : float
        Minimum time to plot as matplotlib date; generates arrows if a family
        extends earlier in time.
    maxtime : float
        Maximum time to plot as matplotlib date; generates arrows if a family
        extends later in time.
    minplot : int
        Minimum number of members in a family to include on occurence plot.
    fixedheight : bool
        True if the occurrence plot height should be the same height as the
        other plots, or False for variable in height.
    filepath : str
        Output file location and name.
    htmltitle : str
        Title of html page.
    divtitle : str
        Title used in div container at top left of plots.
    config : Config object
        Describes the run parameters.

    """

    plot_types = plotformat.replace('+',',').split(',')
    plots = []
    tabtitles = []

    # Create each of the subplots specified in the configuration file
    for p in plot_types:

        if p == 'eqrate':
            # Plot EQ Rates (Repeaters and Orphans)
            plots.append(subplot_rate(ttimes, rtimes_mpl, binsize_hist,
                                                       mintime, maxtime, config))
            tabtitles = tabtitles+['Event Rate']

        elif p == 'fi':
            # Plot Frequency Index
            plots.append(subplot_fi(ttimes, rtimes, rtimes_mpl, fi, mintime,
                                                                maxtime, config))
            tabtitles = tabtitles+['FI']

        elif p == 'longevity':
            # Plot Cluster Longevity
            plots.append(subplot_longevity(ttimes, famstarts, longevity,
                                               mintime, maxtime, barpad, config))
            tabtitles = tabtitles+['Longevity']

        elif p == 'occurrence':
            # Plot family occurrence
            plots.append(subplot_occurrence(ttimes, rtimes_mpl, famstarts,
                             longevity, fi, ftable, mintime, maxtime, minplot,
                             binsize_occur, barpad, 'rate', fixedheight, config))
            tabtitles = tabtitles+['Occurrence (Color by Rate)']

        elif p == 'occurrencefi':
            # Plot family occurrence with color by FI
            plots.append(subplot_occurrence(ttimes, rtimes_mpl, famstarts,
                             longevity, fi, ftable, mintime, maxtime, minplot,
                             binsize_occur, barpad, 'fi', fixedheight, config))
            tabtitles = tabtitles+['Occurrence (Color by FI)']

        else:
            print(f'{p} is not a valid plot type. Moving on.')

    # Set ranges
    for p in plots:
        p.x_range = plots[0].x_range

    # Add annotations
    for p in plots:
        p = add_bokeh_annotations(p, config)

    # Create output and save
    gridplot_items = [[Div(text=divtitle, width=1000, margin=(-40,5,-10,5))]]
    pnum = 0
    for pf in plotformat.split(','):
        # '+' groups plots into tabs
        if '+' in pf:
            tabs = []
            for pft in range(len(pf.split('+'))):
                tabs = tabs + [Panel(child=plots[pnum+pft],
                                     title=tabtitles[pnum+pft])]
            gridplot_items = gridplot_items + [[Tabs(tabs=tabs)]]
            pnum = pnum+pft+1
        else:
            gridplot_items = gridplot_items + [[plots[pnum]]]
            pnum = pnum+1

    o = gridplot(gridplot_items)
    output_file(filepath,title=htmltitle)
    save(o)


def assemble_bokeh_timeline_report(rtimes, rtimes_mpl, fam, core_idx,
    plotformat, fnum, config, windowAmps=None, ccc_full=None):
    """
    Assembles an interactive HTML timeline with Bokeh for a family report.

    Parameters
    ----------
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    fam : int ndarray
        Indices of family members within the Repeaters table ordered by time.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    plotformat : str
        Formatted list of plots to be rendered, separated by ',' or '+' where
        ',' denotes a new row and '+' groups the plots in tabs.
    fnum : int
        Family to be inspected.
    config : Config object
        Describes the run parameters.
    windowAmps : float ndarray, optional
        'windowAmp' column from rtable for all stations.
    ccc_full : float ndarray, optional
        Filled correlation matrix.

    """

    plots = []
    tabtitles = []

    plot_types = plotformat.split(',')

    # Create each of the subplots specified in the configuration file
    for p in plot_types:

        if p == 'amplitude':
            # Plot EQ Rates (Repeaters and Orphans)
            plots.append(subplot_amplitude(rtimes, rtimes_mpl, windowAmps,
                                           fam, core_idx, config, useBokeh=True))
            tabtitles = tabtitles+['Amplitude']

        elif p == 'spacing':
            # Plot Frequency Index
            plots.append(subplot_spacing(rtimes, rtimes_mpl, fam, core_idx,
                                                          config, useBokeh=True))
            tabtitles = tabtitles+['Spacing']

        elif p == 'correlation':
            # Plot Cluster Longevity
            plots.append(subplot_correlation(rtimes, rtimes_mpl, fam, [], [],
                             core_idx, config, ccc_full=ccc_full, useBokeh=True))
            tabtitles = tabtitles+['Correlation']

        else:
            print(f'{p} is not a valid plot type. Moving on.')

    # Set ranges
    for p in plots:
        p.x_range = plots[0].x_range

    # Add annotations
    for p in plots:
        p = add_bokeh_annotations(p, config)

    # Create output and save
    gridplot_items = []
    pnum = 0
    for pf in plotformat.split(','):
        # '+' groups plots into tabs
        if '+' in pf:
            tabs = []
            for pft in range(len(pf.split('+'))):
                tabs = tabs + [Panel(child=plots[pnum+pft],
                                     title=tabtitles[pnum+pft])]
            gridplot_items = gridplot_items + [[Tabs(tabs=tabs)]]
            pnum = pnum+pft+1
        else:
            gridplot_items = gridplot_items + [[plots[pnum]]]
            pnum = pnum+1

    o = gridplot(gridplot_items)

    filepath = os.path.join(config.get('output_folder'), 'reports',
                            f'{fnum}-report-bokeh.html')
    output_file(filepath,
                title=f'{config.get("title")} - Cluster {fnum} Detailed Report')
    save(o)


def assemble_family_image(bboxes, rtable, ftable, rtimes, rtimes_mpl,
    windowAmp, ids, ccc_sparse, oformat, dpi, fnum, tmin, tmax, config):
    """
    Creates a multi-paneled family plot for the specified family 'fnum'.

    This function allows some flexibility in the output format
    (e.g., .png, .pdf) as well as resolution. Many inputs are columns from
    rtable to reduce overhead back to the file when calling this function for
    many families.

    Current format for the image is the following:
        Top row: Waveforms, stacked FFT.
        Second row: Timeline of amplitude.
        Third row: Timeline of event spacing.
        Last row: Correlation with time relative to best-correlated event
            (has most measurements in Correlation table), with core event in
            black and events with missing correlation values as open
            circles (either were never correlated or were below threshold).

    Parameters
    ----------
    bboxes : list of Bbox objects
        List of bounding box positions for each axis.
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    windowAmp : float ndarray
        'windowAmp' column from rtable for single station.
    ids : int ndarray
        'id' column from Repeaters table.
    ccc_sparse : float csr_matrix
        Sparse correlation matrix with id as rows/columns.
    oformat : str
        output file format (e.g., 'png' or 'pdf')
    dpi : int
        Dots per inch resolution of raster file.
    fnum : int
        Family number to plot.
    tmin : float
        Minimum time on timeline axes as matplotlib date (0 for default tmin).
    tmax : float
        Maximum time on timeline axes as matplotlib date (0 for default tmax).
    config : Config object
        Describes the run parameters.

    """

    fig, axes, bboxes = initialize_family_image(config, bboxes=bboxes)

    # Extract family members
    fam = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')

    # Order by time
    fam = fam[np.argsort(rtimes_mpl[fam])]

    # Get core location within fam
    core_idx = np.where(fam==ftable[fnum]['core'])[0][0]


    # Plot waveforms
    subplot_waveforms(rtable[fam], core_idx, axes[0], config)

    # Plot FFTs
    subplot_fft(rtable[fam], core_idx, axes[1], config)

    # Plot amplitude timeline
    subplot_amplitude(rtimes, rtimes_mpl, windowAmp, fam, core_idx, config,
                                                                   ax=axes[2])

    # Plot spacing timeline
    subplot_spacing(rtimes, rtimes_mpl, fam, core_idx, config, ax=axes[3])

    # Plot correlation timeline
    subplot_correlation(rtimes, rtimes_mpl, fam, ids, ccc_sparse, core_idx,
                                                              config, ax=axes[4])

    # General formatting
    axes = format_family_image(axes, config)

    # Enforce time limits for timeline plots
    if tmin and tmax:
        axes[2].set_xlim(tmin, tmax)
    elif tmin:
        axes[2].set_xlim(tmin, axes[2].get_xlim()[1])
    elif tmax:
        axes[2].set_xlim(axes[2].get_xlim()[0], tmax)
    axes[3].set_xlim(axes[2].get_xlim())
    axes[4].set_xlim(axes[2].get_xlim())

    # Save
    plt.savefig(os.path.join(config.get('output_folder'), 'clusters',
                             f'fam{fnum}.{oformat}'), dpi=dpi)


def assemble_pdf_overview(rtable, ftable, ttimes, rtimes, rtimes_mpl, fi,
                tmin, tmax, binsize, minmembers, occurheight, plotformat, config):
    """
    Generate a static PDF version of the overview plot for publication.

    Plot is saved in the usual outputs folder.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    rtimes : datetime ndarray
        Times of repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of repeaters as matplotlib dates.
    fi : float ndarray
        Frequency index values for repeaters.
    tmin : float
        Minimum time on timeline axes as matplotlib date (0 for default tmin).
    tmax : float
        Maximum time on timeline axes as matplotlib date (0 for default tmax).
    binsize : float
        Width (in days) of both rate and occurrence histogram time bins.
    minmembers : int
        Minimum number of members for a family to be included in occurrence
        timeline.
    occurheight : int
        Integer multiplier for how much taller than the other timelines the
        occurrence timeline should be; determines figure aspect ratio/size.
    plotformat : str
        Formatted list of plots to be rendered, separated by ',' or '+' where
        ',' denotes a new row and '+' groups the plots in tabs.
    config : Config object
        Describes the run parameters.

    """

    # !!! Add functionality to separate binsizes
    binsize_hist = binsize
    binsize_occur = binsize

    longevity = ftable.cols.longevity[:]
    famstarts = ftable.cols.startTime[:]

    # Custom tmin/tmax functionality
    if tmin:
        mintime = tmin
    else:
        mintime = min(ttimes)

    if tmax:
        maxtime = tmax
    else:
        maxtime = max(ttimes)
    barpad = 0.01*(maxtime-mintime)

    plot_types = plotformat.replace('+',',').split(',')

    # Determine the height of the plot
    nsub = 0
    for p in plot_types:
        if 'occurrence' in p:
            nsub = nsub + occurheight
        else:
            nsub = nsub + 1
    figheight = 2*nsub+4

    # Hack a reference axis
    figref = plt.figure(figsize=(9,1))
    axref = figref.add_subplot(1,1,1)
    axref = subplot_rate(ttimes, rtimes_mpl, binsize_hist, mintime, maxtime,
        config, useBokeh=False, ax=axref)

    fig = plt.figure(figsize=(9, figheight))

    pnum = 0
    for p in plot_types:

        if p == 'eqrate':
            ### Rate ###
            ax = fig.add_subplot(nsub, 1, pnum+1, sharex=axref)
            ax = add_pdf_annotations(ax, mintime, maxtime, config)
            ax = subplot_rate(ttimes, rtimes_mpl, binsize_hist, mintime,
                maxtime, config, useBokeh=False, ax=ax)
            pnum = pnum + 1

        elif p == 'fi':
            ### FI ###
            ax = fig.add_subplot(nsub, 1, pnum+1, sharex=axref)
            ax = add_pdf_annotations(ax, mintime, maxtime, config)
            ax = subplot_fi(ttimes, rtimes, rtimes_mpl, fi, mintime, maxtime,
                                                   config, useBokeh=False, ax=ax)
            pnum = pnum + 1

        elif p == 'occurrence':
            ### Occurrence ###
            ax = fig.add_subplot(nsub, 1, (pnum+1, pnum+occurheight),
                sharex=axref)
            ax = add_pdf_annotations(ax, mintime, maxtime, config)
            ax = subplot_occurrence(ttimes, rtimes_mpl, famstarts, longevity,
                fi, ftable, mintime, maxtime, minmembers, binsize_occur,
                barpad, 'rate', True, config, useBokeh=False, ax=ax)
            add_pdf_colorbar(fig, figheight, pnum, nsub, 'rate',
                binsize_occur, config)
            pnum = pnum + occurheight

        elif p == 'occurrencefi':
            ### Occurrence ###
            ax = fig.add_subplot(nsub, 1, (pnum+1, pnum+occurheight),
                sharex=axref)
            ax = add_pdf_annotations(ax, mintime, maxtime, config)
            ax = subplot_occurrence(ttimes, rtimes_mpl, famstarts, longevity,
                fi, ftable, mintime, maxtime, minmembers, binsize_occur,
                barpad, 'fi', True, config, useBokeh=False, ax=ax)
            add_pdf_colorbar(fig, figheight, pnum, nsub, 'fi',
                binsize_occur, config)
            pnum = pnum + occurheight

        elif p == 'longevity':
            ### Longevity ###
            ax = fig.add_subplot(nsub, 1, pnum+1, sharex=axref)
            ax = add_pdf_annotations(ax, mintime, maxtime, config)
            ax = subplot_longevity(ttimes, famstarts, longevity, mintime,
                maxtime, barpad, config, useBokeh=False, ax=ax)
            pnum = pnum + 1

        else:
            print(f'{p} is not a valid plot type. Moving on.')

    # Clean up and save
    plt.tight_layout()
    plt.savefig(os.path.join(config.get('output_folder'), 'overview.pdf'))
    plt.close(fig)


def subplot_rate(ttimes, rtimes_mpl, binsize_hist, mintime, maxtime, config,
                                                      useBokeh=True, ax=None):
    """
    Creates subplot for rate of orphans and repeaters.

    Parameters
    ----------
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    rtimes_mpl : float ndarray
        Times of repeaters as matplotlib dates.
    binsize_hist : float
        Width (in days) of time bins for rate plot histogram.
    mintime : float
        Minimum time to plot as matplotlib date.
    maxtime : float
        Maximum time to plot as matplotlib date.
    config : Config object
        Describes the run parameters.
    useBokeh : bool, optional
        True if to render Bokeh figure, or False for matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    fig : Figure object
        If useBokeh is True, returns a bokeh Figure.
    ax : Axis object
        If useBokeh is False, returns a matplotlib Axis handle.

    """

    dt_offset = binsize_hist/2 # used to create the lines

    # Create histogram of events with time
    hist_trigs, bin_times = np.histogram(ttimes, bins=np.arange(mintime,
        maxtime+binsize_hist, binsize_hist))
    hist_repeaters, bin_times = np.histogram(rtimes_mpl, bins=np.arange(mintime,
        maxtime+binsize_hist, binsize_hist))

    if config.get('timeline_vs') == 'triggers':
        trigorph = 'Total Triggers'
        hist_trigorph = hist_trigs
    else:
        trigorph = 'Orphans'
        hist_trigorph = hist_trigs-hist_repeaters

    # Determine title
    hr_days = 'Day Bin' if binsize_hist>=1 else 'Hour Bin'
    bin_text = binsize_hist if binsize_hist >= 1 else binsize_hist*24
    title = f'Repeaters vs. {trigorph} by {bin_text:.1f} {hr_days}'

    if useBokeh:
        fig = bokeh_figure(title=title)
        fig.yaxis.axis_label = 'Events'
        fig.line(mdates.num2date(bin_times[0:-1]+dt_offset), hist_trigorph,
                color='black', legend_label=trigorph)
        fig.line(mdates.num2date(bin_times[0:-1]+dt_offset), hist_repeaters,
            color='red', legend_label='Repeaters', line_width=2)
        fig.legend.location = 'top_left'

        return fig

    else:
        ax.plot(mdates.num2date(bin_times[0:-1]+dt_offset), hist_trigorph,
            color='black', label=trigorph, lw=0.5)
        ax.plot(mdates.num2date(bin_times[0:-1]+dt_offset), hist_repeaters,
            color='red', label='Repeaters', lw=2)
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_ylabel('Events', style='italic')
        ax.set_xlabel('Date', style='italic')
        ax.legend(loc='upper left', frameon=False)

        return ax


def subplot_fi(ttimes, rtimes, rtimes_mpl, fi, mintime, maxtime, config,
                                                      useBokeh=True, ax=None):
    """
    Creates subplot for frequency index scatterplot.

    Parameters
    ----------
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    rtimes : datetime ndarray
        Times of repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of repeaters as matplotlib dates.
    fi : float ndarray
        Frequency index values for repeaters.
    mintime : float
        Minimum time to plot as matplotlib date.
    maxtime : float
        Maximum time to plot as matplotlib date.
    config : Config object
        Describes the run parameters.
    useBokeh : bool, optional
        True if to render Bokeh figure, or False for matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    fig : Figure object
        If useBokeh is True, returns a bokeh Figure.
    ax : Axis object
        If useBokeh is False, returns a matplotlib Axis handle.

    """

    if useBokeh:
        fig = bokeh_figure(title='Frequency Index')
        fig.yaxis.axis_label = 'FI'
        # Always plot at least one invisible point
        fig.circle(mdates.num2date(maxtime), 0,
            line_alpha=0, fill_alpha=0)
    else:
        ax.set_title('Frequency Index', loc='left', fontweight='bold')
        ax.set_ylabel('FI', style='italic')
        ax.set_xlabel('Date', style='italic')

    idxs = np.where((rtimes_mpl>=mintime) & (rtimes_mpl<=maxtime))[0]

    if useBokeh:
        fig.circle(rtimes[idxs], fi[idxs], color='red', line_alpha=0,
            size=3, fill_alpha=0.5)
        return fig
    else:
        ax.scatter(rtimes[idxs], fi[idxs], 2, c='red', alpha=0.25)
        # Need to call get_ylim() or y-limits sometimes freak out
        axlims = ax.get_ylim()
        return ax


def subplot_longevity(ttimes, famstarts, longevity, mintime, maxtime,
                                         barpad, config, useBokeh=True, ax=None):
    """
    Creates subplot for longevity.

    Parameters
    ----------
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    famstarts : float ndarray
        Start times of all families as matplotlib dates.
    longevity : float ndarray
        Longevity values for all families.
    mintime : float
        Minimum time to plot as matplotlib date.
    maxtime : float
        Maximum time to plot as matplotlib date.
    barpad : float
        Horizontal padding for hover and arrows (usually ~1% of window range).
    config : Config object
        Describes the run parameters.
    useBokeh : bool, optional
        True if to render Bokeh figure, or False for matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    fig : Figure object
        If useBokeh is True, returns a bokeh Figure.
    ax : Axis object
        If useBokeh is False, returns a matplotlib Axis handle.

    """

    if useBokeh:
        fig = bokeh_figure(y_axis_type='log',
            y_range=[0.01, np.sort(ttimes)[-1]-np.sort(ttimes)[0]],
            title='Cluster Longevity')
        fig.yaxis.axis_label = 'Days'
        # Always plot at least one invisible point
        fig.circle(mdates.num2date(maxtime), 1,
            line_alpha=0, fill_alpha=0)
    else:
        ax.set_title('Longevity', loc='left', fontweight='bold')
        ax.set_ylabel('Days', style='italic')
        ax.set_xlabel('Date', style='italic')

    # Plot Data
    for n in range(len(famstarts)):

        add_line, add_larrow, add_rarrow, x1, x2 = determine_lines(mintime,
            maxtime, barpad, famstarts[n], longevity[n])

        # Draw a line for the longevity data (turns off if data don't fall
        # within time window) and draw an arrow if longevity line extends
        # beyond the data window
        if add_line:

            if useBokeh:
                source = ColumnDataSource(dict(
                    x=np.array(
                    (mdates.num2date(x1),
                    mdates.num2date(x2))),
                    y=np.array((longevity[n], longevity[n]))))
                fig.add_glyph(source, Line(x='x', y='y', line_color='red',
                    line_alpha=0.5))
            else:
                ax.semilogy(np.array((mdates.num2date(x1),
                    mdates.num2date(x2))),
                    np.array((longevity[n], longevity[n])), 'red', alpha=0.75,
                                                                       lw=0.5)

            if add_larrow:
                if useBokeh:
                    fig.add_layout(Arrow(end=VeeHead(size=5, fill_color='red',
                        line_color='red', line_alpha=0.5), line_alpha=0,
                        x_start=mdates.num2date(famstarts[n] + \
                        longevity[n]), x_end=mdates.num2date(
                        mintime - barpad), y_start=longevity[n],
                        y_end=longevity[n]))
                else:
                    ax.annotate('', xy=(mdates.num2date(
                        mintime - barpad), longevity[n]), xytext=(
                        mdates.num2date(mintime - 2*barpad),
                        longevity[n]), arrowprops=dict(arrowstyle='<-',
                        color='red', alpha=0.75))

            if add_rarrow:
                ax.annotate('', xy=(mdates.num2date(maxtime + \
                    barpad), longevity[n]), xytext=(mdates.num2date(
                    maxtime + 2*barpad), longevity[n]), arrowprops=dict(
                    arrowstyle='<-', color='red', alpha=0.75))

    if useBokeh:
        return fig
    else:
        return ax


def subplot_occurrence(ttimes, rtimes_mpl, famstarts, longevity, fi, ftable,
    mintime, maxtime, minplot, binsize_occur, barpad, colorby, fixedheight,
                                                 config, useBokeh=True, ax=None):
    """
    Creates subplot for family occurrence.

    Parameters
    ----------
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    rtimes_mpl : float ndarray
        Times of repeaters as matplotlib dates.
    famstarts : float ndarray
        Start times of all families as matplotlib dates.
    longevity : float ndarray
        Longevity values for all families.
    fi : float ndarray
        Frequency index values for repeaters.
    ftable : Table object
        Handle to the Families table.
    mintime : float
        Minimum time to plot as matplotlib date; families starting before this
        time will not be plotted if they also end before this time, and will
        have left arrows if they end after it.
    maxtime : float
        Maximum time to plot as matplotlib date.
    minplot : int
        Minimum number of members in a family to include.
    binsize_occur : float
        Width (in days) of time bins for occurrence plot histogram.
    barpad : float
        Horizontal padding for hover and arrows (usually ~1% of window range).
    colorby : str
        Determines color in histograms: 'rate' (YlOrRd) or 'fi' (coolwarm)
    fixedheight : bool
        True if the occurrence plot height should be the same height as the
        other plots, or False for variable in height.
    config : Config object
        Describes the run parameters.
    useBokeh : bool, optional
        True if to render Bokeh figure, or False for matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    fig : Figure object
        If useBokeh is True, returns a bokeh Figure.
    ax : Axis object
        If useBokeh is False, returns a matplotlib Axis handle.

    """

    if useBokeh:
        fig = bokeh_figure(tools=[family_hover_tool(),
            'pan,box_zoom,reset,save,tap'], title='Occurrence Timeline',
            plot_height=250, plot_width=1250)
        fig.yaxis.axis_label = 'Cluster by Date' + (
            f' ({minplot}+ Members)' if minplot>0 else '')
        # Always plot at least one invisible point
        fig.circle(mdates.num2date(maxtime), 0,
            line_alpha=0, fill_alpha=0)
    else:
        ax.set_title('Occurrence Timeline', loc='left', fontweight='bold')
        ax.set_ylabel('Cluster by Date' + (
            f' ({minplot}+ Members)' if minplot>2 else ''),
            style='italic')
        ax.set_xlabel('Date', style='italic')

    # Steal colormap (len=256) from matplotlib
    if colorby is 'rate':
        colormap = matplotlib.cm.get_cmap('YlOrRd')
    elif colorby is 'fi':
        colormap = matplotlib.cm.get_cmap('coolwarm')
    else:
        print('Unrecognized colorby choice, defaulting to rate')
        colorby = 'rate'
        colormap = matplotlib.cm.get_cmap('YlOrRd')
    bokehpalette = [matplotlib.colors.rgb2hex(m) for m in colormap(
        np.arange(colormap.N)[::-1])]

    n = 0
    xs = []
    ys = []
    famnum = []
    for clustNum in range(ftable.attrs.nClust):

        members = np.fromstring(ftable[clustNum]['members'], dtype=int,
            sep=' ')

        if len(rtimes_mpl[members]) >= minplot:

            if max(rtimes_mpl[members])>mintime:

                # Create histogram of events/hour
                hist, h = np.histogram(rtimes_mpl[members],
                                    bins=np.arange(min(rtimes_mpl[members]),
                                    max(rtimes_mpl[members]+binsize_occur),
                                    binsize_occur))
                if useBokeh:
                    d1 = mdates.num2date(h[np.where(hist>0)])
                    d2 = mdates.num2date(h[np.where(hist>0)]+binsize_occur)
                else:
                    d1 = h[np.where(hist>0)]

                if colorby is 'rate':
                    histlog = np.log10(hist[hist>0])
                    if binsize_occur >= 1:
                        ind = [int(min(255,255*(i/3))) for i in histlog]
                    else:
                        ind = [int(min(255,255*(i/2))) for i in histlog]
                elif colorby is 'fi':
                    h = h[np.where(hist>0)]
                    hist = hist[hist>0]
                    fisum = np.zeros(len(hist))
                    # Loop through bins to get summed fi
                    for i in range(len(hist)):
                        # Find indicies of rtimes_mpl[members] within bins
                        idxs = np.where(np.logical_and(
                                  rtimes_mpl[members] >= h[i],
                                  rtimes_mpl[members] < h[i] + binsize_occur))
                        # Sum fi for those events
                        fisum[i] = np.sum(fi[members[idxs]])
                    # Convert to mean fi
                    histfi = fisum/hist
                    ind = [int(max(min(255,255*(i-config.get('fispanlow'))/(
                        config.get('fispanhigh') - config.get('fispanlow'))), 0)) for i in histfi]

                colors = [bokehpalette[i] for i in ind]

                add_line, add_larrow, add_rarrow, x1, x2 = determine_lines(
                    mintime, maxtime, barpad, famstarts[clustNum],
                    longevity[clustNum])

                if add_line:
                    if useBokeh:
                        source = ColumnDataSource(dict(
                            x=np.array(
                            (mdates.num2date(x1),
                            mdates.num2date(x2))),
                            y=np.array((n,n))))
                        fig.add_glyph(source, Line(x='x', y='y',
                            line_color='black'))
                    else:
                        ax.plot(np.array((mdates.num2date(x1),
                            mdates.num2date(x2))),
                            np.array((n,n)), 'black', lw=0.5, zorder=0)

                    if add_larrow:
                        if useBokeh:
                            fig.add_layout(Arrow(end=VeeHead(size=5,
                                fill_color='black', line_color='black'),
                                line_alpha=0,
                                x_start=mdates.num2date(
                                famstarts[clustNum]+longevity[clustNum]),
                                x_end=mdates.num2date(mintime - \
                                barpad), y_start=n, y_end=n))
                        else:
                            ax.annotate('', xy=(mdates.num2date(
                                mintime-barpad),n),xytext=(
                                mdates.num2date(
                                mintime-2*barpad),n), arrowprops=dict(
                                arrowstyle='<-', color='black', alpha=1))

                    if add_rarrow:
                        ax.annotate('', xy=(mdates.num2date(
                            maxtime + barpad), n), xytext=(
                            mdates.num2date(maxtime+2*barpad),
                            n), arrowprops=dict(arrowstyle='<-',
                            color='black', alpha=1))

                # Add boxes
                if useBokeh:
                    idx = np.where(h[np.where(hist>0)[0]]>mintime)[0]
                    fig.quad(top=n+0.3, bottom=n-0.3,
                        left=np.array(d1)[idx],
                        right=np.array(d2)[idx],
                        color=np.array(colors)[idx])
                else:
                    # Potentially slow if many patches
                    for i in range(len(d1)):
                        x = mdates.num2date(np.array(d1)[i])
                        w = datetime.timedelta(binsize_occur)
                        if (x >= mdates.num2date(mintime)) and (
                            x <= mdates.num2date(maxtime)):
                            ax.add_patch(matplotlib.patches.Rectangle((x,
                                n - 0.3), w,0.6, facecolor=colors[i],
                                edgecolor=None, fill=True))
                            if x+w > mdates.num2date(x2):
                                x2 = np.array(d1)[i] + binsize_occur

                # Add label
                if useBokeh:
                    label = Label(x=max(d2), y=n,
                        text=f'  {len(rtimes_mpl[members])}',
                        text_font_size='9pt', text_baseline='middle')
                    fig.add_layout(label)
                else:
                    ax.annotate(f'  {len(rtimes_mpl[members])}',
                        (mdates.num2date(x2),n), va='center',
                        ha='left')

                if useBokeh:
                    # Build source for hover patches
                    fnum = clustNum
                    xs.append([mdates.num2date(max(min(
                                   rtimes_mpl[members]), mintime) - barpad),
                               mdates.num2date(max(min(
                                   rtimes_mpl[members]), mintime) - barpad),
                               mdates.num2date(
                                   max(rtimes_mpl[members]) + barpad),
                               mdates.num2date(max(
                                   rtimes_mpl[members]) + barpad)])
                    ys.append([n-0.5, n+0.5, n+0.5, n-0.5])
                    famnum.append([fnum])

                n = n+1

    if useBokeh:
        cloc1 = 85

        if (n > 0):
            # Patches allow hovering for image of core and cluster number
            source = ColumnDataSource(data=dict(xs=xs, ys=ys, famnum=famnum))
            fig.patches('xs', 'ys', source=source, name='patch', alpha=0,
                selection_fill_alpha=0, selection_line_alpha=0,
                nonselection_fill_alpha=0, nonselection_line_alpha=0)

            # Tapping on one of the patches will open a window to a file with
            # more information on the cluster in question.
            url = './clusters/@famnum.html'
            renderer = fig.select(name='patch')
            taptool = fig.select(type=TapTool)[0]
            taptool.names.append('patch')
            taptool.callback = OpenURL(url=url)

            if (n > 15) and not fixedheight:
                fig.plot_height = n*15
                fig.y_range = Range1d(-1, n)
                cloc1 = n*15-165

        if colorby is 'rate':
                color_bar = ColorBar(color_mapper=determine_color_mapper(
                    binsize_occur), ticker=LogTicker(),
                    border_line_color='#eeeeee', location=(7,cloc1),
                    orientation='horizontal', width=150, height=15,
                    title='Events per {}'.format(
                    determine_legend_text(binsize_occur)), padding=15,
                    major_tick_line_alpha=0,
                    formatter=LogTickFormatter(min_exponent=4))
        elif colorby is 'fi':
                color_bar = ColorBar(color_mapper=determine_color_mapper_fi(
                    config), border_line_color='#eeeeee', location=(7,cloc1),
                    orientation='horizontal', width=150, height=15,
                    title='Mean Frequency Index', padding=15,
                    major_tick_line_alpha=0)

        fig.add_layout(color_bar)

        return fig

    else:
        return ax


def subplot_amplitude(rtimes, rtimes_mpl, windowAmp, fam, core_idx, config,
                                                     useBokeh=False, ax=None):
    """
    Fills the amplitude timeline subplot.

    Parameters
    ----------
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    windowAmp : float ndarray
        'windowAmp' column from rtable for all stations or a single station.
    fam : int ndarray
        Indices of family members within the Repeaters table ordered by time.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    config : Config object
        Describes the run parameters.
    useBokeh : bool, optional
        True if to render Bokeh figure, or False for matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    fig : Figure object
        If useBokeh is True, returns a bokeh Figure.
    ax : Axis object
        If useBokeh is False, returns a matplotlib Axis handle.

    """

    # Determine ylims
    if config.get('amplims') == 'family':
        windowAmpFam = windowAmp[fam][:]
        ymin = 0.25*np.amin(windowAmpFam[np.nonzero(windowAmpFam)])
        ymax = 4*np.amax(windowAmpFam)
    else:
        # Use global maximum
        ymin = 0.25*np.amin(windowAmp[np.nonzero(windowAmp)])
        ymax = 4*np.amax(windowAmp)

    # Prevent ymin being "too small" to be realistic
    if ymax > 1000:
        ymin = np.max((ymin, 1))
    elif ymax > 1:
        ymin = np.max((ymin, 1e-3))
    elif ymax > 1e-6:
        ymin = np.max((ymin, 1e-12))

    # Plot amplitude timeline
    if useBokeh:

        fig = bokeh_figure(title='Amplitude with Time (Click name to hide)',
            y_axis_type='log', y_range=[ymin,ymax])
        fig.yaxis.axis_label = 'Counts'

        if config.get('nsta') <= 8:
            palette = all_palettes['YlOrRd'][9]
        else:
            palette = inferno(config.get('nsta')+1)
        for sta, staname in enumerate(config.get('station').split(',')):
            fig.circle(rtimes[fam], windowAmp[fam][:,sta],
                color=palette[sta], line_alpha=0, size=4, fill_alpha=0.5,
                legend_label='{}.{}'.format(
                    staname,config.get('channel').split(',')[sta]))
        fig.legend.location='bottom_left'
        fig.legend.orientation='horizontal'
        fig.legend.click_policy='hide'

        return fig

    else:

        ax.plot_date(rtimes_mpl[fam], windowAmp[fam],
                'ro', alpha=0.5, markeredgecolor='r', markeredgewidth=0.5,
                markersize=3)
        ax.plot_date(rtimes_mpl[fam[core_idx]], windowAmp[fam[core_idx]],
                'ko', markeredgecolor='k', markeredgewidth=0.5,
                markersize=3)
        ax.set_ylim(ymin, ymax)

        return ax


def subplot_spacing(rtimes, rtimes_mpl, fam, core_idx, config, useBokeh=False,
                                                                     ax=None):
    """
    Fills the temporal spacing timeline subplot.

    Parameters
    ----------
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    fam : int ndarray
        Indices of family members within the Repeaters table ordered by time.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    config : Config object
        Describes the run parameters.
    useBokeh : bool, optional
        True if to render Bokeh figure, or False for matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    fig : Figure object
        If useBokeh is True, returns a bokeh Figure.
    ax : Axis object
        If useBokeh is False, returns a matplotlib Axis handle.

    """

    # Spacing between events in hours
    catalog = rtimes_mpl[fam]
    spacing = np.diff(catalog)*24

    if useBokeh:

        fig = bokeh_figure(title='Time since Previous Event',
            y_axis_type='log', y_range=[1e-3, 2*np.max(spacing)])
        fig.yaxis.axis_label = 'Interval (hr)'
        fig.circle(rtimes[fam[1:]], spacing, color='red', line_alpha=0,
            size=4, fill_alpha=0.5)

        return fig

    else:
        ax.plot_date(catalog[1:], spacing, 'ro', alpha=0.5,
            markeredgecolor='r', markeredgewidth=0.5, markersize=3)
        if core_idx > 0:
            ax.plot_date(catalog[core_idx], spacing[core_idx-1], 'ko',
                markeredgecolor='k', markeredgewidth=0.5, markersize=3)
        ax.set_ylim(1e-3, np.max(spacing)*2)

        return ax


def subplot_correlation(rtimes, rtimes_mpl, fam, ids, ccc_sparse, core_idx,
                                 config, useBokeh=False, ax=None, ccc_full=None):
    """
    Fills the cross-correlation timeline subplot.

    If using matplotlib: Plots a single row from the full cross-correlation
    matrix corresponding to whichever row has the highest sum. If the value
    is stored in the Correlation table it will be plotted as a filled red
    circle, otherwise it will be plotted with an open red circle at the value
    of config.get('cmin'). The core is denoted with a black symbol.

    If using Bokeh: Plots the row from the full cross-correlation matrix that
    corresponds to the current core. All are plotted as filled red circles.

    Parameters
    ----------
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    fam : int ndarray
        Indices of family members within the Repeaters table ordered by time.
    ids : int ndarray
        'id' column from Repeaters table.
    ccc_sparse : float csr_matrix
        Sparse correlation matrix with id as rows/columns.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    config : Config object
        Describes the run parameters.
    useBokeh : bool, optional
        True if to render Bokeh figure, or False for matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.
    ccc_full : float ndarray, optional
        Filled correlation matrix.

    Returns
    -------
    fig : Figure object
        If useBokeh is True, returns a bokeh Figure.
    ax : Axis object
        If useBokeh is False, returns a matplotlib Axis handle.

    """

    if useBokeh:

        fig = bokeh_figure(y_range=[-0.02, 1.02],
            title='Cross-Correlation Coefficient with Core Event')
        fig.yaxis.axis_label = 'CCC'
        fig.circle(rtimes[fam], ccc_full[core_idx,:].tolist()[0], color='red',
            line_alpha=0, size=4, fill_alpha=0.5)

        return fig

    else:

        # Get correlation values for row with highest sum
        Cprint = redpy.correlation.subset_matrix(ids[fam], ccc_sparse, config)

        # Create mask to only plot each point once
        cmask = np.zeros(len(Cprint), dtype=bool)
        cmask[Cprint>=config.get('cmin')] = True

        catalog = rtimes_mpl[fam]

        # Plot closed circles for values that exist, open where undefined
        ax.plot_date(catalog[cmask], Cprint[cmask], 'ro',
            alpha=0.5, markeredgecolor='r', markeredgewidth=0.5, markersize=3)
        ax.plot_date(catalog[~cmask], 0*Cprint[~cmask]+config.get('cmin'), 'wo',
            alpha=0.5, markeredgecolor='r', markeredgewidth=0.5)

        # Plot black dot for core event
        if Cprint[core_idx] >= config.get('cmin'):
            ax.plot_date(catalog[core_idx], Cprint[core_idx], 'ko',
                markeredgecolor='k', markeredgewidth=0.5, markersize=3)
        else:
            ax.plot_date(catalog[core_idx], config.get('cmin'), 'wo',
                markeredgecolor='k', markeredgewidth=0.5, markersize=3)

        return ax


def subplot_waveforms(rtable_fam, core_idx, ax, config, plot_single=False, sta=0):
    """
    Fills the waveform subplot.

    Has different behaviors based on whether a single station or many stations
    are part of the run. If a single station, will show all waveforms either
    as wiggle plots (<12 events) or an image. If multiple stations are queried
    then the core will be plotted in black and the stack of all other events
    in red as wiggles.

    Parameters
    ----------
    rtable_fam : Table object
        Handle to a subset of the Repeaters table containing all members of
        a family.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    ax : Axis object
        Subplot axis to modify in place.
    config : Config object
        Describes the run parameters.
    plot_single : bool, optional
        Plots all events from a single station, False by default.
    sta : int, optional
        Station index to plot when using plot_single option, 0 by default.

    Returns
    -------
    ax : Axis object

    """

    # Define time vector
    time_vector = np.arange(-0.5*config.get('winlen')/config.get('samprate'),
                      1.5*config.get('winlen')/config.get('samprate'), 1/config.get('samprate'))

    # If only one station, plot all aligned waveforms
    if config.get('nsta')==1 or plot_single:

        # Prepare data in matrix (row for each event)
        data = np.zeros((len(rtable_fam), int(config.get('winlen')*2)))
        for n, r in enumerate(rtable_fam):
            data[n, :] = prep_wiggle(r['waveform'], sta, r['windowStart'],
                                     r['windowAmp'][sta], config)

        # Plot
        if len(rtable_fam) > 12:
            ax.imshow(data, aspect='auto', vmin=-1, vmax=1, cmap='RdBu',
                interpolation='nearest', extent=[np.min(time_vector),
                np.max(time_vector), n+0.5, -0.5])
        else:
            for n in range(len(rtable_fam)):
                ax.plot(time_vector,data[n,:]/2+n,'k',linewidth=0.25)
            ax.set_xlim([np.min(time_vector),np.max(time_vector)])
            ax.set_ylim([-0.5, 0.5+n])

    # Otherwise, plot cores and stacks from all stations
    else:

        for s in range(config.get('nsta')):

            # Prepare stack
            data_stack = np.zeros((int(config.get('winlen')*2),))
            for r in rtable_fam:
                data_stack += prep_wiggle(r['waveform'], s, r['windowStart'],
                                          r['windowAmp'][s], config)
            data_stack = data_stack/(np.max(np.abs(data_stack))+1e-12)
            data_stack[data_stack>1] = 1
            data_stack[data_stack<-1] = -1

            # Prepare core
            data_core = prep_wiggle(rtable_fam['waveform'][core_idx], s,
                                    rtable_fam['windowStart'][core_idx],
                                    rtable_fam['windowAmp'][core_idx, s], config)

            # Plot
            ax.plot(time_vector, data_stack - 1.75*s, 'r', linewidth=1)
            ax.plot(time_vector, data_core - 1.75*s, 'k', linewidth=0.25)

    return ax


def subplot_fft(rtable_fam, core_idx, ax, config):
    """
    Fills the FFT subplot.

    This plot shows the amplitude spectrum from the Fourier transform summed
    over all stations. The core is always plotted in black, and the sum over
    all events in red.

    Parameters
    ----------
    rtable_fam : Table object
        Handle to a subset of the Repeaters table containing all members of
        a family.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    ax : Axis object
        Subplot axis to modify in place.
    config : Config object
        Describes the run parameters.

    """

    freq = np.linspace(0,config.get('samprate')/2,int(config.get('winlen')/2))
    fftc = np.zeros((int(config.get('winlen')/2),))
    fftm = np.zeros((int(config.get('winlen')/2),))

    for s in range(config.get('nsta')):
        fft = np.abs(np.real(rtable_fam['windowFFT'][core_idx,int(
            s*config.get('winlen')):int(s*config.get('winlen')+config.get('winlen')/2)]))
        fft = fft/(np.amax(fft)+1.0/1000)
        fftc = fftc+fft
        ffts = np.mean(np.abs(np.real(rtable_fam['windowFFT'][:,int(
            s*config.get('winlen')):int(s*config.get('winlen')+config.get('winlen')/2)])),axis=0)
        fftm = fftm + ffts/(np.amax(ffts)+1.0/1000)

    ax.plot(freq,fftm,'r', linewidth=1)
    ax.plot(freq,fftc,'k', linewidth=0.25)


def bokeh_figure(**kwargs):
    """
    Builds foundation for the Bokeh subplots.

    **kwargs can include any keyword argument passable to a Bokeh figure().
    See https://docs.bokeh.org/en/latest/docs/reference/plotting.html for a
    complete list.

    The main argument passed is usually 'title'. If they are not defined,
    'tools', 'plot_width', 'plot_height', and 'x_axis_type' are populated
    with default values.

    Returns
    -------
    fig : Figure object

    """

    # Default values for Bokeh figures
    if 'tools' not in kwargs:
        kwargs['tools'] = ['pan,box_zoom,reset,save,tap']
    if 'plot_width' not in kwargs:
        kwargs['plot_width'] = 1250
    if 'plot_height' not in kwargs:
        kwargs['plot_height'] = 250
    if 'x_axis_type' not in kwargs:
        kwargs['x_axis_type'] = 'datetime'

    # Create figure
    fig = figure(**kwargs)

    # Generate initial style
    fig.grid.grid_line_alpha = 0.3
    fig.xaxis.axis_label = 'Date'
    fig.yaxis.axis_label = ''

    return fig


def determine_legend_text(binsize_occur):
    """
    Determines legend wording based on bin size.

    Parameters
    ----------
    binsize_occur : float
        Width (in days) of time bins for occurrence plot histogram.

    Returns
    -------
    legtext : str
        Formatted string for legend.

    """

    if binsize_occur == 1/24:
        legtext = 'Hour'
    elif binsize_occur == 1:
        legtext = 'Day'
    elif binsize_occur == 7:
        legtext = 'Week'
    elif binsize_occur < 2:
        legtext = f'{binsize_occur*24} Hours'
    else:
        legtext = f'{binsize_occur} Days'

    return legtext


def determine_color_mapper(binsize_occur):
    """
    Determines logarithmic color map for occurrence plot based on bin size.

    Parameters
    ----------
    binsize_occur : float
        Width (in days) of time bins for occurrence plot histogram.

    Returns
    -------
    color_mapper : LogColorMapper object
    """

    # Steal YlOrRd (len=256) colormap from matplotlib
    colormap = matplotlib.cm.get_cmap('YlOrRd')
    bokehpalette = [matplotlib.colors.rgb2hex(m) for m in colormap(
        np.arange(colormap.N)[::-1])]
    if binsize_occur >= 1:
        color_mapper = LogColorMapper(palette=bokehpalette, low=1, high=1000)
    else:
        color_mapper = LogColorMapper(palette=bokehpalette, low=1, high=100)

    return color_mapper


def determine_color_mapper_fi(config):
    """
    Determines color map for occurrencefi plot based on config.fispan.

    Parameters
    ----------
    config : Config object
        Describes the run parameters.

    Returns
    -------
    color_mapper : LinearColorMapper object

    """

    # Steal YlOrRd (len=256) colormap from matplotlib
    colormap = matplotlib.cm.get_cmap('coolwarm')
    bokehpalette = [matplotlib.colors.rgb2hex(m) for m in colormap(
        np.arange(colormap.N)[::-1])]
    color_mapper = LinearColorMapper(palette=bokehpalette,
                                     low=config.get('fispanlow'), high=config.get('fispanhigh'))

    return color_mapper


def determine_lines(mintime, maxtime, barpad, famstart, longev):
    """
    Determines arrows and line positions for occurrence/longevity timelines.

    Parameters
    ----------
    mintime : float
        Minimum time to plot as matplotlib date.
    maxtime : float
        Maximum time to plot as matplotlib date.
    barpad : float
        Horizontal padding for hover and arrows (usually ~1% of window range).
    famstart : float
        Start time of family as matplotlib date.
    longev : float
        Longevity of family in days.

    Returns
    -------
    add_line : bool
        True if line needs to be plotted.
    add_larrow : bool
        True if left arrow needs to be plotted.
    add_rarrow : bool
        True if right arrow needs to be plotted.
    x1 : float
        Date of start of line as matplotlib date.
    x2 : float
        Date of end of line as matplotlib date.

    """

    add_rarrow = False
    add_larrow = False
    add_line = False
    x1 = 0
    x2 = 0

    # Family starts after start of mintime and ends before maxtime
    if (mintime<=famstart) and (maxtime>=famstart+longev):
        add_line = True
        x1 = famstart
        x2 = famstart+longev

    # Family starts after start of mintime but first event is past maxtime
    elif (mintime<=famstart) and (maxtime<=famstart):
        add_line = False

    # Family starts after start of mintime but ends after maxtime
    elif (mintime<=famstart) and (maxtime<=famstart+longev):
        add_line = True
        add_rarrow = True
        x1 = famstart
        x2 = maxtime+barpad

    # Family starts before mintime, ends before maxtime, ends after mintime
    elif (mintime>=famstart) and (maxtime>=famstart+longev) and (
                                                    mintime<=famstart+longev):
        add_line = True
        add_larrow = True
        x1 = mintime-barpad
        x2 = famstart+longev

    # Family starts before mintime and ends after maxtime
    elif (mintime>=famstart) and (maxtime<=famstart+longev):
        add_line = True
        add_larrow = True
        add_rarrow = True
        x1 = mintime-barpad
        x2 = maxtime+barpad

    return add_line, add_larrow, add_rarrow, x1, x2


def family_hover_tool():
    """
    Generates HoverTool for family hover preview.

    Returns
    -------
    hover : HoverTool object

    """

    hover = HoverTool(
        tooltips="""
        <div>
        <div>
            <img src="./clusters/@famnum.png"
                style="height: 100px; width: 500px;
                vertical-align: middle;"/>
            <span style="font-size: 9px;
                font-family: Helvetica;">Cluster ID: </span>
            <span style="font-size: 12px;
                font-family: Helvetica;">@famnum</span>
        </div>
        </div>
        """, names=["patch"])

    return hover


def add_pdf_colorbar(fig, figheight, pnum, nsub, colorby, binsize_occur, config):
    """
    Adds a colorbar to PDF occurrence plots.

    Parameters
    ----------
    fig : Figure object
        Handle to the matplotlib figure.
    figheight : float
        Total height of the figure.
    pnum : int
        Current subplot number.
    nsub : int
        Total number of subplots,
    colorby : str
        Determines colormap to use ('rate' or 'fi')
    binsize_occur : float
        Width (in days) of time bins for occurrence plot histogram.
    config : Config object
        Describes the run parameters.

    """

    # Inset colorbar
    bottom = (nsub-pnum)/nsub - 0.9/figheight
    cax = fig.add_axes([0.1, bottom, 0.2, 0.2/figheight])
    if colorby is 'rate':
        cax.set_title('Events per {}'.format(
            determine_legend_text(binsize_occur)), loc='left', style='italic')
    else:
        cax.set_title('Mean Frequency Index', loc='left', style='italic')
    cax.get_yaxis().set_visible(False)
    gradient = np.linspace(0, 1, 1001)
    gradient = np.vstack((gradient, gradient))
    if colorby is 'rate':
        cax.imshow(gradient, aspect='auto', cmap='YlOrRd_r',
            interpolation='bilinear')
        if binsize_occur >= 1:
            cax.set_xticks((0,333.3,666.6,1000))
            cax.set_xticklabels(('1','10','100','1000'))
        else:
            cax.set_xticks((0,500,1000))
            cax.set_xticklabels(('1','10','100'))
    else:
        cax.imshow(gradient, aspect='auto', cmap='coolwarm_r',
            interpolation='bilinear')
        cax.set_xticks((0,500,1000))
        cax.set_xticklabels((config.get('fispanlow'), np.mean((config.get('fispanlow'),
            config.get('fispanhigh'))), config.get('fispanhigh')))
    cax.set_frame_on(False)
    cax.tick_params(length=0)


def add_pdf_annotations(ax, mintime, maxtime, config):
    """
    Plots annotations on PDF overview figure.

    Parameters
    ----------
    ax : Axis object
        Handle to the matplotlib axis.
    mintime : float
        Minimum time to plot as matplotlib date.
    maxtime : float
        Maximum time to plot as matplotlib date.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    ax : Axis object

    """

    if config.get('anotfile') != '':

        annotations = pd.read_csv(config.get('anotfile'))

        for a in range(len(annotations)):

            plotdate = mdates.date2num(np.datetime64(
                annotations['Time'][a]))

            # If within bounds, add to figure
            if (plotdate >= mintime) and (plotdate <= maxtime):
                ax.axvline(plotdate, color=annotations['Color'][a],
                    lw=annotations['Weight'][a],
                    ls=annotations['Line Type'][a],
                    alpha=annotations['Alpha'][a], zorder=-1)

    return ax


def add_bokeh_annotations(fig, config):
    """
    Plots annotations on bokeh figure.

    Parameters
    ----------
    fig : Figure object
        Handle to the bokeh figure.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    fig : Figure object

    """

    if config.get('anotfile') != '':

        annotations = pd.read_csv(config.get('anotfile'))

        for a in range(len(annotations)):

            # Deal with bokeh's unusual datetime axis
            spantime = (datetime.datetime.strptime(annotations['Time'][a]
                ,'%Y-%m-%dT%H:%M:%S')-datetime.datetime(
                1970, 1, 1)).total_seconds()

            # Add to figure
            fig.add_layout(Span(location=spantime*1000, dimension='height',
                line_color=annotations['Color'][a],
                line_width=annotations['Weight'][a],
                line_dash=annotations['Line Type'][a],
                line_alpha=annotations['Alpha'][a]))

    return fig


def add_horizontal_annotations(ax, evtimes, config):
    """
    Plots annotations horizontally across an image (e.g., waveforms, matrix).

    Parameters
    ----------
    ax : Axis object
        Handle to the matplotlib axis.
    evtimes : float ndarray
        Sorted array of event times plotted on each row of the image.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    ax : Axis object

    """

    if config.get('anotfile')!='':

        annotations = pd.read_csv(config.get('anotfile'))

        for a in range(len(annotations)):

            # Translate from date to vertical position in the image
            vertical_location = np.interp(mdates.date2num(
                pd.to_datetime(annotations['Time'][a])),
                evtimes, np.array(range(len(evtimes))))

            # Plot if within the time span of the image
            if vertical_location != 0:
                ax.axhline(np.floor(vertical_location)+0.5, color='k',
                           linewidth=annotations['Weight'][a]/2.,
                           linestyle=annotations['Line Type'][a])

    return ax


def prep_wiggle(waveform, sta, window_start, normalize_amplitude, config):
    """
    Cuts window around trigger time and normalizes the waveform for plotting.

    The plotting window is always 2*config.get('winlen') samples, with half config.get('winlen')
    on either side of the correlation window. Data are clipped if they are
    above 'windowAmp' in amplitude.

    Parameters
    ----------
    waveform : float ndarray
        Waveform to be plotted, all stations/channels concatenated.
    sta : int
        Station index for station/channel to be used.
    window_start : int
        Sample corresponding to start of correlation window.
    normalize_amplitude : float
        Amplitude to normalize to. If passed 0, uses the maximum of the entire
        window instead with a small epsilon to prevent division by 0 if empty.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    data : float ndarray
        Clipped, normalized, and trimmed waveform for single station/channel.

    """

    # Trim out data for that station/channel
    window = waveform[sta*config.get('wshape'):(sta+1)*config.get('wshape')]

    # Determine window
    minsample = window_start - int(0.5*config.get('winlen'))
    maxsample = window_start + int(1.5*config.get('winlen'))

    # Determine if we need to pad with 0's if the window is outside the bounds
    # of what is saved in the table for each station
    if minsample < 0:
        prepad = -minsample
        minsample = 0
    else:
        prepad = 0
    if maxsample > config.get('wshape'):
        postpad = maxsample - config.get('wshape')
        maxsample = config.get('wshape')
    else:
        postpad = 0

    # Trim window
    data = window[minsample:maxsample]

    # Pad if necessary
    if prepad:
        data = np.append(np.zeros(prepad), data)
    if postpad:
        data = np.append(data, np.zeros(postpad))

    # Normalize
    if normalize_amplitude > 0:
        data = data / normalize_amplitude
    else:
        data = data / np.max(np.abs(data)+1e-12)

    # Clip
    data[data>1] = 1
    data[data<-1] = -1

    return data


def wiggle_plot(data, figsize, outfile, config):
    """
    Plots a waveform with no decorations (e.g., for a core image).

    Parameters
    ----------
    data : float ndarray
        Waveform data to plot.
    figsize : tuple
        Output figure size as (width, height).
    outfile : str
        Path and filename for saving the figure.
    config : Config object
        Describes the run parameters.
    """

    fig = plt.figure(figsize=figsize)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.plot(data,'k',linewidth=0.25)
    plt.autoscale(tight=True)
    plt.savefig(outfile, dpi=100)
    plt.close(fig)


def wiggle_plot_all(rtable, rtimes_mpl, fam, ordered, outfile, config):
    """
    Creates a plot with all waveforms on all stations in separate subplots.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    fam : int ndarray
        Indices of family members within the Repeaters table.
    ordered : int
        1 if members should be ordered by OPTICS, 0 if by time.
    outfile : str
        Path and filename for saving the figure.
    config : Config object
        Describes the run parameters.

    """

    fig = plt.figure(figsize=(10, 4*(np.ceil(config.get('nsta')/2))))
    for sta in range(config.get('nsta')):

        ax = fig.add_subplot(int(np.ceil((config.get('nsta'))/2.)), 2, sta+1)

        title_text = '{}.{}'.format(config.get('station').split(',')[sta],
                                    config.get('channel').split(',')[sta])
        if ordered:
            title_text += ' (Ordered)'
        else:
            ax = add_horizontal_annotations(ax, rtimes_mpl[fam], config)
        plt.title(title_text, fontweight='bold')

        subplot_waveforms(rtable[fam], 0, ax, config, plot_single=True, sta=sta)

        ax.yaxis.set_visible(False)
        plt.xlabel('Time Relative to Trigger (seconds)', style='italic')

    plt.tight_layout()
    plt.savefig(outfile, dpi=100)
    plt.close(fig)


def correlation_matrix_plot(ccc_fam, ccc_full, rtimes_mpl, fam, ordered,
                                          skip_recalculate_ccc, outfile, config):
    """
    Creates a plot of the stored (and optionally the full) correlation matrix.

    Parameters
    ----------
    ccc_fam : float ndarray
        Dense stored correlation matrix.
    ccc_full : float ndarray, optional
        Filled correlation matrix.
    rtimes_mpl : float ndarray
        Times of all repeaters as matplotlib dates.
    fam : int ndarray
        Indices of family members within the Repeaters table.
    ordered : int
        1 if members should be ordered by OPTICS, 0 if by time.
    skip_recalculate_ccc : int
        1 if user wishes to skip recalculating the full correlation matrix.
    outfile : str
        Path and filename for saving the figure.
    config : Config object
        Describes the run parameters.

    """

    # Correlation matrix image(s)
    cmap = plt.get_cmap('inferno_r').copy()
    cmap.set_extremes(under='w')

    fig = plt.figure(figsize=(14,5.4))
    ax1 = fig.add_subplot(1,2,1)
    cax = ax1.imshow(ccc_fam, vmin=config.get('cmin'), vmax=1, cmap=cmap)
    cbar = plt.colorbar(cax, extend='min')
    if ordered:
        plt.title('Stored Correlation Matrix (Ordered)', fontweight='bold')
    else:
        plt.title('Stored Correlation Matrix', fontweight='bold')
        ax1 = add_horizontal_annotations(ax1, rtimes_mpl[fam],
            config)
    if not skip_recalculate_ccc:
        ax2 = fig.add_subplot(1,2,2)
        cax2 = ax2.imshow(ccc_full, vmin=config.get('cmin'), vmax=1, cmap=cmap)
        cbar = plt.colorbar(cax, extend='min')
        if ordered:
            plt.title('Full Correlation Matrix (Ordered)', fontweight='bold')
        else:
            plt.title('Full Correlation Matrix', fontweight='bold')
            ax2 = add_horizontal_annotations(ax2, rtimes_mpl[fam], config)
    plt.tight_layout()
    plt.savefig(outfile, dpi=100)
    plt.close(fig)


def initialize_family_image(config, bboxes=[]):
    """
    Creates figure and axes with the proper layout for the family images.

    This function basically makes it so we only have to call tight_layout()
    once and significantly reduce time spent formatting the plots by using
    the bounding box locations.

    Parameters
    ----------
    config : Config object
        Describes the run parameters.
    bboxes : list of Bbox objects, optional
        List of bounding box positions for each axis.

    Returns
    -------
    fig : Figure object
        Handle to the matplotlib figure.
    axes : list of Axis objects
        List of subplot axes.
    bboxes : list of Bbox objects
        List of bounding box positions for each axis.

    """

    # Close any existing plots
    plt.close()

    fig = plt.figure(figsize=(10, 12))
    axes = [fig.add_subplot(9, 3, (1,8)), fig.add_subplot(9, 3, (3,9)),
            fig.add_subplot(9, 3, (10,15)),
            fig.add_subplot(9, 3, (16,21)),
            fig.add_subplot(9, 3, (22,27))]

    if bboxes:
        # Set positions for axes
        for n in range(len(axes)):
            axes[n].set_position(bboxes[n])
    else:
        # Set up format for axes positions
        axes = format_family_image(axes, config)
        axes[2].set_xlim(0,1) # Ensure that dates near the edge fit
        plt.tight_layout()
        bboxes = [axes[n].get_position() for n in range(len(axes))]

    return fig, axes, bboxes


def format_family_image(axes, config):
    """
    Handles formatting for each axis in the family image.

    Parameters
    ----------
    axes : list of Axis objects
        List of subplot axes to modify.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    axes : list of Axis objects

    """

    date_format = mdates.DateFormatter('%Y-%m-%d\n%H:%M')

    time_vector = np.arange(-0.5*config.get('winlen')/config.get('samprate'),
                      1.5*config.get('winlen')/config.get('samprate'), 1/config.get('samprate'))

    axes[0].get_yaxis().set_visible(False)
    axes[0].set_xlim((np.min(time_vector),np.max(time_vector)))
    axes[0].axvline(x=-0.1*config.get('winlen')/config.get('samprate'), color='k', ls='dotted')
    axes[0].axvline(x=0.9*config.get('winlen')/config.get('samprate'), color='k', ls='dotted')
    if config.get('nsta') > 1:
        axes[0].set_ylim((-1.75*(config.get('nsta')-1)-1,1))
        # Add station labels
        stas = config.get('station').split(',')
        chas = config.get('channel').split(',')
        for s in range(len(stas)):
            axes[0].text(np.min(time_vector)-0.1,-1.75*s,
                f'{stas[s]}\n{chas[s]}', horizontalalignment='right',
                verticalalignment='center')
    axes[0].set_xlabel('Time Relative to Trigger (seconds)', style='italic')

    axes[1].get_yaxis().set_visible(False)
    axes[1].set_xlim(0,config.get('fmax')*1.5)
    axes[1].legend(['Stack','Core'], loc='upper right', frameon=False)
    axes[1].set_xlabel('Frequency (Hz)', style='italic')

    axes[2].xaxis.set_major_formatter(date_format)
    axes[2].margins(0.05)
    axes[2].set_yscale('log')
    axes[2].set_xlabel('Date', style='italic')
    axes[2].set_ylabel('Amplitude (Counts)', style='italic')

    axes[3].xaxis.set_major_formatter(date_format)
    axes[3].margins(0.05)
    axes[3].set_yscale('log')
    axes[3].set_xlabel('Date', style='italic')
    axes[3].set_ylabel('Time since previous event (hours)', style='italic')

    axes[4].xaxis.set_major_formatter(date_format)
    axes[4].margins(0.05)
    axes[4].set_ylim(config.get('cmin')-0.02, 1.02)
    axes[4].set_xlabel('Date', style='italic')
    axes[4].set_ylabel('Cross-correlation coefficient', style='italic')

    return axes


def write_html_header(f, ftable, fnum, rtimes, rtimes_mpl, fi, config,
                                                                 report=False):
    """
    f : file handle
        Handle to output file to write in.
    ftable : Table object
        Handle to the Families table.
    fnum : int
        Family to be written.
    rtimes : datetime ndarray
        Times of all repeaters as datetimes.
    fi : float ndarray
        Frequency index values for repeaters.
    config : Config object
        Describes the run parameters.
    report : bool, optional
        If true, use report-specific formatting.

    """

    fam = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')
    corenum = ftable[fnum]['core']

    # Prep catalog
    catalogind = np.argsort(rtimes_mpl[fam])
    fam = fam[catalogind]
    catalog = rtimes_mpl[fam]
    spacing = np.diff(catalog)*24

    if report:

        htmltitle = f'{config.get("title")} - Cluster {fnum} Detailed Report'
        topline = f'<em>Last updated: {UTCDateTime.now()}</em>'
        header = f'Cluster {fnum} - Detailed Report'
        coreimg = f'{fnum}-report'
        body = f"""
            <img src='{fnum}-reportwaves.png'></br></br>
            <iframe src="{fnum}-report-bokeh.html" width=1350 height=800
            style="border:none"></iframe></br>
            <img src='{fnum}-reportcmat.png'></br></br></br>"""

    else:

        htmltitle = f'{config.get("title")} - Cluster {fnum}'
        if fnum>0:
            prevlink = f"<a href='{fnum-1}.html'>&lt; Cluster {fnum-1}</a>"
        else:
            prevlink = " "
        if fnum < len(ftable)-1:
            nextlink = f"<a href='{fnum+1}.html'>Cluster {fnum+1} &gt;</a>"
        else:
            nextlink = " "
        topline = f'{prevlink} &nbsp; | &nbsp; {nextlink}'
        header = f'Cluster {fnum}'
        coreimg = fnum
        body = f'<img src="fam{fnum}.png"></br>'

    f.write(f"""
            <html><head><title>{htmltitle}</title>
            </head><style>
            a {{color:red;}}
            body {{font-family:Helvetica; font-size:12px}}
            h1 {{font-size: 20px;}}
            </style>
            <body><center>
            {topline}</br>
            <h1>{header}</h1>
            <img src="{coreimg}.png" width=500 height=100></br></br>

                Number of events: {len(fam)}</br>
                Longevity: {ftable[fnum]['longevity']:.2f} days</br>
                Mean event spacing: {np.mean(spacing):.2f} hours</br>
                Median event spacing: {np.median(spacing):.2f} hours</br>
                Mean Frequency Index: {np.nanmean(fi[fam]):.2f}<br></br>
                First event: {UTCDateTime(rtimes[fam[0]]).isoformat()}</br>
                Core event: {UTCDateTime(rtimes[corenum]).isoformat()}</br>
                Last event: {UTCDateTime(rtimes[fam[-1]]).isoformat()}</br>

            {body}
            """)


def create_local_map(local_lats, local_lons, local_deps, outfile, config):
    """
    Makes map centered on local matches from external catalog.

    Parameters
    ----------
    local_lats : float ndarray
        Latitudes of local events to be plotted.
    local_lons : float ndarray
        Longitudes of local events to be plotted.
    local_deps : float ndarray
        Depths of local events to be plotted.
    outfile : str
        Path and filename for saving the figure.
    config : Config object
        Describes the run parameters.

    """

    stalats = np.array(config.get('stalats').split(',')).astype(float)
    stalons = np.array(config.get('stalons').split(',')).astype(float)

    # Set basemap and projection
    stamen_terrain = cimgt.StamenTerrain()

    extent = [np.median(local_lons) - config.get('locdeg')/2,
              np.median(local_lons) + config.get('locdeg')/2,
              np.median(local_lats) - config.get('locdeg')/4,
              np.median(local_lats) + config.get('locdeg')/4]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Shaded terrain background
    ax.add_image(stamen_terrain, 11)

    # Set up ticks
    ax.set_xticks(np.arange(np.floor(10*(extent[0]))/10,
                     np.ceil(10*(extent[1]))/10, 0.1), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.floor(10*(extent[2]))/10,
                     np.ceil(10*(extent[3]))/10, 0.1), crs=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree()) # Reset to fill image
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    plt.yticks(rotation=90, va='center')

    # Seismicity in red (halo of white), stations open black triangles
    ax.scatter(local_lons, local_lats, s=20, marker='o',
                color='white', transform=ccrs.PlateCarree())
    ax.scatter(local_lons, local_lats, s=5, marker='o',
                color='red', transform=ccrs.PlateCarree())
    ax.scatter(stalons, stalats, marker='^', color='k',
                facecolors='None', transform=ccrs.PlateCarree())

    # 10 km scale bar
    scalebar_lat = 0.05*(config.get('locdeg')/2) + extent[2]
    scalebar_lon_left = 0.05*(config.get('locdeg')) + extent[0]
    # Convert to degrees longitude at the latitude of the bar
    scalebar_len = kilometers2degrees(10) / np.cos(scalebar_lat * np.pi/180)

    # Plot and save
    ax.plot((scalebar_lon_left, scalebar_lon_left + scalebar_len),
            (scalebar_lat, scalebar_lat), 'k-',
            transform=ccrs.PlateCarree(), lw=2)
    geodetic_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
    text_transform = offset_copy(geodetic_transform, units='dots', y=5)
    ax.text(scalebar_lon_left + scalebar_len/2,
            scalebar_lat, '10 km', ha='center', transform=text_transform)

    plt.title('{} potential local matches (~{:3.1f} km depth)'.format(
                                    len(local_deps), np.mean(local_deps)))
    plt.tight_layout()
    plt.savefig(outfile, dpi=100)
    plt.close()


def remove_old_files(ftable, config):
    """
    Removes .html and .png files from deleted/moved family pages.

    Deletes removed family files that have family numbers above the
    current maximum family number.

    ftable : Table object
        Handle to the Families table.
    config : Config object
        Describes the run parameters.

    """
    flist = glob.glob(os.path.join(config.get('output_folder'), 'clusters', '*.html'))
    for file in flist:
        fnum = int(os.path.split(file)[1].split('.')[0])
        if fnum >= ftable.attrs.nClust:
            os.remove(file)
            for itype in [f'{fnum}.png', f'fam{fnum}.png', f'map{fnum}.png']:
                img = os.path.join(os.path.split(file)[0], itype)
                if os.path.exists(img):
                    os.remove(img)
