# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import glob, os, shutil
import datetime

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates
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
from obspy.clients.fdsn.client import Client
from obspy.geodetics import locations2degrees, kilometers2degrees
from obspy.taup import TauPyModel
from tables import *

import redpy.cluster
import redpy.correlation
from redpy.optics import *
from redpy.printing import *

matplotlib.use('Agg')

# Adjust rcParams
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 8.0
matplotlib.rcParams['pdf.fonttype'] = 42


def generate_all_outputs(rtable, ftable, ttable, ctable, otable, opt):
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
    opt : Options object
        Describes the run parameters.
    """
    
    if opt.checkComCat==True:
        external_catalogs = prepare_catalog(ttable, opt)
    else:
        external_catalogs = []
    
    # Write trigger and orphan catalogs
    printTriggerCatalog(ttable, opt)
    printOrphanCatalog(otable, opt)
    
    # If there is at least one family
    if len(rtable) > 1:
        
        # Render Bokeh timelines
        create_timelines(rtable, ftable, ttable, opt)
        
        # If there are changes to the catalog
        if np.sum(ftable.cols.printme[:]):
            
            # Write repeater-related catalogs
            if opt.verbosecatalog == True:
                printVerboseCatalog(rtable, ftable, ctable, opt)
            else:
                printCatalog(rtable, ftable, opt)
            printSwarmCatalog(rtable, ftable, ttable, opt)
            printCoresCatalog(rtable, ftable, opt)
            
            # Make images
            create_core_images(rtable, ftable, opt)
            create_family_images(rtable, ftable, ctable, opt)
            
            # Make HTML files
            create_family_html(rtable, ftable, external_catalogs, opt)
            
            # Reset printing columns
            ftable.cols.printme[:] = np.zeros((len(ftable),))
            ftable.cols.lastprint[:] = np.arange(len(ftable))
            
    else:
        print('Nothing to plot!')
    
    # Rename any .tmp files created in create_core_images()
    tmplist = glob.glob(os.path.join('{}{}'.format(opt.outputPath,
                                           opt.groupName),'clusters','*.tmp'))
    for tmp in tmplist:
        os.rename(tmp,tmp[0:-4])


def prepare_catalog(ttable, opt):
    """
    Downloads and formats event catalog from external datacenter.
    
    Data are queried from three regions (local, regional, teleseismic) based
    on the settings in opt. Times are taken from the first and last trigger
    times in ttable, so if there are large gaps in ttable, this function is
    agnostic to them. Updates the catalog if a file exists to reduce query
    overhead.
    
    Parameters
    ----------
    ttable : Table object
        Handle to the Triggers table.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    external_catalogs : list of DataFrame objects
    """
    
    ttimes = ttable.cols.startTimeMPL[:] + opt.ptrig
    tmin = UTCDateTime(matplotlib.dates.num2date(np.min(ttimes))) - 1800
    tmax = UTCDateTime(matplotlib.dates.num2date(np.max(ttimes))) + 30
    
    external_catalogs = []
    
    for region in ['local', 'regional', 'teleseismic']:
        
        fname = os.path.join('{}{}'.format(opt.outputPath, opt.groupName),
                             'external_{}.txt'.format(region))
        
        if os.path.exists(fname):
            
            # Load existing file
            catalog = pd.read_csv(fname, delimiter='|')
            
            # !!! If I wanted to remove some recent events to get the
            # !!! most updated event solutions, I could do it here
            
            # Get missing events before and after currently saved events
            if len(catalog) > 0:
                
                tmin_catalog = UTCDateTime(np.min(catalog['Time']))-1
                tmax_catalog = UTCDateTime(np.max(catalog['Time']))+1
                
                catalog_before = query_external(region, tmin, tmin_catalog,
                                                                          opt)
                catalog_after = query_external(region, tmax_catalog, tmax,
                                                                          opt)
                
                catalog = pd.concat([catalog_before, catalog, catalog_after],
                                                    axis=0, ignore_index=True)
                
            else:
                
                # Query the entire time again
                catalog = query_external(region, tmin, tmax, opt)
                
        else:
            
            # Query in full
            catalog = query_external(region, tmin, tmax, opt)
            
        # Save to file
        catalog.to_csv(fname, index=False, sep='|')
        
        external_catalogs.append(catalog)
    
    return external_catalogs


def query_external(region, tmin, tmax, opt):
    """
    Handles querying and formatting the external event catalog.
    
    Currently only supports querying the USGS FDSN (ComCat). Other datacenters
    could be used, so long as they support the default FDSN 'text' format,
    with columns separated by | instead of commas.
    
    Parameters
    ----------
    region : str
        String describing which of the three distance regions to use.
    tmin : UTCDateTime object
        Start time for catalog query.
    tmax : UTCDateTime object
        End time for catalog query.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    catalog : pandas DataFrame
        Formatted event catalog.
    """
    
    latitude_center = np.mean(np.array(opt.stalats.split(',')).astype(float))
    longitude_center = np.mean(np.array(opt.stalons.split(',')).astype(float))
    
    datacenter = 'USGS' # Eventually may have options to choose here
    
    if region in 'local':
        minrad = 0
        maxrad = opt.locdeg
        minmag = -10
        phase_list=['p','s','P','S']
    elif region in 'regional':
        minrad = opt.locdeg
        maxrad = opt.regdeg
        minmag = opt.regmag
        phase_list = ['p','s','P','S','PP','SS']
    else:
        minrad = opt.regdeg
        maxrad = 180
        minmag = opt.telemag
        phase_list = ['P','S','PP','SS','PcP','ScS','PKiKP','PKIKP']
    
    # This assumes a lot about the path to get stuff
    base_url = Client(datacenter).base_url
    query_url = base_url + '/fdsnws/event/1/query' + \
                '?starttime={}'.format(tmin) + \
                '&endtime={}'.format(tmax) + \
                '&latitude={}'.format(latitude_center) + \
                '&longitude={}'.format(longitude_center) + \
                '&maxradius={}'.format(maxrad) + \
                '&minradius={}'.format(minrad) + \
                '&minmagnitude={}'.format(minmag) + \
                '&orderby=time-asc&format=text&limit=10000'
    
    try:
        catalog = pd.read_csv(query_url, delimiter='|')
        
        # If the limit is returned
        if (len(catalog) == 10000):
            
            # Continue to query until we find the end
            offset = 0
            while not (len(catalog) % 10000):
                
                offset += 10000
                catalog2 = pd.read_csv(query_url+'&offset={}'.format(offset),
                                                                delimiter='|')
                
                if len(catalog2) > 0:
                    catalog = catalog.append(catalog2, ignore_index=True)
                else:
                    # Remainder will still be 0 so we'd be stuck in the loop
                    break
        
    except:
        
        # Pass an empty dataframe with the correct columns
        catalog = pd.DataFrame(columns=['EventID', 'Time', 'Latitude',
                     'Longitude', 'Depth/km','Magnitude','EventLocationName'])
        print('Failed to download {} event catalog from {}'.format(region,
                                                                  datacenter))
    
    # Clean column names
    catalog.columns = catalog.columns.str.replace(' ','')
    catalog.columns = catalog.columns.str.replace('#','')
    
    # Remove 'unnecessary' columns to reduce space
    catalog.drop(columns=['Author', 'Catalog', 'Contributor',
                'ContributorID', 'MagType', 'MagAuthor', 'EventType'],
                errors='ignore', inplace=True)
    
    # Calculate arrivals
    catalog = calculate_arrivals(catalog, latitude_center, longitude_center,
                                                              phase_list, opt)
    
    return catalog


def calculate_arrivals(catalog, latitude_center, longitude_center, phase_list,
                                                                         opt):
    """
    Calculates the arrivals of given phases from source to study area.
    
    Traces rays through a very simple model (iasp91), then appends the times
    of arrivals to the end of the DataFrame.
    
    Parameters
    ----------
    catalog : pandas DataFrame
        Event catalog, with columns for time and location.
    latitude_center : float
        Latitude of study area centroid.
    longitude_center : float
        Longitude of study area centroid.
    phase_list : list
        List of seismic phase arrivals to trace.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    catalog : pandas DataFrame
        Event catalog, with arrival times appended.
    """
    
    # Add columns for predefined phases
    for phase in phase_list:
        catalog['Arrival_{}'.format(phase)] = 'NaN'
            
    # Loop over events to fill those columns
    if len(catalog) > 0:
        
        taupymodel_runs = 0
        model = TauPyModel(model="iasp91")
        
        for i in range(len(catalog)):
            
            # Calculate distance of separation (degrees)
            deg = locations2degrees(catalog['Latitude'][i],
                                    catalog['Longitude'][i],
                                    latitude_center,
                                    longitude_center)
                    
            # TauPyModel misbehaves if it's used too much
            # Determine if it needs to be reloaded (every 100 runs)
            taupymodel_runs += 1
            if np.remainder(taupymodel_runs,100) == 0:
                model = TauPyModel(model="iasp91")
            
            # Calculate arrivals based on event depth
            arrivals = model.get_travel_times(
                        source_depth_in_km=max(0, catalog['Depth/km'][i]),
                        distance_in_degree=deg, phase_list=phase_list)
                    
            if len(arrivals) > 0:
                for a in range(len(arrivals)):
                    catalog['Arrival_{}'.format(arrivals[a].name)][i] = \
                               '{}'.format(UTCDateTime(catalog['Time'][i]) + \
                                                             arrivals[a].time)
    
    return catalog


def create_timelines(rtable, ftable, ttable, opt):
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
    ttable : Table object
        Handle to the Triggers table.
    opt : Options object
        Describes the run parameters.
    """
    
    # Load table columns into memory to reduce I/O overhead
    # !!! This variable technically needs the windowStart added to it
    # !!! Same for alltrigs needing opt.ptrig...
    rtimes = rtable.cols.startTimeMPL[:]# + rtable.cols.windowStart[:] / \
                                        #                86400. / opt.samprate
    fi = np.nanmean(rtable.cols.FI[:], axis=1)
    longevity = ftable.cols.longevity[:]
    famstarts = ftable.cols.startTime[:]
    alltrigs = ttable.cols.startTimeMPL[:]# + opt.ptrig
    
    for i, file in enumerate(['overview', 'overview_recent', 'meta_recent']):
        
        # Set parameters unique to each timeline type
        if i == 0: # overview.html
            barpad = (max(alltrigs)-min(alltrigs))*0.01
            plotformat = opt.plotformat
            binsize_hist = opt.dybin
            binsize_occur = opt.occurbin
            mintime = min(alltrigs)
            minplot = opt.minplot
            fixedheight = opt.fixedheight
            htmltitle = '{} Overview'.format(opt.title)
            divtitle = '<h1>{}</h1>'.format(opt.title)
        elif i == 1: # overview_recent.html
            barpad = opt.recplot*0.01
            plotformat = opt.plotformat
            binsize_hist = opt.hrbin/24
            binsize_occur = opt.recbin
            mintime = max(alltrigs)-opt.recplot
            minplot = 0
            fixedheight = opt.fixedheight
            htmltitle = '{} Overview - Last {:.1f} Days'.format(opt.title,
                                                                opt.recplot)
            divtitle = '<h1>{} - Last {:.1f} Days</h1>'.format(opt.title,
                                                               opt.recplot)
        else: # meta_recent.html
            barpad = opt.mrecplot*0.01
            plotformat = opt.plotformat.replace(',','+')
            binsize_hist = opt.mhrbin/24
            binsize_occur = opt.mrecbin
            mintime = max(alltrigs)-opt.mrecplot
            minplot = opt.mminplot
            fixedheight = True
            htmltitle = '{} Overview - Last {:.1f} Days'.format(opt.title,
                                                                 opt.mrecplot)
            divtitle = """<h1>{} - Last {:.1f} Days |
                          <a href='overview.html' style='color:red'
                          target='_blank'>Full Overview</a> |
                          <a href="overview_recent.html"
                          style="color:red" target="_blank">Recent</a>
                          </h1>""".format(opt.title, opt.mrecplot)
        
        filepath = os.path.join('{}{}'.format(opt.outputPath, opt.groupName),
                            '{}.html'.format(file))
        
        assemble_bokeh_timeline(ftable, rtimes, fi, longevity, famstarts,
            alltrigs, barpad, plotformat, binsize_hist, binsize_occur,
            mintime, minplot, fixedheight, filepath, htmltitle, divtitle, opt)


def assemble_bokeh_timeline(ftable, rtimes, fi, longevity, famstarts,
        alltrigs, barpad, plotformat, binsize_hist, binsize_occur,
        mintime, minplot, fixedheight, filepath, htmltitle, divtitle, opt):
    """
    Assembles an interactive HTML timeline with given parameters using Bokeh.
    
    Parameters
    ----------
    ftable : Table object
        Handle to the Families table.
    rtimes : float ndarray
        Times of repeaters as matplotlib dates.
    fi : float ndarray
        Frequency index values for repeaters.
    longevity : float ndarray
        Longevity values for all families.
    famstarts : float ndarray
        Start times of all families as matplotlib dates.
    alltrigs : float ndarray
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
    opt : Options object
        Describes the run parameters.
    """
    
    plot_types = plotformat.replace('+',',').split(',')
    plots = []
    tabtitles = []
    maxtime = np.max(alltrigs)
    
    # Create each of the subplots specified in the configuration file
    for p in plot_types:
        
        if p == 'eqrate':
            # Plot EQ Rates (Repeaters and Orphans)
            plots.append(subplot_rate(alltrigs, rtimes, binsize_hist, mintime,
                                                                maxtime, opt))
            tabtitles = tabtitles+['Event Rate']
        
        elif p == 'fi':
            # Plot Frequency Index
            plots.append(subplot_fi(alltrigs, rtimes, fi, mintime, maxtime,
                                                                         opt))
            tabtitles = tabtitles+['FI']
        
        elif p == 'longevity':
            # Plot Cluster Longevity
            plots.append(subplot_longevity(alltrigs, famstarts, longevity,
                                               mintime, maxtime, barpad, opt))
            tabtitles = tabtitles+['Longevity']
        
        elif p == 'occurrence':
            # Plot family occurrence
            plots.append(subplot_occurrence(alltrigs, rtimes, famstarts,
                             longevity, fi, ftable, mintime, maxtime, minplot,
                             binsize_occur, barpad, 'rate', fixedheight, opt))
            tabtitles = tabtitles+['Occurrence (Color by Rate)']
        
        elif p == 'occurrencefi':
            # Plot family occurrence with color by FI
            plots.append(subplot_occurrence(alltrigs, rtimes, famstarts,
                             longevity, fi, ftable, mintime, maxtime, minplot,
                             binsize_occur, barpad, 'fi', fixedheight, opt))
            tabtitles = tabtitles+['Occurrence (Color by FI)']
        
        else:
            print('{} is not a valid plot type. Moving on.'.format(p))
    
    # Set ranges
    for p in plots:
        p.x_range = plots[0].x_range
    
    # Add annotations
    for p in plots:
        p = add_bokeh_annotations(p, opt)
    
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


def bokeh_figure(**kwargs):
    """
    Builds foundation for the bokeh subplots.
    
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


def subplot_rate(alltrigs, rtimes, binsize_hist, mintime, maxtime, opt,
                                                      useBokeh=True, ax=None):
    """
    Creates subplot for rate of orphans and repeaters.
    
    Parameters
    ----------
    alltrigs : float ndarray
        Array containing times of all triggers
    rtimes : float ndarray
        Times of repeaters as matplotlib dates.
    binsize_hist : float
        Width (in days) of time bins for rate plot histogram.
    mintime : float
        Minimum time to plot as matplotlib date.
    maxtime : float
        Maximum time to plot as matplotlib date.
    opt : Options object
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
    
    hr_days = 'Day Bin' if binsize_hist>=1 else 'Hour Bin'
    if binsize_hist >= 1:
        title = 'Repeaters vs. Orphans by {:.1f} Day Bin'.format(binsize_hist)
    else:
        title = 'Repeaters vs. Orphans by {:.1f} Hour Bin'.format(
                                                              binsize_hist*24)
    
    # Create histogram of events/dybin
    histT, hT = np.histogram(alltrigs, bins=np.arange(mintime,
        maxtime+binsize_hist, binsize_hist))
    histR, hR = np.histogram(rtimes, bins=np.arange(mintime,
        maxtime+binsize_hist, binsize_hist))
    
    if useBokeh:
        fig = bokeh_figure(title=title)
        fig.yaxis.axis_label = 'Events'
        fig.line(matplotlib.dates.num2date(hT[0:-1]+dt_offset), histT-histR,
            color='black', legend_label='Orphans')
        fig.line(matplotlib.dates.num2date(hR[0:-1]+dt_offset), histR,
            color='red', legend_label='Repeaters', line_width=2)
        fig.legend.location = 'top_left'
        
        return fig
    
    else:
        ax.plot(matplotlib.dates.num2date(hT[0:-1]+dt_offset), histT-histR,
            color='black', label='Orphans', lw=0.5)
        ax.plot(matplotlib.dates.num2date(hR[0:-1]+dt_offset), histR,
            color='red', label='Repeaters', lw=2)
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_ylabel('Events', style='italic')
        ax.set_xlabel('Date', style='italic')
        ax.legend(loc='upper left', frameon=False)
        
        return ax


def subplot_fi(alltrigs, rtimes, fi, mintime, maxtime, opt, useBokeh=True,
                                                                     ax=None):
    """
    Creates subplot for frequency index scatterplot.
    
    Parameters
    ----------
    alltrigs : float ndarray
        Times of all triggers as matplotlib dates.
    rtimes : float ndarray
        Times of repeaters as matplotlib dates.
    fi : float ndarray
        Frequency index values for repeaters.
    mintime : float
        Minimum time to plot as matplotlib date.
    maxtime : float
        Maximum time to plot as matplotlib date.
    opt : Options object
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
        fig.circle(matplotlib.dates.num2date(np.max(alltrigs)), 0,
            line_alpha=0, fill_alpha=0)
    else:
        ax.set_title('Frequency Index', loc='left', fontweight='bold')
        ax.set_ylabel('FI', style='italic')
        ax.set_xlabel('Date', style='italic')
    
    idxs = np.where((rtimes>=mintime) & (rtimes<=maxtime))[0]
    
    if useBokeh:
        fig.circle(matplotlib.dates.num2date(rtimes[idxs]), fi[idxs],
            color='red', line_alpha=0, size=3, fill_alpha=0.5)
        return fig
    else:
        ax.scatter(matplotlib.dates.num2date(rtimes[idxs]), fi[idxs], 2,
            c='red', alpha=0.25)
        # Need to call get_ylim() or y-limits sometimes freak out
        axlims = ax.get_ylim()
        return ax


def subplot_longevity(alltrigs, famstarts, longevity, mintime, maxtime,
                                         barpad, opt, useBokeh=True, ax=None):
    """
    Creates subplot for longevity.
    
    Parameters
    ----------
    alltrigs : float ndarray
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
    opt : Options object
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
            y_range=[0.01, np.sort(alltrigs)[-1]-np.sort(alltrigs)[0]],
            title='Cluster Longevity')
        fig.yaxis.axis_label = 'Days'
        # Always plot at least one invisible point
        fig.circle(matplotlib.dates.num2date(np.max(alltrigs)), 1,
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
                    (matplotlib.dates.num2date(x1),
                    matplotlib.dates.num2date(x2))),
                    y=np.array((longevity[n], longevity[n]))))
                fig.add_glyph(source, Line(x='x', y='y', line_color='red',
                    line_alpha=0.5))
            else:
                ax.semilogy(np.array((matplotlib.dates.num2date(x1),
                    matplotlib.dates.num2date(x2))),
                    np.array((longevity[n], longevity[n])), 'red', alpha=0.75,
                                                                       lw=0.5)
            
            if add_larrow:
                if useBokeh:
                    fig.add_layout(Arrow(end=VeeHead(size=5, fill_color='red',
                        line_color='red', line_alpha=0.5), line_alpha=0,
                        x_start=matplotlib.dates.num2date(famstarts[n] + \
                        longevity[n]), x_end=matplotlib.dates.num2date(
                        mintime - barpad), y_start=longevity[n],
                        y_end=longevity[n]))
                else:
                    ax.annotate('', xy=(matplotlib.dates.num2date(
                        mintime - barpad), longevity[n]), xytext=(
                        matplotlib.dates.num2date(mintime - 2*barpad),
                        longevity[n]), arrowprops=dict(arrowstyle='<-',
                        color='red', alpha=0.75))
            
            if add_rarrow:
                ax.annotate('', xy=(matplotlib.dates.num2date(maxtime + \
                    barpad), longevity[n]), xytext=(matplotlib.dates.num2date(
                    maxtime + 2*barpad), longevity[n]), arrowprops=dict(
                    arrowstyle='<-', color='red', alpha=0.75))
    
    if useBokeh:
        return fig
    else:
        return ax


def subplot_occurrence(alltrigs, rtimes, famstarts, longevity, fi, ftable,
    mintime, maxtime, minplot, binsize_occur, barpad, colorby, fixedheight,
                                                 opt, useBokeh=True, ax=None):
    """
    Creates subplot for family occurrence.
    
    Parameters
    ----------
    alltrigs : float ndarray
        Times of all triggers as matplotlib dates.
    rtimes : float ndarray
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
    opt : Options object
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
            ' ({}+ Members)'.format(minplot) if minplot>0 else '')
        # Always plot at least one invisible point
        fig.circle(matplotlib.dates.num2date(np.max(alltrigs)), 0,
            line_alpha=0, fill_alpha=0)
    else:
        ax.set_title('Occurrence Timeline', loc='left', fontweight='bold')
        ax.set_ylabel('Cluster by Date' + (
            ' ({}+ Members)'.format(minplot) if minplot>2 else ''),
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
        
        # Create histogram of events/hour
        hist, h = np.histogram(rtimes[members], bins=np.arange(min(
            rtimes[members]), max(rtimes[members]+binsize_occur),
            binsize_occur))
        if useBokeh:
            d1 = matplotlib.dates.num2date(h[np.where(hist>0)])
            d2 = matplotlib.dates.num2date(h[np.where(hist>0)]+binsize_occur)
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
                # Find indicies of rtimes[members] within bins
                idxs = np.where(np.logical_and(rtimes[members] >= h[i],
                                      rtimes[members] < h[i] + binsize_occur))
                # Sum fi for those events
                fisum[i] = np.sum(fi[members[idxs]])
            # Convert to mean fi
            histfi = fisum/hist
            ind = [int(max(min(255,255*(i-opt.fispanlow)/(opt.fispanhigh - \
                                         opt.fispanlow)), 0)) for i in histfi]
        
        colors = [bokehpalette[i] for i in ind]
        
        if len(rtimes[members]) >= minplot:
            
            if max(rtimes[members])>mintime:
                
                add_line, add_larrow, add_rarrow, x1, x2 = determine_lines(
                    mintime, maxtime, barpad, famstarts[clustNum],
                    longevity[clustNum])
                
                if add_line:
                    if useBokeh:
                        source = ColumnDataSource(dict(
                            x=np.array(
                            (matplotlib.dates.num2date(x1),
                            matplotlib.dates.num2date(x2))),
                            y=np.array((n,n))))
                        fig.add_glyph(source, Line(x='x', y='y',
                            line_color='black'))
                    else:
                        ax.plot(np.array((matplotlib.dates.num2date(x1),
                            matplotlib.dates.num2date(x2))),
                            np.array((n,n)), 'black', lw=0.5, zorder=0)
                    
                    if add_larrow:
                        if useBokeh:
                            fig.add_layout(Arrow(end=VeeHead(size=5,
                                fill_color='black', line_color='black'),
                                line_alpha=0,
                                x_start=matplotlib.dates.num2date(
                                famstarts[clustNum]+longevity[clustNum]),
                                x_end=matplotlib.dates.num2date(mintime - \
                                barpad), y_start=n, y_end=n))
                        else:
                            ax.annotate('', xy=(matplotlib.dates.num2date(
                                mintime-barpad),n),xytext=(
                                matplotlib.dates.num2date(
                                mintime-2*barpad),n), arrowprops=dict(
                                arrowstyle='<-', color='black', alpha=1))
                    
                    if add_rarrow:
                        ax.annotate('', xy=(matplotlib.dates.num2date(
                            maxtime + barpad), n), xytext=(
                            matplotlib.dates.num2date(maxtime+2*barpad),
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
                        x = matplotlib.dates.num2date(np.array(d1)[i])
                        w = datetime.timedelta(binsize_occur)
                        if (x >= matplotlib.dates.num2date(mintime)) and (
                            x <= matplotlib.dates.num2date(maxtime)):
                            ax.add_patch(matplotlib.patches.Rectangle((x,
                                n - 0.3), w,0.6, facecolor=colors[i],
                                edgecolor=None, fill=True))
                            if x+w > matplotlib.dates.num2date(x2):
                                x2 = np.array(d1)[i] + binsize_occur
                
                # Add label
                if useBokeh:
                    label = Label(x=max(d2), y=n, text='  {}'.format(len(
                        rtimes[members])), text_font_size='9pt',
                        text_baseline='middle')
                    fig.add_layout(label)
                else:
                    ax.annotate('  {}'.format(len(rtimes[members])), (
                        matplotlib.dates.num2date(x2),n), va='center',
                        ha='left')
                
                if useBokeh:
                    # Build source for hover patches
                    fnum = clustNum
                    xs.append([matplotlib.dates.num2date(max(min(
                                   rtimes[members]), mintime) - barpad),
                               matplotlib.dates.num2date(max(min(
                                   rtimes[members]), mintime) - barpad),
                               matplotlib.dates.num2date(
                                   max(rtimes[members]) + barpad),
                               matplotlib.dates.num2date(max(
                                   rtimes[members]) + barpad)])
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
                    opt), border_line_color='#eeeeee', location=(7,cloc1),
                    orientation='horizontal', width=150, height=15,
                    title='Mean Frequency Index', padding=15,
                    major_tick_line_alpha=0)
        
        fig.add_layout(color_bar)
        
        return fig
    
    else:
        return ax


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
        legtext = '{} Hours'.format(binsize_occur*24)
    else:
        legtext = '{} Days'.format(binsize_occur)
    
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


def determine_color_mapper_fi(opt):
    """
    Determines color map for occurrencefi plot based on opt.fispan.
    
    Parameters
    ----------
    opt : Options object
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
                                     low=opt.fispanlow, high=opt.fispanhigh)
    
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


### PDF OVERVIEW ###

def assemble_pdf_overview(rtable, ftable, ttable, tmin, tmax, binsize,
                                    minmembers, occurheight, plotformat, opt):
    """
    Generate a static PDF version of the overview plot for publication.
    
    Plot is saved in the usual outputs folder.
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    ttable : Table object
        Handle to the Triggers table.
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
    opt : Options object
        Describes the run parameters.
    """
    
    # !!! Add functionality to separate binsizes
    binsize_hist = binsize
    binsize_occur = binsize
    
    # !!! This variable technically still needs the windowStart added to it
    # !!! Same for alltrigs needing opt.ptrig...
    rtimes = rtable.cols.startTimeMPL[:]# + rtable.cols.windowStart[:] / \
                                        #                86400. / opt.samprate
    fi = np.nanmean(rtable.cols.FI[:], axis=1)
    longevity = ftable.cols.longevity[:]
    famstarts = ftable.cols.startTime[:]
    alltrigs = ttable.cols.startTimeMPL[:]# + opt.ptrig
    
    # Custom tmin/tmax functionality
    if tmin:
        mintime = tmin
    else:
        mintime = min(alltrigs)
    
    if tmax:
        maxtime = tmax
    else:
        maxtime = max(alltrigs)
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
    axref = subplot_rate(alltrigs, rtimes, binsize_hist, mintime, maxtime,
        opt, useBokeh=False, ax=axref)
    
    fig = plt.figure(figsize=(9, figheight))
    
    pnum = 0
    for p in plot_types:
        
        if p == 'eqrate':
            ### Rate ###
            ax = fig.add_subplot(nsub, 1, pnum+1, sharex=axref)
            ax = add_pdf_annotations(ax, mintime, maxtime, opt)
            ax = subplot_rate(alltrigs, rtimes, binsize_hist, mintime,
                maxtime, opt, useBokeh=False, ax=ax)
            pnum = pnum + 1
        
        elif p == 'fi':
            ### FI ###
            ax = fig.add_subplot(nsub, 1, pnum+1, sharex=axref)
            ax = add_pdf_annotations(ax, mintime, maxtime, opt)
            ax = subplot_fi(alltrigs, rtimes, fi, mintime, maxtime, opt,
                useBokeh=False, ax=ax)
            pnum = pnum + 1
        
        elif p == 'occurrence':
            ### Occurrence ###
            ax = fig.add_subplot(nsub, 1, (pnum+1, pnum+occurheight),
                sharex=axref)
            ax = add_pdf_annotations(ax, mintime, maxtime, opt)
            ax = subplot_occurrence(alltrigs, rtimes, famstarts, longevity,
                fi, ftable, mintime, maxtime, minmembers, binsize_occur,
                barpad, 'rate', True, opt, useBokeh=False, ax=ax)
            add_pdf_colorbar(fig, figheight, pnum, nsub, 'rate',
                binsize_occur, opt)
            pnum = pnum + occurheight
        
        elif p == 'occurrencefi':
            ### Occurrence ###
            ax = fig.add_subplot(nsub, 1, (pnum+1, pnum+occurheight),
                sharex=axref)
            ax = add_pdf_annotations(ax, mintime, maxtime, opt)
            ax = subplot_occurrence(alltrigs, rtimes, famstarts, longevity,
                fi, ftable, mintime, maxtime, minmembers, binsize_occur,
                barpad, 'fi', True, opt, useBokeh=False, ax=ax)
            add_pdf_colorbar(fig, figheight, pnum, nsub, 'fi',
                binsize_occur, opt)
            pnum = pnum + occurheight
        
        elif p == 'longevity':
            ### Longevity ###
            ax = fig.add_subplot(nsub, 1, pnum+1, sharex=axref)
            ax = add_pdf_annotations(ax, mintime, maxtime, opt)
            ax = subplot_longevity(alltrigs, famstarts, longevity, mintime,
                maxtime, barpad, opt, useBokeh=False, ax=ax)
            pnum = pnum + 1
        
        else:
            print('{} is not a valid plot type. Moving on.'.format(p))
    
    # Clean up and save
    plt.tight_layout()
    plt.savefig('{}{}/overview.pdf'.format(opt.outputPath, opt.groupName))
    plt.close(fig)


def add_pdf_colorbar(fig, figheight, pnum, nsub, colorby, binsize_occur, opt):
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
    opt : Options object
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
        cax.set_xticklabels((opt.fispanlow, np.mean((opt.fispanlow,
            opt.fispanhigh)), opt.fispanhigh))
    cax.set_frame_on(False)
    cax.tick_params(length=0)


def add_pdf_annotations(ax, mintime, maxtime, opt):
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
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    ax : Axis object
    """
    
    if opt.anotfile != '':
        
        annotations = pd.read_csv(opt.anotfile)
        
        for a in range(len(annotations)):
            
            plotdate = matplotlib.dates.date2num(np.datetime64(
                annotations['Time'][a]))
            
            # If within bounds, add to figure
            if (plotdate >= mintime) and (plotdate <= maxtime):
                ax.axvline(plotdate, color=annotations['Color'][a],
                    lw=annotations['Weight'][a],
                    ls=annotations['Line Type'][a],
                    alpha=annotations['Alpha'][a], zorder=-1)
    
    return ax


def add_bokeh_annotations(fig, opt):
    """
    Plots annotations on bokeh figure.
    
    Parameters
    ----------
    fig : Figure object
        Handle to the bokeh figure.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    fig : Figure object
    """
    
    if opt.anotfile != '':
        
        annotations = pd.read_csv(opt.anotfile)
        
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


def add_horizontal_annotations(ax, evtimes, opt):
    """
    Plots annotations horizontally across an image (e.g., waveforms, matrix).
    
    Parameters
    ----------
    ax : Axis object
        Handle to the matplotlib axis.
    evtimes : float ndarray
        Sorted array of event times plotted on each row of the image.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    ax : Axis object
    """
    
    if opt.anotfile!='':
    
        annotations = pd.read_csv(opt.anotfile)
    
        for a in range(len(annotations)):
            
            # Translate from date to vertical position in the image
            vertical_location = np.interp(matplotlib.dates.date2num(
                pd.to_datetime(annotations['Time'][a])),
                evtimes, np.array(range(len(evtimes))))
            
            # Plot if within the time span of the image
            if vertical_location != 0:
                ax.axhline(np.floor(vertical_location)+0.5, color='k',
                           linewidth=annotations['Weight'][a]/2.,
                           linestyle=annotations['Line Type'][a])
    
    return ax


### FAMILY PAGES ###

def create_core_images(rtable, ftable, opt):
    """
    Plots core waveforms as *.png files in the clusters folder.
    
    Used for hovering in timeline and header for HTML family pages.
    
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    opt : Options object
        Describes the run parameters.
    """
    
    # Deal with renaming files to reduce plotting time overhead
    for n in range(len(ftable))[::-1]:
        if ftable.cols.lastprint[n] != n and ftable.cols.printme[n] == 0:
            os.rename('{}{}/clusters/{}.png'.format(opt.outputPath,
                opt.groupName, ftable.cols.lastprint[n]),
                '{}{}/clusters/{}.png.tmp'.format(opt.outputPath,
                opt.groupName, n))
            os.rename('{}{}/clusters/fam{}.png'.format(opt.outputPath,
                opt.groupName, ftable.cols.lastprint[n]),
                '{}{}/clusters/fam{}.png.tmp'.format(opt.outputPath,
                opt.groupName, n))
    
    # Iterate and plot
    cores = rtable[ftable.cols.core[:]]
    n = -1
    for r in cores:
        n = n+1
        if ftable.cols.printme[n] == 1:
            data = prep_wiggle(r['waveform'], opt.printsta, r['windowStart'],
                               r['windowAmp'][opt.printsta], opt)
            wiggle_plot(data, (5, 1), '{}{}/clusters/{}.png'.format(
                opt.outputPath, opt.groupName, n), opt)


def prep_wiggle(waveform, sta, window_start, normalize_amplitude, opt):
    """
    Cuts window around trigger time and normalizes the waveform for plotting.
    
    The plotting window is always 2*opt.winlen samples, with half opt.winlen
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
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    data : float ndarray
        Clipped, normalized, and trimmed waveform for single station/channel.
    """
    
    # Determine window
    minsample = window_start - int(0.5*opt.winlen)
    maxsample = window_start + int(1.5*opt.winlen)
    
    # Trim out data for station 's'
    data = waveform[sta*opt.wshape:(sta+1)*opt.wshape]
    
    # Trim window
    data = data[minsample:maxsample]
    
    # Normalize
    if normalize_amplitude > 0:
        data = data / normalize_amplitude
    else:
        data = data / np.max(np.abs(data)+1e-12)
    
    # Clip
    data[data>1] = 1
    data[data<-1] = -1
    
    return data


def wiggle_plot(data, figsize, outfile, opt):
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
    opt : Options object
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


def create_family_images(rtable, ftable, ctable, opt):
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
    ctable : Table object
        Handle to the Correlation table.
    opt : Options object
        Describes the run parameters.
    """
    
    # Load into memory
    startTimeMPL = rtable.cols.startTimeMPL[:]
    windowAmp = rtable.cols.windowAmp[:][:,opt.printsta]
    windowStart = rtable.cols.windowStart[:]
    fi = rtable.cols.FI[:]
    ids = rtable.cols.id[:]
    id1 = ctable.cols.id1[:]
    id2 = ctable.cols.id2[:]
    ccc = ctable.cols.ccc[:]
    
    for fnum in range(ftable.attrs.nClust):
        
        if ftable.cols.printme[fnum] != 0:
            
            assemble_family_image(rtable, ftable, ctable, startTimeMPL,
                windowAmp, windowStart, fi, ids, id1, id2, ccc, 'png', 100,
                fnum, 0, 0, opt)


def assemble_family_image(rtable, ftable, ctable, startTimeMPL, windowAmp,
    windowStart, fi, ids, id1, id2, ccc, oformat, dpi, fnum, tmin, tmax, opt):
    """
    Creates a multi-paneled family plot for the specified family 'fnum'.
    
    This function allows some flexibility in the output format
    (e.g., .png, .pdf) as well as resolution. Many inputs are columns from
    rtable to reduce overhead back to the file when calling this function for
    many families.
    
    Current format for the image is the following:
        Top row: Ordered waveforms, stacked FFT.
        Second row: Timeline of amplitude.
        Third row: Timeline of event spacing.
        Last row: Correlation with time relative to best-correlated event
            (has most measurements in Correlation table), with core event in
            black and events with missing correlation values as open
            circles (either were never correlated or were below threshold).
    
    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    ctable : Table object
        Handle to the Correlation table.
    startTimeMPL : float ndarray
        'startTimeMPL' column from rtable.
    windowAmp : float ndarray
        'windowAmp' column from rtable for single station.
    windowStart : int ndarray
        'windowStart' column from rtable.
    fi : float ndarray
        Frequency index values for repeaters.
    ids : int ndarray
       'id' column from rtable
    id1 : int ndarray
       'id1' column from ctable
    id2 : int ndarray
        'id2' column from ctable
    ccc : float ndarray
        Cross-correlation values from ctable.
    oformat : str
        output file format (e.g., 'png' or 'pdf')
    dpi : int
        Dots per inch resolution of raster file.
    fnum : int
        Family number to plot.
    tmin : 
        Minimum time on timeline axes as matplotlib date (0 for default tmin).
    tmax : 
        Maximum time on timeline axes as matplotlib date (0 for default tmax).
    opt : Options object
        Describes the run parameters.
    """
    
    fam = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')
    famtable = rtable[fam]
    corenum = ftable[fnum]['core']
    core = rtable[corenum]
    
    # Station names
    stas = opt.station.split(',')
    chas = opt.channel.split(',')
    
    # Prep catalog
    catalogind = np.argsort(startTimeMPL[fam])
    catalog = startTimeMPL[fam][catalogind]
    spacing = np.diff(catalog)*24
    coreind = np.where(fam==corenum)[0][0]
    
    fig = plt.figure(figsize=(10, 12))
    
    # Plot waveforms
    ax1 = fig.add_subplot(9, 3, (1,8))
    
    time_vector = np.arange(-0.5*opt.winlen/opt.samprate,
                      1.5*opt.winlen/opt.samprate, 1/opt.samprate)
    
    # If only one station, plot all aligned waveforms
    if opt.nsta==1:
        
        # Prepare data in matrix (row for each event)
        data = np.zeros((len(fam), int(opt.winlen*2)))
        for n, r in enumerate(famtable):
            data[n, :] = prep_wiggle(r['waveform'], 0, r['windowStart'],
                                     r['windowAmp'][0], opt)
        
        # Plot
        if len(fam) > 12:
            ax1.imshow(data, aspect='auto', vmin=-1, vmax=1, cmap='RdBu',
                interpolation='nearest', extent=[np.min(time_vector),
                np.max(time_vector), n + 0.5, -0.5])
        else:
            for n in range(len(fam)):
                dat=data[n,:]
                ax1.plot(time_vector,dat/2-n,'k',linewidth=0.25)
    
    # Otherwise, plot cores and stacks from all stations
    else:
        
        
        for s in range(opt.nsta):
            
            # Prepare stack
            data_stack = np.zeros((int(opt.winlen*2),))
            waveform = famtable['waveform']
            for n in range(len(fam)):
                data_stack += prep_wiggle(waveform[n], s, windowStart[fam[n]],
                                              famtable['windowAmp'][n,s], opt)
            
            data_stack = data_stack/(np.max(np.abs(data_stack))+1e-12)
            data_stack[data_stack>1] = 1
            data_stack[data_stack<-1] = -1
            
            # Prepare core
            data_core = prep_wiggle(core['waveform'], s, core['windowStart'],
                                    core['windowAmp'][s], opt)
            
            # Plot
            ax1.plot(time_vector,data_stack-1.75*s,'r',linewidth=1)
            ax1.plot(time_vector, data_core - 1.75*s, 'k', linewidth=0.25)
            
            # Label stations/channels on left
            ax1.text(np.min(time_vector)-0.1,-1.75*s,'{}\n{}'.format(stas[s],
                chas[s]), horizontalalignment='right',
                verticalalignment='center')
    
    ax1.axvline(x=-0.1*opt.winlen/opt.samprate, color='k', ls='dotted')
    ax1.axvline(x=0.9*opt.winlen/opt.samprate, color='k', ls='dotted')
    ax1.get_yaxis().set_visible(False)
    ax1.set_xlim((np.min(time_vector),np.max(time_vector)))
    if opt.nsta > 1:
        ax1.set_ylim((-1.75*s-1,1))
    ax1.set_xlabel('Time Relative to Trigger (seconds)', style='italic')
    
    # Plot mean FFT
    ax2 = fig.add_subplot(9, 3, (3,9))
    ax2.set_xlabel('Frequency (Hz)', style='italic')
    ax2.get_yaxis().set_visible(False)
    freq = np.linspace(0,opt.samprate/2,int(opt.winlen/2))
    fftc = np.zeros((int(opt.winlen/2),))
    fftm = np.zeros((int(opt.winlen/2),))
    for s in range(opt.nsta):
        fft = np.abs(np.real(core['windowFFT'][int(
            s*opt.winlen):int(s*opt.winlen+opt.winlen/2)]))
        fft = fft/(np.amax(fft)+1.0/1000)
        fftc = fftc+fft
        ffts = np.mean(np.abs(np.real(famtable['windowFFT'][:,int(
            s*opt.winlen):int(s*opt.winlen+opt.winlen/2)])),axis=0)
        fftm = fftm + ffts/(np.amax(ffts)+1.0/1000)
    ax2.plot(freq,fftm,'r', linewidth=1)
    ax2.plot(freq,fftc,'k', linewidth=0.25)
    ax2.set_xlim(0,opt.fmax*1.5)
    ax2.legend(['Stack','Core'], loc='upper right', frameon=False)
    
    # Set min/max for plotting
    if opt.amplims == 'family':
        windowAmpFam = windowAmp[fam[catalogind]]
        try:
            ymin = 0.5*np.min(windowAmpFam[np.nonzero(windowAmpFam)])
            ymax = 2*np.max(windowAmpFam)
        except ValueError:
            # Use global if all zeros
            ymin = 0.5*np.min(windowAmp[np.nonzero(windowAmp)])
            ymax = 2*np.max(windowAmp)
    else:
        # Use global maximum/minimum
        ymin = 0.5*np.min(windowAmp[np.nonzero(windowAmp)])
        ymax = 2*np.max(windowAmp)
    
    # Plot amplitude timeline
    ax3 = fig.add_subplot(9, 3, (10,15))
    ax3.plot_date(catalog, windowAmp[fam[catalogind]],
            'ro', alpha=0.5, markeredgecolor='r', markeredgewidth=0.5,
            markersize=3)
    ax3.plot_date(catalog[coreind], windowAmp[fam[catalogind]][coreind],
            'ko', markeredgecolor='k', markeredgewidth=0.5,
            markersize=3)
    if tmin and tmax:
        ax3.set_xlim(tmin, tmax)
    elif tmin:
        ax3.set_xlim(tmin, ax3.get_xlim()[1])
    elif tmax:
        ax3.set_xlim(ax3.get_xlim()[0], tmax)
    myFmt = matplotlib.dates.DateFormatter('%Y-%m-%d\n%H:%M')
    ax3.xaxis.set_major_formatter(myFmt)
    ax3.set_ylim(ymin, ymax)
    ax3.margins(0.05)
    ax3.set_ylabel('Amplitude (Counts)', style='italic')
    ax3.set_xlabel('Date', style='italic')
    ax3.set_yscale('log')
    
    # Plot spacing timeline
    ax4 = fig.add_subplot(9, 3, (16,21))
    ax4.plot_date(catalog[1:], spacing, 'ro', alpha=0.5, markeredgecolor='r',
        markeredgewidth=0.5, markersize=3)
    if coreind>0:
        ax4.plot_date(catalog[coreind], spacing[coreind-1], 'ko',
            markeredgecolor='k', markeredgewidth=0.5, markersize=3)
    if tmin and tmax:
        ax4.set_xlim(tmin, tmax)
    elif tmin:
        ax4.set_xlim(tmin, ax4.get_xlim()[1])
    elif tmax:
        ax4.set_xlim(ax4.get_xlim()[0], tmax)
    myFmt = matplotlib.dates.DateFormatter('%Y-%m-%d\n%H:%M')
    ax4.xaxis.set_major_formatter(myFmt)
    ax4.set_xlim(ax3.get_xlim())
    ax4.set_ylim(1e-3, max(spacing)*2)
    ax4.margins(0.05)
    ax4.set_ylabel('Time since previous event (hours)', style='italic')
    ax4.set_xlabel('Date', style='italic')
    ax4.set_yscale('log')
    
    # Plot correlation timeline
    idf = ids[fam]
    ix = np.where(np.in1d(id2,idf))
    C = np.eye(len(idf))
    r1 = [np.where(idf==xx)[0][0] for xx in id1[ix]]
    r2 = [np.where(idf==xx)[0][0] for xx in id2[ix]]
    C[r1,r2] = ccc[ix]
    C[r2,r1] = ccc[ix]
    Cprint = C[np.argmax(np.sum(C,0)),:]
    
    ax5 = fig.add_subplot(9, 3, (22,27))
    ax5.plot_date(catalog, Cprint, 'ro', alpha=0.5,
        markeredgecolor='r', markeredgewidth=0.5, markersize=3)
    ax5.plot_date(catalog[coreind], Cprint[coreind], 'ko',
        markeredgecolor='k', markeredgewidth=0.5, markersize=3)
    Cprint[Cprint<opt.cmin] = opt.cmin
    Cprint[Cprint>opt.cmin] = np.nan
    ax5.plot_date(catalog, Cprint, 'wo', alpha=0.5,
        markeredgecolor='r', markeredgewidth=0.5)
    ax5.plot_date(catalog[np.where(fam==corenum)[0][0]], Cprint[coreind], 'wo',
        markeredgecolor='k', markeredgewidth=0.5, markersize=3)
    if tmin and tmax:
        ax5.set_xlim(tmin, tmax)
    elif tmin:
        ax5.set_xlim(tmin, ax5.get_xlim()[1])
    elif tmax:
        ax5.set_xlim(ax5.get_xlim()[0], tmax)
    myFmt = matplotlib.dates.DateFormatter('%Y-%m-%d\n%H:%M')
    ax5.xaxis.set_major_formatter(myFmt)
    ax5.set_xlim(ax3.get_xlim())
    ax5.set_ylim(opt.cmin-0.02, 1.02)
    ax5.margins(0.05)
    ax5.set_ylabel('Cross-correlation coefficient',
                   style='italic')
    ax5.set_xlabel('Date', style='italic')
    
    plt.tight_layout()
    plt.savefig('{}{}/clusters/fam{}.{}'.format(opt.outputPath, opt.groupName,
        fnum, oformat), dpi=dpi)
    plt.close(fig)


def create_family_html(rtable, ftable, external_catalogs, opt):
    """
    Creates the HTML for the individual family pages.
    
    HTML will hold navigation, images, and basic statistics. May also include
    location information if an external catalog is queried.
    
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    external_catalogs : list of DataFrame objects
        External catalogs to check against, with 'Arrivals_' columns.
    opt : Options object
        Describes the run parameters.
    """
    
    # Load into memory
    startTime = rtable.cols.startTime[:]
    startTimeMPL = rtable.cols.startTimeMPL[:]
    windowStart = rtable.cols.windowStart[:]
    windowAmp = rtable.cols.windowAmp[:][:,opt.printsta]
    fi = rtable.cols.FI[:]
    fmembers = ftable.cols.members[:]
    fcores = ftable.cols.core[:]
    printme = ftable.cols.printme[:]
    lastprint = ftable.cols.lastprint[:]
    
    for fnum in range(ftable.attrs.nClust):
        
        fam = np.fromstring(fmembers[fnum], dtype=int, sep=' ')
        core = fcores[fnum]
        
        # Prep catalog
        catalogind = np.argsort(startTimeMPL[fam])
        catalog = startTimeMPL[fam][catalogind]
        longevity = ftable[fnum]['longevity']
        spacing = np.diff(catalog)*24
        minind = fam[catalogind[0]]
        maxind = fam[catalogind[-1]]
        coreind = np.where(fam==core)[0][0]
        
        if printme[fnum] != 0 or lastprint[fnum] != fnum:
            if fnum>0:
                prev = "<a href='{0}.html'>&lt; Cluster {0}</a>".format(
                    fnum-1)
            else:
                prev = " "
            if fnum<len(ftable)-1:
                next = "<a href='{0}.html'>Cluster {0} &gt;</a>".format(
                    fnum+1)
            else:
                next = " "
            # Now write a simple HTML file to show image and catalog
            with open('{}{}/clusters/{}.html'.format(opt.outputPath,
                    opt.groupName, fnum), 'w') as f:
                f.write("""
                <html><head><title>{1} - Cluster {0}</title>
                </head><style>
                a {{color:red;}}
                body {{font-family:Helvetica; font-size:12px}}
                h1 {{font-size: 20px;}}
                </style>
                <body><center>
                {10} &nbsp; | &nbsp; {11}</br>
                <h1>Cluster {0}</h1>
                <img src="{0}.png" width=500 height=100></br></br>
                    Number of events: {2}</br>
                    Longevity: {5:.2f} days</br>
                    Mean event spacing: {7:.2f} hours</br>
                    Median event spacing: {8:.2f} hours</br>
                    Mean Frequency Index: {9:.2f}<br></br>
                    First event: {3}</br>
                    Core event: {6}</br>
                    Last event: {4}</br>
                <img src="fam{0}.png"></br>
                """.format(fnum, opt.title, len(fam), (UTCDateTime(
                    startTime[minind]) + windowStart[
                    minind]/opt.samprate).isoformat(),
                    (UTCDateTime(startTime[maxind]) + windowStart[
                    maxind]/opt.samprate).isoformat(), longevity,
                    (UTCDateTime(startTime[core]) + windowStart[
                    core]/opt.samprate).isoformat(), np.mean(spacing),
                    np.median(spacing), np.mean(np.nanmean(fi[fam],
                    axis=1)),prev,next))
                
                if opt.checkComCat:
                    match_external(windowAmp, ftable, fnum, f, startTime,
                        windowStart, external_catalogs, opt)
                
                f.write("""
                </center></body></html>
                """)


def match_external(windowAmp, ftable, fnum, f, startTime, windowStart,
                                                      external_catalogs, opt):
    """
    Checks repeater trigger times with arrival times from external catalog.
    
    Currently only supports checking the ANSS Comprehensive Earthquake
    Catalog (USGS ComCat). It writes these to HTML and map image files.
    
    windowAmp : float ndarray
        'windowAmp' column from rtable for single station.
    ftable : Table object
        Handle to the Families table.
    fnum : int
        Family number to check.
    f : file handle
        HTML file to write to.
    startTime : str ndarray
        'startTime' column from rtable
    windowStart : int ndarray
        'windowStart' column from rtable
    external_catalogs : list of DataFrame objects
        External catalogs to check against, with 'Arrivals_' columns.
    opt : Options object
        Describes the run parameters.
    """
    
    pc = ['Potential', 'Conflicting']
    model = TauPyModel(model="iasp91")
    mc = 0
    n = 0
    l = 0
    stalats = np.array(opt.stalats.split(',')).astype(float)
    stalons = np.array(opt.stalons.split(',')).astype(float)
    latitude_center = np.mean(stalats)
    longitude_center = np.mean(stalons)
    
    members = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')
    if opt.matchMax == 0 or opt.matchMax > len(members):
        order = np.argsort(startTime[members])
        matchstring = ('</br><b>ComCat matches (all events):</b></br>'
            '<div style="overflow-y: auto; height:100px; width:1200px;">')
    else:
        nlargest = np.argsort(windowAmp[members])[::-1][:opt.matchMax]
        members = members[nlargest]
        order = np.argsort(startTime[members])
        matchstring = ("""
            </br><b>ComCat matches ({} largest events):</b></br>'
            '<div style="overflow-y: auto; height:100px; width:1200px;">
            """).format(opt.matchMax)
    
    pc = ['Potential', 'Conflicting']
    region = ['local', 'regional', 'teleseismic']
    prestring = ['', '<div style="color:red">', '<div style="color:red">']
    poststring = ['</br>', '</div>', '</div>']
    nfound = 0
    local_lats = np.array([])
    local_lons = np.array([])
    local_deps = np.array([])
    
    for m in members[order]:
        
        t = UTCDateTime(startTime[m])+windowStart[m]/opt.samprate
        
        cflag = 0
        
        for r, cat in enumerate(external_catalogs):
            
            # Get the arrival names only from Arrival_ column names
            anames = [cat.filter(like='Arrival').columns[i].split('_')[1] \
                      for i in range(len(cat.filter(like='Arrival').columns))]
            
            # Get arrivals as a numpy array of strings
            arrivals = cat.filter(like='Arrival').to_numpy().astype(str)
            
            # Find arrivals within opt.serr seconds
            matched = np.any(((arrivals >= format(t-opt.serr)) & \
                              (arrivals <= format(t+opt.serr))),axis=1)
            
            # Subset to only those that were found
            found = arrivals[matched]
            
            if len(found) > 0:
                
                # Convert from strings to time differences
                vfunc = np.vectorize(lambda x,t:np.abs(UTCDateTime(x)-t))
                found[found=='NaN'] = 'nan' # Just in case
                found[found!='nan'] = vfunc(found[found!='nan'],t)
                
                # Convert to float, otherwise numpy complains
                found[found=='nan'] = np.nan
                found = found.astype(float)
                
                # Loop over what was found
                for i in range(len(found)):
                    
                    # Get the row and column of the best matching arrival
                    bestmatch = np.argwhere(found == np.nanmin(found))[0]
                    
                    # Write the line here...
                    catmatch = cat[matched].iloc[bestmatch[0]]
                    
                    matchstring += ('{}{} {} match: {} ({:5.6f}, {:6.6f}) '
                            '{:3.1f}km M{:3.1f} - {} - ({}) '
                            '{:4.2f} s{}').format(
                            prestring[r], pc[cflag], region[r],
                            catmatch['Time'],
                            catmatch['Latitude'], catmatch['Longitude'],
                            catmatch['Depth/km'], catmatch['Magnitude'],
                            catmatch['EventLocationName'],
                            anames[bestmatch[1]],np.nanmin(found),
                            poststring[r])
                    
                    if r == 0: # Local catalog
                        local_lats = np.append(local_lats,
                                               catmatch['Latitude'])
                        local_lons = np.append(local_lons,
                                               catmatch['Longitude'])
                        local_deps = np.append(local_deps,
                                               catmatch['Depth/km'])
                    
                    # Overwrite that match with NaNs so we get next best
                    found[bestmatch[0],:] = np.nan
                    
                    # Set cflag = 1 for 'conflicting'
                    cflag = 1
                
                nfound += 1
    
    if nfound > 0:
        
        matchstring += '</div>'
        matchstring += 'Total potential matches: {}</br>'.format(nfound)
        matchstring += 'Potential local matches: {}</br>'.format(
                                                              len(local_deps))
        
        if len(local_deps) > 0:
            
            create_local_map(local_lats, local_lons, local_deps,
                '{}{}/clusters/map{}.png'.format(opt.outputPath,
                opt.groupName, fnum), opt)
            f.write('<img src="map{}.png"></br>'.format(fnum))
            
    else:
        
        matchstring+='No matches found</br></div>'
        
    f.write(matchstring)


def create_local_map(local_lats, local_lons, local_deps, outfile, opt):
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
    opt : Options object
        Describes the run parameters.
    """
    
    stalats = np.array(opt.stalats.split(',')).astype(float)
    stalons = np.array(opt.stalons.split(',')).astype(float)
    
    # Set basemap and projection
    stamen_terrain = cimgt.StamenTerrain()
    
    extent = [np.median(local_lons) - opt.locdeg/2,
              np.median(local_lons) + opt.locdeg/2,
              np.median(local_lats) - opt.locdeg/4,
              np.median(local_lats) + opt.locdeg/4]
    
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
    scalebar_lat = 0.05*(opt.locdeg/2) + extent[2]
    scalebar_lon_left = 0.05*(opt.locdeg) + extent[0]
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


def remove_old_html(oldnClust, newnClust, opt):
    """
    Removes HTML files from deleted/moved family pages.
    
    Deletes removed family .html files that have fnum above the current
    maximum family number.
    
    oldnClust : int
        Previous number of clusters (ftable.attrs.nClust).
    newnClust : int
        New number of clusters.
    opt : Options object
        Describes the run parameters.
    """
    
    for fnum in range(newnClust, oldnClust):
        if os.path.exists('{}{}/clusters/{}.html'.format(opt.outputPath,
            opt.groupName, fnum)):
            os.remove('{}{}/clusters/{}.html'.format(opt.outputPath,
                opt.groupName, fnum))


### USER-GENERATED ###

def create_report(rtable, ftable, ctable, fnum, ordered, matrixtofile, opt):
    """
    Creates more detailed output plots for a single family in '/reports'.
    
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    ctable : Table object
        Handle to the Correlation table.
    fnum : int
        Family to be inspected.
    ordered : int
        1 if members should be ordered by OPTICS, 0 if by time.
    matrixtofile : int
        1 if correlation should be written to file.
    opt : Options object
        Describes the run parameters.
    """
    
    # !!! Functionalize this monstrosity!
    
    # Read in annotation file (if it exists)
    if opt.anotfile != '':
        annotations = pd.read_csv(opt.anotfile)
    
    # Set up variables
    fam = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')
    startTimeMPL = rtable.cols.startTimeMPL[:]
    startTime = rtable.cols.startTime[:]
    windowStart = rtable.cols.windowStart[:]
    windowAmp = rtable.cols.windowAmp[:][:,opt.printsta]
    windowAmps = rtable.cols.windowAmp[:]
    fi = rtable.cols.FI[:]
    ids = rtable.cols.id[:]
    id1 = ctable.cols.id1[:]
    id2 = ctable.cols.id2[:]
    ccc = ctable.cols.ccc[:]
    core = ftable[fnum]['core']
    catalogind = np.argsort(startTimeMPL[fam])
    catalog = startTimeMPL[fam][catalogind]
    famcat = fam[catalogind]
    longevity = ftable[fnum]['longevity']
    spacing = np.diff(catalog)*24
    minind = fam[catalogind[0]]
    maxind = fam[catalogind[-1]]
    
    idf = ids[fam]
    ix = np.where(np.in1d(id2,idf))
    C = np.eye(len(idf))
    r1 = [np.where(idf==xx)[0][0] for xx in id1[ix]]
    r2 = [np.where(idf==xx)[0][0] for xx in id2[ix]]
    C[r1,r2] = ccc[ix]
    C[r2,r1] = ccc[ix]
    
    # Copy static preview image in case cluster changes
    shutil.copy('{}{}/clusters/{}.png'.format(opt.outputPath,
                                              opt.groupName, fnum),
                '{}{}/reports/{}-report.png'.format(opt.outputPath,
                                                    opt.groupName, fnum))
    
    # Fill in full correlation matrix
    print('Computing full correlation matrix; this will take time' + \
          ' if the family is large')
    famtable = rtable[famcat]
    Cind = C[catalogind,:]
    Cind = Cind[:,catalogind]
    Cfull = Cind.copy()
    for i in range(len(famcat)-1):
        for j in range(i+1,len(famcat)):
            if Cfull[i,j]==0:
                # Compute correlation
                cor, lag, nthcor = redpy.correlation.xcorr_1x1(
                    famtable['windowCoeff'][i], famtable['windowCoeff'][j],
                    famtable['windowFFT'][i], famtable['windowFFT'][j], opt)
                Cfull[i,j] = cor
                Cfull[j,i] = cor
    
    ### BOKEH PLOTS
    oTOOLS = ['pan,box_zoom,reset,save,tap']
    
    # Amplitude vs. time on all stations with interactive show/hide
    # Set min/max for plotting
    if opt.amplims == 'family':
        windowAmpFam = windowAmps[fam[catalogind]][:]
        ymin = 0.25*np.amin(windowAmpFam[np.nonzero(windowAmpFam)])
        ymax = 4*np.amax(windowAmpFam)
    else:
        # Use global maximum
        ymin = 0.25*np.amin(windowAmps[np.nonzero(windowAmps)])
        ymax = 4*np.amax(windowAmps)
    
    o0 = figure(tools=oTOOLS, plot_width=1250, plot_height=250,
        x_axis_type='datetime',
        title='Amplitude with Time (Click name to hide)', y_axis_type='log',
        y_range=[ymin,ymax])
    o0.grid.grid_line_alpha = 0.3
    o0.xaxis.axis_label = 'Date'
    o0.yaxis.axis_label = 'Counts'
    
    if opt.nsta <= 8:
        palette = all_palettes['YlOrRd'][9]
    else:
        palette = inferno(opt.nsta+1)
    for sta, staname in enumerate(opt.station.split(',')):
        o0.circle(matplotlib.dates.num2date(startTimeMPL[fam]),
            windowAmps[fam][:,sta], color=palette[sta], line_alpha=0, size=4,
            fill_alpha=0.5, legend_label='{}.{}'.format(
            staname,opt.channel.split(',')[sta]))
    o0.legend.location='bottom_left'
    o0.legend.orientation='horizontal'
    o0.legend.click_policy='hide'
    
    # Time since last event
    o1 = figure(tools=oTOOLS, plot_width=1250, plot_height=250,
        x_axis_type='datetime', title='Time since Previous Event',
        x_range=o0.x_range, y_axis_type='log', y_range=[1e-3,
        2*np.max(spacing)])
    o1.grid.grid_line_alpha = 0.3
    o1.xaxis.axis_label = 'Date'
    o1.yaxis.axis_label = 'Interval (hr)'
    o1.circle(matplotlib.dates.num2date(catalog[1:]), spacing, color='red',
        line_alpha=0, size=4, fill_alpha=0.5)
    
    # Cross-correlation wrt. core
    o2 = figure(tools=oTOOLS, plot_width=1250, plot_height=250,
        x_axis_type='datetime',
        title='Cross-correlation Coefficient with Core Event',
        x_range=o0.x_range, y_range=[0, 1.02])
    o2.grid.grid_line_alpha = 0.3
    o2.xaxis.axis_label = 'Date'
    o2.yaxis.axis_label = 'CCC'
    o2.circle(matplotlib.dates.num2date(catalog), Cfull[np.where(
        famcat==core)[0],:][0], color='red', line_alpha=0, size=4,
        fill_alpha=0.5)
    
    for p in [o0, o1, o2]:
        p = add_bokeh_annotations(p, opt)
    
    # Combine and save
    o = gridplot([[o0],[o1],[o2]])
    output_file('{}{}/reports/{}-report-bokeh.html'.format(opt.outputPath,
        opt.groupName, fnum), title='{} - Cluster {} Detailed Report'.format(
        opt.title, fnum))
    save(o)
    
    ### OPTICS ORDERING (OPTIONAL)
    if ordered:
        # Order by OPTICS rather than by time
        D = 1-Cfull
        s = np.argsort(sum(D))[::-1]
        D = D[s,:]
        D = D[:,s]
        famcat = famcat[s]
        Cind = Cind[s,:]
        Cind = Cind[:,s]
        Cfull = Cfull[s,:]
        Cfull = Cfull[:,s]
        ttree = setOfObjects(D)
        prep_optics(ttree,1)
        build_optics(ttree,1)
        order = np.array(ttree._ordered_list)
        famcat = famcat[order]
        Cind = Cind[order,:]
        Cind = Cind[:,order]
        Cfull = Cfull[order,:]
        Cfull = Cfull[:,order]
    
    ### SAVE FULL CORRELATION MATRIX TO FILE
    if matrixtofile:
        np.save('{}{}/reports/0-Cfull.npy'.format(opt.outputPath,
            opt.groupName, fnum), Cfull)
        np.save('{}{}/reports/0-evTimes.npy'.format(opt.outputPath,
            opt.groupName, fnum), startTime[famcat])
    
    ### CORRELATION MATRIX
    fig = plt.figure(figsize=(14,5.4))
    ax1 = fig.add_subplot(1,2,1)
    cax = ax1.imshow(Cind, vmin=opt.cmin-0.05, cmap='Spectral_r')
    cbar = plt.colorbar(cax, ticks=np.arange(opt.cmin-0.05,1.05,0.05))
    tix = cbar.ax.get_yticklabels()
    tix[0] = 'Undefined'
    cbar.ax.set_yticklabels(tix)
    if ordered:
        plt.title('Stored Correlation Matrix (Ordered)', fontweight='bold')
    else:
        plt.title('Stored Correlation Matrix', fontweight='bold')
        ax1 = add_horizontal_annotations(ax1, startTimeMPL[fam][catalogind],
            opt)
    ax2 = fig.add_subplot(1,2,2)
    cax2 = ax2.imshow(Cfull, vmin=opt.cmin-0.05, cmap='Spectral_r')
    cbar2 = plt.colorbar(cax2, ticks=np.arange(opt.cmin-0.05,1.05,0.05))
    tix = cbar2.ax.get_yticklabels()
    tix[0] = '< {:1.2f}'.format(opt.cmin-0.05)
    cbar2.ax.set_yticklabels(tix)
    if ordered:
        plt.title('Full Correlation Matrix (Ordered)', fontweight='bold')
    else:
        plt.title('Full Correlation Matrix', fontweight='bold')
        ax2 = add_horizontal_annotations(ax2, startTimeMPL[fam][catalogind],
            opt)
    plt.tight_layout()
    plt.savefig('{}{}/reports/{}-reportcmat.png'.format(opt.outputPath,
                                                opt.groupName, fnum), dpi=100)
    plt.close(fig)
    
    ### WAVEFORM IMAGES
    famtable = rtable[famcat]
    fig2 = plt.figure(figsize=(10, 12))
    time_vector = np.arange(-0.5*opt.winlen/opt.samprate,
                             1.5*opt.winlen/opt.samprate, 1/opt.samprate)
    
    for sta in range(opt.nsta):
    
        ax = fig2.add_subplot(int(np.ceil((opt.nsta)/2.)), 2, sta+1)
        
        if ordered:
            plt.title('{}.{} (Ordered)'.format(opt.station.split(',')[sta],
                      opt.channel.split(',')[sta]), fontweight='bold')
        else:
            plt.title('{}.{}'.format(opt.station.split(',')[sta],
                      opt.channel.split(',')[sta]), fontweight='bold')
            ax = add_horizontal_annotations(ax, startTimeMPL[fam][catalogind],
                                                                          opt)
        
        # Prepare data in matrix (row for each element)
        data = np.zeros((len(fam), int(opt.winlen*2)))
        for n, r in enumerate(famtable):
            data[n, :] = prep_wiggle(r['waveform'], sta, r['windowStart'],
                                     r['windowAmp'][sta], opt)
        
        # Plot
        if len(fam) > 12:
            ax.imshow(data, aspect='auto', vmin=-1, vmax=1, cmap='RdBu',
                interpolation='nearest', extent=[np.min(time_vector),
                np.max(time_vector), n + 0.5, -0.5])
        else:
            for n in range(len(fam)):
                dat=data[n,:]
                ax.plot(time_vector,dat/2-n,'k',linewidth=0.25)
            plt.xlim([np.min(time_vector),np.max(time_vector)])
            plt.ylim([-n-0.5,0.5])
        ax.yaxis.set_visible(False)
        plt.xlabel('Time Relative to Trigger (seconds)', style='italic')
    plt.tight_layout()
    plt.savefig('{}{}/reports/{}-reportwaves.png'.format(opt.outputPath,
                                                opt.groupName, fnum), dpi=100)
    plt.close(fig2)
    
    ### HTML OUTPUT PAGE
    tstamp = UTCDateTime.now()
    with open('{}{}/reports/{}-report.html'.format(opt.outputPath,
                                              opt.groupName, fnum), 'w') as f:
        f.write("""
        <html><head><title>{1} - Cluster {0} Detailed Report</title>
        </head><style>
        a {{color:red;}}
        body {{font-family:Helvetica; font-size:12px}}
        h1 {{font-size: 20px;}}
        </style>
        <body><center>
        <em>Last updated: {10}</em></br>
        <h1>Cluster {0} - Detailed Report</h1>
        <img src="{0}-report.png" width=500 height=100></br></br>
            Number of events: {2}</br>
            Longevity: {5:.2f} days</br>
            Mean event spacing: {7:.2f} hours</br>
            Median event spacing: {8:.2f} hours</br>
            Mean Frequency Index: {9:.2f}<br></br>
            First event: {3}</br>
            Core event: {6}</br>
            Last event: {4}</br>
            
            <img src='{11}-reportwaves.png'></br></br>
            
            <iframe src="{11}-report-bokeh.html" width=1350 height=800
            style="border:none"></iframe>
            
            </br>
            <img src='{11}-reportcmat.png'></br></br></br>
        
        """.format(fnum, opt.title, len(fam), (UTCDateTime(
            startTime[minind]) + windowStart[
            minind]/opt.samprate).isoformat(), (UTCDateTime(
            startTime[maxind]) + windowStart[
            maxind]/opt.samprate).isoformat(), longevity, (UTCDateTime(
            startTime[core]) + windowStart[core]/opt.samprate).isoformat(),
            np.mean(spacing), np.median(spacing), np.mean(np.nanmean(fi[fam],
            axis=1)),tstamp,fnum))
        
        f.write("""
        </center></body></html>
        """)


def create_junk_images(jtable, opt):
    """
    Creates images of waveforms contained in the junk table.
    
    File names correspond to the trigger time and the flag for the type of
    junk it was flagged as.
    
    Parameters
    ----------
    jtable : Table object
        Handle to the Junk table.
    opt : Options object
        Describes the run parameters.
    """
    
    # Write out times of junk triggers
    printJunk(jtable, opt)
    
    for r in jtable:
        
        data = np.array([])
        
        for s in range(opt.nsta):
            
            # Concatenate all channels together
            data = np.append(data, prep_wiggle(r['waveform'], s,
                      r['windowStart'] + int(opt.ptrig*opt.samprate), 0, opt))
        
        wiggle_plot(data, (15, 0.5), '{}{}/junk/{}-{}.png'.format(
            opt.outputPath, opt.groupName, (UTCDateTime(r['startTime']) + \
            opt.ptrig).strftime('%Y%m%d%H%M%S'), r['isjunk']), opt)
