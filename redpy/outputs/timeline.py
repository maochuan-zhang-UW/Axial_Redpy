# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling functions related to creating timelines.

The primary function of this module is to support the .output() method of
Detector() objects. The .output() method generates various images and .html
files so the user may easily browse and export the contents of REDPy's
detections.
"""
import os
import datetime
from collections import defaultdict

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.models import Arrow, ColorBar, ColumnDataSource, Div
from bokeh.models import HoverTool, Label, LinearColorMapper, LogColorMapper
from bokeh.models import LogTicker, OpenURL, TabPanel, Range1d, Span, Tabs
from bokeh.models import TapTool, VeeHead
from bokeh.models.glyphs import Line
from bokeh.models.formatters import LogTickFormatter
from bokeh.plotting import figure, gridplot, output_file, save
from obspy import UTCDateTime


def assemble_bokeh_timeline(detector, options, filepath):
    """
    Assembles an interactive timeline with given parameters using Bokeh.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    options : dict
        Describes the options used in the plots.
    filepath : str
        Output file location and name.

    """
    plot_types = options['plotformat'].replace('+', ',').split(',')
    plots = []
    tabtitles = []
    for plot in plot_types:
        if plot == 'eqrate':
            plots.append(subplot_rate(detector, options))
            tabtitles = tabtitles+['Event Rate']
        elif plot == 'fi':
            plots.append(subplot_fi(detector, options))
            tabtitles = tabtitles+['FI']
        elif plot == 'longevity':
            plots.append(subplot_longevity(detector, options))
            tabtitles = tabtitles+['Longevity']
        elif plot == 'occurrence':
            plots.append(subplot_occurrence(
                detector, options, 'rate'))
            tabtitles = tabtitles+['Occurrence (Color by Rate)']
        elif plot == 'occurrencefi':
            plots.append(subplot_occurrence(
                detector, options, 'fi'))
            tabtitles = tabtitles+['Occurrence (Color by FI)']
        else:
            print(f'{plot} is not a valid plot type. Moving on.')
    for fig in plots:
        fig.x_range = plots[0].x_range
        _add_bokeh_annotations(detector, fig)
    _generate_tap_tool(plots)
    gridplot_items = [[Div(
        text=options['divtitle'], width=1000, margin=(-40, 5, -10, 15))]]
    pnum = 0
    for plotgroup in options['plotformat'].split(','):
        if '+' in plotgroup:  # '+' groups plots into tabs
            tabs = []
            for group in range(len(plotgroup.split('+'))):
                tabs = tabs + [TabPanel(child=plots[pnum + group],
                                        title=tabtitles[pnum + group])]
            gridplot_items = gridplot_items + [[Tabs(tabs=tabs)]]
            pnum += group + 1
        else:
            gridplot_items = gridplot_items + [[plots[pnum]]]
            pnum += 1
    output = gridplot(gridplot_items)
    output_file(filepath, title=options['htmltitle'])
    save(output)


def assemble_pdf_timeline(detector, options):
    """
    Generate a static .pdf version of the overview plot for publication.

    Plot is saved in the usual outputs folder.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    options : dict
        Describes the options used in the plots.

    """
    plot_types = options['plotformat'].replace('+', ',').split(',')
    nsub = 0
    for plot in plot_types:
        if 'occurrence' in plot:
            nsub += options['occurheight']
        else:
            nsub += 1
    figheight = 2*nsub+4
    # Hack a reference axis
    figref = plt.figure(figsize=(9, 1))
    axref = figref.add_subplot(1, 1, 1)
    axref = subplot_rate(detector, options, use_bokeh=False, ax=axref)
    fig = plt.figure(figsize=(9, figheight))
    pnum = 0
    for plot in plot_types:
        if plot == 'eqrate':
            ax = fig.add_subplot(nsub, 1, pnum+1, sharex=axref)
            ax = _add_pdf_annotations(detector, ax, options)
            ax = subplot_rate(detector, options, use_bokeh=False, ax=ax)
            pnum += 1
        elif plot == 'fi':
            ax = fig.add_subplot(nsub, 1, pnum+1, sharex=axref)
            ax = _add_pdf_annotations(detector, ax, options)
            ax = subplot_fi(detector, options, use_bokeh=False, ax=ax)
            pnum += 1
        elif plot == 'occurrence':
            ax = fig.add_subplot(
                nsub, 1, (pnum+1, pnum+options['occurheight']), sharex=axref)
            ax = _add_pdf_annotations(detector, ax, options)
            ax = subplot_occurrence(
                detector, options, 'rate', use_bokeh=False, ax=ax)
            _add_pdf_colorbar(detector, ax, 'rate', options)
            pnum += options['occurheight']
        elif plot == 'occurrencefi':
            ax = fig.add_subplot(
                nsub, 1, (pnum+1, pnum+options['occurheight']), sharex=axref)
            ax = _add_pdf_annotations(detector, ax, options)
            ax = subplot_occurrence(
                detector, options, 'fi', use_bokeh=False, ax=ax)
            _add_pdf_colorbar(detector, ax, 'fi', options)
            pnum += options['occurheight']
        elif plot == 'longevity':
            ax = fig.add_subplot(nsub, 1, pnum+1, sharex=axref)
            ax = _add_pdf_annotations(detector, ax, options)
            ax = subplot_longevity(detector, options, use_bokeh=False, ax=ax)
            pnum += 1
        else:
            print(f'{plot} is not a valid plot type. Moving on.')
    plt.tight_layout()
    plt.savefig(os.path.join(detector.get('output_folder'), 'overview.pdf'))
    plt.close(fig)


def bokeh_figure(**kwargs):
    """
    Build foundation for the Bokeh subplots.

    **kwargs can include any keyword argument passable to a Bokeh figure().
    See https://docs.bokeh.org/en/latest/docs/reference/plotting.html for a
    complete list.

    The main argument passed is usually 'title'. If they are not defined,
    'tools', 'width', 'height', and 'x_axis_type' are populated
    with default values.

    Returns
    -------
    Figure object

    """
    if 'tools' not in kwargs:
        kwargs['tools'] = ['pan,box_zoom,reset']
    if 'width' not in kwargs:
        kwargs['width'] = 1250
    if 'height' not in kwargs:
        kwargs['height'] = 250
    if 'x_axis_type' not in kwargs:
        kwargs['x_axis_type'] = 'datetime'
    fig = figure(**kwargs)
    fig.grid.grid_line_alpha = 0.3
    fig.xaxis.axis_label = 'Date'
    fig.yaxis.axis_label = ''
    fig.margin = (0, 0, 0, 5)
    return fig


def generate_pdf_timeline(
        detector, tmin, tmax, binsize, minmembers, occurheight, plotformat):
    """
    Create .pdf version of overview timeline.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    tmin : int
        Matplotlib date of start of plot.
    tmax : int
        Matplotlib date of end of plot.
    binsize : float, optional
        Temporal bin size (in days) for both the rate and occurrence plots.
    minmembers : int, optional
        Minimum number of members in a family to be included in the
        occurrence plot.
    occurheight : float, optional
        How much taller the occurrence plot should be than the other
        subplots.
    plotformat : str
        Comma-separated list of plots to include.

    """
    ttimes = detector.get('plotvars')['ttimes']
    if tmin:
        mintime = tmin
    else:
        mintime = np.min(ttimes) - (
            2 * detector.get('winlen') / detector.get('samprate') / 86400)
    if tmax:
        maxtime = tmax
    else:
        maxtime = np.max(ttimes) + (
            2 * detector.get('winlen') / detector.get('samprate') / 86400)
    options = {
        'mintime': mintime,
        'maxtime': maxtime,
        'barpad': 0.01 * (maxtime-mintime),
        'binsize_hist': binsize,
        'binsize_occur': binsize,
        'minplot': minmembers,
        'plotformat': plotformat,
        'occurheight': occurheight}
    assemble_pdf_timeline(detector, options)


def generate_timelines(detector):
    """
    Create Bokeh timelines: overview, overview_recent, and meta_recent.

    The primary purpose of this function is to sort out the differences in
    the unique behavior of each Bokeh timeline type.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    ttimes = detector.get('plotvars')['ttimes']
    if detector.get('bokehendtime') == 'now':
        maxtime = UTCDateTime().matplotlib_date
    else:
        maxtime = np.max(ttimes) + (
            2 * detector.get('winlen') / detector.get('samprate') / 86400)
    for file in ['overview', 'overview_recent', 'meta_recent']:
        # Set parameters unique to each timeline type
        if file == 'overview':
            mintime = np.min(ttimes) - (
                2 * detector.get('winlen') / detector.get('samprate') / 86400)
            options = {
                'mintime': mintime,
                'maxtime': maxtime,
                'barpad': 0.01 * (maxtime-mintime),
                'plotformat': detector.get('plotformat'),
                'binsize_hist': detector.get('dybin'),
                'binsize_occur': detector.get('occurbin'),
                'minplot': detector.get('minplot'),
                'fixedheight': detector.get('fixedheight'),
                'htmltitle': f'{detector.get("title")} Overview',
                'divtitle': f'<h1>{detector.get("title")}</h1>'}
        elif file == 'overview_recent':
            options = {
                'mintime': maxtime - detector.get('recplot'),
                'maxtime': maxtime,
                'barpad': 0.01 * detector.get('recplot'),
                'plotformat': detector.get('plotformat'),
                'binsize_hist': detector.get('hrbin') / 24,
                'binsize_occur': detector.get('recbin'),
                'minplot': 0,
                'fixedheight': detector.get('fixedheight'),
                'htmltitle': (f'{detector.get("title")} Overview - Last '
                              f'{detector.get("recplot"):.1f} Days'),
                'divtitle': (f'<h1>{detector.get("title")} - Last '
                             f'{detector.get("recplot"):.1f} Days</h1>')}
        else:  # meta_recent.html
            options = {
                'mintime': maxtime - detector.get('mrecplot'),
                'maxtime': maxtime,
                'barpad': 0.01 * detector.get('mrecplot'),
                'plotformat': detector.get('plotformat').replace(',', '+'),
                'binsize_hist': detector.get('mhrbin') / 24,
                'binsize_occur': detector.get('mrecbin'),
                'minplot': detector.get('mminplot'),
                'fixedheight': True,
                'htmltitle': (f'{detector.get("title")} Overview - Last '
                              f'{detector.get("mrecplot"):.1f} Days'),
                'divtitle': f"""
                    <h1>{detector.get('title')} - Last
                        {detector.get('mrecplot'):.1f} Days |
                        <a href='overview.html' style='color:red'
                              target='_blank'>Full Overview</a> |
                        <a href='overview_recent.html'
                              style='color:red' target='_blank'>Recent</a>
                    </h1>"""}
        filepath = os.path.join(detector.get('output_folder'), f'{file}.html')
        assemble_bokeh_timeline(detector, options, filepath)


def subplot_fi(detector, options, use_bokeh=True, ax=None):
    """
    Create subplot for frequency index scatterplot.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    options : dict
        Describes the options used in the plots.
    use_bokeh : bool, optional
        If True, render a Bokeh figure, else matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    Figure object or Axis object
        If use_bokeh is True, returns a Bokeh Figure else returns a
        matplotlib Axis handle.

    """
    mean_fi = detector.get('plotvars')['mean_fi']
    rtimes = detector.get('plotvars')['rtimes']
    rtimes_mpl = detector.get('plotvars')['rtimes_mpl']
    if use_bokeh:
        fig = bokeh_figure(title='Frequency Index')
        fig.yaxis.axis_label = 'FI'
        # Always plot at least one invisible point
        fig.circle(mdates.num2date(options['maxtime']), 0,
                   line_alpha=0, fill_alpha=0)
    else:
        ax.set_title('Frequency Index', loc='left', fontweight='bold')
        ax.set_ylabel('FI', style='italic')
        ax.set_xlabel('Date', style='italic')
    idxs = np.where((rtimes_mpl >= options['mintime']) & (
                    rtimes_mpl <= options['maxtime']))[0]
    if use_bokeh:
        fig.circle(rtimes[idxs], mean_fi[idxs], color='red', line_alpha=0,
                   size=3, fill_alpha=0.5)
        return fig
    ax.scatter(rtimes[idxs], mean_fi[idxs], 2, c='red', alpha=0.25)
    # Need to call get_ylim() or y-limits sometimes freak out
    _ = ax.get_ylim()
    return ax


def subplot_longevity(detector, options, use_bokeh=True, ax=None):
    """
    Create subplot for longevity.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    options : dict
        Describes the options used in the plots.
    use_bokeh : bool, optional
        If True, render a Bokeh figure, else matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    Figure object or Axis object
        If use_bokeh is True, returns a Bokeh Figure else returns a
        matplotlib Axis handle.

    """
    ttimes = detector.get('plotvars')['ttimes']
    longevity = detector.get('ftable', 'longevity')
    if use_bokeh:
        fig = bokeh_figure(
            y_axis_type='log',
            y_range=[0.01, np.sort(ttimes)[-1] - np.sort(ttimes)[0]],
            title='Family Longevity')
        fig.yaxis.axis_label = 'Days'
        # Always plot at least one invisible point
        fig.circle(mdates.num2date(options['maxtime']), 1,
                   line_alpha=0, fill_alpha=0)
    else:
        ax.set_title('Family Longevity', loc='left', fontweight='bold')
        ax.set_ylabel('Days', style='italic')
        ax.set_xlabel('Date', style='italic')
    for fnum in range(len(detector)):
        line = _determine_lines(detector, options, fnum)
        if use_bokeh:
            fig = _draw_lines_bokeh(
                line, longevity[fnum], 'red', 0.5, fig)
        else:
            ax = _draw_lines_mpl(
                line, longevity[fnum], options, 'red', 0.75, True, ax)
    if use_bokeh:
        return fig
    return ax


def subplot_occurrence(detector, options, colorby, use_bokeh=True, ax=None):
    """
    Create subplot for family occurrence.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    options : dict
        Describes the options used in the plots.
    colorby : str
        Determines color in histograms: 'rate' (YlOrRd) or 'fi' (coolwarm)
    use_bokeh : bool, optional
        If True, render a Bokeh figure, else matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    Figure object or Axis object
        If use_bokeh is True, returns a Bokeh Figure else returns a
        matplotlib Axis handle.

    """
    if use_bokeh:
        fig = bokeh_figure(
            tools=[_family_hover_tool(), 'pan,box_zoom,reset'],
            title='Occurrence Timeline',
            height=250, width=1250)
        fig.yaxis.axis_label = 'Family by Date' + (
            f' ({options["minplot"]}+ Members)' if options[
                'minplot'] > 2 else '')
        # Always plot at least one invisible point
        fig.circle(mdates.num2date(
            options['maxtime']), 0, line_alpha=0, fill_alpha=0)
    else:
        ax.set_title('Occurrence Timeline', loc='left', fontweight='bold')
        ax.set_ylabel('Family by Date' + (
            f' ({options["minplot"]}+ Members)' if options[
                'minplot'] > 2 else ''),
            style='italic')
        ax.set_xlabel('Date', style='italic')
        fig = None
    y_pos = 0
    patch = defaultdict(list)
    for fnum in range(len(detector)):
        fig, ax, y_pos, patch = _occurrence_for_family(
            detector, options, fig, ax, y_pos, patch, fnum, colorby, use_bokeh)
    if use_bokeh:
        return _finish_occurrence(
            detector, fig, patch, y_pos, options, colorby)
    return ax


def subplot_rate(detector, options, use_bokeh=True, ax=None):
    """
    Create subplot for rate of orphans and repeaters.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    options : dict
        Describes the options used in the plots.
    use_bokeh : bool, optional
        If True, render a Bokeh figure, else matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    Figure object or Axis object
        If use_bokeh is True, returns a Bokeh Figure else returns a
        matplotlib Axis handle.

    """
    dt_offset = options['binsize_hist']/2  # used to create the lines
    hist_trigs, bin_times = np.histogram(
        detector.get('plotvars')['ttimes'],
        bins=np.arange(options['mintime'],
                       options['maxtime'] + options['binsize_hist'],
                       options['binsize_hist']))
    hist_repeaters, bin_times = np.histogram(
        detector.get('plotvars')['rtimes_mpl'],
        bins=np.arange(options['mintime'],
                       options['maxtime'] + options['binsize_hist'],
                       options['binsize_hist']))
    if detector.get('timeline_vs') == 'triggers':
        trigorph = 'Total Triggers'
        hist_trigorph = hist_trigs
    else:
        trigorph = 'Orphans'
        hist_trigorph = hist_trigs - hist_repeaters
    hr_days = 'Day Bin' if options['binsize_hist'] >= 1 else 'Hour Bin'
    bin_text = options['binsize_hist'] if options[
        'binsize_hist'] >= 1 else options['binsize_hist']*24
    title = f'Repeaters vs. {trigorph} by {bin_text:.1f} {hr_days}'
    if use_bokeh:
        fig = bokeh_figure(title=title)
        fig.yaxis.axis_label = 'Events'
        fig.line(mdates.num2date(bin_times[0:-1] + dt_offset), hist_trigorph,
                 color='black', legend_label=trigorph)
        fig.line(mdates.num2date(bin_times[0:-1] + dt_offset), hist_repeaters,
                 color='red', legend_label='Repeaters', line_width=2)
        fig.legend.location = 'top_left'
        return fig
    ax.plot(mdates.num2date(bin_times[0:-1] + dt_offset), hist_trigorph,
            color='black', label=trigorph, lw=0.5)
    ax.plot(mdates.num2date(bin_times[0:-1] + dt_offset), hist_repeaters,
            color='red', label='Repeaters', lw=2)
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_ylabel('Events', style='italic')
    ax.set_xlabel('Date', style='italic')
    ax.legend(loc='upper left', frameon=False)
    return ax


def _add_bokeh_annotations(detector, fig):
    """Plot annotations on Bokeh figure."""
    if detector.get('anotfile') != '':
        annotations = pd.read_csv(detector.get('anotfile'))
        for i in range(len(annotations)):
            # Deal with bokeh's unusual datetime axis
            spantime = (datetime.datetime.strptime(annotations['Time'][i],
                        '%Y-%m-%dT%H:%M:%S') - datetime.datetime(
                            1970, 1, 1)).total_seconds()
            fig.add_layout(Span(
                location=spantime*1000, dimension='height',
                line_color=annotations['Color'][i],
                line_width=annotations['Weight'][i],
                line_dash=annotations['Line Type'][i],
                line_alpha=annotations['Alpha'][i]))


def _add_pdf_colorbar(detector, ax, colorby, options):
    """
    Add a colorbar to .pdf occurrence plots.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    ax : Axid object
        Handle to the current axis.
    colorby : str
        Determines colormap to use ('rate' or 'fi')
    options : float
        Width (in days) of time bins for occurrence plot histogram.

    """
    cax = ax.inset_axes([0.025, 0.925, 0.25, 0.025])
    if colorby == 'rate':
        cax.set_title(f'Events per {_determine_legend_text(options)}',
                      loc='left', style='italic')
    else:
        cax.set_title('Mean Frequency Index', loc='left', style='italic')
    cax.get_yaxis().set_visible(False)
    gradient = np.linspace(0, 1, 1001)
    gradient = np.vstack((gradient, gradient))
    if colorby == 'rate':
        cax.imshow(gradient, aspect='auto', cmap='YlOrRd_r',
                   interpolation='bilinear')
        if options['binsize_occur'] >= 1:
            cax.set_xticks((0, 333.3, 666.6, 1000))
            cax.set_xticklabels(('1', '10', '100', '1000'))
        else:
            cax.set_xticks((0, 500, 1000))
            cax.set_xticklabels(('1', '10', '100'))
    else:
        cax.imshow(gradient, aspect='auto', cmap='coolwarm_r',
                   interpolation='bilinear')
        cax.set_xticks((0, 500, 1000))
        cax.set_xticklabels(
            (detector.get('fispanlow'), np.mean((detector.get('fispanlow'),
             detector.get('fispanhigh'))), detector.get('fispanhigh')))
    cax.set_frame_on(False)
    cax.tick_params(length=0)


def _add_pdf_annotations(detector, ax, options):
    """Plot annotations on .pdf overview figure."""
    if detector.get('anotfile') != '':
        annotations = pd.read_csv(detector.get('anotfile'))
        for i in range(len(annotations)):
            plotdate = mdates.date2num(np.datetime64(
                annotations['Time'][i]))
            # If within bounds, add to figure
            if options['mintime'] <= plotdate <= options['maxtime']:
                ax.axvline(plotdate, color=annotations['Color'][i],
                           lw=annotations['Weight'][i],
                           ls=annotations['Line Type'][i],
                           alpha=annotations['Alpha'][i], zorder=-1)
    return ax


def _build_occurrence_histogram(detector, options, members, colorby):
    """Build patch definitions for the occurrence histogram."""
    hist, hist_x = np.histogram(
        detector.get('plotvars')['rtimes_mpl'][members],
        bins=np.arange(
            min(detector.get('plotvars')['rtimes_mpl'][members]),
            max(detector.get('plotvars')['rtimes_mpl'][members]
                + options['binsize_occur']),
            options['binsize_occur']))
    left = hist_x[np.where(hist > 0)]
    right = hist_x[np.where(hist > 0)] + options['binsize_occur']
    if colorby == 'rate':
        if options['binsize_occur'] >= 1:
            ind = [int(min(255, 255*(i/3))) for i in np.log10(hist[hist > 0])]
        else:
            ind = [int(min(255, 255*(i/2))) for i in np.log10(hist[hist > 0])]
    elif colorby == 'fi':
        hist_x = hist_x[np.where(hist > 0)]
        hist = hist[hist > 0]
        fisum = np.zeros(len(hist))
        # Loop through bins to get summed fi
        for i in range(len(hist)):
            # Find indicies of within bins
            idx = np.where(np.logical_and(
                detector.get('plotvars')['rtimes_mpl'][members] >= hist_x[i],
                detector.get('plotvars')['rtimes_mpl'][members] < hist_x[i]
                + options['binsize_occur']))
            # Sum fi for those events
            fisum[i] = np.sum(
                detector.get('plotvars')['mean_fi'][members[idx]])
        # Convert to mean fi
        ind = [int(max(min(
            255,
            255*(i - detector.get('fispanlow'))
            / (detector.get('fispanhigh') - detector.get('fispanlow'))
            ), 0)) for i in fisum/hist]
    if colorby == 'rate':
        colormap = matplotlib.cm.get_cmap('YlOrRd')
    elif colorby == 'fi':
        colormap = matplotlib.cm.get_cmap('coolwarm')
    else:
        print('Unrecognized colorby choice, defaulting to rate')
        colorby = 'rate'
        colormap = matplotlib.cm.get_cmap('YlOrRd')
    palette = [matplotlib.colors.rgb2hex(i) for i in colormap(
        np.arange(colormap.N)[::-1])]
    colors = np.array([palette[i] for i in ind])
    idx = np.where(hist_x[np.where(hist > 0)[0]] > options['mintime'])[0]
    return left[idx], right[idx], colors[idx]


def _build_patch(patch, y_pos, left, right, fnum,
                 options):
    """Build source for Bokeh hover patches."""
    patch['xs'].append([
        np.min(left) - datetime.timedelta(days=options['barpad']),
        np.min(left) - datetime.timedelta(days=options['barpad']),
        np.max(right) + datetime.timedelta(days=options['barpad']),
        np.max(right) + datetime.timedelta(days=options['barpad'])])
    patch['ys'].append(
        [y_pos-0.5, y_pos+0.5, y_pos+0.5, y_pos-0.5])
    patch['famnum'].append([fnum])
    return patch


def _determine_color_mapper(options):
    """Define LogColorMapper for occurrence plot."""
    colormap = matplotlib.cm.get_cmap('YlOrRd')
    bokehpalette = [matplotlib.colors.rgb2hex(m) for m in colormap(
        np.arange(colormap.N)[::-1])]
    if options['binsize_occur'] >= 1:
        return LogColorMapper(palette=bokehpalette, low=1, high=1000)
    return LogColorMapper(palette=bokehpalette, low=1, high=100)


def _determine_color_mapper_fi(detector):
    """Define LinearColorMapper for occurrencefi plot."""
    colormap = matplotlib.cm.get_cmap('coolwarm')
    bokehpalette = [matplotlib.colors.rgb2hex(m) for m in colormap(
        np.arange(colormap.N)[::-1])]
    return LinearColorMapper(palette=bokehpalette,
                             low=detector.get('fispanlow'),
                             high=detector.get('fispanhigh'))


def _determine_legend_text(options):
    """Determine legend wording based on bin size."""
    if options['binsize_occur'] == 1/24:
        return 'Hour'
    if options['binsize_occur'] == 1:
        return 'Day'
    if options['binsize_occur'] == 7:
        return 'Week'
    if options['binsize_occur'] < 2:
        return f'{options["binsize_occur"]*24} Hours'
    return f'{options["binsize_occur"]} Days'


def _determine_lines(detector, options, fnum):
    """
    Determine arrows and line positions for occurrence/longevity timelines.

    Parameters
    ----------
    options : dict
        Describes the options used in the plots.
    famstart : float
        Start time of family as matplotlib date.
    longev : float
        Longevity of family in days.

    Returns
    -------
    dict
        Dictionary filled with start and end times of the line and whether
        it should have arrows. Empty if there should be no line.

    """
    famstart = detector.get('ftable', 'startTime', fnum)
    longev = detector.get('ftable', 'longevity', fnum)
    add_rarrow = False
    add_larrow = False
    add_line = False
    line_start = 0
    line_end = 0
    # Family starts after start of mintime and ends before maxtime
    if (options['mintime'] <= famstart) and (
            options['maxtime'] >= famstart + longev):
        add_line = True
        line_start = famstart
        line_end = famstart + longev
    # Family starts after start of mintime but first event is past maxtime
    elif (options['mintime'] <= famstart) and (options['maxtime'] <= famstart):
        add_line = False
    # Family starts after start of mintime but ends after maxtime
    elif (options['mintime'] <= famstart) and (
            options['maxtime'] <= famstart + longev):
        add_line = True
        add_rarrow = True
        line_start = famstart
        line_end = options['maxtime'] + options['barpad']
    # Family starts before mintime, ends before maxtime, ends after mintime
    elif (options['mintime'] >= famstart) and (
            options['maxtime'] >= famstart + longev) and (
            options['mintime'] <= famstart + longev):
        add_line = True
        add_larrow = True
        line_start = options['mintime'] - options['barpad']
        line_end = famstart + longev
    # Family starts before mintime and ends after maxtime
    elif (options['mintime'] >= famstart) and (
            options['maxtime'] <= famstart + longev):
        add_line = True
        add_larrow = True
        add_rarrow = True
        line_start = options['mintime'] - options['barpad']
        line_end = options['maxtime'] + options['barpad']
    if add_line:
        return {'start': line_start, 'end': line_end,
                'larrow': add_larrow, 'rarrow': add_rarrow}
    return {}


def _draw_lines_bokeh(line, y_pos, color, alpha, fig):
    """Draw lines into Bokeh figure."""
    source = ColumnDataSource({
        'x': np.array((mdates.num2date(line['start']),
                      mdates.num2date(line['end']))),
        'y': np.array((y_pos, y_pos))})
    fig.add_glyph(
        source, Line(x='x', y='y', line_color=color, line_alpha=alpha))
    if line['larrow']:
        fig.add_layout(Arrow(
            end=VeeHead(
                size=5, fill_color=color, line_color=color, line_alpha=alpha),
            line_alpha=0,
            x_start=mdates.num2date(line['end']),
            x_end=mdates.num2date(line['start']),
            y_start=y_pos, y_end=y_pos))
    return fig


def _draw_lines_mpl(line, y_pos, options, color, alpha, log, ax):
    """Draw lines into matplotlib axis."""
    if line:
        if log:
            ax.semilogy(np.array((mdates.num2date(line['start']),
                                  mdates.num2date(line['end']))),
                        np.array((y_pos, y_pos)),
                        color, lw=0.5, zorder=0, alpha=alpha)
        else:
            ax.plot(np.array((mdates.num2date(line['start']),
                              mdates.num2date(line['end']))),
                    np.array((y_pos, y_pos)),
                    color, lw=0.5, zorder=0, alpha=alpha)
        if line['larrow']:
            ax.annotate(
                '',
                xy=(mdates.num2date(line['start']), y_pos),
                xytext=(mdates.num2date(
                    line['start'] - options['barpad']), y_pos),
                arrowprops={
                    'arrowstyle': '<-', 'color': color, 'alpha': alpha})
        if line['rarrow']:  # Only for matplotlib
            ax.annotate(
                '',
                xy=(mdates.num2date(line['end']), y_pos),
                xytext=(mdates.num2date(
                    line['end'] + options['barpad']), y_pos),
                arrowprops={
                    'arrowstyle': '<-', 'color': color, 'alpha': alpha})
    return ax


def _generate_tap_tool(plots):
    """Create a TapTool to open family pages."""
    renderers = []
    for fig in plots:
        hover = fig.select(type=HoverTool)
        if hover and len(hover[0].renderers):
            renderers.append(hover[0].renderers[0])
    if renderers:
        taptool = TapTool(renderers=renderers,
                          callback=OpenURL(url='./families/@famnum.html'))
    else:
        taptool = TapTool()
    for fig in plots:
        fig.add_tools(taptool)


def _family_hover_tool():
    """Generate HoverTool for family hover preview."""
    return HoverTool(
        tooltips="""
        <div>
        <div>
            <img src="./families/@famnum.png"
                style="height: 100px; width: 500px;
                vertical-align: middle;"/>
            <span style="font-size: 9px;
                font-family: Helvetica;">Family ID: </span>
            <span style="font-size: 12px;
                font-family: Helvetica;">@famnum</span>
        </div>
        </div>
        """, renderers=[])


def _finish_occurrence(detector, fig, patch, y_pos, options, colorby):
    """Finish adding hover patches and colorbar to occurrence plot."""
    cbar_location = 85
    if y_pos > 0:
        # Patches allow hovering for image of core and family number
        source = ColumnDataSource(
            data=patch)
        renderer = fig.patches(
            'xs', 'ys', source=source, alpha=0,
            selection_fill_alpha=0, selection_line_alpha=0,
            nonselection_fill_alpha=0, nonselection_line_alpha=0)
        hovertool = fig.select(type=HoverTool)[0]
        hovertool.renderers.append(renderer)
        if (y_pos > 15) and not options['fixedheight']:
            fig.height = y_pos*15
            fig.y_range = Range1d(-1, y_pos)
            cbar_location = y_pos*15 - 165
    if colorby == 'rate':
        color_bar = ColorBar(
            color_mapper=_determine_color_mapper(options),
            ticker=LogTicker(),
            border_line_color='#eeeeee', location=(7, cbar_location),
            orientation='horizontal', width=150, height=15,
            title=f'Events per {_determine_legend_text(options)}',
            padding=15, major_tick_line_alpha=0,
            formatter=LogTickFormatter(min_exponent=4))
    elif colorby == 'fi':
        color_bar = ColorBar(
            color_mapper=_determine_color_mapper_fi(detector),
            border_line_color='#eeeeee', location=(7, cbar_location),
            orientation='horizontal', width=150, height=15,
            title='Mean Frequency Index', padding=15,
            major_tick_line_alpha=0)
    fig.add_layout(color_bar)
    return fig


def _occurrence_for_family(
        detector, options, fig, ax, y_pos, patch, fnum, colorby, use_bokeh):
    """Prepare and draw occurrence for a single family."""
    members = detector.get_members(fnum)
    if (len(members) >= options['minplot']) and (
            max(detector.get('plotvars')[
                'rtimes_mpl'][members]) > options['mintime']):
        left, right, colors = _build_occurrence_histogram(
            detector, options, members, colorby)
        if use_bokeh:
            fig = _draw_lines_bokeh(
                _determine_lines(detector, options, fnum),
                y_pos, 'black', 1, fig)
            left = mdates.num2date(left)
            right = mdates.num2date(right)
            fig.quad(top=y_pos+0.3, bottom=y_pos-0.3,
                     left=np.array(left),
                     right=np.array(right),
                     color=np.array(colors))
            fig.add_layout(Label(
                x=max(right), y=y_pos,
                text=f'  {len(members)}',
                text_font_size='9pt', text_baseline='middle'))
            patch = _build_patch(
                patch, y_pos, left, right, fnum, options)
        else:
            ax = _draw_lines_mpl(
                _determine_lines(detector, options, fnum),
                y_pos, options, 'black', 1, False, ax)
            # Potentially slow if many patches
            for i in range(len(left)):
                j = mdates.num2date(np.array(left)[i])
                if (mdates.num2date(options[
                        'mintime']) <= j <= mdates.num2date(
                            options['maxtime'])):
                    ax.add_patch(matplotlib.patches.Rectangle(
                        (j, y_pos - 0.3),
                        datetime.timedelta(options['binsize_occur']),
                        0.6, facecolor=colors[i],
                        edgecolor=None, fill=True))
            ax.annotate(
                f'  {len(members)}',
                (mdates.num2date(max(right)), y_pos), va='center',
                ha='left')
        y_pos += 1
    return fig, ax, y_pos, patch
