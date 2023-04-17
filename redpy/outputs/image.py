# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling functions related to creating family image files.

The primary function of this module is to support the .output() method of
Detector() objects. The .output() method generates various images and .html
files so the user may easily browse and export the contents of REDPy's
detections.
"""
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.palettes import all_palettes, inferno
from obspy import UTCDateTime

import redpy.correlation
from redpy.outputs.timeline import bokeh_figure


def assemble_family_image(detector, fnum, tmin, tmax, bboxes, oformat, dpi):
    """
    Create a multi-paneled family plot for the specified family 'fnum'.

    This function allows some flexibility in the output format
    (e.g., .png, .pdf) as well as resolution.

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
    detector : Detector object
        Primary interface for handling detections.
    fnum : int
        Family number to plot.
    tmin : float
        Minimum time on timeline axes as matplotlib date (0 for default
        tmin).
    tmax : float
        Maximum time on timeline axes as matplotlib date (0 for default
        tmax).
    bboxes : list of Bbox objects
        List of bounding box positions for each axis.
    oformat : str
        output file format (e.g., 'png' or 'pdf')
    dpi : int
        Dots per inch resolution of raster file.

    """
    axes, bboxes = initialize_family_image(detector, bboxes=bboxes)
    members = detector.get_members(fnum)
    members = members[
        np.argsort(detector.get('plotvars')['rtimes_mpl'][members])]
    rtable_fam = detector.get('rtable', row=members)
    core_idx = np.where(members == detector.get('ftable', 'core', fnum))[0][0]
    subplot_waveforms(detector, rtable_fam, core_idx, ax=axes[0])
    subplot_fft(detector, rtable_fam, core_idx, ax=axes[1])
    subplot_amplitude(detector, rtable_fam, members, core_idx, ax=axes[2])
    subplot_spacing(detector, members, core_idx, ax=axes[3])
    subplot_correlation(detector, members, core_idx, ax=axes[4])
    axes = _format_family_image(axes, detector)
    if tmin and tmax:
        axes[2].set_xlim(tmin, tmax)
    elif tmin:
        axes[2].set_xlim(tmin, axes[2].get_xlim()[1])
    elif tmax:
        axes[2].set_xlim(axes[2].get_xlim()[0], tmax)
    axes[3].set_xlim(axes[2].get_xlim())
    axes[4].set_xlim(axes[2].get_xlim())
    plt.savefig(os.path.join(detector.get('output_folder'), 'families',
                             f'fam{fnum}.{oformat}'), dpi=dpi)


def correlation_matrix_plot(detector, ccc_fam, ccc_full, members, ordered,
                            skip_recalculate_ccc, outfile):
    """
    Create a plot of the stored and full correlation matrices.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    ccc_fam : float ndarray
        Dense stored correlation matrix.
    ccc_full : float ndarray, optional
        Filled correlation matrix.
    members : int ndarray
        Indices of family members within the Repeaters table.
    ordered : bool
        True if members have been ordered (e.g., by OPTICS), else by time.
    skip_recalculate_ccc : bool
        True if user wishes to skip recalculating the full correlation matrix.
    outfile : str
        Path and filename for saving the figure.

    """
    cmap = plt.get_cmap('inferno_r').copy()
    cmap.set_extremes(under='w')
    fig = plt.figure(figsize=(14, 5.4))
    ax1 = fig.add_subplot(1, 2, 1)
    cax = ax1.imshow(ccc_fam, vmin=detector.get('cmin'), vmax=1, cmap=cmap)
    plt.colorbar(cax, extend='min')
    if ordered:
        plt.title('Stored Correlation Matrix (Ordered)', fontweight='bold')
    else:
        plt.title('Stored Correlation Matrix', fontweight='bold')
        ax1 = _add_horizontal_annotations(
            ax1, detector.get('plotvars')['rtimes_mpl'][members], detector)
    if not skip_recalculate_ccc:
        ax2 = fig.add_subplot(1, 2, 2)
        cax2 = ax2.imshow(
            ccc_full, vmin=detector.get('cmin'), vmax=1, cmap=cmap)
        plt.colorbar(cax2, extend='min')
        if ordered:
            plt.title('Full Correlation Matrix (Ordered)', fontweight='bold')
        else:
            plt.title('Full Correlation Matrix', fontweight='bold')
            ax2 = _add_horizontal_annotations(
                ax2, detector.get('plotvars')['rtimes_mpl'][members], detector)
    plt.tight_layout()
    plt.savefig(outfile, dpi=100)
    plt.close(fig)


def create_core_images(detector):
    """
    Plot core waveforms as *.png files in the families folder.

    Used for hovering in timeline and header for family pages.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    # Iterate and plot
    for fnum, core_idx in enumerate(detector.get('ftable', 'core')):
        if detector.get('ftable', 'printme', fnum) == 1:
            core = detector.get('rtable', row=core_idx)
            data = prep_wiggle(
                core['waveform'], detector.get('printsta'),
                core['windowStart'],
                core['windowAmp'][detector.get('printsta')], detector)
            wiggle_plot(
                data, (5, 1), os.path.join(
                    detector.get('output_folder'), 'families', f'{fnum}.png'))


def create_family_images(detector):
    """
    Create multi-paneled family plots for all families that need plotting.

    This function wraps assemble_family_image() and outputs all files as
    *.png files in the families folder.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    _, bboxes = initialize_family_image(detector)
    for fnum in range(len(detector)):
        if detector.get('ftable', 'printme', fnum) != 0:
            assemble_family_image(detector, fnum, 0, 0, bboxes, 'png', 100)


def create_junk_images(detector):
    """
    Create images of waveforms contained in the junk table.

    File names correspond to the trigger time and the flag for the type of
    junk it was flagged as.

    Parameters
    ----------
    jtable : Table object
        Handle to the Junk table.
    detector : Detector object
        Primary interface for handling detections.

    """
    if detector.get('verbose'):
        print('Creating junk plots...')
    for row in detector.get('jtable').table:
        data = np.array([])
        for sta in range(detector.get('nsta')):
            # Concatenate all channels together
            data = np.append(
                data, prep_wiggle(
                    row['waveform'], sta,
                    row['windowStart'] + int(detector.get(
                        'ptrig')*detector.get('samprate')), 0, detector))
        jtime = (UTCDateTime(
            row['startTime']) + detector.get('ptrig')).strftime('%Y%m%d%H%M%S')
        jtype = row['isjunk']
        wiggle_plot(data, (15, 0.5), os.path.join(
            detector.get('output_folder'), 'junk', f'{jtime}-{jtype}.png'))


def generate_images(detector):
    """
    Make both core and family images based on new changes.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    move_images(detector)
    create_core_images(detector)
    create_family_images(detector)


def initialize_family_image(detector, bboxes=None):
    """
    Create figure and axes with the proper layout for the family images.

    This function basically makes it so we only have to call tight_layout()
    once and significantly reduce time spent formatting the plots by using
    the bounding box locations.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    bboxes : list of Bbox objects, optional
        List of bounding box positions for each axis.

    Returns
    -------
    list of Axis objects
        List of subplot axes.
    ist of Bbox objects
        List of bounding box positions for each axis.

    """
    # Close any existing plots
    plt.close()
    fig = plt.figure(figsize=(10, 12))
    axes = [fig.add_subplot(9, 3, (1, 8)), fig.add_subplot(9, 3, (3, 9)),
            fig.add_subplot(9, 3, (10, 15)),
            fig.add_subplot(9, 3, (16, 21)),
            fig.add_subplot(9, 3, (22, 27))]
    if bboxes:
        # Set positions for axes
        for i, ax in enumerate(axes):
            ax.set_position(bboxes[i])
    else:
        # Set up format for axes positions
        axes = _format_family_image(axes, detector)
        axes[2].set_xlim(0, 1)  # Ensure that dates near the edge fit
        plt.tight_layout()
        bboxes = [ax.get_position() for ax in axes]
    return axes, bboxes


def move_images(detector):
    """
    Move files with family numbers that have changed but contents have not.

    This step can potentially save a lot of plotting overhead.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    opath = os.path.join(detector.get('output_folder'), 'families')
    # Deal with renaming files to reduce plotting time overhead
    for fnum in range(len(detector))[::-1]:
        if (detector.get('ftable', 'lastprint', fnum) != fnum) and (
                detector.get('ftable', 'printme', fnum) == 0):
            lastprint = detector.get('ftable', 'lastprint', fnum)
            os.rename(os.path.join(opath, f'{lastprint}.png'),
                      os.path.join(opath, f'{fnum}.png.tmp'))
            os.rename(os.path.join(opath, f'fam{lastprint}.png'),
                      os.path.join(opath, f'map{fnum}.png.tmp'))
            os.rename(os.path.join(opath, f'map{lastprint}.png'),
                      os.path.join(opath, f'map{fnum}.png.tmp'))


def prep_wiggle(waveform, sta, window_start, normalize_amplitude, detector):
    """
    Cut window around trigger time and normalizes the waveform for plotting.

    The plotting window is always 2*detector.get('winlen') samples, with
    half detector.get('winlen') on either side of the correlation window.
    Data are clipped if they are above 'windowAmp' in amplitude.

    Parameters
    ----------
    waveform : float ndarray
        Waveform to be plotted, all stations/channels concatenated.
    sta : int
        Station index for station/channel to be used.
    window_start : int
        Sample corresponding to start of correlation window.
    normalize_amplitude : float
        Amplitude to normalize to. If passed 0, uses the maximum of the
        entire window instead with a small epsilon to prevent division by 0
        if empty.
    detector : Detector object
        Primary interface for handling detections.

    Returns
    -------
    float ndarray
        Clipped, normalized, and trimmed waveform for single
        station/channel.

    """
    window = waveform[sta*detector.get('wshape'):(
        sta+1)*detector.get('wshape')]
    minsample = window_start - int(0.5*detector.get('winlen'))
    maxsample = window_start + int(1.5*detector.get('winlen'))
    if minsample < 0:
        prepad = -minsample
        minsample = 0
    else:
        prepad = 0
    if maxsample > detector.get('wshape'):
        postpad = maxsample - detector.get('wshape')
        maxsample = detector.get('wshape')
    else:
        postpad = 0
    data = window[minsample:maxsample]
    if prepad:
        data = np.append(np.zeros(prepad), data)
    if postpad:
        data = np.append(data, np.zeros(postpad))
    if normalize_amplitude > 0:
        data = data / normalize_amplitude
    else:
        data = data / np.max(np.abs(data)+1e-12)
    data[data > 1] = 1
    data[data < -1] = -1
    return data


def subplot_amplitude(
        detector, rtable_fam, members, core_idx, use_bokeh=False, ax=None):
    """
    Fill the amplitude timeline subplot.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    rtable_fam : structured array
        Handle to a subset of the Repeaters table containing all members of
        a family.
    members : int ndarray
        Indices of family members within the Repeaters table ordered by
        time.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    use_bokeh : bool, optional
        True if to render Bokeh figure, or False for matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    Figure object or Axis object

    """
    if use_bokeh:
        ymin = 0.25*np.amin(rtable_fam['windowAmp'][np.nonzero(
            rtable_fam['windowAmp'])])
        ymax = 4*np.amax(rtable_fam['windowAmp'])
    elif detector.get('amplims') == 'family':
        ymin = 0.25*np.amin(
            rtable_fam['windowAmp'][:, detector.get('printsta')][np.nonzero(
                rtable_fam['windowAmp'][:, detector.get('printsta')])])
        ymax = 4*np.amax(
            rtable_fam['windowAmp'][:, detector.get('printsta')])
    else:
        ymin = 0.25*np.amin(
            detector.get('rtable', 'windowAmp')[:, detector.get('printsta')][
                np.nonzero(detector.get('rtable', 'windowAmp')[:,
                           detector.get('printsta')])])
        ymax = 4*np.amax(
            detector.get('rtable', 'windowAmp')[:, detector.get('printsta')])
    # Prevent ymin being "too small" to be realistic
    if ymax > 1000:
        ymin = np.max((ymin, 1))
    elif ymax > 1:
        ymin = np.max((ymin, 1e-3))
    elif ymax > 1e-6:
        ymin = np.max((ymin, 1e-12))
    if use_bokeh:
        fig = bokeh_figure(
            title='Amplitude with Time (Click name to hide)',
            y_axis_type='log', y_range=[ymin, ymax])
        fig.yaxis.axis_label = 'Counts'
        if detector.get('nsta') <= 8:
            palette = all_palettes['YlOrRd'][9]
        else:
            palette = inferno(detector.get('nsta')+1)
        for sta, staname in enumerate(detector.get('station')):
            fig.circle(
                detector.get('plotvars')['rtimes'][members],
                rtable_fam['windowAmp'][:, sta],
                color=palette[sta], line_alpha=0, size=4, fill_alpha=0.5,
                legend_label=f'{staname}.{detector.get("channel")[sta]}')
        fig.legend.location = 'bottom_left'
        fig.legend.orientation = 'horizontal'
        fig.legend.click_policy = 'hide'
        return fig
    ax.plot_date(
        detector.get('plotvars')['rtimes_mpl'][members],
        rtable_fam['windowAmp'][:, detector.get('printsta')],
        'ro', alpha=0.5, markeredgecolor='r', markeredgewidth=0.5,
        markersize=3)
    ax.plot_date(
       detector.get('plotvars')['rtimes_mpl'][members[core_idx]],
       rtable_fam['windowAmp'][core_idx, detector.get('printsta')],
       'ko', markeredgecolor='k', markeredgewidth=0.5, markersize=3)
    ax.set_ylim(ymin, ymax)
    return ax


def subplot_spacing(
        detector, members, core_idx, use_bokeh=False, ax=None):
    """
    Fill the temporal spacing timeline subplot.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    members : int ndarray
        Indices of family members within the Repeaters table ordered by
        time.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    use_bokeh : bool, optional
        True if to render Bokeh figure, or False for matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.

    Returns
    -------
    Figure object or Axis object

    """
    catalog = detector.get('plotvars')['rtimes_mpl'][members]
    spacing = np.diff(catalog)*24
    if use_bokeh:
        fig = bokeh_figure(
            title='Time since Previous Event',
            y_axis_type='log', y_range=[1e-3, 2*np.max(spacing)])
        fig.yaxis.axis_label = 'Interval (hr)'
        fig.circle(
            detector.get('plotvars')['rtimes'][members[1:]], spacing,
            color='red', line_alpha=0, size=4, fill_alpha=0.5)
        return fig
    ax.plot_date(
        catalog[1:], spacing, 'ro', alpha=0.5,
        markeredgecolor='r', markeredgewidth=0.5, markersize=3)
    if core_idx > 0:
        ax.plot_date(
            catalog[core_idx], spacing[core_idx-1], 'ko',
            markeredgecolor='k', markeredgewidth=0.5, markersize=3)
    ax.set_ylim(1e-3, np.max(spacing)*2)
    return ax


def subplot_correlation(
        detector, members, core_idx, use_bokeh=False, ax=None,
        ccc_full=None):
    """
    Fill the cross-correlation timeline subplot.

    If using matplotlib:
        Plots a single row from the full cross-correlation matrix
        corresponding to whichever row has the highest sum. If the value
        is stored in the Correlation table it will be plotted as a filled
        red circle, otherwise it will be plotted with an open red circle at
        the value of detector.get('cmin'). The core is denoted with a black
        symbol.

    If using Bokeh:
        Plots the row from the full cross-correlation matrix that
        corresponds to the current core. All are plotted as filled red
        circles.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    members : int ndarray
        Indices of family members within the Repeaters table ordered by
        time.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    use_bokeh : bool, optional
        True if to render Bokeh figure, or False for matplotlib figure.
    ax : Axis object, optional
        If using matplotlib, the Axis handle in which to plot.
    ccc_full : float ndarray, optional
        Filled correlation matrix, must be passed with use_bokeh.

    Returns
    -------
    Figure object or Axis object

    """
    if use_bokeh:
        fig = bokeh_figure(
            y_range=[-0.02, 1.02],
            title='Cross-Correlation Coefficient with Core Event')
        fig.yaxis.axis_label = 'CCC'
        fig.circle(detector.get('plotvars')['rtimes'][members],
                   ccc_full[core_idx, :].tolist()[0], color='red',
                   line_alpha=0, size=4, fill_alpha=0.5)
        return fig
    # Get correlation values for row with highest sum
    ccc_maxrow = redpy.correlation.subset_matrix(
        detector.get('plotvars')['ids'][members],
        detector.get('plotvars')['ccc_sparse'])
    # Create mask to only plot each point once
    cmask = np.zeros(len(ccc_maxrow), dtype=bool)
    cmask[ccc_maxrow >= detector.get('cmin')] = True
    catalog = detector.get('plotvars')['rtimes_mpl'][members]
    # Plot closed circles for values that exist, open where undefined
    ax.plot_date(catalog[cmask], ccc_maxrow[cmask], 'ro', alpha=0.5,
                 markeredgecolor='r', markeredgewidth=0.5, markersize=3)
    ax.plot_date(catalog[~cmask], 0*ccc_maxrow[~cmask]+detector.get('cmin'),
                 'wo', alpha=0.5, markeredgecolor='r', markeredgewidth=0.5)
    # Plot black dot for core event
    if ccc_maxrow[core_idx] >= detector.get('cmin'):
        ax.plot_date(catalog[core_idx], ccc_maxrow[core_idx], 'ko',
                     markeredgecolor='k', markeredgewidth=0.5, markersize=3)
    else:
        ax.plot_date(catalog[core_idx], detector.get('cmin'), 'wo',
                     markeredgecolor='k', markeredgewidth=0.5, markersize=3)
    return ax


def subplot_waveforms(
        detector, rtable_fam, core_idx, ax, plot_single=False, sta_idx=0):
    """
    Fill the waveform subplot.

    Has different behaviors based on whether a single station or many
    stations are part of the run. If a single station, will show all
    waveforms either as wiggle plots (<12 events) or an image. If multiple
    stations are queried then the core will be plotted in black and the
    stack of all other events in red as wiggles.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    rtable_fam : structured array
        Handle to a subset of the Repeaters table containing all members of
        a family.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    ax : Axis object
        Subplot axis to modify in place.
    plot_single : bool, optional
        Plots all events from a single station, False by default.
    sta : int, optional
        Station index to plot when using plot_single option, 0 by default.

    Returns
    -------
    ax : Axis object

    """
    time_vector = np.arange(
        -0.5*detector.get('winlen')/detector.get('samprate'),
        1.5*detector.get('winlen')/detector.get('samprate'),
        1/detector.get('samprate'))
    if detector.get('nsta') == 1 or plot_single:
        data = np.zeros((len(rtable_fam), int(detector.get('winlen')*2)))
        for i, row in enumerate(rtable_fam):
            data[i, :] = prep_wiggle(
                row['waveform'], sta_idx, row['windowStart'],
                row['windowAmp'][sta_idx], detector)
        if len(rtable_fam) > 12:
            ax.imshow(
                data, aspect='auto', vmin=-1, vmax=1, cmap='RdBu',
                interpolation='nearest',
                extent=[np.min(time_vector), np.max(time_vector),
                        len(rtable_fam)-0.5, -0.5])
        else:
            for i in range(len(rtable_fam)):
                ax.plot(time_vector, data[i, :]/2+i, 'k', linewidth=0.25)
            ax.set_xlim([np.min(time_vector), np.max(time_vector)])
            ax.set_ylim([-0.5, 0.5+i])
    else:
        for sta in range(detector.get('nsta')):
            data_stack = np.zeros((int(detector.get('winlen')*2), ))
            for row in rtable_fam:
                data_stack += prep_wiggle(
                    row['waveform'], sta, row['windowStart'],
                    row['windowAmp'][sta], detector)
            data_stack = data_stack/(np.max(np.abs(data_stack))+1e-12)
            data_stack[data_stack > 1] = 1
            data_stack[data_stack < -1] = -1
            data_core = prep_wiggle(
                rtable_fam['waveform'][core_idx], sta,
                rtable_fam['windowStart'][core_idx],
                rtable_fam['windowAmp'][core_idx, sta], detector)
            # Plot
            ax.plot(time_vector, data_stack - 1.75*sta, 'r', linewidth=1)
            ax.plot(time_vector, data_core - 1.75*sta, 'k', linewidth=0.25)
    return ax


def subplot_fft(detector, rtable_fam, core_idx, ax):
    """
    Fill the FFT subplot.

    This plot shows the amplitude spectrum from the Fourier transform summed
    over all stations. The core is always plotted in black, and the sum over
    all events in red.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    rtable_fam : structured array
        Handle to a subset of the Repeaters table containing all members of
        a family.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    ax : Axis object
        Subplot axis to modify in place.

    """
    freq = np.linspace(0, detector.get('samprate')/2,
                       int(detector.get('winlen')/2))
    fftc = np.zeros((int(detector.get('winlen')/2),))
    fftm = np.zeros((int(detector.get('winlen')/2),))
    for sta in range(detector.get('nsta')):
        fft = np.abs(np.real(rtable_fam['windowFFT'][core_idx, int(
            sta*detector.get('winlen')):int(sta*detector.get(
                'winlen')+detector.get('winlen')/2)]))
        fft = fft/(np.amax(fft)+1.0/1000)
        fftc = fftc+fft
        ffts = np.mean(np.abs(np.real(rtable_fam['windowFFT'][:, int(
            sta*detector.get('winlen')):int(sta*detector.get(
                'winlen')+detector.get('winlen')/2)])), axis=0)
        fftm = fftm + ffts/(np.amax(ffts)+1.0/1000)
    ax.plot(freq, fftm, 'r', linewidth=1)
    ax.plot(freq, fftc, 'k', linewidth=0.25)


def wiggle_plot(data, figsize, outfile):
    """
    Plot a waveform with no decorations (e.g., for a core image).

    Parameters
    ----------
    data : float ndarray
        Waveform data to plot.
    figsize : tuple
        Output figure size as (width, height).
    outfile : str
        Path and filename for saving the figure.
    detector : Detector object
        Primary interface for handling detections.
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.plot(data, 'k', linewidth=0.25)
    plt.autoscale(tight=True)
    plt.savefig(outfile, dpi=100)
    plt.close(fig)


def wiggle_plot_all(detector, rtable_fam, members, ordered, outfile):
    """
    Create a plot with all waveforms on all stations in separate subplots.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    rtable_fam : structured array
        Handle to a subset of the Repeaters table containing all members of
        a family.
    members : int ndarray
        Indices of family members within the Repeaters table.
    ordered : bool
        True if members have been ordered (e.g., by OPTICS), False if by
        time.
    outfile : str
        Path and filename for saving the figure.


    """
    fig = plt.figure(figsize=(10, 4*(np.ceil(detector.get('nsta')/2))))
    for sta in range(detector.get('nsta')):
        ax = fig.add_subplot(int(np.ceil((detector.get('nsta'))/2.)), 2, sta+1)
        title_text = (
            f"{detector.get('station')[sta]}.{detector.get('channel')[sta]}")
        if ordered:
            title_text += ' (Ordered)'
        else:
            ax = _add_horizontal_annotations(
                ax, detector.get('plotvars')['rtimes_mpl'][members], detector)
        plt.title(title_text, fontweight='bold')
        subplot_waveforms(
            detector, rtable_fam, 0, ax, plot_single=True, sta_idx=sta)
        ax.yaxis.set_visible(False)
        plt.xlabel('Time Relative to Trigger (seconds)', style='italic')
    plt.tight_layout()
    plt.savefig(outfile, dpi=100)
    plt.close(fig)


def _add_horizontal_annotations(ax, evtimes, detector):
    """
    Plot annotations horizontally across an image (e.g., waveforms, matrix).

    Parameters
    ----------
    ax : Axis object
        Handle to the matplotlib axis.
    evtimes : float ndarray
        Sorted array of event times plotted on each row of the image.
    detector : Detector object
        Primary interface for handling detections.

    Returns
    -------
    Axis object

    """
    if detector.get('anotfile') != '':
        annotations = pd.read_csv(detector.get('anotfile'))
        for anot in range(len(annotations)):
            # Translate from date to vertical position in the image
            vertical_location = np.interp(mdates.date2num(
                pd.to_datetime(annotations['Time'][anot])),
                evtimes, np.array(range(len(evtimes))))
            # Plot if within the time span of the image
            if vertical_location != 0:
                ax.axhline(np.floor(vertical_location)+0.5, color='k',
                           linewidth=annotations['Weight'][anot]/2.,
                           linestyle=annotations['Line Type'][anot])
    return ax


def _format_family_image(axes, detector):
    """
    Handle formatting for each axis in the family image.

    Parameters
    ----------
    axes : list of Axis objects
        List of subplot axes to modify.
    detector : Detector object
        Primary interface for handling detections.

    Returns
    -------
    axes : list of Axis objects

    """
    date_format = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    time_vector = np.arange(
        -0.5*detector.get('winlen')/detector.get('samprate'),
        1.5*detector.get('winlen')/detector.get('samprate'),
        1/detector.get('samprate'))
    axes[0].get_yaxis().set_visible(False)
    axes[0].set_xlim((np.min(time_vector), np.max(time_vector)))
    axes[0].axvline(
        x=-0.1*detector.get('winlen') / detector.get('samprate'),
        color='k', ls='dotted')
    axes[0].axvline(
        x=0.9*detector.get('winlen') / detector.get('samprate'),
        color='k', ls='dotted')
    if detector.get('nsta') > 1:
        axes[0].set_ylim((-1.75*(detector.get('nsta')-1)-1, 1))
        # Add station labels
        stas = detector.get('station')
        chas = detector.get('channel')
        for i, sta in enumerate(stas):
            axes[0].text(
                np.min(time_vector)-0.1, -1.75*i,
                f'{sta}\n{chas[i]}', horizontalalignment='right',
                verticalalignment='center')
    axes[0].set_xlabel('Time Relative to Trigger (seconds)', style='italic')
    axes[1].get_yaxis().set_visible(False)
    axes[1].set_xlim(0, detector.get('fmax')*1.5)
    axes[1].legend(['Stack', 'Core'], loc='upper right', frameon=False)
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
    axes[4].set_ylim(detector.get('cmin')-0.02, 1.02)
    axes[4].set_xlabel('Date', style='italic')
    axes[4].set_ylabel('Cross-correlation coefficient', style='italic')
    return axes
