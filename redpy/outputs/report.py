# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling functions related to creating detailed family reports.

The primary function of this module is to support the .output() method of
Detector() objects. The .output() method generates various images and .html
files so the user may easily browse and export the contents of REDPy's
detections.
"""
import os
import shutil

import numpy as np
from bokeh.models import Panel, Tabs
from bokeh.plotting import gridplot, output_file, save

import redpy.correlation
import redpy.optics
import redpy.outputs.html
import redpy.outputs.image
import redpy.outputs.timeline


def assemble_bokeh_timeline_report(
        detector, members, rtable_fam, core_idx, fnum, ccc_full=None):
    """
    Assemble an interactive HTML timeline with Bokeh for a family report.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    fnum : int
        Family to be inspected.
    members : int ndarray
        Indices of family members within the Repeaters table ordered by time.
    core_idx : int
        Index corresponding to position of core event within ordered family.
    ccc_full : float ndarray, optional
        Filled correlation matrix.

    """
    plots = []
    tabtitles = []
    plotformat = 'amplitude,spacing,correlation'
    # Create each of the subplots specified in the configuration file
    for plot in plotformat.split(','):
        if plot == 'amplitude':
            # Plot EQ Rates (Repeaters and Orphans)
            plots.append(redpy.outputs.image.subplot_amplitude(
                detector, rtable_fam, members, core_idx, use_bokeh=True))
            tabtitles += ['Amplitude']
        elif plot == 'spacing':
            # Plot Frequency Index
            plots.append(redpy.outputs.image.subplot_spacing(
                detector, members, core_idx, use_bokeh=True))
            tabtitles += ['Spacing']
        elif plot == 'correlation':
            # Plot Cluster Longevity
            plots.append(redpy.outputs.image.subplot_correlation(
                detector, members, core_idx, use_bokeh=True,
                ccc_full=ccc_full))
            tabtitles += ['Correlation']
        else:
            print(f'{plot} is not a valid plot type. Moving on.')
    for plot in plots:
        plot.x_range = plots[0].x_range
    for plot in plots:
        # pylint: disable=W0212
        # I trust use of protected member here.
        plot = redpy.outputs.timeline._add_bokeh_annotations(detector, plot)
        # pylint: enable=W0212
    gridplot_items = []
    pnum = 0
    for plot in plotformat.split(','):
        # '+' groups plots into tabs
        if '+' in plot:
            tabs = []
            for pft in range(len(plot.split('+'))):
                tabs = tabs + [Panel(child=plots[pnum+pft],
                                     title=tabtitles[pnum+pft])]
            gridplot_items = gridplot_items + [[Tabs(tabs=tabs)]]
            pnum += pft+1
        else:
            gridplot_items = gridplot_items + [[plots[pnum]]]
            pnum += 1
    plot = gridplot(gridplot_items)
    filepath = os.path.join(detector.get('output_folder'), 'reports',
                            f'{fnum}-report-bokeh.html')
    output_file(filepath, title=(f'{detector.get("title")} - '
                                 f'Cluster {fnum} Detailed Report'))
    save(plot)


def create_report(detector, fnum, ordered=False, skip_recalculate_ccc=False,
                  matrixtofile=False):
    """
    Create more detailed output plots for a single family in '/reports'.

    detector : Detector object
        Primary interface for handling detections.
    fnum : int
        Family to render a report for.
    ordered : bool, optional
        True if members should be ordered by OPTICS, else by time.
    skip_recalculate_ccc : bool, optional
        True if user wishes to skip recalculating the full correlation matrix.
    matrixtofile : bool, optional
        True if correlation should be written to file.

    """
    if detector.get('verbose'):
        print(f'Creating report for family {fnum}...')
    rpath = os.path.join(detector.get('output_folder'), 'reports')
    shutil.copy(os.path.join(detector.get(
        'output_folder'), 'families', f'{fnum}.png'), os.path.join(
            rpath, f'{fnum}-report.png'))
    corenum = detector.get('ftable', 'core', fnum)
    members = detector.get_members(fnum)
    members = members[
        np.argsort(detector.get('plotvars')['rtimes_mpl'][members])]
    rtable_fam = detector.get('rtable', row=members)
    ccc_fam = redpy.correlation.subset_matrix(
        detector.get('plotvars')['ids'][members],
        detector.get('plotvars')['ccc_sparse'], return_type='matrix')
    if not skip_recalculate_ccc:
        if len(members) > 1000:
            print('There are a lot of members in this family! '
                  'Consider using the option to skip recalculating '
                  'the cross-correlation matrix...')
        print('Computing full correlation matrix; this will take time'
              ' if the family is large')
        ccc_full = redpy.correlation.make_full(
            detector, rtable_fam, ccc_fam)
    else:
        ccc_full = ccc_fam.copy()
    core_idx = np.where(members == corenum)[0]
    assemble_bokeh_timeline_report(
        detector, members, rtable_fam, core_idx, fnum, ccc_full=ccc_full)
    if ordered:
        members, rtable_fam, ccc_fam, ccc_full = _reorder_by_optics(
            members, rtable_fam, ccc_fam, ccc_full)
    if matrixtofile:
        np.save(os.path.join(rpath, f'{fnum}-ccc_full.npy'), ccc_full)
        np.save(os.path.join(rpath, f'{fnum}-evTimes.npy'), detector.get(
            'plotvars')['rtimes'][members])
    redpy.outputs.image.correlation_matrix_plot(
        detector, ccc_fam, ccc_full, members, ordered, skip_recalculate_ccc,
        os.path.join(rpath, f'{fnum}-reportcmat.png'))
    redpy.outputs.image.wiggle_plot_all(
        detector, rtable_fam, members, ordered,
        os.path.join(rpath, f'{fnum}-reportwaves.png'))
    with open(os.path.join(rpath, f'{fnum}-report.html'), 'w',
              encoding='utf-8') as file:
        redpy.outputs.html.write_html_header(detector, fnum, file, report=True)
        file.write('</center></body></html>')


def _reorder_by_optics(members, rtable_fam, ccc_fam, ccc_full):
    """Order variables by OPTICS rather than by time."""
    distance_matrix = 1 - ccc_full
    sort = np.argsort(sum(distance_matrix))[::-1]
    distance_matrix = distance_matrix[sort, :]
    distance_matrix = distance_matrix[:, sort]
    members = members[sort]
    rtable_fam = rtable_fam[sort]
    ccc_fam = ccc_fam[sort, :]
    ccc_fam = ccc_fam[:, sort]
    ccc_full = ccc_full[sort, :]
    ccc_full = ccc_full[:, sort]
    order = redpy.optics.OPTICS(distance_matrix).run(1)
    members = members[order]
    ccc_fam = ccc_fam[order, :]
    ccc_fam = ccc_fam[:, order]
    ccc_full = ccc_full[order, :]
    ccc_full = ccc_full[:, order]
    rtable_fam = rtable_fam[order]
    return members, rtable_fam, ccc_fam, ccc_full
