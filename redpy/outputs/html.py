"""
Module for handling functions related to creating family .html files.

The primary function of this module is to support the .output() method of
Detector() objects. The .output() method generates various images and .html
files so the user may easily browse and export the contents of REDPy's
detections.
"""
import os

import numpy as np
from obspy import UTCDateTime

import redpy.locate
import redpy.outputs.mapping


FM_MAT_PATH = ('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/'
               'G_2015_HASH_Po_Clu_FM.mat')


def generate_html(detector, catalog_csv='axial_catalog_dd.csv'):
    """
    Write the .html files for the individual family pages.

    This file holds navigation, images, and basic statistics. May also
    include location information if an external catalog is queried.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    catalog_csv : str, optional
        Path to DD catalog CSV for location maps.

    """
    if detector.get('checkcomcat'):
        external_catalogs = redpy.locate.prepare_catalog(detector)
        redpy.outputs.mapping.get_tiles(detector)
    else:
        external_catalogs = []

    # Load FM catalog once for all families
    fm_catalog = {}
    if os.path.exists(FM_MAT_PATH):
        fm_catalog = redpy.outputs.mapping.load_fm_catalog(FM_MAT_PATH)

    printme = detector.get('ftable', 'printme')
    lastprint = detector.get('ftable', 'lastprint')
    for fnum in range(len(detector)):
        if printme[fnum] != 0 or lastprint[fnum] != fnum:
            # Generate Axial location map
            if os.path.exists(catalog_csv):
                redpy.outputs.mapping.create_axial_map(
                    detector, fnum, catalog_csv)
            # Generate focal mechanism plot
            if fm_catalog and os.path.exists(catalog_csv):
                redpy.outputs.mapping.create_axial_fm_plot(
                    detector, fnum, catalog_csv, fm_catalog)
            with open(os.path.join(detector.get('output_folder'), 'families',
                      f'{fnum}.html'), 'w', encoding='utf-8') as file:
                write_html_header(detector, fnum, file)
                if detector.get('checkcomcat'):
                    local = match_external_to_html(
                        detector, fnum, external_catalogs, file)
                    redpy.outputs.mapping.create_local_map(
                        detector, fnum, local)
                file.write('</center></body></html>')


def make_meta(runs='', path='./runs', topath='.', verbose=False):
    """
    Make "meta.html" to hold multiple meta overview pages.

    This page gathers the 'overview_meta.html' tabbed overviews within the
    output directories into a single page. This is intended to be used
    to monitor several runs simultaneously.

    Parameters
    ----------
    runs : str, optional
        Comma separated list of runs to include, matching the "groupname"
        in their configuration files.
    path : str, optional
        Relative path to where "meta.html" should be created; defaults to
        './runs' which is the default location for new run outputs.
    topath : str, optional
        Relative path from "meta.html" to the runs; defaults to same
        path.
    verbose : bool, optional
        Increase written print statements.

    """
    filename = os.path.join(path, 'meta.html')
    if verbose:
        print(f'Creating {filename}...')
    if not runs:
        print('No runs supplied, assuming "default" only')
        runs = 'default'
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(r'<html><head><title>REDPy Meta Overview</title></head>')
        file.write(r'<body style="padding:0;margin:0">')
        for run in runs.split(','):
            runpath = r'/'.join([topath, run.strip(), 'overview_meta.html'])
            file.write(rf"""
                <iframe src="{runpath}" title="{run}"
                    style="height:350px;width:1300px;border:none;"></iframe>
                    """)
        file.write('</body></html>')


def match_external_to_html(detector, fnum, external_catalogs, file):
    """
    Check repeater trigger times with arrival times from external catalog.

    Currently only supports checking the ANSS Comprehensive Earthquake
    Catalog (USGS ComCat). It writes these to .html and returns a dictionary
    of local matches to support mapping image files.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    fnum : int
        Family number.
    external_catalogs : list of DataFrame objects
        External catalogs to check against, with 'Arrivals_' columns.
    file : file handle
        Handle to open .html file to write to.

    Returns
    -------
    dict
        Dictionary of local match locations.

    """
    rtimes = detector.get('plotvars')['rtimes']
    members = detector.get_members(fnum)
    if (detector.get('matchmax') == 0) or (
            detector.get('matchmax') > len(members)):
        order = np.argsort(rtimes[members])
        matchstring = (
            '</br><b>ComCat matches (all events):</b></br>'
            '<div style="overflow-y: auto; height:100px; width:1200px;">')
    else:
        nlargest = np.argsort(detector.get(
            'plotvars')['amps'][members])[::-1][:detector.get('matchmax')]
        members = members[nlargest]
        order = np.argsort(rtimes[members])
        matchstring = f"""
            </br>
            <b>ComCat matches ({detector.get('matchmax')} largest events):</b>
            </br>
            <div style="overflow-y: auto; height:100px; width:1200px;">
            """
    nfound, matchstring, local = redpy.locate.match_external(
        detector, rtimes[members[order]], external_catalogs, matchstring)
    if nfound > 0:
        matchstring += '</div>'
        matchstring += f'Total potential matches: {nfound}</br>'
        matchstring += f'Potential local matches: {len(local["deps"])}</br>'
        if len(local['deps']) > 0:
            file.write(f'<img src="map{fnum}.png"></br>')
    else:
        matchstring += 'No matches found</br></div>'
    file.write(matchstring)
    return local


def write_html_header(detector, fnum, file, report=False):
    """
    Write the header containing statistics to the .html file.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    fnum : int
        Family number.
    file : file handle
        Handle to open .html file to write to.
    report : bool, optional
        If true, use report-specific formatting.

    """
    rtimes = detector.get('plotvars')['rtimes']
    corenum = detector.get('ftable', 'core', fnum)
    fam = detector.get_members(fnum)
    catalog = detector.get('plotvars')['rtimes_mpl'][fam]
    fam = fam[np.argsort(catalog)]
    catalog = np.sort(catalog)
    spacing = np.diff(catalog) * 24
    file.write('<html><head><title>')
    if report:
        file.write(f'{detector.get("title")} - Family {fnum} Detailed Report')
        topline = f'<em>Last updated: {UTCDateTime.now()}</em>'
        header = f'Family {fnum} - Detailed Report'
        coreimg = f'{fnum}-report'
        body = f"""
            <img src='{fnum}-reportwaves.png'></br></br>
            <iframe src="{fnum}-report-bokeh.html" width=1350 height=800
            style="border:none"></iframe></br>
            <img src='{fnum}-reportcmat.png'></br></br></br>"""
    else:
        file.write(f'{detector.get("title")} - Family {fnum}')
        if fnum > 0:
            prevlink = f"<a href='{fnum-1}.html'>&lt; Family {fnum-1}</a>"
        else:
            prevlink = " "
        if fnum < len(detector)-1:
            nextlink = f"<a href='{fnum+1}.html'>Family {fnum+1} &gt;</a>"
        else:
            nextlink = " "
        topline = f'{prevlink} &nbsp; | &nbsp; {nextlink}'
        header = f'Family {fnum}'
        coreimg = fnum
        families_path = os.path.abspath(os.path.join(
            detector.get('output_folder'), 'families', f'{fnum}.html'))
        file_url = f'file://{families_path}'
        body = (f'<img src="fam{fnum}.png"></br>'
                f'<small><a href="{file_url}">{file_url}</a></small></br>'
                f'<img src="map{fnum}.png" onerror="this.style.display=\'none\'">'
                f'</br>'
                f'<img src="fm{fnum}.png" onerror="this.style.display=\'none\'">'
                f'</br>')
    file.write(
        f"""</title>
        </head>
        <style>
            a {{color:red;}}
            body {{font-family:Helvetica; font-size:12px}}
            h1 {{font-size: 20px;}}
        </style>
        <body><center>
        {topline}</br>
        <h1>{header}</h1>
        <img src="{coreimg}.png" width=500 height=100></br></br>
        Number of events: {len(fam)}</br>
        Longevity: {detector.get('ftable', 'longevity', fnum):.2f} days</br>
        Mean event spacing: {np.mean(spacing):.2f} hours</br>
        Median event spacing: {np.median(spacing):.2f} hours</br>
        Mean frequency index: {np.mean(
            detector.get('plotvars')['mean_fi'][fam]):.2f}
        </br></br>
        First event: {UTCDateTime(rtimes[fam[0]]).isoformat()}</br>
        Core event: {UTCDateTime(rtimes[corenum]).isoformat()}</br>
        Last event: {UTCDateTime(rtimes[fam[-1]]).isoformat()}</br>
        {body}""")
