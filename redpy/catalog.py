# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import os

import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

import redpy


def prepare_catalog(ttimes, opt):
    """
    Downloads and formats event catalog from external datacenter.
    
    Data are queried from three regions (local, regional, teleseismic) based
    on the settings in opt. Times are taken from the first and last trigger
    times in ttable, so if there are large gaps in ttable, this function is
    agnostic to them. Updates the catalog if a file exists to reduce query
    overhead.
    
    Parameters
    ----------
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    opt : Options object
        Describes the run parameters.
    
    Returns
    -------
    external_catalogs : list of DataFrame objects
    
    """
    tmin = UTCDateTime(mdates.num2date(np.min(ttimes))) - 1800
    tmax = UTCDateTime(mdates.num2date(np.max(ttimes))) + 30
    external_catalogs = []
    for region in ['local', 'regional', 'teleseismic']:
        fname = os.path.join(opt.output_folder, f'external_{region}.txt')
        if os.path.exists(fname):
            catalog = pd.read_csv(fname, delimiter='|')
            # Get missing events before and after currently saved events
            if len(catalog) > 0:
                tmin_catalog = UTCDateTime(np.min(catalog['Time']))-1
                tmax_catalog = UTCDateTime(np.max(catalog['Time']))+1
                catalog_before = query_external(
                    region, tmin, tmin_catalog,opt)
                catalog_after = query_external(
                    region, tmax_catalog, tmax, opt)
                catalog = pd.concat([catalog_before, catalog, catalog_after],
                                    axis=0, ignore_index=True)
            else:
                catalog = query_external(region, tmin, tmax, opt)
        else:
            catalog = query_external(region, tmin, tmax, opt)
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
    base_url = Client(datacenter).base_url
    query_url = base_url + ('/fdsnws/event/1/query'
                + f'?starttime={tmin}'
                + f'&endtime={tmax}'
                + f'&latitude={latitude_center}'
                + f'&longitude={longitude_center}'
                + f'&maxradius={maxrad}'
                + f'&minradius={minrad}'
                + f'&minmagnitude={minmag}'
                + '&orderby=time-asc&format=text&limit=10000')
    try:
        catalog = pd.read_csv(query_url, delimiter='|')
        # If the limit is returned
        if (len(catalog) == 10000):
            offset = 0
            while not (len(catalog) % 10000):
                offset += 10000
                catalog2 = pd.read_csv(query_url+f'&offset={offset}',
                                       delimiter='|')
                if len(catalog2) > 0:
                    catalog = catalog.append(catalog2, ignore_index=True)
                else: # Remainder will still be 0 so we'd be stuck in the loop
                    break
    except:
        # Pass an empty dataframe with the correct columns
        catalog = pd.DataFrame(
            columns=['EventID', 'Time', 'Latitude', 'Longitude', 'Depth/km',
                     'Magnitude','EventLocationName'])
        print(f'Failed to download {region} event catalog from {datacenter}')
    catalog.columns = catalog.columns.str.replace(' ','')
    catalog.columns = catalog.columns.str.replace('#','')
    catalog.drop(
        columns=['Author', 'Catalog', 'Contributor', 'ContributorID',
                 'MagType', 'MagAuthor', 'EventType'],
        errors='ignore', inplace=True)
    catalog = calculate_arrivals(
        catalog, latitude_center, longitude_center, phase_list, opt)
    return catalog


def calculate_arrivals(
        catalog, latitude_center, longitude_center, phase_list, opt):
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
    for phase in phase_list:
        catalog[f'Arrival_{phase}'] = 'NaN'
    if len(catalog) > 0:
        taupymodel_runs = 0
        model = TauPyModel(model="iasp91")
        for i in range(len(catalog)):
            deg = locations2degrees(catalog['Latitude'][i],
                                    catalog['Longitude'][i],
                                    latitude_center,
                                    longitude_center)
            # TauPyModel misbehaves if it's used too much
            # Determine if it needs to be reloaded (every 100 runs)
            taupymodel_runs += 1
            if np.remainder(taupymodel_runs,100) == 0:
                model = TauPyModel(model="iasp91")
            arrivals = model.get_travel_times(
                source_depth_in_km=max(0, catalog['Depth/km'][i]),
                distance_in_degree=deg, phase_list=phase_list)
            if len(arrivals) > 0:
                for arrival in arrivals:
                    catalog[f'Arrival_{arrival.name}'][i] = (
                        f'{UTCDateTime(catalog["Time"][i]) + arrival.time}')
    return catalog


def match_external(windowAmp, ftable, fnum, f, rtimes, external_catalogs, opt):
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
    rtimes : datetime ndarray
        Times of repeaters as datetimes.
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
        order = np.argsort(rtimes[members])
        matchstring = ('</br><b>ComCat matches (all events):</b></br>'
            '<div style="overflow-y: auto; height:100px; width:1200px;">')
    else:
        nlargest = np.argsort(windowAmp[members])[::-1][:opt.matchMax]
        members = members[nlargest]
        order = np.argsort(rtimes[members])
        matchstring = (f"""
            </br><b>ComCat matches ({opt.matchMax} largest events):</b></br>'
            '<div style="overflow-y: auto; height:100px; width:1200px;">
            """)
    pc = ['Potential', 'Conflicting']
    region = ['local', 'regional', 'teleseismic']
    prestring = ['', '<div style="color:red">', '<div style="color:red">']
    poststring = ['</br>', '</div>', '</div>']
    nfound = 0
    local_lats = np.array([])
    local_lons = np.array([])
    local_deps = np.array([])
    for m in members[order]:
        t = UTCDateTime(rtimes[m])
        cflag = 0
        for r, cat in enumerate(external_catalogs):
            # Get the arrival names only from Arrival_ column names
            anames = [cat.filter(like='Arrival').columns[i].split('_')[1] \
                      for i in range(len(cat.filter(like='Arrival').columns))]
            arrivals = cat.filter(like='Arrival').to_numpy().astype(str)
            matched = np.any(((arrivals >= format(t-opt.serr)) & \
                              (arrivals <= format(t+opt.serr))),axis=1)
            found = arrivals[matched]
            if len(found) > 0:
                # Convert from strings to time differences
                vfunc = np.vectorize(lambda x,t:np.abs(UTCDateTime(x)-t))
                found[found=='NaN'] = 'nan' # Just in case
                found[found!='nan'] = vfunc(found[found!='nan'],t)
                # Convert to float, otherwise numpy complains
                found[found=='nan'] = np.nan
                found = found.astype(float)
                for i in range(len(found)):
                    bestmatch = np.argwhere(found == np.nanmin(found))[0]
                    catmatch = cat[matched].iloc[bestmatch[0]]
                    matchstring += ('{}{} {} match: {} ({:5.6f}, {:6.6f}) '
                            '{:3.1f}km M{:3.1f} - {} - ({}) '
                            '{:4.2f} s{}').format(
                            prestring[r], pc[cflag], region[r],
                            catmatch['Time'],
                            catmatch['Latitude'], catmatch['Longitude'],
                            catmatch['Depth/km'], catmatch['Magnitude'],
                            catmatch['EventLocationName'],
                            anames[bestmatch[1]], np.nanmin(found),
                            poststring[r])
                    if r == 0: # Local catalog
                        local_lats = np.append(local_lats,
                                               catmatch['Latitude'])
                        local_lons = np.append(local_lons,
                                               catmatch['Longitude'])
                        local_deps = np.append(local_deps,
                                               catmatch['Depth/km'])
                    found[bestmatch[0],:] = np.nan
                    # Set cflag = 1 for 'conflicting'
                    cflag = 1
                nfound += 1
    if nfound > 0:
        matchstring += '</div>'
        matchstring += f'Total potential matches: {nfound}</br>'
        matchstring += f'Potential local matches: {len(local_deps)}</br>'
        if len(local_deps) > 0:
            redpy.plotting.create_local_map(
                local_lats, local_lons, local_deps,
                os.path.join(opt.output_folder, 'clusters', f'map{fnum}.png'),
                opt)
            f.write(f'<img src="map{fnum}.png"></br>')
    else:
        matchstring+='No matches found</br></div>'
    f.write(matchstring)
