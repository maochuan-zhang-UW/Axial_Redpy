# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

import redpy


def get_event_times_from_csv(
        csvfile, time_column_name, sep, config, start_time=None, end_time=None,
        arrivals=False):
    """
    Reads event times from a catalog.

    Parameters
    ----------
    csvfile : str
        Path to a .csv-like catalog.
    time_column_name : str
        Name of the column that contains event times.
    sep : str
        Separator (or delimiter) between columns.
    config : Config object
        Describes the run parameters.
    start_time : UTCDateTime object, optional
        Events returned must have happened after this date.
    end_time : UTCDateTime object, optional
        Events returned must have happened before this date.
    arrivals : bool, optional
        Use P-wave arrival time instead of origin time.

    Returns
    -------
    event_list : ndarray of UTCDateTime objects

    """
    catalog = pd.read_csv(csvfile, sep=sep)
    if arrivals:
        handle_arrivals(catalog, time_column_name, config)
    event_list = np.array(
        [UTCDateTime(event) for event in catalog[time_column_name]])
    event_list.sort()
    if start_time:
        event_list = event_list[event_list >= start_time]
    if end_time:
        event_list = event_list[event_list <= end_time]
    return event_list


def handle_arrivals(catalog, time_column_name, config, write_to_column=None):
    """
    Handle getting the P-wave arrivals into a single column.

    If arrival column doesn't exist in the catalog, it is calculated. By
    default, the combined arrival column overwrites the 'Time' column, but
    can be appended to the end by using the 'write_to_column' option.

    Parameters
    ----------
    catalog : DataFrame object
        Catalog with event times and locations.
    time_column_name : str
        Name of the event time column.
    config : Config object
        Describes the run parameters.
    write_to_column : str, optional
        Column name to append to overwrite or append.

    Returns
    -------
    catalog : DataFrame object
        Modified catalog.

    """
    if 'Arrival_p' not in catalog.columns:
        latitude_center = np.mean(
            np.array(config.get('stalats').split(',')).astype(float))
        longitude_center = np.mean(
            np.array(config.get('stalons').split(',')).astype(float))
        catalog = calculate_arrivals(
            catalog, latitude_center, longitude_center, ['p','P'], config,
            time_column_name=time_column_name)
    if write_to_column:
        time_column_name = write_to_column
    catalog[time_column_name] = catalog['Arrival_p'].replace(
        to_replace='NaN', value=np.nan).fillna(
            catalog['Arrival_P'])
    return catalog


def prepare_catalog(ttimes, config):
    """
    Downloads and formats event catalog from external datacenter.

    Data are queried from three regions (local, regional, teleseismic) based
    on the settings in config. Times are taken from the first and last trigger
    times in ttable, so if there are large gaps in ttable, this function is
    agnostic to them. Updates the catalog if a file exists to reduce query
    overhead.

    Parameters
    ----------
    ttimes : float ndarray
        Times of all triggers as matplotlib dates.
    config : Config object
        Describes the run parameters.

    Returns
    -------
    external_catalogs : list of DataFrame objects

    """
    tmin = UTCDateTime(mdates.num2date(np.min(ttimes))) - 1800
    tmax = UTCDateTime(mdates.num2date(np.max(ttimes))) + 30
    external_catalogs = []
    for region in ['local', 'regional', 'teleseismic']:
        fname = os.path.join(config.get('output_folder'), f'external_{region}.txt')
        if os.path.exists(fname):
            catalog = pd.read_csv(fname, delimiter='|')
            # Get missing events before and after currently saved events
            if len(catalog) > 0:
                tmin_catalog = UTCDateTime(np.min(catalog['Time']))-1
                tmax_catalog = UTCDateTime(np.max(catalog['Time']))+1
                catalog_before = query_external(
                    region, tmin, tmin_catalog,config)
                catalog_after = query_external(
                    region, tmax_catalog, tmax, config)
                catalog = pd.concat([catalog_before, catalog, catalog_after],
                                    axis=0, ignore_index=True)
            else:
                catalog = query_external(region, tmin, tmax, config)
        else:
            catalog = query_external(region, tmin, tmax, config)
        catalog.to_csv(fname, index=False, sep='|')
        external_catalogs.append(catalog)
    return external_catalogs


def save_external_catalog(csvfile, config, arrivals=False, start_time=None,
                          end_time=None, rtable=None, delimiter=','):
    """
    Download and save catalog from external FDSN webservice to csvfile.

    Parameters
    ----------
    csvfile : str
        File to save catalog to.
    config : Config object
        Describes the run parameters.
    arrivals : bool, optional
        If True, calculate P-wave arrival to center of network.
    starttime : str, optional
        Start time for catalog query.
    endtime : str, optional
        End time for catalog query.
    rtable : Table object, optional
        Handle to the Repeaters table.
    delimiter : str, optional
        Delimiter to use between columns in output .csv file.

    """
    if not end_time:
        end_time = UTCDateTime()
        print(f'Defaulting to end time of "now" ({end_time})')
    if not start_time:
        start_time = end_time - config.get('nsec')
        if rtable:
            if rtable.attrs.ptime:
                start_time = rtable.attrs.ptime
        print(f'Defaulting to start time of {start_time}')
    catalog = query_external('local', UTCDateTime(start_time),
                             UTCDateTime(end_time), config, arrivals)
    if len(catalog) == 0:
        print('No events found!')
    catalog.to_csv(csvfile, index=False, sep=delimiter)


def query_external(region, tmin, tmax, config, arrivals=True):
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
    config : Config object
        Describes the run parameters.

    Returns
    -------
    catalog : pandas DataFrame
        Formatted event catalog.

    """
    latitude_center = np.mean(np.array(config.get('stalats').split(',')).astype(float))
    longitude_center = np.mean(np.array(config.get('stalons').split(',')).astype(float))
    if region in 'local':
        minrad = 0
        maxrad = config.get('locdeg')
        minmag = -10
        phase_list=['p','s','P','S']
    elif region in 'regional':
        minrad = config.get('locdeg')
        maxrad = config.get('regdeg')
        minmag = config.get('regmag')
        phase_list = ['p','s','P','S','PP','SS']
    else:
        minrad = config.get('regdeg')
        maxrad = 180
        minmag = config.get('telemag')
        phase_list = ['P','S','PP','SS','PcP','ScS','PKiKP','PKIKP']
    base_url = Client(config.get('datacenter')).base_url
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
        if len(catalog) == 10000:
            offset = 0
            while not len(catalog) % 10000:
                offset += 10000
                catalog2 = pd.read_csv(query_url+f'&offset={offset}',
                                       delimiter='|')
                if len(catalog2) > 0:
                    catalog = catalog.append(catalog2, ignore_index=True)
                else: # Remainder will still be 0 so we'd be stuck in the loop
                    break
    except: # pylint: disable=bare-except
        # Pass an empty dataframe with the correct columns
        catalog = pd.DataFrame(
            columns=['EventID', 'Time', 'Latitude', 'Longitude', 'Depth/km',
                     'Magnitude','EventLocationName'])
        print((f'Failed to download {region} event catalog from '
               f'{config.get("datacenter")}'))
    catalog.columns = catalog.columns.str.replace(' ','')
    catalog.columns = catalog.columns.str.replace('#','')
    catalog.drop(
        columns=['Author', 'Catalog', 'Contributor', 'ContributorID',
                 'MagType', 'MagAuthor', 'EventType'],
        errors='ignore', inplace=True)
    if arrivals:
        catalog = calculate_arrivals(
            catalog, latitude_center, longitude_center, phase_list, config)
    return catalog


def calculate_arrivals(
        catalog, latitude_center, longitude_center, phase_list, config,
        time_column_name='Time'):
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
    config : Config object
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
        latitudes = np.squeeze(catalog.filter(regex='[lL]at.*'))
        longitudes = np.squeeze(catalog.filter(regex='[lL]on.*'))
        depths = np.squeeze(catalog.filter(regex='[dD]ep.*m'))
        if len(depths) < 1:
            depths = np.squeeze(catalog.filter(regex='[dD]ep.*'))
        for i in range(len(catalog)):
            deg = locations2degrees(latitudes[i], longitudes[i],
                                    latitude_center, longitude_center)
            # TauPyModel misbehaves if it's used too much
            # Determine if it needs to be reloaded (every 100 runs)
            taupymodel_runs += 1
            if np.remainder(taupymodel_runs,100) == 0:
                model = TauPyModel(model="iasp91")
            arrivals = model.get_travel_times(
                source_depth_in_km=max(0, depths[i]),
                distance_in_degree=deg, phase_list=phase_list)
            if len(arrivals) > 0:
                for arrival in arrivals:
                    atime = (UTCDateTime(catalog[time_column_name][i])
                             + arrival.time)
                    catalog[f'Arrival_{arrival.name}'][i] = f'{atime}'
    return catalog


def match_external(windowAmp, ftable, fnum, file, rtimes, external_catalogs,
                   config):
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
    file : file
        HTML file to write to.
    rtimes : datetime ndarray
        Times of repeaters as datetimes.
    external_catalogs : list of DataFrame objects
        External catalogs to check against, with 'Arrivals_' columns.
    config : Config object
        Describes the run parameters.

    """
    members = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')
    if config.get('matchmax') == 0 or config.get('matchmax') > len(members):
        order = np.argsort(rtimes[members])
        matchstring = ('</br><b>ComCat matches (all events):</b></br>'
            '<div style="overflow-y: auto; height:100px; width:1200px;">')
    else:
        nlargest = np.argsort(windowAmp[members])[::-1][:config.get('matchmax')]
        members = members[nlargest]
        order = np.argsort(rtimes[members])
        matchstring = f"""
            </br><b>ComCat matches ({config.get('matchmax')} largest events):</b></br>
            <div style="overflow-y: auto; height:100px; width:1200px;">
            """
    pc_string = ['Potential', 'Conflicting']
    region = ['local', 'regional', 'teleseismic']
    prestring = ['', '<div style="color:red">', '<div style="color:red">']
    poststring = ['</br>', '</div>', '</div>']
    nfound = 0
    local_lats = np.array([])
    local_lons = np.array([])
    local_deps = np.array([])
    for i in members[order]:
        ev_time = UTCDateTime(rtimes[i])
        cflag = 0
        for j, cat in enumerate(external_catalogs):
            # Get the arrival names only from Arrival_ column names
            anames = [cat.filter(like='Arrival').columns[i].split('_')[1] \
                      for i in range(len(cat.filter(like='Arrival').columns))]
            arrivals = cat.filter(like='Arrival').to_numpy().astype(str)
            matched = np.any(((arrivals >= format(ev_time-config.get('serr'))) & \
                              (arrivals <= format(ev_time+config.get('serr')))),axis=1)
            found = arrivals[matched]
            if len(found) > 0:
                # Convert from strings to time differences
                vfunc = np.vectorize(
                    lambda x, ev_time:np.abs(UTCDateTime(x)-ev_time))
                found[found=='NaN'] = 'nan' # Just in case
                found[found!='nan'] = vfunc(found[found!='nan'],ev_time)
                # Convert to float, otherwise numpy complains
                found[found=='nan'] = np.nan
                found = found.astype(float)
                for i in range(len(found)):
                    bestmatch = np.argwhere(found == np.nanmin(found))[0]
                    catmatch = cat[matched].iloc[bestmatch[0]]
                    matchstring += (
                        f'{prestring[j]}{pc_string[cflag]} {region[j]} match: '
                        f'{catmatch["Time"]} ({catmatch["Latitude"]:5.6f}, '
                        f'{catmatch["Longitude"]:6.6f}) '
                        f'{catmatch["Depth/km"]:3.1f}km '
                        f'M{catmatch["Magnitude"]:3.1f} - '
                        f'{catmatch["EventLocationName"]} - '
                        f'({anames[bestmatch[1]]}) '
                        f'{np.nanmin(found):4.2f} s{poststring[j]}')
                    if j == 0: # Local catalog
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
                os.path.join(config.get('output_folder'), 'clusters', f'map{fnum}.png'),
                config)
            file.write(f'<img src="map{fnum}.png"></br>')
    else:
        matchstring+='No matches found</br></div>'
    file.write(matchstring)


def get_median_locations(config, regional=False, distant=False):
    """
    Parse .html output files for locations and create a DataFrame of medians.

    Parameters
    ----------
    config : Config object
        Describes the run parameters.
    regional : bool, optional
        If True, include regional matches.
    distant : bool, optional
        If True, include regional and teleseismic matches.

    Returns
    -------
    df : DataFrame object

    """
    flist = np.array(glob.glob(os.path.join(config.get('output_folder'), 'clusters',
                                            '*.html')))
    fnums = [int(os.path.basename(fname).split('.')[0]) for fname in flist]
    df = pd.DataFrame(columns=['Latitude', 'Longitude', r'Depth/km'],
                      index=range(max(fnums)))
    for i in np.argsort(fnums):
        with open(flist[i], 'r', encoding='utf-8') as famfile:
            data = famfile.readlines()
        famfile.close()
        locs = {'lats': np.array([]),
                'lons': np.array([]),
                'deps': np.array([])}
        lines = data[-1].split('>')
        for line in lines:
            line = line.strip()
            if line.count('Potential local match:'):
                locs = append_location(locs, line)
            elif line.count('regional') and (distant or regional):
                locs = append_location(locs, line)
            elif line.count('teleseismic') and distant:
                locs = append_location(locs, line)
        if len(locs['lats']) > 0:
            df.loc[fnums[i]] = [np.median(locs['lats']),
                                np.median(locs['lons']),
                                np.median(locs['deps'])]
    return df


def append_location(locs, line):
    """
    Append location contained in text line to location arrays.

    Parameters
    ----------
    locs : dict
        Dictionary containing float ndarrays for latitudes, longitudes, and
        depths of matches.
    line : str
        String from .html file containing matched event locations.

    Returns
    -------
    locs : dict
        Modified dictionary of locations.

    """
    locs['lats'] = np.append(locs['lats'],float(line.split()[4].strip('(,')))
    locs['lons'] = np.append(locs['lons'],float(line.split()[5].strip(')')))
    locs['deps'] = np.append(locs['deps'],float(line.split()[6].strip('km')))
    return locs
