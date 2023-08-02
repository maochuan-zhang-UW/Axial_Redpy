# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling functions related to external catalog locations.

The primary function of this module is to support the .locate() method of
Detector() objects. The .locate() method uses various approaches to find
potential matches for REDPy's detections in an external catalog,
and assigning a likely location. This module also handles downloading
external catalogs and approximating arrival times.
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel


DISTANT_TYPES = {'tele': 'Teleseismic',
                 'distant': 'Regional+Teleseismic',
                 'regional': 'Regional',
                 'regional_notele': 'Regional (ignore Teleseisms)',
                 'regional3': '3+ Regional Matches',
                 'findphrase': 'Containing Phrase'}
REGIONS = ['local', 'regional', 'teleseismic']
_PC_STRING = ['Potential', 'Conflicting']
_PRESTRING = ['', '<div style="color:red">', '<div style="color:red">']
_POSTSTRING = ['</br>', '</div>', '</div>']


class DistantCounter():
    """Counter for distant matches within family .html files."""

    def __init__(self, filename, findphrase='', percent=90.):
        """
        Read file and save counts as attributes.

        Parameters
        ----------
        filename : str
            Path to family .html file to open.
        findphrase : str, optional
            Specific phrase to find within the location string.
        percent : float, optional
            Minimum percentage of matches required to add a family to the
            list; 90 by default.

        """
        self.percent = percent
        self.findphrase = findphrase
        self.fnum = int(os.path.basename(filename).split('.')[0])
        self.count_dict = dict.fromkeys(['regional', 'tele', 'local',
                                         'findphrase', 'total'], 0)
        with open(filename, 'r', encoding='utf-8') as file:
            data = file.read()
            self.count_dict['regional'] = data.count('regional')
            self.count_dict['tele'] = data.count('teleseismic')
            self.count_dict['local'] = data.count('Potential local match:')
            if self.findphrase:
                self.count_dict['findphrase'] = data.count(self.findphrase)
            self.count_dict['total'] = (self.count_dict['regional']
                                        + self.count_dict['tele']
                                        + self.count_dict['local'])

    def get_percent(self, kind):
        """
        Get the percentage of a certain 'kind' of match out of the total.

        Parameters
        ----------
        kind : str
            Phrase corresponding to the 'kind' of earthquake. See global
            variable DISTANT_TYPES for supported types.

        Returns
        -------
        float
            Percentage of lines that match the 'kind' specified.

        """
        if self.count_dict['total'] > 0:
            if kind == 'distant':
                return 100*(
                    (self.count_dict['regional'] + self.count_dict['tele'])
                    / self.count_dict['total'])
            if kind == 'regional':
                return 100*self.count_dict['regional']/self.count_dict['total']
            if kind == 'tele':
                return 100*self.count_dict['tele']/self.count_dict['total']
            if kind == 'findphrase':
                return 100*(self.count_dict['findphrase']
                            / self.count_dict['total'])
            if (kind == 'regional_notele') and (
                    (self.count_dict['total'] - self.count_dict['tele']) > 0):
                return 100*(
                    self.count_dict['regional']
                    / (self.count_dict['total'] - self.count_dict['tele']))
        return 0  # pragma: no cover

    def print_stats(self):
        """Print a line describing what was found."""
        found = (self.count_dict['total'] - self.count_dict['local']
                 + self.count_dict['findphrase'])
        if found > 0:
            print_str = (f'Family {self.fnum:4} : L '
                         f'{self.count_dict["local"]:2} | '
                         f'R {self.count_dict["regional"]:2} | T '
                         f'{self.count_dict["tele"]:2}')
            if self.findphrase:
                print_str += f' | F {self.count_dict["findphrase"]:2}'
            print_str += f' | Distant {self.get_percent("distant"):5.1f}%'
            if self.findphrase:
                print_str += f' | "{self.findphrase}" '
                print_str += f'{self.get_percent("findphrase"):5.1f}%'
            print(print_str)

    def append_fam(self, fam_string, kind):
        """
        Append family number to a string if it meets the criteria.

        Parameters
        ----------
        fam_string : str
            String to append family number to.
        kind : str
            Phrase corresponding to the 'kind' of earthquake. See global
            variable DISTANT_TYPES for supported types.

        Returns
        -------
        str
            Modified fam_string.

        """
        if kind == 'regional3':
            if self.count_dict['regional'] >= 3:  # pragma: no cover
                fam_string += f' {self.fnum}'
        else:
            if self.get_percent(kind) >= self.percent:
                fam_string += f' {self.fnum}'
        return fam_string


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
    dict
        Modified dictionary of locations.

    """
    locs['lats'] = np.append(locs['lats'], float(line.split()[4].strip('(,')))
    locs['lons'] = np.append(locs['lons'], float(line.split()[5].strip(')')))
    locs['deps'] = np.append(locs['deps'], float(line.split()[6].strip('km')))
    return locs


def calculate_arrivals(detector, catalog, phase_list, time_column_name='Time'):
    """
    Calculate the arrivals of given phases from source to study area.

    Traces rays through a very simple model (iasp91), then appends the times
    of arrivals to the end of the DataFrame.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    catalog : DataFrame object
        Event catalog, with columns for time and location.
    phase_list : list
        List of seismic phase arrivals to trace.
    time_column_name : str, optional
        Name of the event time column.

    Returns
    -------
    DataFrame object
        Event catalog, with arrival times appended.

    """
    for phase in phase_list:
        catalog[f'Arrival_{phase}'] = 'NaN'
    if len(catalog) > 0:
        taupymodel_runs = 0
        model = TauPyModel(model='iasp91')
        latitudes = np.squeeze(
            catalog.filter(regex='[lL]at.*').to_numpy(), axis=1)
        longitudes = np.squeeze(
            catalog.filter(regex='[lL]on.*').to_numpy(), axis=1)
        depths = np.squeeze(
            catalog.filter(regex='[dD]ep.*').to_numpy(), axis=1)
        for i in range(len(catalog)):
            deg = locations2degrees(latitudes[i], longitudes[i],
                                    detector.get('latitude_center'),
                                    detector.get('longitude_center'))
            # TauPyModel misbehaves if it's used too much
            # Determine if it needs to be reloaded (every 100 runs)
            taupymodel_runs += 1
            if np.remainder(taupymodel_runs, 100) == 0:
                model = TauPyModel(model='iasp91')
            arrivals = model.get_travel_times(
                source_depth_in_km=max(0, depths[i]),
                distance_in_degree=deg, phase_list=phase_list)
            if len(arrivals) > 0:
                for arrival in arrivals:
                    atime = (UTCDateTime(catalog[time_column_name][i])
                             + arrival.time)
                    catalog.loc[i, f'Arrival_{arrival.name}'] = f'{atime}'
    return catalog


def compare_catalog(detector, catalog, arrival=False, delimiter=',',
                    include_missing=True, junk=False, maxdtoffset=-1.,
                    name='Time', outfile=''):
    """
    Compare an independent text catalog with a REDPy catalog.

    The input catalog is appended with the best matching event, with columns
    for times, time difference (negative being that REDPy triggered early),
    which family or trigger type the best event corresponds to, and for
    repeaters, the frequency index and amplitudes. The combined catalog may
    be saved.

    Called by redpy.Detector.locate('compare', ...)

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    catalog : str or DataFrame object
        Catalog to compare with; either a file name to be read or a
        DataFrame.
    arrival : bool, optional
        If True, use P-wave arrival as match time.
    delimiter : str, optional
        Delimiter between columns of the catalog.
    include_missing : bool, optional
        If True, append events that were not found in catalog.
    junk : bool, optional
        If True, include possible matches with junk triggers.
    maxdtoffset : float, optional
        Maximum time offset in seconds to be considered a match. If
        negative, reverts to the default of 75% of the correlation window
        length, which is also the minimum time between consecutive triggers.
    name : str, optional
        Name of the 'Time' column in catalog.
    outfile : str, optional
        Path and filename to write output catalog to.

    Returns
    -------
    DataFrame object
        Original catalog from catalog with matches appended at right and,
        optionally, missing events appended below.

    """
    df = detector.to_dataframe(junk)
    mask = np.full(len(df), True)
    if maxdtoffset < 0:  # pragma: no cover
        maxdtoffset = detector.get('mintrig')
    if isinstance(catalog, str):
        catalog = pd.read_csv(catalog, sep=delimiter)
    if arrival:  # pragma: no cover
        catalog = handle_arrivals(detector, catalog, name,
                                  write_to_column='Match Time')
        catalog['Match Time'] = pd.to_datetime(catalog['Match Time'], utc=True)
    else:
        catalog['Match Time'] = pd.to_datetime(catalog[name], utc=True)
    catalog['Trigger Time'] = ''
    catalog['Trigger Time'] = pd.to_datetime(catalog['Trigger Time'], utc=True)
    catalog['dt (s)'] = ''
    catalog['Family'] = ''
    catalog['FI'] = ''
    catalog['Amplitudes'] = ''
    if detector.get('verbose'):
        print('Matching...')
    for i in range(len(catalog)):
        if detector.get('verbose'):
            if i % 1000 == 0 and i > 0:  # pragma: no cover
                print(f'{100.0*i/len(catalog):3.2f}% complete')
        time_delta = df['Trigger Time'] - catalog['Match Time'][i]
        idx = np.argmin(np.abs(time_delta))
        if np.abs(time_delta[idx].total_seconds()) <= maxdtoffset:
            catalog.loc[i, ['Trigger Time', 'Family',
                            'FI', 'Amplitudes']] = df.loc[idx]
            catalog.loc[i, 'dt (s)'] = time_delta[idx].total_seconds()
            mask[idx] = False
    if include_missing:
        catalog = pd.concat([catalog, df.iloc[np.where(mask)[0]]])
    catalog.reset_index(drop=True, inplace=True)
    if outfile:
        if detector.get('verbose'):
            print(f'Saving to {outfile}')
        catalog.to_csv(outfile, index=False,
                       date_format='%Y-%m-%dT%H:%M:%S.%fZ')
    return catalog


def distant_families(detector, findphrase='', percent=90.):
    """
    Find families with distant catalog matches by parsing their .html files.

    Prints family numbers that match the criteria that can then be copied as
    arguments to other removal functions, while also allowing the user to
    vet those matches prior to removal.

    Called by redpy.Detector.locate('distant', ...)

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    findphrase : str, optional
        Specific phrase to find within the location string.
    percent : float, optional
        Minimum percentage of matches required to add a family to the list;
        90% by default.

    Returns
    -------
    dict
        Dictionary keyed by distant 'type' with numpy arrays of family
        numbers that match the criteria. See global variable DISTANT_TYPES
        for list of types.

    """
    fam_dict = dict.fromkeys(DISTANT_TYPES.keys(), '')
    flist = np.array(glob.glob(os.path.join(detector.get('output_folder'),
                                            'families', '*.html')))
    fnums = [int(os.path.basename(fname).split('.')[0]) for fname in flist]
    for fname in flist[np.argsort(fnums)]:
        counter = DistantCounter(fname, findphrase, percent)
        if counter.count_dict['total'] > 0:
            if detector.get('verbose'):
                counter.print_stats()
            for key in fam_dict:
                fam_dict[key] = counter.append_fam(fam_dict[key], key)
    for (key, fam_str) in fam_dict.items():
        if key == 'regional3':
            print(f'{DISTANT_TYPES[key]}: \n{fam_str}\n')
        elif (key == 'findphrase') and findphrase:
            print(f'{percent}%+ {DISTANT_TYPES[key]} "{findphrase}": \n'
                  f'{fam_str}\n')
        else:
            print(f'{percent}%+ {DISTANT_TYPES[key]}:\n{fam_str}\n')
        fam_dict[key] = np.fromstring(fam_str, dtype=int, sep=' ')
    return fam_dict


def event_times_from_catalog(
        detector, catalog, time_column_name, start_time=None, end_time=None,
        arrivals=False, delimiter=','):
    """
    Read event times from a csv-like catalog.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    catalog : str or DataFrame object
        Catalog to pull times from; either a file name to be read or a
        DataFrame.
    time_column_name : str
        Name of the column that contains event times.
    start_time : UTCDateTime object, optional
        Events returned must have happened after this date.
    end_time : UTCDateTime object, optional
        Events returned must have happened before this date.
    arrivals : bool, optional
        Use P-wave arrival time instead of origin time.
    delimiter : str, optional
        Delimiter (or separating character) between columns.

    Returns
    -------
    ndarray of UTCDateTime objects
        List of event times.

    """
    if isinstance(catalog, str):
        catalog = pd.read_csv(catalog, sep=delimiter)
    if arrivals:
        handle_arrivals(detector, catalog, time_column_name)
    event_list = np.array(
        [UTCDateTime(event) for event in catalog[time_column_name]])
    event_list.sort()
    if start_time:
        event_list = event_list[event_list >= start_time]
    if end_time:
        event_list = event_list[event_list <= end_time]
    return event_list


def get_catalog(detector, csvfile, arrival=False, query=False,
                delimiter=',', name='Time', starttime=None, endtime=None):
    """
    Get a catalog of event times from a local file or query from the web.

    Always saves the file from the web locally, and can optionally calculate
    arrival times.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    csvfile : str
        Name of catalog csv file to read or save to.
    arrival : bool, optional
        Calculate and use P-wave arrival to center of network.
    query : bool, optional
        Query an external webservice with parameters in configfile.
    delimiter : str, optional
        Custom delimiter between columns in csvfile.
    name : str, optional
        Custom name of event time column.
    starttime : str, optional
        Subsets catalog to begin at this time.
    endtime : str, optional
        Subsets catalog to end at this time.

    Returns
    -------
    ndarray of UTCDateTime objects
        List of event times.

    """
    if query:
        query_arrivals(detector, starttime, endtime, outfile=csvfile)
    try:
        event_list = event_times_from_catalog(
            detector, csvfile, name, starttime, endtime,
            arrival, delimiter)
    except KeyError as exc:
        raise KeyError(
            f'Could not find "{name}" column in {csvfile}. Check file, '
            'column name, and delimiter!') from exc
    return event_list


def get_median_locations(detector, regional=False, distant=False):
    """
    Parse html output files for locations and create a DataFrame of medians.

    Called by redpy.Detector.locate('median', ...)

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    regional : bool, optional
        If True, include regional matches.
    distant : bool, optional
        If True, include regional and teleseismic matches.

    Returns
    -------
    DataFrame object
        Rows by family with columns for median latitude, longitude, and
        depth, as well as total number of members and number located.

    """
    flist = np.array(glob.glob(os.path.join(detector.get('output_folder'),
                                            'families', '*.html')))
    fnums = [int(os.path.basename(fname).split('.')[0]) for fname in flist]
    df = pd.DataFrame(columns=['Latitude', 'Longitude', r'Depth/km',
                               r'#Members', r'#Located'],
                      index=range(max(fnums))).rename_axis(index='Family')
    for fam in np.argsort(fnums):
        with open(flist[fam], 'r', encoding='utf-8') as famfile:
            data = famfile.readlines()
        famfile.close()
        locs = {'lats': np.array([]),
                'lons': np.array([]),
                'deps': np.array([])}
        lines = ' '.join(data).split('>')
        nmem = 0
        for line in lines:
            line = line.strip()
            if line.count('Potential local match:'):
                locs = append_location(locs, line)
            elif line.count('regional') and (distant or regional):
                locs = append_location(locs, line)  # pragma: no cover
            elif line.count('teleseismic') and distant:
                locs = append_location(locs, line)  # pragma: no cover
            elif line.count('Number of events:'):
                nmem = int(line.split(': ')[1][:-4])
        if len(locs['lats']):
            df.loc[fnums[fam]] = [
                np.median(locs['lats']), np.median(locs['lons']),
                np.median(locs['deps']), nmem, len(locs['lats'])]
        else:
            df.loc[fnums[fam]] = [np.nan, np.nan, np.nan, nmem, 0]
    return df.astype({r'#Members': int, r'#Located': int})


def handle_arrivals(detector, catalog, time_column_name, write_to_column=None):
    """
    Handle getting the P-wave arrivals into a single column.

    If arrival column doesn't exist in the catalog, it is calculated. By
    default, the combined arrival column overwrites the 'Time' column, but
    can be appended to the end by using the 'write_to_column' option.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    catalog : DataFrame object
        Catalog with event times and locations.
    time_column_name : str
        Name of the event time column.
    write_to_column : str, optional
        Column name to overwrite or append.

    Returns
    -------
    DataFrame object
        Modified catalog.

    """
    if 'Arrival_p' not in catalog.columns:  # pragma: no cover
        catalog = calculate_arrivals(detector, catalog, ['p', 'P'],
                                     time_column_name)
    if write_to_column:
        time_column_name = write_to_column
    catalog[time_column_name] = catalog['Arrival_p'].replace(
        to_replace='NaN', value=np.nan).fillna(catalog['Arrival_P'])
    return catalog


def match_external(detector, rtimes, external_catalogs, matchstring):
    """
    Find matches to repeater times within external catalogs.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    rtimes :
        Trigger times for repeaters to match.
    external_catalogs : DataFrame list
        List of catalogs, one for each region (local, regional, teleseismic).
    matchstring : str
        String to append matches to.

    Returns
    -------
    int
        Number of matches found.
    str
        Modified 'matchstring' with matches appended.
    dict
        Dictionary containing locations of local matches.

    """
    nfound = 0
    local = dict.fromkeys(['lats', 'lons', 'deps'], np.array([]))
    for ev_time in [UTCDateTime(i) for i in rtimes]:
        cflag = 0
        for j, cat in enumerate(external_catalogs):
            # Get the arrival names only from Arrival_ column names
            anames = [cat.filter(like='Arrival').columns[i].split('_')[1]
                      for i in range(len(cat.filter(like='Arrival').columns))]
            arrivals = cat.filter(like='Arrival').to_numpy().astype(str)
            matched = np.any(
                ((arrivals >= format(ev_time-detector.get('serr'))) & (
                    arrivals <= format(ev_time+detector.get('serr')))), axis=1)
            found = arrivals[matched]
            if len(found) > 0:
                cflag, matchstring, local = _write_matches(
                    j, anames, ev_time, matched, cat, found, cflag,
                    matchstring, local)
                nfound += 1
    return (nfound, matchstring, local)


def prepare_catalog(detector):
    """
    Download and format event catalog from external datacenter.

    Data are queried from three regions (local, regional, teleseismic) based
    on the settings in the configuration. Times are taken from the first and
    last trigger times in ttable, so if there are large gaps in ttable, this
    function is agnostic to them. Updates the catalog if a file exists to
    reduce query overhead.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    Returns
    -------
    list of DataFrame objects
        Catalogs for local, regional, and teleseismic events.

    """
    ttimes = detector.get('ttable', 'startTimeMPL')
    tmin = UTCDateTime(mdates.num2date(np.min(ttimes))) - 1800
    tmax = UTCDateTime(mdates.num2date(np.max(ttimes))) + 30
    external_catalogs = []
    for region in REGIONS:
        fname = os.path.join(detector.get('output_folder'),
                             f'external_{region}.txt')
        if os.path.exists(fname):  # pragma: no cover
            catalog = pd.read_csv(fname, delimiter='|')
            # Get missing events before and after currently saved events
            if len(catalog) > 0:
                tmin_catalog = UTCDateTime(np.min(catalog['Time']))-1
                tmax_catalog = UTCDateTime(np.max(catalog['Time']))+1
                catalog_before = query_external(detector, region, tmin,
                                                tmin_catalog)
                catalog_after = query_external(detector, region, tmax_catalog,
                                               tmax)
                catalog = pd.concat([catalog_before, catalog, catalog_after],
                                    axis=0, ignore_index=True)
            else:
                catalog = query_external(detector, region, tmin, tmax)
        else:
            catalog = query_external(detector, region, tmin, tmax)
        catalog.to_csv(fname, index=False, sep='|')
        external_catalogs.append(catalog)
    return external_catalogs


def query_arrivals(detector, tmin, tmax, outfile=None):
    """
    Query an external event catalog for a list of P-wave arrival times.

    Basically combines query_external() and handle_arrivals(), and returns
    an event_list that can be used in Detector.update().

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    tmin : UTCDateTime object, str
        Start time for catalog query.
    tmax : UTCDateTime object, str
        End time for catalog query.
    outfile : str, optional
        Where to save the catalog to file.

    Returns
    -------
    str list
        List of P-wave arrival times for local events.

    """
    tmin = UTCDateTime(tmin)
    tmax = UTCDateTime(tmax)
    catalog = query_external(detector, 'local', tmin, tmax, True)
    catalog = handle_arrivals(detector, catalog, 'Time', 'Arrival')
    if outfile:
        catalog.to_csv(outfile, index=False)
    return list(catalog['Arrival'])


def query_external(detector, region, tmin, tmax, arrivals=True):
    """
    Query and format an external event catalog using a web service.

    Currently only supports querying datacenters that support the default
    FDSN 'text' format, with columns separated by | instead of commas.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    region : str
        String describing which of the three distance regions to use.
    tmin : UTCDateTime object
        Start time for catalog query.
    tmax : UTCDateTime object
        End time for catalog query.
    arrivals : bool, optional
        If True, calculate phase arrivals.

    Returns
    -------
    DataFrame object
        Event catalog.

    """
    if region == 'local':
        minrad = 0
        maxrad = detector.get('locdeg')
        minmag = -10
        phase_list = ['p', 's', 'P', 'S']
    elif region == 'regional':
        minrad = detector.get('locdeg')
        maxrad = detector.get('regdeg')
        minmag = detector.get('regmag')
        phase_list = ['p', 's', 'P', 'S', 'PP', 'SS']
    else:
        minrad = detector.get('regdeg')
        maxrad = 180
        minmag = detector.get('telemag')
        phase_list = ['P', 'S', 'PP', 'SS', 'PcP', 'ScS', 'PKiKP', 'PKIKP']
    base_url = Client(detector.get('datacenter')).base_url
    query_url = base_url + ('/fdsnws/event/1/query'
                            + f'?starttime={tmin}'
                            + f'&endtime={tmax}'
                            + f'&latitude={detector.get("latitude_center")}'
                            + f'&longitude={detector.get("longitude_center")}'
                            + f'&maxradius={maxrad}'
                            + f'&minradius={minrad}'
                            + f'&minmagnitude={minmag}'
                            + '&orderby=time-asc&format=text&limit=10000')
    try:
        catalog = pd.read_csv(query_url, delimiter='|')
        # If the limit is returned
        if len(catalog) == 10000:  # pragma: no cover
            offset = 0
            while not len(catalog) % 10000:
                offset += 10000
                catalog2 = pd.read_csv(query_url+f'&offset={offset}',
                                       delimiter='|')
                if len(catalog2) > 0:
                    catalog = catalog.append(catalog2, ignore_index=True)
                else:  # Remainder will still be 0 so we'd be stuck in the loop
                    break
    except Exception as exc:  # pragma: no cover
        # Pass an empty dataframe with the correct columns
        catalog = pd.DataFrame(
            columns=['EventID', 'Time', 'Latitude', 'Longitude', 'Depth/km',
                     'Magnitude', 'EventLocationName'])
        print((f'Failed to download {region} event catalog from '
               f'{detector.get("datacenter")} - {exc}'))
    catalog.columns = catalog.columns.str.replace(' ', '')
    catalog.columns = catalog.columns.str.replace('#', '')
    catalog.drop(
        columns=['Author', 'Catalog', 'Contributor', 'ContributorID',
                 'MagType', 'MagAuthor', 'EventType'],
        errors='ignore', inplace=True)
    if arrivals:
        catalog = calculate_arrivals(detector, catalog, phase_list)
    return catalog


def save_external_catalog(detector, csvfile, arrivals=False, start_time=None,
                          end_time=None, delimiter=','):
    """
    Download and save catalog from external FDSN webservice to csvfile.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    csvfile : str
        File to save catalog to.
    arrivals : bool, optional
        If True, calculate P-wave arrival to center of network.
    starttime : str, optional
        Start time for catalog query.
    endtime : str, optional
        End time for catalog query.
    delimiter : str, optional
        Delimiter to use between columns in output .csv file.

    Returns
    -------
    DataFrame object
        Catalog of local events.

    """
    if not end_time:
        end_time = UTCDateTime()
        print(f'Defaulting to end time of "now" ({end_time})')
    if not start_time:
        start_time = end_time - detector.get('nsec')
        if detector.get('rtable').table.attrs.ptime:  # pragma: no cover
            start_time = detector.get('rtable').table.attrs.ptime
        print(f'Defaulting to start time of {start_time}')
    catalog = query_external(detector, 'local', UTCDateTime(start_time),
                             UTCDateTime(end_time), arrivals)
    if len(catalog) == 0:
        print('No events found!')
    catalog.to_csv(csvfile, index=False, sep=delimiter)
    return catalog


def _write_matches(j, anames, ev_time, matched, cat, found, cflag,
                   matchstring, local):
    """Append best matches to string, return updated local dictionary."""
    # Convert from strings to time differences
    vfunc = np.vectorize(lambda x, ev_time: np.abs(UTCDateTime(x) - ev_time))
    found[found == 'NaN'] = 'nan'  # Just in case
    found[found != 'nan'] = vfunc(found[found != 'nan'], ev_time)
    # Convert to float, otherwise numpy complains
    found[found == 'nan'] = np.nan
    found = found.astype(float)
    for _ in range(len(found)):
        bestmatch = np.argwhere(found == np.nanmin(found))[0]
        catmatch = cat[matched].iloc[bestmatch[0]]
        matchstring += (
            f'{_PRESTRING[j]}{_PC_STRING[cflag]} '
            f'{REGIONS[j]} match: '
            f'{catmatch["Time"]} ({catmatch["Latitude"]:5.6f}, '
            f'{catmatch["Longitude"]:6.6f}) '
            f'{catmatch["Depth/km"]:3.1f}km '
            f'M{catmatch["Magnitude"]:3.1f} - '
            f'{catmatch["EventLocationName"]} - '
            f'({anames[bestmatch[1]]}) '
            f'{np.nanmin(found):4.2f} s{_POSTSTRING[j]}')
        if j == 0:  # Local catalog
            local['lats'] = np.append(local['lats'],
                                      catmatch['Latitude'])
            local['lons'] = np.append(local['lons'],
                                      catmatch['Longitude'])
            local['deps'] = np.append(local['deps'],
                                      catmatch['Depth/km'])
        found[bestmatch[0], :] = np.nan
        # Set cflag = 1 for 'conflicting'
        cflag = 1
    return (cflag, matchstring, local)
