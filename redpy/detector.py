# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling REDPy Detector() object.

The Detector() object holds all of the references to configurations, data
stored on disk for a single run, and has methods to update or create
outputs for those runs. This object is the primary user interface for REDPy.
"""
import os
import sys
from copy import deepcopy

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from scipy.sparse import coo_matrix
from tables import open_file

import redpy.locate
import redpy.output
from redpy.config import Config
from redpy.table import Table


_UNKNOWN_METHOD = ("Unknown 'method' given. Options are {}; see "
                   "documentation for help.")


class Detector():
    """
    Primary interface for handling detections.

    Attributes
    ----------
    config : Config object
        Container for all configuration parameters describing a single run.
    h5file : PyTables File object
        In-memory representation of the hdf5 file on disk.
    tables : dict
        Dictionary containing a Table object for each table type.
    """

    def __init__(self, configfile='settings.cfg', verbose=False,
                 opened=False, troubleshoot=False):
        """
        Load configuration file and set attribute structure.

        Parameters
        ----------
        configfile : str, optional
            Name of configuration file to read.
        verbose : bool, optional
            Enable additional print statements.
        opened : bool, optional
            If True, return opened.
        troubleshoot : bool, optional
            Escape try/except statements to diagnose problems.

        """
        np.set_printoptions(threshold=sys.maxsize)
        np.set_printoptions(linewidth=sys.maxsize)
        self.config = Config(configfile, verbose, troubleshoot)
        self.h5file = None
        self.tables = {}
        self.plotvars = {}
        self.waveforms = {}
        if opened:
            self.open()
        self._check_output_folders()

    def __len__(self):
        """Define length as length of Families table."""
        return len(self.get('ftable'))

    def __str__(self):
        """Define print string."""
        if self.tables:
            return '\n'.join([str(self.get(table)) for table in self.tables])
        return 'Tables are not open; use .open() or .initialize()'

    def __repr__(self):
        """Define representation string."""
        string = f'redpy.Detector(configfile="{self.get("configfile")}"'
        if self.get('verbose'):
            string += ', verbose=True'
        if self.get('troubleshoot'):
            string += ', troubleshoot=True'
        if self.tables:
            string += ', opened=True'
        string += ')'
        return self.config.append_custom(string)

    def close(self):
        """Gracefully close the tables and empty variables in memory."""
        if self.tables:
            self.h5file.close()
            self.tables = {}
            self.plotvars = {}
            self.waveforms = {}

    def expand(self, config_to, update_outputs=False):
        """
        Copy the contents of the current hdf5 file into an expanded one.

        A limited number of checks are made on the destination's
        configuration. It can be very similar or very different from the
        current configuration, and the user is warned that this method is
        simultaneously powerful and naive. Note that the current Detector
        object must be closed, and all columns in memory will be lost.

        Parameters
        ----------
        config_to : Config object
            Configuration of the destination run.
        update_outputs : bool, optional
            If True, encourages all outputs to be updated.

        """
        # Enforce largest max_famlen
        config_to.set('max_famlen', max(self.get('max_famlen'),
                                        config_to.get('max_famlen')))
        self.close()
        detector_from = deepcopy(self)
        self.config = config_to
        if self.get('filename') == detector_from.get('filename'):
            if self.get('verbose'):
                print(f'Copying {self.get("filename")} to '
                      f'{self.get("filename")}.tmp')
            os.rename(self.get('filename'), f"{self.get('filename')}.tmp")
            detector_from.set('filename', f"{self.get('filename')}.tmp")
        self.initialize()
        detector_from.open()
        self._copy_contents(detector_from, update_outputs)
        detector_from.close()
        if detector_from.get('filename').endswith('.tmp'):
            if self.get('verbose'):
                print(f'Removing {detector_from.get("filename")}')
            os.remove(detector_from.get('filename'))

    def get(self, key, col=None, row=None):
        """
        Get various items within the object.

        This function allows the user to more easily navigate the complex
        structure of this object.

        Parameters
        ----------
        key : str
            Either the name of an attribute of this object, a configuration
            setting, or a Table name (e.g., 'rtable').
        col : str or int, optional
            Column name or index; only supported if key is a Table name.
            Only supports a single column at a time.
        row : int or int ndarray, optional
            Row index or slice; only supported if key is a Table name.

        Returns
        -------
        varies
            If key is an attribute of the Detector object, returns the
            value of that attribute. If key is a configuration setting
            or an attribute of the Config object, returns the value of that
            setting or attribute. If a Table name is given without row or
            column, returns the Table object in full. Otherwise, if col
            is specified without a row, the entire column within that table
            is returned. If row is specified without col, full row(s) within
            the table as a numpy array are returned. If both col and row are
            specified, the column is sliced with row.

        """
        if key in ['config', 'h5file', 'tables', 'plotvars', 'waveforms']:
            return getattr(self, key)
        if 'table' in key:
            if not self.tables:
                self.open()
            return self.tables[key].get(col, row)
        return self.config.get(key)

    def get_matrix(self):
        """
        Return the contents of the Correlation table as a sparse matrix.

        Returns
        -------
        int ndarray
            'id' column from Repeaters table.
        float csr_matrix
            Sparse correlation matrix with id as rows/columns.

        """
        ids = self.get('rtable', 'id')
        id1 = self.get('ctable', 'id1')
        id2 = self.get('ctable', 'id2')
        ccc = self.get('ctable', 'ccc')
        maxid = np.max((np.max(ids), np.max(id2)))+1
        all_ids = np.vstack([id1, id2]).T.copy()
        dtypes = all_ids.dtype.descr * 2
        i = np.unique(all_ids.view(dtypes), return_index=True)[1]
        ccc_sparse = coo_matrix((ccc[i], (id1[i], id2[i])),
                                shape=(maxid, maxid)).tocsr()
        return ids, ccc_sparse

    def get_members(self, fnum):
        """
        Get the members of a family as an array.

        Parameters
        ----------
        fnum : int
            Family number to query.

        Returns
        -------
        int ndarray
            Rows of 'rtable' that are members of this family.

        """
        return np.fromstring(
            self.get('ftable', 'members', fnum), dtype=int, sep=' ')

    def get_small_families(self, minmembers=5, maxage=0, seedtime=''):
        """
        Search for old, small families.

        If verbose, prints a summary table.

        Parameters
        ----------
        minmembers : int
            Minimum number of members needed to keep a family.
        maxage : float, optional
            Maximum age relative to seedtime (days) measured from the most
            recent member of the family.
        seedtime : str, optional
            Date from which to measure maxage; defaults to the latest
            trigger time.

        Returns
        -------
        int ndarray
            List of family numbers that met the criteria.

        """
        if seedtime:
            seedtime = UTCDateTime(seedtime)
        else:
            seedtime = UTCDateTime(
                mdates.num2date(self.get('ttable', 'startTimeMPL', -1)))
        if self.get('verbose'):
            print('\n::: .get_small_families()')
            print(f'::: - minmembers    : {minmembers}')
            print(f'::: - maxage (days) : {maxage}')
            print(f'::: - seedtime      : {seedtime}\n')
            print('::: Member count per Family :::')
            print(f'#{"Family #":>12s} | {"Members":>12s} | '
                  f'{"Age (d)":>12s} | {"Fate":<12s}')
        small_families = []
        i = 0
        rtimes = self.get('rtable', 'startTimeMPL')
        for fnum in range(len(self.get('ftable'))):
            fate = 'keep'
            members = self.get_members(fnum)
            nmembers = len(members)
            age = (seedtime - (UTCDateTime(mdates.num2date(
                np.max(rtimes[members]))))) / 86400  # Gets most recent member
            if nmembers < minmembers and age > maxage:
                small_families.append(fnum)
                fate = 'REMOVE'
                i += nmembers
            if self.get('verbose'):
                print(f'#{fnum:>12d} | {nmembers:12d} | {age:>12.2f} |  '
                      f'{fate:<12s}')
        small_families = np.array(small_families)
        if self.get('verbose'):
            print(f'\nSmall families : {" ".join(small_families.astype(str))}')
            print(('# Families     : '
                   f'{len(small_families)}/{fnum+1}'))
            percent_removed = i/len(rtimes)*100
            print(('# Repeaters    : '
                   f'{i}/{len(rtimes)} ({percent_removed:2.1f}%)\n'))
        return small_families

    def initialize(self):
        """Write empty hdf5 file to disk, overwriting any existing file."""
        if self.get('verbose'):
            if os.path.exists(self.get('filename')):
                print(
                    f'Overwriting existing hdf5 file: {self.get("filename")}')
            else:
                print(f'Initializing empty hdf5 file: {self.get("filename")}')
        self.h5file = open_file(
            self.get('filename'), mode='w', title=self.get('title'))
        group = self.h5file.create_group(
            r'/', self.get('groupname'), self.get('title'))
        for name in ['ctable', 'dtable', 'ftable', 'jtable', 'otable',
                     'rtable', 'ttable']:
            self.tables[name] = Table(name)
            self.tables[name].initialize(self.h5file, group, self.config)
        self._create_folder()
        self._create_folder('families')

    def locate(self, method, *args, **kwargs):
        """
        Associate REDPy detections with locations from an external catalog.

        Currently, three methods are supported, of which two rely on the
        configuration parameter 'checkcomcat' to be True, as they parse the
        .html output files.

        'catalog':
            calls redpy.locate.compare_catalog()
        'distant':
            calls redpy.locate.distant_families()
        'median':
            calls redpy.locate.get_median_locations()

        See the documentation for those functions for a full explanation of
        the required arguments and format of outputs.

        Parameters
        ----------
        method : str
            Controls which method of locate to apply.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        dict or DataFrame object
            If using method 'distant' returns a dictionary, else returns a
            DataFrame. Format of returned DataFrame varies by method.

        Raises
        ------
        ValueError
            If method is not recognized.

        """
        if method == 'catalog':
            return redpy.locate.compare_catalog(self, *args, **kwargs)
        if method == 'distant':
            return redpy.locate.distant_families(self, *args, **kwargs)
        if method == 'median':
            return redpy.locate.get_median_locations(self, *args, **kwargs)
        raise ValueError(_UNKNOWN_METHOD.format(
            "'catalog', 'distant', or 'median'"))

    def open(self):
        """Open connection to hdf5 file and populate Table links."""
        if self.get('verbose'):
            print(f'Opening hdf5 file: {self.get("filename")}')
        self.h5file = open_file(self.get('filename'), 'a')
        for name in ['ttable', 'otable', 'rtable', 'ftable', 'jtable',
                     'dtable', 'ctable']:
            self.tables[name] = Table(name)
            self.tables[name].open(self.h5file, self.config)
        self._check_max_famlen()
        if self.get('verbose'):
            print(self)

    def output(self, method=None, **kwargs):
        """
        Generate outputs as dictated.

        A variety of methods are supported, which alter the functionality in
        some way. Many utilize at least one keyword argument.

        None:
            calls redpy.output.generate() - without any arguments, creates
            the default outputs as dictated by changes to the tables.
        'force':
            calls redpy.output.force() - without any arguments, creates all
            default outputs from scratch. Otherwise, accepts keyword
            arguments from both redpy.output.set_print_cols() and
            redpy.output.generate().
        'junk':
            calls redpy.output.junk() - does not accept keyword arguments.
        'pdf_family':
            calls redpy.output.pdf_family() - must supply at least one
            argument 'fnum' with a list of family members to create .pdf
            versions of the family plots for.
        'pdf_timeline':
            calls redpy.output.pdf_timeline() - without any arguments,
            recreates the current full overview timeline as a .pdf.
        'report':
            calls redpy.output.create_report() - requires at least one
            argument 'fnum' with a list of family numbers to create reports
            for.

        See the documentation for those functions for a full explanation of
        the keyword arguments and format of outputs.

        Parameters
        ----------
        method : str, optional
            Controls which method of output to apply.
        **kwargs
            Arbitrary keyword arguments.

        Raises
        ------
        ValueError
            If method is not recognized.

        """
        if method is None:
            redpy.output.generate(self, **kwargs)
        elif method == 'force':
            redpy.output.force(self, **kwargs)
        elif method == 'junk':
            self._create_folder('junk')
            redpy.output.junk(self)
        elif method == 'pdf_family':
            redpy.output.pdf_family(self, **kwargs)
        elif method == 'pdf_timeline':
            redpy.output.pdf_timeline(self, **kwargs)
        elif method == 'report':
            self._create_folder('reports')
            redpy.output.report(self, **kwargs)
        else:
            raise ValueError(_UNKNOWN_METHOD.format(
                "'force', 'junk', 'pdf_family', 'pdf_timeline', or 'report'"))

    def remove(self, method, fnums=None, skip_dtable=False):
        """
        Remove items from further consideration.

        Currently, three methods are supported:

        'expire':
            remove orphans with expiration dates older than latest trigger
        'junk':
            empties the 'Junk' table of all triggers
        'family' or 'families':
            removes the family or families listed from the 'Families',
            'Repeater', and 'Correlation' tables, and by default moves the
            core of each family to the 'Deleted' table.

        Parameters
        ----------
        method : str
            Controls which method of remove to apply.
        fnums : int, int list, or int ndarray, optional
            Family or families to remove for 'family' method.
        skip_dtable : bool, optional
            If True, does not write cores to the Deleted table.

        Raises
        ------
        ValueError
            If method is not recognized or if no families are provided with
            the 'family'/'families' method.

        """
        if method == 'expire':
            print('deletes orphans with expiration date before latest trigger')
            print('not yet implemented')
        if method == 'junk':
            self.get('jtable').remove('all')
        elif method in ['family', 'families']:
            if fnums is not None:
                self._remove_families(fnums, skip_dtable)
            else:
                raise ValueError("Specify at least one family with 'fnum'")
        else:
            raise ValueError(_UNKNOWN_METHOD.format(
                "'expire', 'junk', 'family', or 'families'"))

    def set(self, key, value, col=None, row=None):
        """
        Update attributes, settings, or data in table.

        Parameters
        ----------
        key : str
            Either the name of an attribute of this object, a configuration
            setting, or a Table name (e.g., 'rtable').
        value : array_like
            Data to be written to the table for a single column. Should
            match the type and shape of the destination.
        col : str or int, optional
            Column name or index; only supported if key is a Table name.
            Only supports a single column at a time.
        row : int or int ndarray, optional
            Row index or slice; only supported if key is a Table name. If an
            integer, this refers to a single row, and thus value should be a
            single cell. If an array, these are row slices, and the length
            of value should match.

        """
        if key in ['config', 'h5file', 'tables']:
            setattr(self, key, value)
        if 'table' in key:
            self.tables[key].set(value, col, row)
        self.config.set(key, value)

    def stats(self):
        """Print lengths of the three most important tables."""
        if self.tables:
            print('\n'.join([str(self.get(table)) for table in [
                'otable', 'rtable', 'ftable']]))
        else:
            print(self)

    def to_dataframe(self, junk=False):
        """
        Create a DataFrame from triggers and repeaters on disk.

        Specifically, this DataFrame has columns for the trigger time,
        family number (or whether it is a trigger (former orphans or deleted
        events), current orphan, or categorized as junk), frequency index,
        and amplitudes on all stations.

        Parameters
        ----------
        junk : bool, optional
            If True, output junk triggers.

        Returns
        -------
        DataFrame object
            Tabular summary of all triggers and subset of metadata.

        """
        return self._build_dataframe(junk)

    def update(self):
        """Docstring when written."""
        print('.update()')

    def _build_dataframe(self, junk):
        """Build the dataframe representation of the detector."""
        if self.get('verbose'):
            print('Building DataFrame of events...')
        if junk:
            jdates = [UTCDateTime(j).matplotlib_date for j in self.get(
                'jtable', 'startTime')]
        else:
            jdates = []
        rtimes_mpl = self.get('rtable', 'startTimeMPL')
        fi_mean = np.nanmean(self.get('rtable', 'FI'), axis=1)
        amps = np.array([str(list(i)) for i in self.get(
            'rtable', 'windowAmp')])
        df = pd.DataFrame(
            columns=['Trigger Time', 'Family', 'FI', 'Amplitudes'],
            index=np.concatenate((self.get('ttable', 'startTimeMPL'),
                                  np.array(jdates))))
        df['Family'] = 'trigger'
        df['Family'][jdates] = 'junk'
        df['Family'][self.get('otable', 'startTimeMPL')] = 'orphan'
        df['Trigger Time'] = mdates.num2date(
            df.index + self.get('ptrig') / 86400)
        df.loc[rtimes_mpl, 'Trigger Time'] = mdates.num2date(
            rtimes_mpl + self.get('rtable', 'windowStart')
            / self.get('samprate') / 86400)
        for fnum in range(len(self.get('ftable'))):
            members = self.get_members(fnum)
            df.loc[rtimes_mpl[members], 'Family'] = fnum
            df.loc[rtimes_mpl[members], 'FI'] = fi_mean[members]
            df.loc[rtimes_mpl[members], 'Amplitudes'] = amps[members]
        df = df.astype({r'Family': str})
        df.sort_values(['Family', 'Trigger Time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _check_famlen(self):
        """
        Check if the string holding family members is too long.

        When the current maximum family string length exceeds 45% of the
        current allotted maximum length, the Families table needs to be
        expanded.

        Unfortunately, in order to do that we need to copy everything over
        into a new file. Additionally, we use 45% instead of something
        higher like 95% to allow for the edge case that the largest family
        and the second largest family with similar lengths are merged.

        It's still possible this is too generous based on how often the
        check is done.

        """
        if (self.get('ftable').table.attrs.current_max_famlen
                >= 0.45*self.get('ftable').table.attrs.allowed_max_famlen):
            if self.get('verbose'):
                print('Approaching maximum family length Families table '
                      r'can hold!\nAutomatically expanding to compensate...')
            config_to = self.get('config').copy()
            config_to.set('max_famlen',
                          3*self.get('ftable').table.attrs.allowed_max_famlen)
            self.expand(config_to)

    def _check_max_famlen(self):
        """Ensure the maximum family length setting is up to date."""
        allowed_max_famlen = self.get('ftable').table.attrs.allowed_max_famlen
        if allowed_max_famlen > self.get('max_famlen'):
            self.set('max_famlen', allowed_max_famlen)
            self.config.custom_settings.append('max_famlen')

    def _check_output_folders(self):
        """Rename 'clusters' output folder from older version to 'families'."""
        if os.path.isdir(os.path.join(self.get('output_folder'), 'clusters')):
            os.rename(os.path.join(self.get('output_folder'), 'clusters'),
                      os.path.join(self.get('output_folder'), 'families'))

    def _copy_contents(self, detector_from, update_outputs):
        """Copy the contents of one Detector() into this one."""
        dsta = self.get('nsta') - detector_from.get('nsta')
        if dsta < 0:
            raise ValueError(
                f'{self.get("configfile")} must have nsta >= '
                f'{detector_from.get("nsta")} (current: {self.get("nsta")})')
        if self.get('verbose'):
            print('Copying data... please wait...')
        for tname in self.get('tables'):
            self.get(tname).populate_from_table(
                detector_from.get(tname), self.config, dsta)
        if update_outputs:
            # !!! Replace with force
            self.set('ftable', np.ones(len(self.get('ftable'))), 'printme')
            self.output()

    def _create_folder(self, subfolder=None):
        """Create folder for outputs."""
        folder = self.get('output_folder')
        if subfolder:
            folder = os.path.join(folder, subfolder)
        if not os.path.isdir(folder):
            if self.get('verbose'):
                print(f"Creating output folder: '{folder}'")
            os.mkdir(folder)

    def _remove_families(self, fnums, skip_dtable=False):
        """
        Remove families from catalog.

        Specifically, it removes the families from the Families table,
        removes the cross-correlation values from the Correlation table for
        members of those families, moves the core of the families into the
        Deleted table, and removes the rest of the members from the Repeaters
        table.

        Parameters
        ----------
        fnums : int ndarray
            List of families to remove from the Families table.
        skip_dtable : bool, optional
            If True, do not append core to the Deleted table.

        """
        if self.get('verbose'):
            if isinstance(fnums, int):
                print(f'Removing family: {fnums}')
                fnums = np.array([fnums])
            else:
                fnums = np.array(fnums)
                print(f'Removing families: {fnums}')
        members = np.array([])
        for fnum in fnums:
            members = np.append(members, self.get_members(fnum))
        members = np.sort(members).astype('uint32')
        rtable_len = len(self.get('rtable'))
        ids = self.get('rtable', 'id')[members]
        id2 = self.get('ctable', 'id2')
        self.get('ctable').remove(np.where(np.in1d(id2, ids))[0])
        cores = np.array([])
        if not skip_dtable:
            cores = np.intersect1d(members, self.get('ftable', 'core'))
            for core in cores:
                self.get('rtable').move(self.get('dtable'), core)
        transform = np.zeros(rtable_len).astype(int)
        transform[np.delete(list(range(rtable_len)), members)] = range(
            rtable_len-len(members))
        members = np.setdiff1d(members, cores)
        self.get('rtable').remove(members)
        self.get('ftable').remove(fnums)
        for fnum in range(len(self.get('ftable'))):
            fmembers = self.get_members(fnum)
            core = self.get('ftable', 'core', fnum)
            self.set('ftable',
                     bytes(np.array2string(
                         transform[fmembers])[1:-1], 'utf-8'),
                     'members', fnum)
            self.set('ftable', transform[core], 'core', fnum)
        self.set('ftable', 1, 'printme', len(self.get('ftable'))-1)
        if self.get('verbose'):
            print('Done removing families!')
